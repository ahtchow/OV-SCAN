import numpy as np
import copy
from PIL import Image
from enum import Enum, IntEnum
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation
import torch
import open_clip

class BoxVisibility(IntEnum):
    """ Borrowed from NuScenes DevKit: Enumerates the various level of box visibility in an image """
    ALL = 0  # Requires all corners are inside the image.
    ANY = 1  # Requires at least one corner visible in the image.
    NONE = 2  # Requires no corners to be inside, i.e. box can be fully outside the image.

class Box:
    """ Simple data class representing a 3d box including, label, score and velocity. """

    def __init__(self, center, size, orientation, label=np.nan, score=np.nan,
                 velocity=(np.nan, np.nan, np.nan), name=None, token=None):
        """
        :param center: Center of box given as x, y, z.
        :param size: Size of box in width, length, height.
        :param orientation: Box orientation.
        :param label: Integer label, optional.
        :param score: Classification score, optional.
        :param velocity: Box velocity in x, y, z direction.
        :param name: Box name, optional. Can be used e.g. for denote category name.
        :param token: Unique string identifier from DB.
        """
        assert not np.any(np.isnan(center))
        assert not np.any(np.isnan(size))
        assert len(center) == 3
        assert len(size) == 3
        assert type(orientation) == Quaternion

        self.center = np.array(center)
        self.wlh = np.array(size)
        self.orientation = orientation
        self.label = int(label) if not np.isnan(label) else label
        self.score = float(score) if not np.isnan(score) else score
        self.velocity = np.array(velocity)
        self.name = name
        self.token = token

    def __eq__(self, other):
        center = np.allclose(self.center, other.center)
        wlh = np.allclose(self.wlh, other.wlh)
        orientation = np.allclose(self.orientation.elements, other.orientation.elements)
        label = (self.label == other.label) or (np.isnan(self.label) and np.isnan(other.label))
        score = (self.score == other.score) or (np.isnan(self.score) and np.isnan(other.score))
        vel = (np.allclose(self.velocity, other.velocity) or
               (np.all(np.isnan(self.velocity)) and np.all(np.isnan(other.velocity))))

        return center and wlh and orientation and label and score and vel

    def __repr__(self):
        repr_str = 'label: {}, score: {:.2f}, xyz: [{:.2f}, {:.2f}, {:.2f}], wlh: [{:.2f}, {:.2f}, {:.2f}], ' \
                   'rot axis: [{:.2f}, {:.2f}, {:.2f}], ang(degrees): {:.2f}, ang(rad): {:.2f}, ' \
                   'vel: {:.2f}, {:.2f}, {:.2f}, name: {}, token: {}'

        return repr_str.format(self.label, self.score, self.center[0], self.center[1], self.center[2], self.wlh[0],
                               self.wlh[1], self.wlh[2], self.orientation.axis[0], self.orientation.axis[1],
                               self.orientation.axis[2], self.orientation.degrees, self.orientation.radians,
                               self.velocity[0], self.velocity[1], self.velocity[2], self.name, self.token)

    def translate(self, x):
        """
        Applies a translation.
        :param x: <np.float: 3, 1>. Translation in x, y, z direction.
        """
        self.center += x

    def rotate(self, quaternion):
        """
        Rotates box.
        :param quaternion: Rotation to apply.
        """
        self.center = np.dot(quaternion.rotation_matrix, self.center)
        self.orientation = quaternion * self.orientation
        self.velocity = np.dot(quaternion.rotation_matrix, self.velocity)

    def corners(self, wlh_factor=1.0):
        """
        Returns the bounding box corners.
        :param wlh_factor: Multiply w, l, h by a factor to scale the box.
        :return: <np.float: 3, 8>. First four corners are the ones facing forward.
            The last four are the ones facing backwards.
        """
        w, l, h = self.wlh * wlh_factor

        # 3D bounding box corners. (Convention: x points forward, y to the left, z up.)
        x_corners = l / 2 * np.array([1,  1,  1,  1, -1, -1, -1, -1])
        y_corners = w / 2 * np.array([1, -1, -1,  1,  1, -1, -1,  1])
        z_corners = h / 2 * np.array([1,  1, -1, -1,  1,  1, -1, -1])
        corners = np.vstack((x_corners, y_corners, z_corners))

        # Rotate
        corners = np.dot(self.orientation.rotation_matrix, corners)

        # Translate
        x, y, z = self.center
        corners[0, :] = corners[0, :] + x
        corners[1, :] = corners[1, :] + y
        corners[2, :] = corners[2, :] + z

        return corners

    def copy(self):
        """
        Create a copy of self.
        :return: A copy.
        """
        return copy.deepcopy(self)

class AlignmentStatus(Enum):
    NO_OBJECTS_FROM_DETECTION = 0
    FIT_FOR_ALIGNMENT = 1
    HEAVILY_OCCLUDED = 2
    NO_OBJECT_OF_INTEREST = 3
    INCONSISTENT_SHAPE = 4 # when box ratio is off between GS and detection

class AlignmentSelector:

    def __init__(self, selection_config, data_root, tag="train"):

        self.data_root = data_root
        self.tag = tag
        self.SEARCH_SCALE = selection_config.SEARCH_SCALE # Search radius for matching
        self.BOX_SIMILARITY_THRESH = selection_config.BOX_SIMILARITY_THRESH # Threshold for box similarity during matching
        self.IMAGE_CROP_MARGIN = selection_config.IMAGE_CROP_MARGIN # Margin for cropping the image
        self.MAX_ALIGNMENT_RANGE = selection_config.MAX_ALIGNMENT_RANGE # Maximum object distance from lidar used for alignment
        self.MIN_BOX_IN_IMAGE_RATIO = selection_config.MIN_BOX_IN_IMAGE_RATIO # Minimum ratio of box in image for alignment
        self.CLIP_MODEL = selection_config.CLIP_MODEL # CLIP Tokenizer for Images
        self.CLIP_MODEL_TAG = selection_config.CLIP_MODEL_TAG # Tag for CLIP Model

        self.clip_model, _,  self.clip_preprocess = open_clip.create_model_and_transforms('hf-hub:apple/DFN5B-CLIP-ViT-H-14')
        self.clip_model.eval().cuda()
        self.object_crop_save_path = self.data_root / f'ov_alignment/{self.CLIP_MODEL_TAG}/object_crops/{self.tag}'
        self.object_crop_save_path.mkdir(parents=True, exist_ok=True)
        self.object_count = 0

        # Score threshold for grounding sam detections
        self.GROUNDING_SAM_THRESHOLDS = {
            "car": 0.35,
            "pedestrian": 0.35,
            "person sitting": 0.35,
            "van": 0.35,
            "truck": 0.35,
            "tram": 0.35,
            "person on a bicycle": 0.35,
            "person on a motorcycle": 0.35,
        }

    def process_predictions(self, gt_info):
        """
        Process the predictions and filter out the boxes that are not in the image

        Parameters:
            boxes (np.array): The predicted boxes
    
        Returns:
            list: The image crops
        """
        # Filter boxes based on distance
        boxes_3d = gt_info['annos']['gt_boxes_lidar']
        boxes_2d = gt_info['annos']['bbox']
        dist_filter_mask = [np.linalg.norm(box[:2]) < self.MAX_ALIGNMENT_RANGE for box in boxes_3d]
        keep_index = np.arange(len(boxes_3d))[dist_filter_mask]
        boxes_3d = boxes_3d[dist_filter_mask]
        boxes_2d = boxes_2d[dist_filter_mask]
    
        # Extract Image from GTs
        image_crops = []
        for i, box in enumerate(boxes_2d):
            image_crop_dict = {}
            image_crop_dict['box_index'] = i
            image_crop_dict['dist_from_ego'] = np.linalg.norm(boxes_3d[i][:2])
            xmin, ymin, xmax, ymax = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            image_crop_dict['corners_2d'] = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
            image_crop_dict['box'] = Box(boxes_3d[i][:3], boxes_3d[i][3:6],  Quaternion(Quaternion(axis=[0, 0, 1], angle=boxes_3d[i][6])))
            image_crop_dict['img_size'] = gt_info['image']['image_shape']
            image_crop_dict['box_in_image_ratio'] = self.get_box_in_image_ratio(image_crop_dict['corners_2d'], image_crop_dict['img_size'])
            image_crop_dict['img_path'] = f'{self.data_root}/{self.tag}/image_2/{gt_info["image"]["image_idx"]}.png'
            image_crops.append(image_crop_dict)

        return image_crops

    def get_box_in_image_ratio(self, corners_2d, imgsize):
        """
        Determine ratio between the box area and the image area

        Parameters:
            corners_2d (list): The 2D corners of the box
            imgsize (tuple): The image size
        
        Returns:
            float: The ratio between the box area and the image area
        """
        min_x, min_y = corners_2d[0]
        max_x, max_y = corners_2d[2]
        len_y, len_x = imgsize
        
        # Calculate the area of the box
        box_area = (max_x - min_x) * (max_y - min_y)
        
        # Determine the intersection coordinates
        inter_min_x = max(min_x, 0)
        inter_max_x = min(max_x, len_x)
        inter_min_y = max(min_y, 0)
        inter_max_y = min(max_y, len_y)
        
        # Check if the intersection is valid
        if inter_min_x >= inter_max_x or inter_min_y >= inter_max_y:
            return 0.0  # No intersection, the box is completely outside the image
        
        # Calculate the intersection area
        intersection_area = (inter_max_x - inter_min_x) * (inter_max_y - inter_min_y)
        
        # Calculate the percentage of the box within the image
        percentage_in_image = (intersection_area / box_area)
        return percentage_in_image

    def get_alignment_status(self, image_crops):
        """
        Get the alignment status for the image crops. The alignment status is based on the fitness of the object
        in the image crop. The fitness is determined by the percentage of the object that is visible in the image crop.
        Grounding SAM detections are used to determine the fitness of the object in the image crop.

        Parameters:
            image_crops (list): The image crops
            gs_info (dict): The grounding sam data

        Returns:
            list: The alignment status for the image crops
        """

        if len(image_crops) == 0:
            return []

        image_height, image_width = image_crops[0]['img_size']

        selection_results = []
        # Iterate through each image crop and match with the GS object
        for image_crop in image_crops:
            # Align all
            object_of_interest = {'object': None, 'crop_index': -1, 'status': AlignmentStatus.FIT_FOR_ALIGNMENT, 'align': True}

            # Get Cropped Image from 3D Box 
            (min_x, min_y), _, (max_x, max_y), _ = image_crop['corners_2d']
            min_x, min_y, max_x, max_y = int(min_x), int(min_y), int(max_x), int(max_y)
            min_x, min_y = max(0, min_x), max(0, min_y)
            max_x, max_y = min(image_width, max_x), min(image_height, max_y)
            object_crop = np.array(Image.open(image_crop['img_path']))[min_y:max_y, min_x:max_x]
            preprocessed_object_crop = self.clip_preprocess(Image.fromarray(object_crop)).unsqueeze(0).cuda()

            with torch.no_grad():
                object_embedding = self.clip_model.encode_image(preprocessed_object_crop)
                object_embedding = object_embedding / object_embedding.norm(dim=1, keepdim=True)
                object_embedding = object_embedding.squeeze().cpu()
            save_path = self.object_crop_save_path / f'object_crop_{self.object_count}.pt'
            object_of_interest['object_embedding_path'] = save_path
            torch.save(object_embedding, save_path)
            self.object_count += 1

            object_of_interest['box_index'] = image_crop['box_index']
            object_of_interest['dist_from_ego'] = image_crop['dist_from_ego']
            object_of_interest['pred_corners'] = image_crop['corners_2d']
            object_of_interest['box_in_image_ratio'] = image_crop['box_in_image_ratio']
            object_of_interest['img_path'] = image_crop['img_path']
            selection_results.append(object_of_interest)

        return selection_results


    def __call__(self, gt_info):
        candidate_crops = self.process_predictions(gt_info)
        alignment_results = self.get_alignment_status(candidate_crops)
        alignment_results = [result for result in alignment_results if result['align']]
        matches = [None] * len(gt_info['annos']['gt_boxes_lidar'])
        for result in alignment_results:
            matches[result['box_index']] = result
        return matches
