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
            "vehicle": 0.35,
            "pedestrian": 0.35,
            "traffic cone" : 0.35,
            "concrete block barrier": 0.35,
            "traffic block barrier": 0.35,
            # Additionals
            "animal": 0.35,
            "standing traffic sign": 0.35,
            "standing traffic crossing light": 0.35,
            "fire hydrant": 0.35,
            "mail box": 0.35,
            "electrical box": 0.35,
            "parking meter": 0.35,
            "bench": 0.35,
            "trash can": 0.35,
        }

        self.CAM_IDS = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
   
    def process_predictions(self, boxes, grounding_sam, lidar2camera_calibs, cam_intrinstics):
        """
        Process the predictions and filter out the boxes that are not in the image

        Parameters:
            boxes (np.array): The predicted boxes
            grounding_sam (dict): The grounding sam data
            lidar2camera_calibs (dict): The lidar to camera calibration matrices
            cam_intrinstics (dict): The camera intrinsics
        
        Returns:
            list: The image crops
        """
        
        # Filter boxes based on distance
        dist_filter_mask = [np.linalg.norm(box[:2]) < self.MAX_ALIGNMENT_RANGE for box in boxes]
        keep_index = np.arange(len(boxes))[dist_filter_mask]
        boxes = boxes[dist_filter_mask]
    
        # Extract Image from GTs
        image_crops = []
        for i, box in enumerate(boxes):
            box_ = Box(tuple(box[:3]), tuple(box[[4,3,5]]),
                      Quaternion(Quaternion(axis=[0, 0, 1], angle=box[6])))
            dist_from_ego = np.linalg.norm(box_.center[:2])
            # Get 2D crop of bounding box from image. If not in any images, skip.
            image_crop_dict = self.get_image_crop(box_, grounding_sam, lidar2camera_calibs, cam_intrinstics)
            if image_crop_dict:
                image_crop_dict['box_index'] = keep_index[i]
                image_crop_dict['dist_from_ego'] = dist_from_ego
                image_crops.append(image_crop_dict)
    
        return image_crops


    def get_image_crop(self, box, grounding_sam, lidar2camera_calibs, cam_intrinstics):
        """
        Get the image crop for a given box

        Parameters:
            box (Box): The box to get the image crop for
            grounding_sam (dict): The grounding sam data
            lidar2camera_calibs (dict): The lidar to camera calibration matrices
            cam_intrinstics (dict): The camera intrinsics
        
        Returns:
            dict: The image crop data
        """
        # Get best camera for this predictions
        best_cam_id = None
        best_cam_info = None
    
        for i, cam_id in enumerate(self.CAM_IDS):

            box_cam = copy.deepcopy(box)

            lidar2camera = lidar2camera_calibs[i] # homogeneous transformation matrix
            quat = Rotation.from_matrix(lidar2camera[:3, :3]).as_quat()
            box_cam.rotate(Quaternion([quat[3], quat[0], quat[1], quat[2]]))
            box_cam.translate(np.array(lidar2camera[:3, 3]))

            img_path = self.data_root / grounding_sam[cam_id]['image_path']
            imgsize = Image.open(img_path).size
            cam_intrinsic = cam_intrinstics[i]

            if self.is_box_in_image(box_cam, cam_intrinsic, imgsize, vis_level=BoxVisibility.ANY):
                
                # Get corners in 2D
                corners = self.view_points(box_cam.corners(), cam_intrinsic, normalize=True)[:2, :].T
                max_x = int(np.max(corners[:, 0]))
                min_x = int(np.min(corners[:, 0]))
                max_y = int(np.max(corners[:, 1]))
                min_y = int(np.min(corners[:, 1]))
                corners_2d = [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)]
                box_in_image_ratio = self.get_box_in_image_ratio(corners_2d, imgsize)
    
                if not best_cam_id or box_in_image_ratio > best_cam_info['box_in_image_ratio']:
                    best_cam_id = cam_id
                    best_cam_info = {
                        'cam_id': cam_id,
                        'box': box_cam,
                        'box_in_image_ratio': box_in_image_ratio,
                        'corners_2d': corners_2d,
                        'imgsize': imgsize,
                        'img_path': img_path,
                        'img_size': imgsize
                    }
    
        # Use best_cam for extraction, if it exists
        if best_cam_id and best_cam_info['box_in_image_ratio'] > self.MIN_BOX_IN_IMAGE_RATIO:
            
            min_x, min_y = best_cam_info['corners_2d'][0]
            max_x, max_y = best_cam_info['corners_2d'][2]
        
            # Validate the coordinates to ensure they are within the image bounds
            min_x = max(0, min_x)
            min_y = max(0, min_y)
            max_x = min(best_cam_info['imgsize'][0], max_x)
            max_y = min(best_cam_info['imgsize'][1], max_y)
            corners_2d = [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)]
            
            if min_x > max_x or min_y > max_y:
                return None
            
            # Crop the image using the validated coordinates
            candidate = {}
            candidate['img_path'] = best_cam_info['img_path']
            candidate['cam_id'] = best_cam_info['cam_id']
            candidate['box_in_image_ratio'] = best_cam_info['box_in_image_ratio']
            candidate['corners_2d'] = corners_2d
            candidate['img_size'] = best_cam_info['imgsize']
            return candidate
    
        else: # Cant find object in image
            return None

    def is_box_in_image(self, box, intrinsic, imsize, vis_level=BoxVisibility.ANY):
        """
        Borrowed from NuScenes DevKit:

        Check if a box is visible inside an image. A box is considered visible if all corners are inside the image.
        
        Parameters:
            box (Box): The box to check.
            intrinsic (np.array): The camera intrinsic matrix.
            imsize (tuple): The image size.
            vis_level (BoxVisibility): The visibility level.

        Returns:
            bool: True if the box is visible in the image, False otherwise.
        """
        corners_3d = box.corners()
        corners_img = self.view_points(corners_3d, intrinsic, normalize=True)[:2, :]

        visible = np.logical_and(corners_img[0, :] > 0, corners_img[0, :] < imsize[0])
        visible = np.logical_and(visible, corners_img[1, :] < imsize[1])
        visible = np.logical_and(visible, corners_img[1, :] > 0)
        visible = np.logical_and(visible, corners_3d[2, :] > 1)

        in_front = corners_3d[2, :] > 0.1  # True if a corner is at least 0.1 meter in front of the camera.

        if vis_level == BoxVisibility.ALL:
            return all(visible) and all(in_front)
        elif vis_level == BoxVisibility.ANY:
            return any(visible) and all(in_front)
        elif vis_level == BoxVisibility.NONE:
            return True
        else:
            raise ValueError("vis_level: {} not valid".format(vis_level))

    def view_points(self, points, view, normalize):
        """
        Taken from NuScenes DevKit:

        This is a helper class that maps 3d points to a 2d plane. It can be used to implement both perspective and
        orthographic projections. It first applies the dot product between the points and the view. By convention,
        the view should be such that the data is projected onto the first 2 axis. It then optionally applies a
        normalization along the third dimension.

        For a perspective projection the view should be a 3x3 camera matrix, and normalize=True
        For an orthographic projection with translation the view is a 3x4 matrix and normalize=False
        For an orthographic projection without translation the view is a 3x3 matrix (optionally 3x4 with last columns
        all zeros) and normalize=False

        Parameters:
            points (np.array): The points to map.
            view (np.array): The view matrix.
            normalize (bool): Whether to normalize the points.
        
        Returns:
            np.array: The mapped points.
        """

        assert view.shape[0] <= 4
        assert view.shape[1] <= 4
        assert points.shape[0] == 3

        viewpad = np.eye(4)
        viewpad[:view.shape[0], :view.shape[1]] = view

        nbr_points = points.shape[1]

        # Do operation in homogenous coordinates.
        points = np.concatenate((points, np.ones((1, nbr_points))))
        points = np.dot(viewpad, points)
        points = points[:3, :]

        if normalize:
            points = points / points[2:3, :].repeat(3, 0).reshape(3, nbr_points)

        return points

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
        len_x, len_y = imgsize
        
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

    def get_alignment_status(self, image_crops, gs_info):
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

        image_width, image_height = image_crops[0]['img_size']

        selection_results = []
        # Iterate through each image crop and match with the GS object
        for image_crop in image_crops:

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

            # # Save image to view
            # img_save_path = self.object_crop_save_path / f'object_crop_{self.object_count}.png'
            # Image.fromarray(object_crop).save(img_save_path)

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

    def __call__(self, boxes, grounding_sam, lidar2camera_calibs, cam_intrinstics):
        candidate_crops = self.process_predictions(boxes, grounding_sam, lidar2camera_calibs, cam_intrinstics)
        alignment_results = self.get_alignment_status(candidate_crops, grounding_sam)
        alignment_results = [result for result in alignment_results if result['align']]
        matches = [None] * len(boxes)
        for result in alignment_results:
            matches[result['box_index']] = result
        return matches
