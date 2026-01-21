import copy
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_
from ..model_utils.transfusion_utils import clip_sigmoid
from ..model_utils.basic_block_2d import BasicBlock2D
from ..model_utils.transfusion_utils import PositionEmbeddingLearned, TransformerDecoderLayer
from .target_assigner.hungarian_assigner import HungarianAssigner3D
from ...utils import loss_utils
from ..model_utils import centernet_utils

import open_clip
from PIL import Image

class SeparateHead_Transfusion(nn.Module):
    def __init__(self, input_channels, head_channels, kernel_size, sep_head_dict, init_bias=-2.19, use_bias=False):
        super().__init__()
        self.sep_head_dict = sep_head_dict

        for cur_name in self.sep_head_dict:
            output_channels = self.sep_head_dict[cur_name]['out_channels']
            num_conv = self.sep_head_dict[cur_name]['num_conv']

            fc_list = []
            for k in range(num_conv - 1):
                fc_list.append(nn.Sequential(
                    nn.Conv1d(input_channels, head_channels, kernel_size, stride=1, padding=kernel_size//2, bias=use_bias),
                    nn.BatchNorm1d(head_channels),
                    nn.ReLU()
                ))
            fc_list.append(nn.Conv1d(head_channels, output_channels, kernel_size, stride=1, padding=kernel_size//2, bias=True))
            fc = nn.Sequential(*fc_list)
            if 'hm' in cur_name:
                fc[-1].bias.data.fill_(init_bias)
            else:
                for m in fc.modules():
                    if isinstance(m, nn.Conv2d):
                        kaiming_normal_(m.weight.data)
                        if hasattr(m, "bias") and m.bias is not None:
                            nn.init.constant_(m.bias, 0)

            self.__setattr__(cur_name, fc)

    def forward(self, x):
        ret_dict = {}
        for cur_name in self.sep_head_dict:
            ret_dict[cur_name] = self.__getattr__(cur_name)(x)

        return ret_dict


class MLPAlignmentHead(nn.Module):
    """
        This module implements AlignmentHead.
    """
    def __init__(self, input_dim=128, hidden_dim=512, output_dim=1024):
        super(MLPAlignmentHead, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.do1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.GELU()

    def forward(self, x):
        batch_size, num_objects, input_feature_dim = x.size()
        x = x.reshape(-1, input_feature_dim)
        x = self.activation(self.norm1(self.fc1(x)))
        x = self.do1(x)
        x = self.activation(self.norm2(self.fc2(x)))
        x = self.fc3(x)
        x = x.reshape(batch_size, num_objects, -1)
        return x

class OVScanAlignmentHead(nn.Module):
    def __init__(self,
                 class_text_embeddings,
                 align_out_dim=1024,
                 object_input_dim=128,
                 num_heads=4,
                 dropout=0.1,
                 activation='gelu'):
        super(OVScanAlignmentHead, self).__init__()

        self.class_text_embeddings = class_text_embeddings
        self.class_text_embedding_dim = class_text_embeddings.size(-1)
        self.num_classes = class_text_embeddings.size(0)

        # Different transforms for the class_text_embeddings based on the class
        self.text_embed_fc_1 = nn.Linear(self.class_text_embedding_dim, object_input_dim)
        self.text_embed_fc_2 = nn.Linear(self.class_text_embedding_dim, 2 * object_input_dim)
        self.text_embed_fc_3 = nn.Linear(self.class_text_embedding_dim, 4 * object_input_dim)

        # Upscale layers with text as query and concatenated object features
        self.upscale_mha_1 = nn.MultiheadAttention(object_input_dim, num_heads, dropout=dropout)
        self.upscale_mha_ln_1 = nn.LayerNorm(object_input_dim)
        self.upscale_ffn_1 = nn.Sequential(
            nn.Linear(object_input_dim + object_input_dim, 2 * object_input_dim),
            self.get_activation_fn(activation),
            nn.Dropout(dropout)
        )
        self.upscale_ffn_ln_1 = nn.LayerNorm(2 * object_input_dim)

        self.upscale_mha_2 = nn.MultiheadAttention(2 * object_input_dim, num_heads, dropout=dropout)
        self.upscale_mha_ln_2 = nn.LayerNorm(2 * object_input_dim)
        self.upscale_ffn_2 = nn.Sequential(
            nn.Linear(2 * object_input_dim + object_input_dim, 4 * object_input_dim),
            self.get_activation_fn(activation),
            nn.Dropout(dropout)
        )
        self.upscale_ffn_ln_2 = nn.LayerNorm(4 * object_input_dim)

        self.upscale_mha_3 = nn.MultiheadAttention(4 * object_input_dim, num_heads, dropout=dropout)
        self.upscale_mha_ln_3 = nn.LayerNorm(4 * object_input_dim)
        self.out = nn.Linear(4 * object_input_dim + object_input_dim, align_out_dim)

        self.init_weights()

    def get_activation_fn(self, activation):
        """Return an activation function given a string"""
        if activation == "relu":
            return nn.ReLU()
        if activation == "gelu":
            return nn.GELU()
        if activation == "glu":
            return nn.GLU()
        raise RuntimeError(f"activation should be relu/gelu, not {activation}.")

    def init_weights(self):
        for m in self.upscale_mha_1.parameters():
            if m.dim() > 1:
                nn.init.xavier_uniform_(m)
        for m in self.upscale_ffn_1.parameters():
            if m.dim() > 1:
                nn.init.xavier_uniform_(m)
        for m in self.upscale_mha_2.parameters():
            if m.dim() > 1:
                nn.init.xavier_uniform_(m)
        for m in self.upscale_ffn_2.parameters():
            if m.dim() > 1:
                nn.init.xavier_uniform_(m)
        for m in self.upscale_mha_3.parameters():
            if m.dim() > 1:
                nn.init.xavier_uniform_(m)
        for m in self.out.parameters():
            if m.dim() > 1:
                nn.init.xavier_uniform_(m)

    def forward(self, input, pred_class):
        # Global Feature
        global_obj_feats = input.permute(0, 2, 1)
        batch_size = global_obj_feats.size(0)
        num_proposals = global_obj_feats.size(1)
        global_obj_feats = global_obj_feats.reshape(batch_size * num_proposals, -1)
        
        # Get class-specific text embeddings and transform for each upscale stage
        class_text_embeddings = self.class_text_embeddings[pred_class].reshape(batch_size * num_proposals, -1)

        # Upscale 1 with text as query, concatenating object features before FFN
        text_guide_1 = self.text_embed_fc_1(class_text_embeddings)
        upscale_1, _ = self.upscale_mha_1(query=text_guide_1.unsqueeze(0),
                                          key=global_obj_feats.unsqueeze(0),
                                          value=global_obj_feats.unsqueeze(0))
        upscale_1 = upscale_1.squeeze(0)
        upscale_1_mid = self.upscale_mha_ln_1(upscale_1 + text_guide_1)
        upscale_1_concat = torch.cat([upscale_1_mid, global_obj_feats], dim=-1)  # Concatenate object features
        upscale_1_post = self.upscale_ffn_ln_1(self.upscale_ffn_1(upscale_1_concat))

        # Upscale 2 with text as query, concatenating object features before FFN
        text_guide_2 = self.text_embed_fc_2(class_text_embeddings)
        upscale_2, _ = self.upscale_mha_2(query=text_guide_2.unsqueeze(0),
                                          key=upscale_1_post.unsqueeze(0),
                                          value=upscale_1_post.unsqueeze(0))
        upscale_2 = upscale_2.squeeze(0)
        upscale_2_mid = self.upscale_mha_ln_2(upscale_2 + text_guide_2)
        upscale_2_concat = torch.cat([upscale_2_mid, global_obj_feats], dim=-1)  # Concatenate object features
        upscale_2_post = self.upscale_ffn_ln_2(self.upscale_ffn_2(upscale_2_concat))

        # Upscale 3 with text as query, concatenating object features before FFN
        text_guide_3 = self.text_embed_fc_3(class_text_embeddings)
        upscale_3, _ = self.upscale_mha_3(query=text_guide_3.unsqueeze(0),
                                          key=upscale_2_post.unsqueeze(0),
                                          value=upscale_2_post.unsqueeze(0))
        upscale_3 = upscale_3.squeeze(0)
        upscale_3_mid = self.upscale_mha_ln_3(upscale_3 + text_guide_3)
        upscale_3_concat = torch.cat([upscale_3_mid, global_obj_feats], dim=-1)  # Concatenate object features
        
        # Final output projection
        out = self.out(upscale_3_concat)
        out = out.view(batch_size, num_proposals, -1)  # Reshape to original batch
        return out

class OVScanHead(nn.Module):
    def __init__(
        self,
        model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range, voxel_size, predict_boxes_when_training=True,
    ):
        super(OVScanHead, self).__init__()

        self.grid_size = grid_size
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size

        self.model_cfg = model_cfg
        self.class_names = class_names
        self.num_class = self.model_cfg.NUM_CLASSES
        self.feature_map_stride = self.model_cfg.TARGET_ASSIGNER_CONFIG.get('FEATURE_MAP_STRIDE', None)
        self.dataset_name = self.model_cfg.TARGET_ASSIGNER_CONFIG.get('DATASET', 'nuScenes')

        hidden_channel = self.model_cfg.HIDDEN_CHANNEL
        self.num_proposals = self.model_cfg.NUM_PROPOSALS
        self.bn_momentum = self.model_cfg.BN_MOMENTUM
        self.nms_kernel_size = self.model_cfg.NMS_KERNEL_SIZE

        num_heads = self.model_cfg.NUM_HEADS
        dropout = self.model_cfg.DROPOUT
        activation = self.model_cfg.ACTIVATION
        ffn_channel = self.model_cfg.FFN_CHANNEL
        bias = self.model_cfg.get('USE_BIAS_BEFORE_NORM', False)

        loss_cls = self.model_cfg.LOSS_CONFIG.LOSS_SIZE_CLS
        self.use_sigmoid_cls = loss_cls.get("use_sigmoid", False)
        if not self.use_sigmoid_cls:
            self.num_class += 1
        self.loss_cls = loss_utils.SigmoidFocalClassificationLoss(gamma=loss_cls.gamma,alpha=loss_cls.alpha)
        self.loss_cls_weight = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['cls_weight']
        self.loss_align = nn.CosineEmbeddingLoss()
        self.loss_align_weight = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['align_weight']
        self.loss_bbox = loss_utils.L1Loss()
        self.loss_bbox_weight = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['bbox_weight']
        self.loss_heatmap = loss_utils.GaussianFocalLoss()
        self.loss_heatmap_weight = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['hm_weight']
        self.code_weights = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['code_weights']
        self.code_size = 10

        # a shared convolution
        self.shared_conv = nn.Conv2d(in_channels=input_channels,out_channels=hidden_channel,kernel_size=3,padding=1)
        layers = []
        layers.append(BasicBlock2D(hidden_channel,hidden_channel, kernel_size=3,padding=1,bias=bias))
        layers.append(nn.Conv2d(in_channels=hidden_channel,out_channels=self.num_class,kernel_size=3,padding=1))
        self.heatmap_head = nn.Sequential(*layers)
        self.class_encoding = nn.Conv1d(self.num_class, hidden_channel, 1)

        # transformer decoder layers for object query with LiDAR feature
        self.object_decoder = TransformerDecoderLayer(hidden_channel, num_heads, ffn_channel, dropout, activation,
                self_posembed=PositionEmbeddingLearned(2, hidden_channel),
                cross_posembed=PositionEmbeddingLearned(2, hidden_channel),
            )
        # Prediction Head
        heads = copy.deepcopy(self.model_cfg.SEPARATE_HEAD_CFG.HEAD_DICT)
        heads['heatmap'] = dict(out_channels=self.num_class, num_conv=self.model_cfg.NUM_HM_CONV)
        self.prediction_head = SeparateHead_Transfusion(hidden_channel, 64, 1, heads, use_bias=bias)

        self.init_weights()
        # self.model_cfg.TARGET_ASSIGNER_CONFIG.HUNGARIAN_ASSIGNER.cls_cost['weight'] = 0.0 # Class should be irrelevant for matcher
        self.bbox_assigner = HungarianAssigner3D(**self.model_cfg.TARGET_ASSIGNER_CONFIG.HUNGARIAN_ASSIGNER)

        # Position Embedding for Cross-Attention, which is re-used during training
        x_size = self.grid_size[0] // self.feature_map_stride
        y_size = self.grid_size[1] // self.feature_map_stride
        self.bev_pos = self.create_2D_grid(x_size, y_size)

        # Frozen CLIP for alignment
        self.clip_name = self.model_cfg.ALIGNMENT.CLIP_MODEL
        clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(self.clip_name)
        self.clip_img_encoder = copy.deepcopy(clip_model.visual)
        self.clip_context_length = clip_model.context_length
        self.clip_tokenizer = open_clip.get_tokenizer(self.model_cfg.ALIGNMENT.TOKENIZER)
        self.clip_logit_scale_exp = clip_model.logit_scale.exp().item()
        for param in self.clip_img_encoder.parameters():
            param.requires_grad = False
        self.clip_img_encoder.eval().cuda()
        # Get class names and create clip prompt templates
        self.clip_prompt_template = str(self.model_cfg.ALIGNMENT.PROMPT)
        self.nuscenes_to_ov_classes = self.model_cfg.ALIGNMENT.NUSCENES_TO_OV_CLASSES
        self.ov_classes = []
        for class_name in self.nuscenes_to_ov_classes.values():
            self.ov_classes.extend(class_name)
        self.ov_to_nuscenes_classes = {class_name: key for key, value in self.nuscenes_to_ov_classes.items() for class_name in value}
        self.nuscenes_to_label_id = {class_name: idx+1 for idx, class_name in enumerate(self.class_names)}

        self.clip_texts = [self.clip_prompt_template.replace('CLASS', class_name) for class_name in self.ov_classes]
        self.text_inputs = self.clip_tokenizer(self.clip_texts, context_length=self.clip_context_length)
        clip_model.eval().cuda()
        self.text_features = clip_model.encode_text(self.text_inputs.cuda()).detach()
        self.text_features = self.text_features / self.text_features.norm(dim=-1, keepdim=True)
        
        # For Alignment Head
        self.class_texts = [self.clip_prompt_template.replace('CLASS', class_name.replace('_', ' ')) for class_name in self.class_names]
        self.class_text_inputs = self.clip_tokenizer(self.class_texts, context_length=self.clip_context_length)
        self.class_text_features = clip_model.encode_text(self.class_text_inputs.cuda()).detach()
        self.class_text_features = self.class_text_features / self.class_text_features.norm(dim=-1, keepdim=True)


        self.alignment_head = OVScanAlignmentHead(self.class_text_features,
                                                   align_out_dim=self.model_cfg.ALIGNMENT.CLIP_EMBED_DIM,
                                                   object_input_dim=hidden_channel,
                                                   num_heads=num_heads,
                                                   dropout=dropout,
                                                   activation='gelu')

        self.forward_ret_dict = {}

    def create_2D_grid(self, x_size, y_size):
        meshgrid = [[0, x_size - 1, x_size], [0, y_size - 1, y_size]]
        # NOTE: modified
        batch_x, batch_y = torch.meshgrid(
            *[torch.linspace(it[0], it[1], it[2]) for it in meshgrid],
            indexing='ij'
        )
        batch_x = batch_x + 0.5
        batch_y = batch_y + 0.5
        coord_base = torch.cat([batch_x[None], batch_y[None]], dim=0)[None]
        coord_base = coord_base.view(1, 2, -1).permute(0, 2, 1)
        return coord_base

    def init_weights(self):
        # initialize transformer
        for m in self.object_decoder.parameters():
            if m.dim() > 1:
                nn.init.xavier_uniform_(m)
        if hasattr(self, "query"):
            nn.init.xavier_normal_(self.query)
        self.init_bn_momentum()

    def init_bn_momentum(self):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.momentum = self.bn_momentum

    def predict(self, inputs):
        batch_size = inputs.shape[0]
        lidar_feat = self.shared_conv(inputs)

        lidar_feat_flatten = lidar_feat.view(
            batch_size, lidar_feat.shape[1], -1
        )
        bev_pos = self.bev_pos.repeat(batch_size, 1, 1).to(lidar_feat.device)

        # query initialization
        dense_heatmap = self.heatmap_head(lidar_feat)
        heatmap = dense_heatmap.detach().sigmoid()
        padding = self.nms_kernel_size // 2
        local_max = torch.zeros_like(heatmap)
        local_max_inner = F.max_pool2d(
            heatmap, kernel_size=self.nms_kernel_size, stride=1, padding=0
        )
        local_max[:, :, padding:(-padding), padding:(-padding)] = local_max_inner
        # for Pedestrian & Traffic_cone in nuScenes
        if self.dataset_name == "nuScenes":
            local_max[ :, 8, ] = F.max_pool2d(heatmap[:, 8], kernel_size=1, stride=1, padding=0)
            local_max[ :, 9, ] = F.max_pool2d(heatmap[:, 9], kernel_size=1, stride=1, padding=0)
        # for Pedestrian & Cyclist in Waymo
        elif self.dataset_name == "Waymo":
            local_max[ :, 1, ] = F.max_pool2d(heatmap[:, 1], kernel_size=1, stride=1, padding=0)
            local_max[ :, 2, ] = F.max_pool2d(heatmap[:, 2], kernel_size=1, stride=1, padding=0)
        heatmap = heatmap * (heatmap == local_max)
        heatmap = heatmap.view(batch_size, heatmap.shape[1], -1)
 
        # top num_proposals among all classes
        top_proposals = heatmap.view(batch_size, -1).argsort(dim=-1, descending=True)[
            ..., : self.num_proposals
        ]
        top_proposals_class = top_proposals // heatmap.shape[-1]
        top_proposals_index = top_proposals % heatmap.shape[-1]
        query_feat = lidar_feat_flatten.gather(
            index=top_proposals_index[:, None, :].expand(-1, lidar_feat_flatten.shape[1], -1),
            dim=-1,
        )
        # add category embedding
        one_hot = F.one_hot(top_proposals_class, num_classes=self.num_class).permute(0, 2, 1)
        query_cat_encoding = self.class_encoding(one_hot.float())
        query_cat_feat = query_feat + query_cat_encoding

        query_pos = bev_pos.gather(
            index=top_proposals_index[:, None, :].permute(0, 2, 1).expand(-1, -1, bev_pos.shape[-1]),
            dim=1,
        )
        # convert to xy
        query_pos = query_pos.flip(dims=[-1])
        bev_pos = bev_pos.flip(dims=[-1])

        global_query_feat = self.object_decoder(
            query_cat_feat, lidar_feat_flatten, query_pos, bev_pos
        )

        res_layer = self.prediction_head(global_query_feat)
        res_layer["center"] = res_layer["center"] + query_pos.permute(0, 2, 1)
        res_layer["query_labels"] = top_proposals_class

        res_layer["query_heatmap_score"] = heatmap.gather(
            index=top_proposals_index[:, None, :].expand(-1, self.num_class, -1),
            dim=-1,
        )
        res_layer["dense_heatmap"] = dense_heatmap
        res_layer["object_feats"] = query_cat_feat.permute(0, 2, 1)

        res_layer['clip_preds'] = self.alignment_head(global_query_feat, top_proposals_class)
    
        clip_preds = res_layer['clip_preds'].detach()
        normalized_clip_preds = clip_preds / clip_preds.norm(dim=-1, keepdim=True)
        res_layer["clip_logits"] = self.clip_logit_scale_exp * normalized_clip_preds @ self.text_features.t()
        res_layer["ov_labels"] = torch.argmax(res_layer["clip_logits"], dim=-1)
        return res_layer

    def forward(self, batch_dict):
        feats = batch_dict['spatial_features_2d']
        res = self.predict(feats)
        if not self.training:
            bboxes = self.get_bboxes(res)
            batch_dict['final_box_dicts'] = bboxes
        else:
            gt_boxes = batch_dict['gt_boxes']
            gt_bboxes_3d = gt_boxes[...,:-1]
            gt_labels_3d =  gt_boxes[...,-1].long() - 1
            gt_grounding_sam = batch_dict['gt_grounding_sam']
            # gt_size_labels = batch_dict['gt_size_labels']
            images_dict = []
            camera_ids = batch_dict['cam_ids']
            camera_images = batch_dict['raw_imgs']
            for batch_id in range(len(camera_ids)):
                cam_frame_dict = {}
                for image_id in range(len(camera_ids[batch_id])):
                    cam_frame_dict[camera_ids[batch_id][image_id]] = camera_images[batch_id][image_id]
                images_dict.append(cam_frame_dict)
            loss, tb_dict = self.loss(gt_bboxes_3d, gt_labels_3d, res, gt_grounding_sam, images_dict)
            batch_dict['loss'] = loss
            batch_dict['tb_dict'] = tb_dict
        return batch_dict

    def get_targets(self, gt_bboxes_3d, gt_labels_3d, gt_grounding_sam, pred_dicts):
        assign_results = []
        for batch_idx in range(len(gt_bboxes_3d)):
            pred_dict = {}
            for key in pred_dicts.keys():
                pred_dict[key] = pred_dicts[key][batch_idx : batch_idx + 1]
            gt_bboxes = gt_bboxes_3d[batch_idx]
            valid_idx = []
            # filter empty boxes
            for i in range(len(gt_bboxes)):
                if gt_bboxes[i][3] > 0 and gt_bboxes[i][4] > 0:
                    valid_idx.append(i)
            gt_grounding_sam_filtered = [gt_grounding_sam[batch_idx][idx] for idx in valid_idx]
            assign_result = self.get_targets_single(gt_bboxes[valid_idx], gt_labels_3d[batch_idx][valid_idx], gt_grounding_sam_filtered, pred_dict)
            assign_results.append(assign_result)

        res_tuple = tuple(map(list, zip(*assign_results)))
        label_targets = torch.cat(res_tuple[0], dim=0)
        label_weights = torch.cat(res_tuple[1], dim=0)
        bbox_targets = torch.cat(res_tuple[2], dim=0)
        bbox_weights = torch.cat(res_tuple[3], dim=0)
        num_pos = np.sum(res_tuple[4])
        matched_ious = np.mean(res_tuple[5])
        heatmap = torch.cat(res_tuple[6], dim=0)
        pos_inds = res_tuple[7]
        bbox_grounding_sam = res_tuple[8]
        return label_targets, label_weights, bbox_targets, bbox_weights, num_pos, matched_ious, heatmap, pos_inds, bbox_grounding_sam
        

    def get_targets_single(self, gt_bboxes_3d, gt_labels_3d, gt_grounding_sam, preds_dict):
        
        num_proposals = preds_dict["center"].shape[-1]
        score = copy.deepcopy(preds_dict["heatmap"].detach())
        center = copy.deepcopy(preds_dict["center"].detach())
        height = copy.deepcopy(preds_dict["height"].detach())
        dim = copy.deepcopy(preds_dict["dim"].detach())
        rot = copy.deepcopy(preds_dict["rot"].detach())
        if "vel" in preds_dict.keys():
            vel = copy.deepcopy(preds_dict["vel"].detach())
        else:
            vel = None
        object_feats = copy.deepcopy(preds_dict["object_feats"].detach())
        clip_preds = copy.deepcopy(preds_dict["clip_preds"].detach())
        ov_labels = copy.deepcopy(preds_dict["ov_labels"].detach())
        clip_logits = copy.deepcopy(preds_dict["clip_logits"].detach())

        boxes_dict = self.decode_bbox(score, rot, dim, center, height, vel, 
                                      object_feats, clip_preds, ov_labels, clip_logits,
                                      filter=False)

        bboxes_tensor = boxes_dict[0]["pred_boxes"]
        gt_bboxes_tensor = gt_bboxes_3d.to(score.device)
        
        assigned_gt_inds, ious = self.bbox_assigner.assign(
            bboxes_tensor, gt_bboxes_tensor, gt_labels_3d,
            score, self.point_cloud_range,
        )

        pos_inds = torch.nonzero(assigned_gt_inds > 0, as_tuple=False).squeeze(-1).unique()
        neg_inds = torch.nonzero(assigned_gt_inds == 0, as_tuple=False).squeeze(-1).unique()
        pos_assigned_gt_inds = assigned_gt_inds[pos_inds] - 1
        if gt_bboxes_3d.numel() == 0:
            assert pos_inds.numel() == 0
            pos_gt_bboxes = torch.empty_like(gt_bboxes_3d).view(-1, 9)
            pos_gt_grounding_sam = [None] * len(pos_assigned_gt_inds)
        else:
            pos_gt_bboxes = gt_bboxes_3d[pos_assigned_gt_inds.long(), :]
            pos_gt_grounding_sam = [gt_grounding_sam[idx] for idx in pos_assigned_gt_inds]

        # create target for loss computation
        bbox_targets = torch.zeros([num_proposals, self.code_size]).to(center.device)
        bbox_weights = torch.zeros([num_proposals, self.code_size]).to(center.device)
        ious = torch.clamp(ious, min=0.0, max=1.0)
        labels = bboxes_tensor.new_zeros(num_proposals, dtype=torch.long)
        label_weights = bboxes_tensor.new_zeros(num_proposals, dtype=torch.long)

        if gt_labels_3d is not None:  # default label is -1
            labels += self.num_class

        # both pos and neg have classification loss, only pos has regression and iou loss
        if len(pos_inds) > 0:
            pos_bbox_targets = self.encode_bbox(pos_gt_bboxes)
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0

            if gt_labels_3d is None:
                labels[pos_inds] = 1
            else:
                labels[pos_inds] = gt_labels_3d[pos_assigned_gt_inds]
            label_weights[pos_inds] = 1.0

        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # compute dense heatmap targets
        device = labels.device
        target_assigner_cfg = self.model_cfg.TARGET_ASSIGNER_CONFIG
        feature_map_size = (self.grid_size[:2] // self.feature_map_stride) 
        heatmap = gt_bboxes_3d.new_zeros(self.num_class, feature_map_size[1], feature_map_size[0])
        for idx in range(len(gt_bboxes_3d)):
            width = gt_bboxes_3d[idx][3]
            length = gt_bboxes_3d[idx][4]
            width = width / self.voxel_size[0] / self.feature_map_stride
            length = length / self.voxel_size[1] / self.feature_map_stride
            if width > 0 and length > 0:
                radius = centernet_utils.gaussian_radius(length.view(-1), width.view(-1), target_assigner_cfg.GAUSSIAN_OVERLAP)[0]
                radius = max(target_assigner_cfg.MIN_RADIUS, int(radius))
                x, y = gt_bboxes_3d[idx][0], gt_bboxes_3d[idx][1]

                coor_x = (x - self.point_cloud_range[0]) / self.voxel_size[0] / self.feature_map_stride
                coor_y = (y - self.point_cloud_range[1]) / self.voxel_size[1] / self.feature_map_stride

                center = torch.tensor([coor_x, coor_y], dtype=torch.float32, device=device)
                center_int = center.to(torch.int32)
                centernet_utils.draw_gaussian_to_heatmap(heatmap[gt_labels_3d[idx]], center_int, radius)

        mean_iou = ious[pos_inds].sum() / max(len(pos_inds), 1)
        return (labels[None], label_weights[None], bbox_targets[None], bbox_weights[None], int(pos_inds.shape[0]), float(mean_iou), heatmap[None], pos_inds, pos_gt_grounding_sam)

    def loss(self, gt_bboxes_3d, gt_labels_3d, pred_dicts, gt_grounding_sam, images_dict, **kwargs):
        labels, label_weights, bbox_targets, bbox_weights, num_pos, matched_ious, heatmap, pos_indices, bbox_grounding_sam = \
            self.get_targets(gt_bboxes_3d, gt_labels_3d, gt_grounding_sam, pred_dicts)
        loss_dict = dict()
        loss_all = 0

        # compute heatmap loss
        loss_heatmap = self.loss_heatmap(
            clip_sigmoid(pred_dicts["dense_heatmap"]),
            heatmap,
        ).sum() / max(heatmap.eq(1).float().sum().item(), 1)
        loss_dict["loss_heatmap"] = loss_heatmap.item() * self.loss_heatmap_weight
        loss_all += loss_heatmap * self.loss_heatmap_weight

        # compute classification loss
        labels_for_loss = labels.reshape(-1)
        label_weights_for_loss = label_weights.reshape(-1)
        cls_score = pred_dicts["heatmap"].permute(0, 2, 1).reshape(-1, self.num_class)
        one_hot_targets = torch.zeros(*list(labels_for_loss.shape), self.num_class+1, dtype=cls_score.dtype, device=labels_for_loss.device)
        one_hot_targets.scatter_(-1, labels_for_loss.unsqueeze(dim=-1).long(), 1.0)
        one_hot_targets = one_hot_targets[..., :-1]
        loss_cls = self.loss_cls(
            cls_score, one_hot_targets, label_weights_for_loss
        ).sum() / max(num_pos, 1)
        loss_dict["loss_cls"] = loss_cls.item() * self.loss_cls_weight
        loss_all += loss_cls * self.loss_cls_weight

        # compute box loss
        preds = torch.cat([pred_dicts[head_name] for head_name in self.model_cfg.SEPARATE_HEAD_CFG.HEAD_ORDER], dim=1).permute(0, 2, 1)
        reg_weights = bbox_weights * bbox_weights.new_tensor(self.code_weights)
        loss_bbox = self.loss_bbox(preds, bbox_targets)
        loss_bbox = (loss_bbox * reg_weights).sum() / max(num_pos, 1)
        loss_dict["loss_bbox"] = loss_bbox.item() * self.loss_bbox_weight
        loss_all += loss_bbox * self.loss_bbox_weight

        # Selective alignment for alignment loss
        bboxes_formated = self.get_bboxes(pred_dicts, filter=False)
        clip_target_features = []
        clip_pred_features = []
        batch_ids = np.arange(len(bboxes_formated))
        pred_labels_list = torch.tensor([], dtype=torch.long).cuda()
        gt_labels_list = torch.tensor([], dtype=torch.long).cuda()
        alignment_batch = zip(batch_ids, pos_indices, bbox_grounding_sam, images_dict)

        gt_labels_interest = []
        pred_labels_interest = []

        for batch_id, pos_ind, grounding_sam, image_dict in alignment_batch:

            # Get the classes
            pred_labels = bboxes_formated[batch_id]['pred_labels'][pos_ind].clone().detach()
            gt_labels  = labels[batch_id][pos_ind] + 1
            pred_labels_list = torch.cat((pred_labels_list, pred_labels), dim=0)
            gt_labels_list = torch.cat((gt_labels_list, gt_labels), dim=0)
            
            # Alignment selection
            clip_predictions = pred_dicts['clip_preds'][batch_id][pos_ind]
            clip_matched_preds = []
            clip_image_embeddings = []
            for alignment_crop, clip_pred, gt_label, pred_labels in zip(grounding_sam, clip_predictions, gt_labels, pred_labels):
                if not alignment_crop:
                    continue
                # Load object image embedding from CLIP
                alignment_obj_embedding = torch.load(alignment_crop['object_embedding_path']).cuda()
                clip_image_embeddings.append(alignment_obj_embedding)
                clip_matched_preds.append(clip_pred)
                gt_labels_interest.append(gt_label)
                pred_labels_interest.append(pred_labels)

            clip_target_features.extend(clip_image_embeddings)
            clip_pred_features.extend(clip_matched_preds)

        # Compute alignment loss
        if len(clip_target_features) != 0:
            clip_target_features = torch.stack(clip_target_features).cuda()
            clip_pred_features = torch.stack(clip_pred_features).cuda()
            loss_align = self.loss_align(clip_pred_features, clip_target_features, torch.ones(clip_pred_features.shape[0]).cuda())
            loss_dict["loss_align"] = loss_align.item() * self.loss_align_weight
            loss_all += loss_align * self.loss_align_weight

        else:
            # Ensure the clip predictions are apart of loss, even if not used
            dummy_loss = pred_dicts['clip_preds'].sum() * 0.0
            loss_dict["loss_align"] = 0.0
            loss_all += dummy_loss * self.loss_align_weight

        # Compute classification accuracy percentage
        assert len(pred_labels_list) == len(gt_labels_list)

        if len(gt_labels_list) != 0:
            loss_dict['cls_acc_ovr'] = (pred_labels_list == gt_labels_list).sum().item() / len(pred_labels_list)
        else:
            loss_dict['cls_acc_ovr'] = 0.0
        gt_labels_interest = torch.tensor(gt_labels_interest)
        pred_labels_interest = torch.tensor(pred_labels_interest)
        if len(gt_labels_interest) != 0:
            loss_dict['cls_acc_selective'] = (gt_labels_interest == pred_labels_interest).sum().item() / len(gt_labels_interest)
        else:
            loss_dict['cls_acc_selective'] = 0.0
                
        loss_dict['loss_ovr'] = loss_all.item()

        return loss_all, loss_dict

    def encode_bbox(self, bboxes):
        code_size = 10
        targets = torch.zeros([bboxes.shape[0], code_size]).to(bboxes.device)
        targets[:, 0] = (bboxes[:, 0] - self.point_cloud_range[0]) / (self.feature_map_stride * self.voxel_size[0])
        targets[:, 1] = (bboxes[:, 1] - self.point_cloud_range[1]) / (self.feature_map_stride * self.voxel_size[1])
        targets[:, 3:6] = bboxes[:, 3:6].log()
        targets[:, 2] = bboxes[:, 2]
        targets[:, 6] = torch.sin(bboxes[:, 6])
        targets[:, 7] = torch.cos(bboxes[:, 6])
        if code_size == 10:
            targets[:, 8:10] = bboxes[:, 7:]
        return targets

    def decode_bbox(self, heatmap, rot, dim, center, height, vel, 
                    object_feats, clip_preds, ov_labels, clip_logits, filter=False):
        
        post_process_cfg = self.model_cfg.POST_PROCESSING
        score_thresh = post_process_cfg.SCORE_THRESH
        post_center_range = post_process_cfg.POST_CENTER_RANGE
        post_center_range = torch.tensor(post_center_range).cuda().float()
        final_scores = heatmap.max(1, keepdims=False).values

        center[:, 0, :] = center[:, 0, :] * self.feature_map_stride * self.voxel_size[0] + self.point_cloud_range[0]
        center[:, 1, :] = center[:, 1, :] * self.feature_map_stride * self.voxel_size[1] + self.point_cloud_range[1]
        dim = dim.exp()
        rots, rotc = rot[:, 0:1, :], rot[:, 1:2, :]
        rot = torch.atan2(rots, rotc)

        if vel is None:
            final_box_preds = torch.cat([center, height, dim, rot], dim=1).permute(0, 2, 1)
        else:
            final_box_preds = torch.cat([center, height, dim, rot, vel], dim=1).permute(0, 2, 1)

        predictions_dicts = []
        for i in range(heatmap.shape[0]):
            boxes3d = final_box_preds[i]
            scores = final_scores[i]
            obj_feats = object_feats[i]
            clip_pred = clip_preds[i]
            labels = ov_labels[i]
            logits = clip_logits[i]
            
            predictions_dict = {
                'pred_boxes': boxes3d,
                'pred_scores': scores, # (box/ objectness score)
                'object_feats': obj_feats,
                'clip_preds': clip_pred,
                'ov_labels': labels,
                'clip_logits': logits,
            }
            # Convert OV Classes to Nuscenes Classes
            nusc_labels = [self.nuscenes_to_label_id[self.ov_to_nuscenes_classes[self.ov_classes[label]]] for label in predictions_dict['ov_labels']]
            predictions_dict["pred_labels"] = torch.tensor(nusc_labels).cuda()
            predictions_dict["ov_str_labels"] = [self.ov_classes[label] for label in predictions_dict['ov_labels']]
            predictions_dicts.append(predictions_dict)

        if filter is False:
            return predictions_dicts

        thresh_mask = final_scores > score_thresh        
        mask = (final_box_preds[..., :3] >= post_center_range[:3]).all(2)
        mask &= (final_box_preds[..., :3] <= post_center_range[3:]).all(2)

        predictions_dicts = []
        for i in range(heatmap.shape[0]):
            cmask = mask[i, :]
            cmask &= thresh_mask[i]

            boxes3d = final_box_preds[i, cmask]
            scores = final_scores[i, cmask]
            obj_feats = object_feats[i, cmask]
            clip_pred = clip_preds[i, cmask]
            labels = ov_labels[i, cmask]
            logits = clip_logits[i, cmask]

            predictions_dict = {
                'pred_boxes': boxes3d,
                'pred_scores': scores,
                'object_feats': obj_feats,
                'clip_preds': clip_pred,
                'ov_labels': labels,
                'clip_logits': logits,
            }

            # Convert OV Classes to Nuscenes Classes
            nusc_labels = [self.nuscenes_to_label_id[self.ov_to_nuscenes_classes[self.ov_classes[label]]] for label in predictions_dict['ov_labels']]
            predictions_dict["pred_labels"] = torch.tensor(nusc_labels).cuda()
            predictions_dict["ov_str_labels"] = [self.ov_classes[label] for label in predictions_dict['ov_labels']]
            predictions_dicts.append(predictions_dict)

        return predictions_dicts

    def get_bboxes(self, preds_dicts, filter=True):

        batch_size = preds_dicts["heatmap"].shape[0]
        batch_score = preds_dicts["heatmap"].sigmoid()
        one_hot = F.one_hot(
            preds_dicts['query_labels'], num_classes=self.num_class
        ).permute(0, 2, 1)
        # batch_score = preds_dicts["query_heatmap_score"] * one_hot
        batch_score = batch_score * preds_dicts["query_heatmap_score"] * one_hot
        batch_center = preds_dicts["center"]
        batch_height = preds_dicts["height"]
        batch_dim = preds_dicts["dim"]
        batch_rot = preds_dicts["rot"]
        batch_vel = None
        if "vel" in preds_dicts:
            batch_vel = preds_dicts["vel"]

        batch_object_feats = preds_dicts["object_feats"]
        batch_clip_pred = preds_dicts["clip_preds"]
        batch_pred_labels = preds_dicts["ov_labels"]
        batch_clip_logits = preds_dicts["clip_logits"]

        ret_dict = self.decode_bbox(
            batch_score, batch_rot, batch_dim, batch_center,
            batch_height, batch_vel, batch_object_feats,
            batch_clip_pred, batch_pred_labels, batch_clip_logits, filter=filter,
        )

        return ret_dict 
