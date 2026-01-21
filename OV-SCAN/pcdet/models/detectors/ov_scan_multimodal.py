import torch
import os

from .detector3d_template import Detector3DTemplate
from .. import backbones_image, view_transforms
from ..backbones_image import img_neck
from ..backbones_2d import fuser

class OVScanMultiModal(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_topology = [
            'vfe', 'backbone_3d', 'map_to_bev_module', 'pfe',
            'image_backbone','neck','vtransform','fuser',
            'backbone_2d', 'dense_head',  'point_head', 'roi_head'
        ]
        self.module_list = self.build_networks()
       
    def build_neck(self,model_info_dict):
        if self.model_cfg.get('NECK', None) is None:
            return None, model_info_dict
        neck_module = img_neck.__all__[self.model_cfg.NECK.NAME](
            model_cfg=self.model_cfg.NECK
        )
        model_info_dict['module_list'].append(neck_module)

        return neck_module, model_info_dict
    
    def build_vtransform(self,model_info_dict):
        if self.model_cfg.get('VTRANSFORM', None) is None:
            return None, model_info_dict
        
        vtransform_module = view_transforms.__all__[self.model_cfg.VTRANSFORM.NAME](
            model_cfg=self.model_cfg.VTRANSFORM
        )
        model_info_dict['module_list'].append(vtransform_module)

        return vtransform_module, model_info_dict

    def build_image_backbone(self, model_info_dict):
        if self.model_cfg.get('IMAGE_BACKBONE', None) is None:
            return None, model_info_dict
        image_backbone_module = backbones_image.__all__[self.model_cfg.IMAGE_BACKBONE.NAME](
            model_cfg=self.model_cfg.IMAGE_BACKBONE
        )
        image_backbone_module.init_weights()
        model_info_dict['module_list'].append(image_backbone_module)

        return image_backbone_module, model_info_dict
    
    def build_fuser(self, model_info_dict):
        if self.model_cfg.get('FUSER', None) is None:
            return None, model_info_dict
    
        fuser_module = fuser.__all__[self.model_cfg.FUSER.NAME](
            model_cfg=self.model_cfg.FUSER
        )
        model_info_dict['module_list'].append(fuser_module)
        model_info_dict['num_bev_features'] = self.model_cfg.FUSER.OUT_CHANNEL
        return fuser_module, model_info_dict

    def forward(self, batch_dict):

        for i,cur_module in enumerate(self.module_list):
            batch_dict = cur_module(batch_dict)
        
        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss(batch_dict)

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self,batch_dict):
        disp_dict = {}

        loss_trans, tb_dict = batch_dict['loss'],batch_dict['tb_dict']
        tb_dict = {
            'loss_trans': loss_trans.item(),
            **tb_dict
        }

        loss = loss_trans
        return loss, tb_dict, disp_dict

    def post_processing(self, batch_dict):
        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size']
        final_pred_dict = batch_dict['final_box_dicts']
        recall_dict = {}
        for index in range(batch_size):
            pred_boxes = final_pred_dict[index]['pred_boxes']

            recall_dict = self.generate_recall_record(
                box_preds=pred_boxes,
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=post_process_cfg.RECALL_THRESH_LIST
            )

        return final_pred_dict, recall_dict
    
    # Overwritten to allow parameters to be loaded from a checkpoint file using Agnostic Head. The head weights are not loaded.
    def load_params_from_file(self, filename, logger, to_cpu=False, pre_trained_path=None):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        model_state_disk = checkpoint['model_state']
        if not pre_trained_path is None:
            pretrain_checkpoint = torch.load(pre_trained_path, map_location=loc_type)
            pretrain_model_state_disk = pretrain_checkpoint['model_state']
            model_state_disk.update(pretrain_model_state_disk)

        version = checkpoint.get("version", None)
        if version is not None:
            logger.info('==> Checkpoint trained from version: %s' % version)

        if self.model_cfg.DENSE_HEAD.get('RETRAIN_HEAD', False) and self.training:
            model_state_disk_headless = {key: value for key, value in model_state_disk.items() if "dense_head" not in key}
            model_state_disk  = model_state_disk_headless
            logger.info('==> Retraining head from scratch ...')

        state_dict, update_model_state = self._load_state_dict(model_state_disk, strict=False)

        for key in state_dict:
            if key not in update_model_state:
                logger.info('Not updated weight %s: %s' % (key, str(state_dict[key].shape)))

        logger.info('==> Done (loaded %d/%d)' % (len(update_model_state), len(state_dict)))
