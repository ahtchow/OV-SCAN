# --------------------------------------------------------
# DCNv4
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
import torch
import torch.nn as nn
from collections import OrderedDict
import torch.utils.checkpoint as checkpoint
from timm.models.layers import trunc_normal_, DropPath
from mmengine.runner.checkpoint import _load_checkpoint
from mmengine.model import constant_init, trunc_normal_init
import DCNv4

from ..model_utils.flash_intern_image_utils import build_norm_layer
from ..model_utils.flash_intern_image_utils import StemLayer, DownsampleLayer, MLPLayer

class InternImageLayer(nn.Module):
    r""" Basic layer of InternImage
    Args:
        core_op (nn.Module): core operation of InternImage
        channels (int): number of input channels
        groups (list): Groups of each block.
        mlp_ratio (float): ratio of mlp hidden features to input channels
        drop (float): dropout rate
        drop_path (float): drop path rate
        act_layer (str): activation layer
        norm_layer (str): normalization layer
        post_norm (bool): whether to use post normalization
        layer_scale (float): layer scale
        offset_scale (float): offset scale
        with_cp (bool): whether to use checkpoint
    """

    def __init__(self,
                 core_op,
                 channels,
                 groups,
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.,
                 act_layer='GELU',
                 norm_layer='LN',
                 post_norm=False,
                 layer_scale=None,
                 offset_scale=1.0,
                 with_cp=False,
                 dcn_output_bias=False,
                 mlp_fc2_bias=False,
                 dw_kernel_size=None, # for InternImage-H/G
                 res_post_norm=False, # for InternImage-H/G
                 center_feature_scale=False): # for InternImage-H/G
        super().__init__()
        self.channels = channels
        self.groups = groups
        self.mlp_ratio = mlp_ratio
        self.with_cp = with_cp

        self.norm1 = build_norm_layer(channels, 'LN')
        self.post_norm = post_norm
        self.dcn = core_op(
            channels=channels,
            group=groups,
            offset_scale=offset_scale,
            dw_kernel_size=dw_kernel_size,
            output_bias=dcn_output_bias,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.norm2 = build_norm_layer(channels, 'LN')
        self.mlp = MLPLayer(in_features=channels,
                            hidden_features=int(channels * mlp_ratio),
                            act_layer=act_layer,
                            drop=drop,
                            mlp_fc2_bias=mlp_fc2_bias
                            )
        self.layer_scale = layer_scale is not None
        if self.layer_scale:
            self.gamma1 = nn.Parameter(layer_scale * torch.ones(channels),
                                       requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(channels),
                                       requires_grad=True)
        self.res_post_norm = res_post_norm
        if res_post_norm:
            self.res_post_norm1 = build_norm_layer(channels, 'LN')
            self.res_post_norm2 = build_norm_layer(channels, 'LN')
    def forward(self, x, shape, level_idx=0):

        def _inner_forward(x, shape, level_idx):
            if not self.layer_scale:
                if self.post_norm:
                    x = x + self.drop_path(self.norm1(self.dcn(x, shape, level_idx)))
                    x = x + self.drop_path(self.norm2(self.mlp(x, shape, level_idx)))
                elif self.res_post_norm: # for InternImage-H/G
                    x = x + self.drop_path(self.res_post_norm1(self.dcn(self.norm1(x), shape, level_idx)))
                    x = x + self.drop_path(self.res_post_norm2(self.mlp(self.norm2(x), shape, level_idx)))

                else:
                    x = x + self.drop_path(self.dcn(self.norm1(x), shape, level_idx))
                    x = x + self.drop_path(self.mlp(self.norm2(x), shape, level_idx))
                return x
            if self.post_norm:
                x = x + self.drop_path(self.gamma1 * self.norm1(self.dcn(x, shape)))
                x = x + self.drop_path(self.gamma2 * self.norm2(self.mlp(x, shape, level_idx)))
            else:
                x = x + self.drop_path(self.gamma1 * self.dcn(self.norm1(x), shape))
                x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x), shape, level_idx))
            return x

        if self.with_cp and x.requires_grad:
            x = checkpoint.checkpoint(_inner_forward, x, shape, level_idx)
        else:
            x = _inner_forward(x, shape, level_idx)

        return x


class InternImageBlock(nn.Module):
    r""" Block of InternImage
    Args:
        core_op (nn.Module): core operation of InternImage
        channels (int): number of input channels
        depths (list): Depth of each block.
        groups (list): Groups of each block.
        mlp_ratio (float): ratio of mlp hidden features to input channels
        drop (float): dropout rate
        drop_path (float): drop path rate
        act_layer (str): activation layer
        norm_layer (str): normalization layer
        post_norm (bool): whether to use post normalization
        layer_scale (float): layer scale
        offset_scale (float): offset scale
        with_cp (bool): whether to use checkpoint
    """

    def __init__(self,
                 core_op,
                 channels,
                 depth,
                 groups,
                 downsample=True,
                 downsample_layer=DownsampleLayer,
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.,
                 act_layer='GELU',
                 norm_layer='LN',
                 post_norm=False,
                 offset_scale=0.5,
                 layer_scale=None,
                 with_cp=False,
                 dcn_output_bias=False,
                 mlp_fc2_bias=False,
                 dw_kernel_size=None, # for InternImage-H/G
                 post_norm_block_ids=None, # for InternImage-H/G
                 res_post_norm=False, # for InternImage-H/G
                 center_feature_scale=False): # for InternImage-H/G
        super().__init__()
        self.channels = channels
        self.depth = depth
        self.post_norm = post_norm
        self.center_feature_scale = center_feature_scale

        self.blocks = nn.ModuleList([
            InternImageLayer(
                core_op=core_op,
                channels=channels,
                groups=groups,
                mlp_ratio=mlp_ratio,
                drop=drop,
                drop_path=drop_path[i] if isinstance(
                    drop_path, list) else drop_path,
                act_layer=act_layer,
                norm_layer=norm_layer,
                post_norm=post_norm,
                layer_scale=layer_scale,
                offset_scale=offset_scale,
                with_cp=with_cp,
                dcn_output_bias=dcn_output_bias,
                mlp_fc2_bias=mlp_fc2_bias,
                dw_kernel_size=dw_kernel_size, # for InternImage-H/G
                res_post_norm=res_post_norm, # for InternImage-H/G
                center_feature_scale=center_feature_scale # for InternImage-H/G
            ) for i in range(depth)
        ])
        if not self.post_norm or center_feature_scale:
            self.norm = build_norm_layer(channels, 'LN')
        self.post_norm_block_ids = post_norm_block_ids
        if post_norm_block_ids is not None: # for InternImage-H/G
            self.post_norms = nn.ModuleList(
                [build_norm_layer(channels, 'LN', eps=1e-6) for _ in post_norm_block_ids]
            )
        self.downsample = downsample_layer(
            channels=channels, norm_layer=norm_layer) if downsample else None


    def forward(self, x, return_wo_downsample=False, shape=None, level_idx=0
    ):
        for i, blk in enumerate(self.blocks):
            x = blk(x, shape=shape, level_idx=level_idx)
            if (self.post_norm_block_ids is not None) and (i in self.post_norm_block_ids):
                index = self.post_norm_block_ids.index(i)
                x = self.post_norms[index](x) # for InternImage-H/G
        if not self.post_norm or self.center_feature_scale:
            x = self.norm(x)
        if return_wo_downsample:
            x_ = x.clone()
        if self.downsample is not None:
            x, shape = self.downsample(x, shape=shape)

        if return_wo_downsample:
            return x, x_, shape
        return x, shape

class FlashInternImage(nn.Module):
    r""" FlashInternImage
        A PyTorch impl based on :
            `InternImage: Exploring Large-Scale Vision Foundation Models with Deformable Convolutions`  -
            https://arxiv.org/pdf/2103.14030
            'DCNv4': TODO: add arxiv
    Args:
        core_op (str): Core operator. Default: 'DCNv4'
        channels (int): Number of the first stage. Default: 64
        depths (list): Depth of each block. Default: [3, 4, 18, 5]
        groups (list): Groups of each block. Default: [3, 6, 12, 24]
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        drop_rate (float): Probability of an element to be zeroed. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2
        act_layer (str): Activation layer. Default: 'GELU'
        norm_layer (str): Normalization layer. Default: 'LN'
        layer_scale (bool): Whether to use layer scale. Default: False
        cls_scale (bool): Whether to use class scale. Default: False
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
        dw_kernel_size (int): Size of the dwconv. Default: None
        use_clip_projector (bool): Whether to use clip projector. Default: False
        level2_post_norm (bool): Whether to use level2 post norm. Default: False
        level2_post_norm_block_ids (list): Indexes of post norm blocks. Default: None
        res_post_norm (bool): Whether to use res post norm. Default: False
        center_feature_scale (bool): Whether to use center feature scale. Default: False
    """

    def __init__(self, model_cfg):
        super().__init__()

        self.model_cfg = model_cfg
        core_op = self.model_cfg.get('CORE_OP', 'DCNv4')
        channels = self.model_cfg.get('CHANNELS', 64)
        depths = self.model_cfg.get('DEPTHS', [3, 4, 18, 5])
        groups = self.model_cfg.get('GROUPS', [3, 6, 12, 24])
        mlp_ratio = self.model_cfg.get('MLP_RATIO', 4.)
        drop_rate = self.model_cfg.get('DROP_RATE', 0.)
        drop_path_rate = self.model_cfg.get('DROP_PATH_RATE', 0.2)
        drop_path_type = self.model_cfg.get('DROP_PATH_TYPE', 'linear')
        act_layer = self.model_cfg.get('ACT_LAYER', 'GELU')
        norm_layer = self.model_cfg.get('NORM_LAYER', 'LN')
        layer_scale = self.model_cfg.get('LAYER_SCALE', None)
        offset_scale = self.model_cfg.get('OFFSET_SCALE', 0.5)
        post_norm = self.model_cfg.get('POST_NORM', False)
        with_cp = self.model_cfg.get('WITH_CP', False)
        mlp_fc2_bias = self.model_cfg.get('MLP_FC2_BIAS', False)
        dcn_output_bias = self.model_cfg.get('DCN_OUTPUT_BIAS', False)
        dw_kernel_size = self.model_cfg.get('DW_KERNEL_SIZE', None)
        level2_post_norm = self.model_cfg.get('LEVEL2_POST_NORM', False)
        level2_post_norm_block_ids = self.model_cfg.get('LEVEL2_POST_NORM_BLOCK_IDS', None)
        res_post_norm = self.model_cfg.get('RES_POST_NORM', False)
        center_feature_scale = self.model_cfg.get('CENTER_FEATURE_SCALE', False)
        out_indices = self.model_cfg.get('OUT_INDICES', (0, 1, 2, 3))
        init_cfg = self.model_cfg.get('INIT_CFG', None)

        self.core_op = core_op
        self.num_levels = len(depths)
        self.depths = depths
        self.channels = channels
        self.num_features = int(channels * 2**(self.num_levels - 1))
        self.post_norm = post_norm
        self.mlp_ratio = mlp_ratio
        self.init_cfg = init_cfg
        self.out_indices = out_indices
        self.level2_post_norm_block_ids = level2_post_norm_block_ids

        in_chans = 3
        self.patch_embed = StemLayer(in_chans=in_chans,
                                     out_chans=channels,
                                     act_layer=act_layer,
                                     norm_layer=norm_layer)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]
        if drop_path_type == 'uniform':
            for i in range(len(dpr)):
                dpr[i] = drop_path_rate
   
        self.levels = nn.ModuleList()
        for i in range(self.num_levels):
            post_norm_block_ids = level2_post_norm_block_ids if level2_post_norm and (
                i == 2) else None # for InternImage-H/G

            level = InternImageBlock(
                core_op=getattr(DCNv4, core_op),
                channels=int(channels * 2**i),
                depth=depths[i],
                groups=groups[i],
                mlp_ratio=self.mlp_ratio,
                drop=drop_rate,
                drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                act_layer=act_layer,
                norm_layer=norm_layer,
                post_norm=post_norm,
                downsample=(i < self.num_levels - 1),
                downsample_layer = DownsampleLayer,
                layer_scale=layer_scale,
                offset_scale=offset_scale,
                with_cp=with_cp,
                mlp_fc2_bias=mlp_fc2_bias,
                dcn_output_bias=dcn_output_bias,
                dw_kernel_size=dw_kernel_size,  # for InternImage-H/G
                post_norm_block_ids=post_norm_block_ids, # for InternImage-H/G
                res_post_norm=res_post_norm, # for InternImage-H/G
                center_feature_scale=center_feature_scale # for InternImage-H/G
            )
            self.levels.append(level)

        self.num_layers = len(depths)
        self.apply(self._init_weights)
        self.apply(self._init_deform_weights)

    def init_weights(self):
        if self.init_cfg is None:
            print(f'No pre-trained weights for '
                        f'{self.__class__.__name__}, '
                        f'training start from scratch')
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, 1.0)        
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            ckpt = _load_checkpoint(self.init_cfg.checkpoint,
                                    logger=None,
                                    map_location='cpu')
            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            elif 'model_ema' in ckpt:
                _state_dict = ckpt['model_ema']
            elif 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt                
            
            state_dict = OrderedDict()
            for k, v in _state_dict.items():
                if k.startswith('backbone.'):
                    state_dict[k[9:]] = v
                else:
                    state_dict[k] = v

            # strip prefix of state_dict
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[7:]: v for k, v in state_dict.items()}

            # load state_dict
            meg = self.load_state_dict(state_dict, False)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _init_deform_weights(self, m):
        if isinstance(m, getattr(DCNv4, self.core_op)):
            m._reset_parameters()

    @torch.jit.ignore
    def lr_decay_keywards(self, decay_ratio=0.87):
        lr_ratios = {}

        # blocks
        idx = 0
        for i in range(4):
            layer_num = 3 - i  # 3 2 1 0
            for j in range(self.depths[layer_num]):
                block_num = self.depths[layer_num] - j - 1
                tag = 'levels.{}.blocks.{}.'.format(layer_num, block_num)
                decay = 1.0 * (decay_ratio**idx)
                lr_ratios[tag] = decay
                idx += 1
        # patch_embed (before stage-1)
        lr_ratios["patch_embed"] = lr_ratios['levels.0.blocks.0.']
        # levels.0.downsample (between stage-1 and stage-2)
        lr_ratios["levels.0.downsample"] = lr_ratios['levels.1.blocks.0.']
        lr_ratios["levels.0.norm"] = lr_ratios['levels.1.blocks.0.']
        # levels.1.downsample (between stage-2 and stage-3)
        lr_ratios["levels.1.downsample"] = lr_ratios['levels.2.blocks.0.']
        lr_ratios["levels.1.norm"] = lr_ratios['levels.2.blocks.0.']
        # levels.2.downsample (between stage-3 and stage-4)
        lr_ratios["levels.2.downsample"] = lr_ratios['levels.3.blocks.0.']
        lr_ratios["levels.2.norm"] = lr_ratios['levels.3.blocks.0.']
        return lr_ratios

    def forward(self, batch_dict):
        x = batch_dict['camera_imgs']
        B, N, C, H, W = x.size()
        x = x.view(B*N, C, H, W)
        x = self.patch_embed(x)
        _, H_, W_, C = x.shape
        x = x.view(B*N, H_*W_, C)

        shape=(H_, W_)
        seq_out = []
        for level_idx, level in enumerate(self.levels):
            old_shape = shape
            x, x_ , shape = level(x, return_wo_downsample=True, shape=shape, level_idx=level_idx)   
            if level_idx in self.out_indices:
                h, w= old_shape
                seq_out.append(x_.reshape(B*N, h, w, -1).permute(0, 3, 1, 2))
        batch_dict['image_features'] = seq_out
        return batch_dict
