# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import (
    Dict,
    List,
    Tuple,
    Optional
)
# from mmcv.cnn import xavier_init, constant_init, kaiming_init
from mmcv.cnn import Conv2d, Linear, bias_init_with_prob
from mmcv.cnn.bricks.transformer import build_positional_encoding
from mmcv.runner import force_fp32
from mmdet.core import (
    # bbox_cxcywh_to_xyxy,
    # bbox_xyxy_to_cxcywh,
    build_assigner,
    build_sampler,
    multi_apply,
    reduce_mean,
)
from mmdet.models.utils import build_transformer, NormedLinear
from mmdet.models import (
    HEADS,
    build_loss,
)
from mmdet.models.dense_heads.anchor_free_head import AnchorFreeHead
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet3d.core.bbox.coders import build_bbox_coder
from mmdet3d.models.builder import build_neck
from mmdet3d.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes
from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox
from projects.mmdet3d_plugin.models.utils import depth_utils
from projects.mmdet3d_plugin.models.utils.projection_utils import NormalizeMode, convert_to_homogeneous, project_ego_to_image


def pos2posemb3d(pos, num_pos_feats=128, temperature=10000):
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
    pos_x = pos[..., 0, None] / dim_t
    pos_y = pos[..., 1, None] / dim_t
    pos_z = pos[..., 2, None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_z = torch.stack((pos_z[..., 0::2].sin(), pos_z[..., 1::2].cos()), dim=-1).flatten(-2)
    posemb = torch.cat((pos_y, pos_x, pos_z), dim=-1)
    return posemb


@HEADS.register_module()
class DepthrHead(AnchorFreeHead):
    """Implements the DETR transformer head.
    See `paper: End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.
    Args:
        num_classes (int): Number of categories excluding the background.
        in_channels (int): Number of channels in the input feature map.
        num_query (int): Number of query in Transformer.
        num_reg_fcs (int, optional): Number of fully-connected layers used in
            `FFN`, which is then used for the regression head. Default 2.
        transformer (obj:`mmcv.ConfigDict`|dict): Config for transformer.
            Default: None.
        sync_cls_avg_factor (bool): Whether to sync the avg_factor of
            all ranks. Default to False.
        positional_encoding (obj:`mmcv.ConfigDict`|dict):
            Config for position encoding.
        loss_cls (obj:`mmcv.ConfigDict`|dict): Config of the
            classification loss. Default `CrossEntropyLoss`.
        loss_bbox (obj:`mmcv.ConfigDict`|dict): Config of the
            regression loss. Default `L1Loss`.
        loss_iou (obj:`mmcv.ConfigDict`|dict): Config of the
            regression iou loss. Default `GIoULoss`.
        tran_cfg (obj:`mmcv.ConfigDict`|dict): Training config of
            transformer head.
        test_cfg (obj:`mmcv.ConfigDict`|dict): Testing config of
            transformer head.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None

        depth_predictor (obj:`ConfigDict`): ConfigDict is used for building
            depth_predictor, which use feature map to predict weight depth
            distribution and depth embedding.
            `Optional[ConfigDict]`
        depth_gt_encoder (obj:`ConfigDict`): ConfigDict is used for building
            depth_gt_encoder, which use convolutional layer to suppress gt_depth_maps
            to gt_depth_embedding.
            `Optional[ConfigDict]`
        loss_ddn (obj:`ConfigDict`): ConfigDict is used for building
            loss_ddn, which use for the supervision of predicted depth maps
            `Optional[ConfigDict]`
        loss_depth (bool): whether to generate depth_ave from sampling predicted depth maps.
            Default to False.
    """
    _version = 2

    def __init__(self,
                 num_classes,
                 in_channels,
                 num_query=100,
                 num_reg_fcs=2,
                 transformer=None,
                 sync_cls_avg_factor=False,
                 positional_encoding=dict(
                     type='SinePositionalEncoding',
                     num_feats=128,
                     normalize=True),
                 code_weights=None,
                 bbox_coder=None,
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     bg_cls_weight=0.1,
                     use_sigmoid=False,
                     loss_weight=1.0,
                     class_weight=1.0),
                 loss_bbox=dict(type='L1Loss', loss_weight=5.0),
                 loss_iou=dict(type='GIoULoss', loss_weight=2.0),
                 train_cfg=dict(
                     assigner=dict(
                         type='HungarianAssigner',
                         cls_cost=dict(type='ClassificationCost', weight=1.),
                         reg_cost=dict(type='BBoxL1Cost', weight=5.0),
                         iou_cost=dict(
                             type='IoUCost', iou_mode='giou', weight=2.0))),
                 test_cfg=dict(max_per_img=100),
                 with_position=True,
                 with_multiview=False,
                 depth_step=0.8,
                 depth_num=64,
                 LID=False,
                 depth_start=1,
                 position_range=[-65, -65, -8.0, 65, 65, 8.0],
                 init_cfg=None,
                 normedlinear=False,
                 depth_predictor=None,
                 depth_gt_encoder=None,
                 loss_ddn=None,
                 loss_depth=False,
                 **kwargs):
        # NOTE here use `AnchorFreeHead` instead of `TransformerHead`,
        # since it brings inconvenience when the initialization of
        # `AnchorFreeHead` is called.
        if 'code_size' in kwargs:
            self.code_size = kwargs['code_size']
        else:
            self.code_size = 10
        if code_weights is not None:
            self.code_weights = code_weights
        else:
            self.code_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]
        self.code_weights = self.code_weights[:self.code_size]
        self.bg_cls_weight = 0
        self.sync_cls_avg_factor = sync_cls_avg_factor
        class_weight = loss_cls.get('class_weight', None)
        if class_weight is not None and (self.__class__ is DepthrHead):
            assert isinstance(class_weight, float), 'Expected ' \
                'class_weight to have type float. Found ' \
                f'{type(class_weight)}.'
            # NOTE following the official DETR rep0, bg_cls_weight means
            # relative classification weight of the no-object class.
            bg_cls_weight = loss_cls.get('bg_cls_weight', class_weight)
            assert isinstance(bg_cls_weight, float), 'Expected ' \
                'bg_cls_weight to have type float. Found ' \
                f'{type(bg_cls_weight)}.'
            class_weight = torch.ones(num_classes + 1) * class_weight
            # set background class as the last indice
            class_weight[num_classes] = bg_cls_weight
            loss_cls.update({'class_weight': class_weight})
            if 'bg_cls_weight' in loss_cls:
                loss_cls.pop('bg_cls_weight')
            self.bg_cls_weight = bg_cls_weight

        if train_cfg:
            assert 'assigner' in train_cfg, 'assigner should be provided '\
                'when train_cfg is set.'
            assigner = train_cfg['assigner']
            assert loss_cls['loss_weight'] == assigner['cls_cost']['weight'], \
                'The classification weight for loss and matcher should be' \
                'exactly the same.'
            assert loss_bbox['loss_weight'] == assigner['reg_cost'][
                'weight'], 'The regression L1 weight for loss and matcher ' \
                'should be exactly the same.'
            # assert loss_iou['loss_weight'] == assigner['iou_cost']['weight'], \
            #     'The regression iou weight for loss and matcher should be' \
            #     'exactly the same.'
            self.assigner = build_assigner(assigner)
            # DETR sampling=False, so use PseudoSampler
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)

        self.num_query = num_query
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.num_reg_fcs = num_reg_fcs
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False

        self.embed_dims = 256
        self.depth_step = depth_step
        self.depth_num = depth_num
        self.position_dim = 3 * self.depth_num
        self.position_range = position_range
        self.LID = LID
        self.depth_start = depth_start
        self.position_level = 0
        self.with_position = with_position
        self.with_multiview = with_multiview
        assert 'num_feats' in positional_encoding
        num_feats = positional_encoding['num_feats']
        assert num_feats * 2 == self.embed_dims, 'embed_dims should' \
            f' be exactly 2 times of num_feats. Found {self.embed_dims}' \
            f' and {num_feats}.'
        self.act_cfg = transformer.get('act_cfg',
                                       dict(type='ReLU', inplace=True))
        self.num_pred = 6
        self.normedlinear = normedlinear
        super(DepthrHead, self).__init__(num_classes, in_channels, init_cfg=init_cfg)

        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_iou = build_loss(loss_iou)

        if self.loss_cls.use_sigmoid:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1
        # self.activate = build_activation_layer(self.act_cfg)
        # if self.with_multiview or not self.with_position:
        #     self.positional_encoding = build_positional_encoding(
        #         positional_encoding)
        self.positional_encoding = build_positional_encoding(
            positional_encoding)
        self.transformer = build_transformer(transformer)
        self.code_weights = nn.Parameter(torch.tensor(
            self.code_weights, requires_grad=False), requires_grad=False)
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.pc_range = self.bbox_coder.pc_range

        """
        Operation for depth embedding
            depth_bin_cfg: Config for depth_utils.bin_depths
            depth_predictor (obj:`ConfigDict`): ConfigDict is used for building
                depth_predictor, which use feature map to predict weight depth
                distribution and depth embedding.
                `Optional[ConfigDict]`
            depth_gt_encoder (obj:`ConfigDict`): ConfigDict is used for building
                deoth_gt_encoder, which use convolutional layer to suppress gt_depth_maps
                to gt_depth_embedding.
                `Optional[ConfigDict]`
        """
        self.depth_bin_cfg = None
        self.depth_predictor = None
        self.depth_gt_encoder = None
        self.loss_ddn = None
        self.depth_maps_down_scale = 8
        self.gt_depth_maps_down_scale = 8

        if depth_predictor is not None:
            self.depth_predictor = build_neck(depth_predictor)
            self.depth_bin_cfg = dict(
                mode="LID",
                depth_min=depth_predictor.get("depth_min"),
                depth_max=depth_predictor.get("depth_max"),
                num_depth_bins=depth_predictor.get("num_depth_bins"),
            )

        if depth_gt_encoder is not None:
            self.depth_gt_encoder = build_neck(depth_gt_encoder)
            self.depth_bin_cfg = dict(
                mode="LID",
                depth_min=depth_gt_encoder.get("depth_min"),
                depth_max=depth_gt_encoder.get("depth_max"),
                num_depth_bins=depth_gt_encoder.get("num_depth_bins"),
            )
            self.gt_depth_maps_down_scale = depth_gt_encoder.get("gt_depth_maps_down_scale")

        if loss_ddn is not None:
            self.loss_ddn = build_loss(loss_ddn)
            self.depth_maps_down_scale = loss_ddn.get("downsample_factor")

        self.loss_depth = loss_depth

        self._init_layers()

    def _init_layers(self):
        """Initialize layers of the transformer head."""
        if self.with_position:
            self.input_proj = Conv2d(
                self.in_channels, self.embed_dims, kernel_size=1)
        else:
            self.input_proj = Conv2d(
                self.in_channels, self.embed_dims, kernel_size=1)

        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        if self.normedlinear:
            cls_branch.append(NormedLinear(self.embed_dims, self.cls_out_channels))
        else:
            cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)

        self.cls_branches = nn.ModuleList(
            [fc_cls for _ in range(self.num_pred)])
        self.reg_branches = nn.ModuleList(
            [reg_branch for _ in range(self.num_pred)])

        if self.with_multiview:
            self.adapt_pos3d = nn.Sequential(
                nn.Conv2d(self.embed_dims * 3 // 2, self.embed_dims * 4, kernel_size=1, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(self.embed_dims * 4, self.embed_dims, kernel_size=1, stride=1, padding=0),
            )
        else:
            self.adapt_pos3d = nn.Sequential(
                nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=1, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=1, stride=1, padding=0),
            )

        if self.with_position:
            self.position_encoder = nn.Sequential(
                nn.Conv2d(self.position_dim, self.embed_dims * 4, kernel_size=1, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(self.embed_dims * 4, self.embed_dims, kernel_size=1, stride=1, padding=0),
            )

        self.reference_points = nn.Embedding(self.num_query, 3)
        self.query_embedding = nn.Sequential(
            nn.Linear(self.embed_dims * 3 // 2, self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims),
        )

    def init_weights(self):
        """Initialize weights of the transformer head."""
        # The initialization for transformer is important
        self.transformer.init_weights()
        nn.init.uniform_(self.reference_points.weight.data, 0, 1)
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)

    def position_embeding(self, img_feats, img_metas, masks=None):
        eps = 1e-5
        pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]
        B, N, C, H, W = img_feats[self.position_level].shape
        coords_h = torch.arange(H, device=img_feats[0].device).float() * pad_h / H
        coords_w = torch.arange(W, device=img_feats[0].device).float() * pad_w / W

        if self.LID:
            index = torch.arange(start=0, end=self.depth_num, step=1, device=img_feats[0].device).float()
            index_1 = index + 1
            bin_size = (self.position_range[3] - self.depth_start) / (self.depth_num * (1 + self.depth_num))
            coords_d = self.depth_start + bin_size * index * index_1

            # print(f'self.position_range[3]: {self.position_range[3]}')
            # print(f'self.depth_start: {self.depth_start}')

            # print(f'bin_size: {bin_size}')
            # print(f'coords_d: {coords_d}')

        else:
            index = torch.arange(start=0, end=self.depth_num, step=1, device=img_feats[0].device).float()
            bin_size = (self.position_range[3] - self.depth_start) / self.depth_num
            coords_d = self.depth_start + bin_size * index

        D = coords_d.shape[0]
        coords = torch.stack(torch.meshgrid([coords_w, coords_h, coords_d])).permute(1, 2, 3, 0)  # W, H, D, 3
        coords = torch.cat((coords, torch.ones_like(coords[..., :1])), -1)
        coords[..., :2] = coords[..., :2] * torch.maximum(coords[..., 2:3], torch.ones_like(coords[..., 2:3]) * eps)

        img2lidars = []
        for img_meta in img_metas:
            img2lidar = []
            for i in range(len(img_meta['lidar2img'])):
                img2lidar.append(np.linalg.inv(img_meta['lidar2img'][i]))
            img2lidars.append(np.asarray(img2lidar))
        img2lidars = np.asarray(img2lidars)
        img2lidars = coords.new_tensor(img2lidars)  # (B, N, 4, 4)

        coords = coords.view(1, 1, W, H, D, 4, 1).repeat(B, N, 1, 1, 1, 1, 1)
        img2lidars = img2lidars.view(B, N, 1, 1, 1, 4, 4).repeat(1, 1, W, H, D, 1, 1)
        coords3d = torch.matmul(img2lidars, coords).squeeze(-1)[..., :3]
        coords3d[..., 0:1] = (coords3d[..., 0:1] - self.position_range[0]) / \
            (self.position_range[3] - self.position_range[0])
        coords3d[..., 1:2] = (coords3d[..., 1:2] - self.position_range[1]) / \
            (self.position_range[4] - self.position_range[1])
        coords3d[..., 2:3] = (coords3d[..., 2:3] - self.position_range[2]) / \
            (self.position_range[5] - self.position_range[2])

        coords_mask = (coords3d > 1.0) | (coords3d < 0.0)
        coords_mask = coords_mask.flatten(-2).sum(-1) > (D * 0.5)
        coords_mask = masks | coords_mask.permute(0, 1, 3, 2)
        coords3d = coords3d.permute(0, 1, 4, 5, 3, 2).contiguous().view(B * N, -1, H, W)
        coords3d = inverse_sigmoid(coords3d)
        coords_position_embeding = self.position_encoder(coords3d)

        return coords_position_embeding.view(B, N, self.embed_dims, H, W), coords_mask

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """load checkpoints."""
        # NOTE here use `AnchorFreeHead` instead of `TransformerHead`,
        # since `AnchorFreeHead._load_from_state_dict` should not be
        # called here. Invoking the default `Module._load_from_state_dict`
        # is enough.

        # Names of some parameters in has been changed.
        version = local_metadata.get('version', None)
        if (version is None or version < 2) and self.__class__ is DepthrHead:
            convert_dict = {
                '.self_attn.': '.attentions.0.',
                # '.ffn.': '.ffns.0.',
                '.multihead_attn.': '.attentions.1.',
                '.decoder.norm.': '.decoder.post_norm.'
            }
            state_dict_keys = list(state_dict.keys())
            for k in state_dict_keys:
                for ori_key, convert_key in convert_dict.items():
                    if ori_key in k:
                        convert_key = k.replace(ori_key, convert_key)
                        state_dict[convert_key] = state_dict[k]
                        del state_dict[k]

        super(AnchorFreeHead,
              self)._load_from_state_dict(state_dict, prefix, local_metadata,
                                          strict, missing_keys,
                                          unexpected_keys, error_msgs)

    def forward(
        self,
        mlvl_feats,
        img_metas,
        gt_bboxes_3d=None,
    ):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
            img_metas: A list of dict containing the `lidar2img` tensor.
            gt_bboxes_3d: The ground truth list of `LiDARInstance3DBoxes`.

        Returns:
            all_cls_scores (Tensor): Outputs from the classification head,
                shape [nb_dec, bs, num_query, cls_out_channels]. Note
                cls_out_channels should includes background.
            all_bbox_preds(torch.Tensor): Sigmoid outputs from the regression
                head with normalized coordinate format
                (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy).
                `[num_layer, B, num_queries, 10]`
        """

        x = mlvl_feats[0]
        batch_size, num_cams = x.size(0), x.size(1)
        input_img_h, input_img_w, _ = img_metas[0]['pad_shape'][0]
        masks = x.new_ones(
            (batch_size, num_cams, input_img_h, input_img_w))
        for img_id in range(batch_size):
            for cam_id in range(num_cams):
                img_h, img_w, _ = img_metas[img_id]['img_shape'][cam_id]
                masks[img_id, cam_id, :img_h, :img_w] = 0
        x = self.input_proj(x.flatten(0, 1))
        x = x.view(batch_size, num_cams, *x.shape[-3:])
        # interpolate masks to have the same spatial shape with x
        masks = F.interpolate(
            masks, size=x.shape[-2:]).to(torch.bool)

        if self.with_position:
            coords_position_embeding, _ = self.position_embeding(mlvl_feats, img_metas, masks)
            pos_embed = coords_position_embeding
            if self.with_multiview:
                sin_embed = self.positional_encoding(masks)
                sin_embed = self.adapt_pos3d(sin_embed.flatten(0, 1)).view(x.size())
                pos_embed = pos_embed + sin_embed
            else:
                pos_embeds = []
                for i in range(num_cams):
                    xy_embed = self.positional_encoding(masks[:, i, :, :])
                    pos_embeds.append(xy_embed.unsqueeze(1))
                sin_embed = torch.cat(pos_embeds, 1)
                sin_embed = self.adapt_pos3d(sin_embed.flatten(0, 1)).view(x.size())
                pos_embed = pos_embed + sin_embed
        else:
            if self.with_multiview:
                pos_embed = self.positional_encoding(masks)
                pos_embed = self.adapt_pos3d(pos_embed.flatten(0, 1)).view(x.size())
            else:
                pos_embeds = []
                for i in range(num_cams):
                    pos_embed = self.positional_encoding(masks[:, i, :, :])
                    pos_embeds.append(pos_embed.unsqueeze(1))
                pos_embed = torch.cat(pos_embeds, 1)

        reference_points = self.reference_points.weight
        query_embeds = self.query_embedding(pos2posemb3d(reference_points))
        reference_points = reference_points.unsqueeze(0).repeat(batch_size, 1, 1)  # .sigmoid()

        # Operations for depth embedding
        depth_pos_embed = None
        pred_depth_map_logits = None
        weighted_depth = None
        if self.depth_predictor is not None:
            # pred_depth_map_logits: [B, N, D, H, W]
            # depth_pos_embed: [B, N, C, H, W]
            # weighted_depth(pred_depth_map_values): [B, N, H, W]
            pred_depth_map_logits, depth_pos_embed, weighted_depth = self.depth_predictor(
                mlvl_feats=mlvl_feats,
                mask=None,
                pos=None,
            )
            # print(f'pred_depth_map_logits: {pred_depth_map_logits.shape[:]}')

        if self.depth_gt_encoder is not None:
            assert gt_bboxes_3d is not None
            # gt_depth_maps(gt_depth_map_probs) with depth_gt_encoder: [B, N, H, W, num_depth_bins], dtype: torch.float32
            gt_depth_maps, gt_bboxes_2d = self.get_depth_map_and_gt_bboxes_2d(
                gt_bboxes_list=gt_bboxes_3d,
                img_metas=img_metas,
                target=False,
                device=mlvl_feats[0].device,
                depth_maps_down_scale=self.gt_depth_maps_down_scale,
            )
            gt_depth_maps = gt_depth_maps.to(mlvl_feats[0].device)
            # print(f'gt_depth_maps: {gt_depth_maps.shape}')
            # We do not need pred_depth_map_logits and weighted_depth to compute
            # loss_ddn when using gt_depth_maps
            _, depth_pos_embed, _ = self.depth_gt_encoder(
                mlvl_feats=mlvl_feats,
                mask=None,
                pos=None,
                gt_depth_maps=gt_depth_maps,
            )

        # out_dec: [num_layers, num_query, bs, dim]
        outs_dec, _ = self.transformer(
            x=x,
            mask=masks,
            query_embed=query_embeds,
            pos_embed=pos_embed,
            reg_branch=self.reg_branches,
            depth_pos_embed=depth_pos_embed,
        )

        outs_dec = torch.nan_to_num(outs_dec)

        outputs_classes = []
        outputs_coords = []

        for lvl in range(outs_dec.shape[0]):
            reference = inverse_sigmoid(reference_points.clone())
            assert reference.shape[-1] == 3
            outputs_class = self.cls_branches[lvl](outs_dec[lvl])
            tmp = self.reg_branches[lvl](outs_dec[lvl])

            tmp[..., 0:2] += reference[..., 0:2]
            tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
            tmp[..., 4:5] += reference[..., 2:3]
            tmp[..., 4:5] = tmp[..., 4:5].sigmoid()

            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        all_cls_scores = torch.stack(outputs_classes)
        all_bbox_preds = torch.stack(outputs_coords)

        all_bbox_preds[..., 0:1] = (all_bbox_preds[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0])
        all_bbox_preds[..., 1:2] = (all_bbox_preds[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1])
        all_bbox_preds[..., 4:5] = (all_bbox_preds[..., 4:5] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2])

        outs = {
            'all_cls_scores': all_cls_scores,
            'all_bbox_preds': all_bbox_preds,
            'enc_cls_scores': None,
            'enc_bbox_preds': None,
            # pred_depth_map_logits
            'pred_depth_map_logits': pred_depth_map_logits,
            # pred_depth_map_values
            'weighted_depth': weighted_depth,
        }
        return outs

    def get_depth_map_and_gt_bboxes_2d(
        self,
        gt_bboxes_list: List[LiDARInstance3DBoxes],
        img_metas: List[Dict[str, torch.Tensor]],
        target: bool = True,
        device: Optional[torch.device] = None,
        depth_maps_down_scale: int = 8,
    ) -> Tuple[torch.Tensor, List[List[torch.Tensor]]]:
        """Get depth map and the 2D ground truth bboxes.

        Args:
            gt_bboxes_list: The ground truth list of `LiDARInstance3DBoxes`.
            img_metas: A list of dict containing the `lidar2img` tensor.
            target: If true, the returned `gt_depth_maps` will only have indices instead of another class dimension. Default: True.
            device: The device of the input image feature map.
            depth_maps_down_scale: The down scale of gt_depth_maps. Default: 8.
        Returns:
            gt_depth_maps: Thr ground truth depth maps with shape
                `gt_depth_map_indices: `[batch, num_cameras, depth_map_H, depth_map_W] if `target` is true.
                Otherwise, 
                `gt_depth_map_probs:` [batch, num_cameras, depth_map_H, depth_map_W, num_depth_bins].
            gt_bboxes_2d: A list of list of tensor containing 2D ground truth bboxes (x, y, w, h)
                for each sample and each camera. Each tensor has shape [N_i, 4].
                Below is the brief explanation of a single batch:
                [B, N, ....]
                [[gt_bboxes_tensor_camera_0, gt_bboxes_tensor_camera_1, ..., gt_bboxes_tensor_camera_n] (sample 0),
                 [gt_bboxes_tensor_camera_0, gt_bboxes_tensor_camera_1, ..., gt_bboxes_tensor_camera_n] (sample 1),
                 ...].
        """
        img_H, img_W, _ = img_metas[0]['img_shape'][0]
        depth_map_H, depth_map_W = img_H // depth_maps_down_scale, img_W // depth_maps_down_scale

        gt_depth_maps = []
        gt_bboxes_2d = []

        resize_scale = img_H // depth_map_H
        assert resize_scale == img_W // depth_map_W

        for gt_bboxes, img_meta in zip(gt_bboxes_list, img_metas):
            # Check the gt_bboxes.tensor in case the empty bboxes
            # new version of mmdetection3d do not provide empety tensor
            if len(gt_bboxes.tensor) != 0:
                # [num_objects, 8, 3]
                gt_bboxes_corners = gt_bboxes.corners
                # [num_objects, 3].
                gt_bboxes_centers = gt_bboxes.gravity_center
            else:
                # [num_objects, 8, 3]
                gt_bboxes_corners = torch.empty([0, 8, 3], device=device)
                # [num_objects, 3].
                gt_bboxes_centers = torch.empty([0, 3], device=device)

            # [num_cameras, 3, 4]
            lidar2img = gt_bboxes_corners.new_tensor(img_meta['lidar2img'])[:, :3]
            assert tuple(lidar2img.shape) == (6, 3, 4)

            # Convert to homogeneous coordinate. [num_objects, 8, 4]
            gt_bboxes_corners = torch.cat([
                gt_bboxes_corners,
                gt_bboxes_corners.new_ones((*gt_bboxes_corners.shape[:-1], 1))
            ], dim=-1)

            # Convert to homogeneous coordinate. [num_objects, 4]
            gt_bboxes_centers = torch.cat([
                gt_bboxes_centers,
                gt_bboxes_centers.new_ones((*gt_bboxes_centers.shape[:-1], 1))
            ], dim=-1)

            # [num_cameras, num_objects, 8, 3]
            corners_uvd: torch.Tensor = torch.einsum('nij,mlj->nmli', lidar2img, gt_bboxes_corners)
            # [num_cameras, num_objects, 3]
            centers_uvd: torch.Tensor = torch.einsum('nij,mj->nmi', lidar2img, gt_bboxes_centers)
            # [num_cameras, num_objects]
            depth_targets = centers_uvd[..., 2]
            # [num_cameras, num_objects, 8]
            corners_depth_targets = corners_uvd[..., 2]

            # [num_cameras, num_objects, 8, 2]
            # fix for devide to zero
            corners_uv = corners_uvd[..., :2] / (corners_uvd[..., -1:] + 1e-8)

            depth_maps_all_camera = []
            gt_bboxes_all_camera = []
            # Generate depth maps and gt_bboxes for each camera.
            for corners_uv_per_camera, depth_target, corners_depth_target in zip(corners_uv, depth_targets, corners_depth_targets):
                # [num_objects, 8]
                visible = (corners_uv_per_camera[..., 0] > 0) & (corners_uv_per_camera[..., 0] < img_W) & \
                    (corners_uv_per_camera[..., 1] > 0) & (corners_uv_per_camera[..., 1] < img_H) & \
                    (corners_depth_target > 1)

                # [num_objects, 8]
                in_front = (corners_depth_target > 0.1)

                # [N,]
                # Filter num_objects in each camera
                mask = visible.any(dim=-1) & in_front.all(dim=-1)

                # [N, 8, 2]
                corners_uv_per_camera = corners_uv_per_camera[mask]

                # [N,]
                depth_target = depth_target[mask]

                # Resize corner for bboxes
                corners_uv_per_camera = (corners_uv_per_camera / resize_scale)

                # Clamp for depth
                corners_uv_per_camera[..., 0] = torch.clamp(corners_uv_per_camera[..., 0], 0, depth_map_W)
                corners_uv_per_camera[..., 1] = torch.clamp(corners_uv_per_camera[..., 1], 0, depth_map_H)

                # [N, 4]: (x_min, y_min, x_max, y_max)
                xy_min, _ = corners_uv_per_camera.min(dim=1)
                xy_max, _ = corners_uv_per_camera.max(dim=1)
                bboxes = torch.cat([xy_min, xy_max], dim=1).int()

                # [N, 4]: (x_min, y_min, w, h)
                bboxes[:, 2:] -= bboxes[:, :2]

                sort_by_depth = torch.argsort(depth_target, descending=True)
                bboxes = bboxes[sort_by_depth]
                depth_target = depth_target[sort_by_depth]

                # Fill into resize depth map = origin img /  resize_scale^2
                # depth_map = gt_bboxes_corners.new_zeros((img_H, img_W))
                depth_map = gt_bboxes_corners.new_zeros((depth_map_H, depth_map_W))

                for bbox, depth in zip(bboxes, depth_target):
                    x, y, w, h = bbox
                    depth_map[y:y + h, x:x + w] = depth

                gt_bboxes_all_camera.append(bboxes)
                depth_maps_all_camera.append(depth_map)

            # Visualizatioin for debugging
            # for i in range(6):
            #     print(f'i: {i+1}')
            #     heatmap = depth_maps_all_camera[i].detach().cpu().numpy().astype(np.uint8)
            #     print(f'type: {type(heatmap)}, shape: {heatmap.shape}')

            #     print(heatmap.min(), heatmap.max())
            #     heatmap = ((heatmap - heatmap.min()) / (heatmap.max() - heatmap.min()) * 255).astype(np.uint8)
            #     print(f'heatmap: {heatmap}')
            #     heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_OCEAN)
            #     print(f'heatmap.shape: {heatmap.shape}')
            #     cv2.imwrite(f'/home/cytseng/dhm{i + 1}.jpg', heatmap)

            # exit()

            # [num_cameras, depth_map_H, depth_map_W]
            depth_maps_all_camera = torch.stack(depth_maps_all_camera)

            gt_depth_maps.append(depth_maps_all_camera)
            gt_bboxes_2d.append(gt_bboxes_all_camera)

        # [batch, num_cameras, depth_map_H, depth_map_W]
        gt_depth_maps = torch.stack(gt_depth_maps)
        # [batch, num_cameras, depth_map_H, depth_map_W], dtype: torch.long if `target` is true.
        # Otherwise [batch, num_cameras, depth_map_H, depth_map_W, num_depth_bins], dtype: torch.float
        gt_depth_maps = depth_utils.bin_depths(gt_depth_maps, **self.depth_bin_cfg, target=target)
        return gt_depth_maps, gt_bboxes_2d

    def _get_target_single(self,
                           cls_score,
                           bbox_pred,
                           gt_labels,
                           gt_bboxes,
                           gt_bboxes_ignore=None):
        """"Compute regression and classification targets for one image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for 6 images(single sample), Outputs from the regression head with
                normalized coordinate format
                (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy).
                `[num_queries, 10]`
            gt_bboxes (Tensor): Ground truth bboxes for one image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth class indices for one image
                with shape (num_gts, ).
            gt_bboxes_ignore (Tensor, optional): Bounding boxes
                which can be ignored. Default None.
        Returns:
            tuple[Tensor]: a tuple containing the following for one image.
                - labels (Tensor): Labels of each image.
                - label_weights (Tensor]): Label weights of each image.
                - bbox_targets (Tensor): BBox targets of each image.
                - bbox_weights (Tensor): BBox weights of each image.
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
        """

        num_bboxes = bbox_pred.size(0)
        # assigner and sampler
        assign_result = self.assigner.assign(bbox_pred, cls_score, gt_bboxes,
                                             gt_labels, gt_bboxes_ignore)
        sampling_result = self.sampler.sample(assign_result, bbox_pred,
                                              gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        labels = gt_bboxes.new_full((num_bboxes, ),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        code_size = gt_bboxes.size(1)
        bbox_targets = torch.zeros_like(bbox_pred)[..., :code_size]
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0
        # print(gt_bboxes.size(), bbox_pred.size())
        # DETR
        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
        return (labels, label_weights, bbox_targets, bbox_weights,
                pos_inds, neg_inds)

    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None):
        """"Compute regression and classification targets for a batch image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each sample, with normalized coordinate
                normalized coordinate format
                (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy).
                `[num_queries, 10]`
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            tuple: a tuple containing the following targets.
                - labels_list (list[Tensor]): Labels for all images.
                - label_weights_list (list[Tensor]): Label weights for all \
                    images.
                - bbox_targets_list (list[Tensor]): BBox targets for all \
                    images.
                - bbox_weights_list (list[Tensor]): BBox weights for all \
                    images.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
        """
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, pos_inds_list, neg_inds_list) = multi_apply(
             self._get_target_single, cls_scores_list, bbox_preds_list,
             gt_labels_list, gt_bboxes_list, gt_bboxes_ignore_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg)

    def loss_single(self,
                    cls_scores,
                    bbox_preds,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images.
                Shape `[B, num_query, cls_out_channels]`.
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images(single sample), Outputs from the regression head with
                normalized coordinate format
                (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy).
                `[B, num_queries, 10]`
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = cls_scores.size(0)

        # Operation to transfrom from torch.Tensor[B, num_query, 10] to
        # list(B) of torch.Tensor[num_query, 10]

        # list(B) of tensor cls_scores_list: [num_queries, 10]
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        # list(B) of tensor bbox_preds_list: [num_queries, 10]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]

        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                           gt_bboxes_list, gt_labels_list,
                                           gt_bboxes_ignore_list)

        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_bbox(bbox_targets, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights

        loss_bbox = self.loss_bbox(
            bbox_preds[isnotnan, :10], normalized_bbox_targets[isnotnan, :10], bbox_weights[isnotnan, :10], avg_factor=num_total_pos)

        loss_cls = torch.nan_to_num(loss_cls)
        loss_bbox = torch.nan_to_num(loss_bbox)
        return loss_cls, loss_bbox

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             gt_bboxes_list,
             gt_labels_list,
             img_metas,
             preds_dicts,
             gt_bboxes_ignore=None):
        """"Loss function.
        Args:
            gt_bboxes_list (list[Tensor]): The ground truth list of
            `LiDARInstance3DBoxes`.

            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).

            img_metas: A list of dict containing the `lidar2img` tensor.

            preds_dicts:
                all_cls_scores (Tensor): Classification score of all
                    decoder layers, has shape
                    `[num_layer, bs, num_query, cls_out_channels]`.
                all_bbox_preds(torch.Tensor): Sigmoid regression outputs
                    of all decode layers. Outputs from the regression head with
                    normalized coordinate format
                    (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy).
                    `[num_layer, B, num_queries, 10]`
                enc_cls_scores (Tensor): Classification scores of
                    points on encode feature map , has shape
                    (N, h*w, num_classes). Only be passed when as_two_stage is
                    True, otherwise is None.
                enc_bbox_preds (Tensor): Regression results of each points
                    on the encode feature map, has shape (N, h*w, 4). Only be
                    passed when as_two_stage is True, otherwise is None.
                pred_depth_map_logits (Tensor): one hot encoding to represent the predict depth_maps,
                    in depth_gt_encoder default is None, defualt downsample to 1/32
                    `[B, N, D, H, W]`
                weighted_depth (Tensor): weighted sum value (real depth value) of predicted_depth_maps or gt_depth_maps
                    `[B, N, H, W]`
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'
        gt_bboxes_3d = gt_bboxes_list

        # Operation from prediction

        # all_cls_scores(torch.Tensor): [num_layer, B, num_queries, 10]
        all_cls_scores = preds_dicts['all_cls_scores']
        # all_bbox_preds(torch.Tensor): [num_layer, B, num_queries, 10]
        all_bbox_preds = preds_dicts['all_bbox_preds']
        # print(f'all_bbox_preds: {all_bbox_preds.shape}')

        enc_cls_scores = preds_dicts['enc_cls_scores']
        enc_bbox_preds = preds_dicts['enc_bbox_preds']

        # Operation for loss_depth
        # we transform all_bbox_preds from lidar coordinate into camera coordinate to recompute
        # depth_ave = (d + d_from_weighted_depth) / 2
        if self.loss_depth:
            all_bbox_preds = self.compute_d_ave(
                img_metas=img_metas,
                depth_map_values=preds_dicts['weighted_depth'],
                all_bbox_preds=preds_dicts['all_bbox_preds'],
            )

        # Operation from Ground True
        num_dec_layers = len(all_cls_scores)
        device = gt_labels_list[0].device
        gt_bboxes_list = [torch.cat(
            (gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]),
            dim=1).to(device) for gt_bboxes in gt_bboxes_list]

        # list(6 layer) of list(B) of tensor gt_labels_list: [num_objects, 9]
        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        # list(6 layer) of list(B) of tensor gt_labels_list: [num_objects,]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]

        losses_cls, losses_bbox = multi_apply(
            self.loss_single, all_cls_scores, all_bbox_preds,
            all_gt_bboxes_list, all_gt_labels_list,
            all_gt_bboxes_ignore_list)

        loss_dict = dict()
        # loss of proposal generated from encode feature map.
        # print(f'enc_cls_scores: {enc_cls_scores}')
        # enc_cls_scores is None
        if enc_cls_scores is not None:
            binary_labels_list = [
                torch.zeros_like(gt_labels_list[i])
                for i in range(len(all_gt_labels_list))
            ]
            enc_loss_cls, enc_losses_bbox = \
                self.loss_single(enc_cls_scores, enc_bbox_preds,
                                 gt_bboxes_list, binary_labels_list, gt_bboxes_ignore)
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_bbox'] = enc_losses_bbox

        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]

        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i in zip(losses_cls[:-1],
                                           losses_bbox[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            num_dec_layer += 1

        # Operation for depth_map surpervised
        if self.loss_ddn is not None:
            # loss from pred_depth_map_logits:
            # pred_depth_map_logits: [B, N, D+1, H, W]
            pred_depth_map_logits = preds_dicts['pred_depth_map_logits']
            assert pred_depth_map_logits is not None
            # print(f'pred_depth_map_logits: {pred_depth_map_logits.shape}')

            # get gt_depth_maps:
            # gt_depth_maps with depth_gt_encoder: [B, N, H, W, num_depth_bins], dtype: torch.float32
            # gt_depth_maps with normal depth encoder: [B, N, H, W], dtype: torch.long
            """
                gt_depth_maps: Thr ground truth depth maps with shape
                    `gt_depth_map_indices: `[batch, num_cameras, depth_map_H, depth_map_W] if `target` is true.
                    Otherwise, 
                    `gt_depth_map_probs:` [batch, num_cameras, depth_map_H, depth_map_W, num_depth_bins].
                gt_bboxes_2d: A list of list of tensor containing 2D ground truth bboxes(x, y, w, h)
                    for each sample and each camera. Each tensor has shape [N_i, 4].
                    Below is the brief explanation of a single batch:
                    [B, N, ....]
                    [[gt_bboxes_tensor_camera_0, gt_bboxes_tensor_camera_1, ..., gt_bboxes_tensor_camera_n](sample 0),
                        [gt_bboxes_tensor_camera_0, gt_bboxes_tensor_camera_1, ..., gt_bboxes_tensor_camera_n](sample 1),
                        ...].
            """

            gt_depth_maps, gt_bboxes_2d = self.get_depth_map_and_gt_bboxes_2d(
                gt_bboxes_list=gt_bboxes_3d,
                img_metas=img_metas,
                # TODO: `target` should be true after removing depth_gt_encoder
                target=(self.depth_gt_encoder is None),
                device=device,
                depth_maps_down_scale=self.depth_maps_down_scale,
            )
            gt_depth_maps = gt_depth_maps.to(device)
            # print(f'gt_depth_maps: {gt_depth_maps.shape}')

            loss_ddn = self.loss_ddn(
                depth_logits=pred_depth_map_logits,
                depth_target=gt_depth_maps,
                gt_bboxes_2d=gt_bboxes_2d,
            )
            loss_dict['loss_ddn'] = loss_ddn
        # print(f'loss_dict: {loss_dict}')

        return loss_dict

    def compute_d_ave(
        self,
        img_metas: List[Dict[str, torch.Tensor]],
        depth_map_values: torch.Tensor,
        all_bbox_preds: torch.Tensor,
    ) -> torch.Tensor:
        """Compute d_ave for loss_depth from depth_map_values and return new all_bbox_preds
            with update cz

        Args:
            img_metas: A list of dict containing the `lidar2img` tensor.
            depth_map_values (torch.Tensor): weighted sum value (real depth value) of predicted_depth_maps or gt_depth_maps
                which reperesent the z value in camera coordinate in depth_map format:
                `depth_maps_value(depth_map_coord): [B, num_cameras, H, W]`
            all_bbox_preds(torch.Tensor): Sigmoid regression outputs
                of all decode layers. Outputs from the regression head with
                normalized coordinate format
                (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy).
                `[num_layers, B, num_queries, 10]`

        Returns:
            all_bbox_preds_new(torch.Tensor): new all_bbox_preds
                with update cz and normalized coordinate format
                (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy).
                `[num_layers, B, num_queries, 10]`
        """

        # depth_map_values: [B, num_cameras, H, W] -> [B * num_cameras, 1, H, W]
        depth_map_values = depth_map_values.flatten(0, 1).unsqueeze(1)
        # lidar2img: [B, num_cameras, 4, 4]
        lidar2img = all_bbox_preds.new_tensor([img_meta['lidar2img'] for img_meta in img_metas])
        B, num_cameras, _, _ = lidar2img.shape
        num_layers, _, num_queries, _ = all_bbox_preds.shape
        # print(f'lidar2img: {lidar2img.shape}, B: {B}, N: {num_cameras}, Q: {num_queries}, L: {num_layers}')

        # Extract the x, y, z columns
        # outputs_centers: [num_layers, B, num_queries, 3] -> [B, num_queries, num_layers, 3]
        outputs_centers = torch.cat(
            (all_bbox_preds[..., 0:2], all_bbox_preds[..., 4:5]), dim=-1).permute(1, 2, 0, 3)

        ret_dict = project_ego_to_image(
            outputs_centers,
            lidar2img,
            img_shape=img_metas[0]['img_shape'][0],
            return_depth=True,
            return_normalized_uv=True,
            normalize_method=NormalizeMode.MINUS_ONE_TO_ONE,
            return_batch_first=True
        )
        # [B, num_cameras, num_queries, num_levels, 2]
        uv = ret_dict['uv']
        # [B, num_cameras, num_queries, num_levels]
        mask = ret_dict['mask']
        # [B, num_cameras, num_queries, num_levels]
        depth_in_cameras = ret_dict['depth']
        # [B, num_cameras, num_queries, num_levels, 2]
        normalized_uv = ret_dict['normalized_uv']
        # print(f'mask: {mask.shape}')

        # # convert to homogeneous coordinate. [batch, num_queries, num_layer, 4]
        # outputs_centers = torch.cat([
        #     outputs_centers,
        #     outputs_centers.new_ones((*outputs_centers.shape[:-1], 1))
        # ], dim=-1)
        # # print(f'outputs_centers: {outputs_centers.shape}')

        # # uvd: [num_cameras, batch, num_queries, num_layer, 3]
        # uvd: torch.Tensor = torch.einsum('bnij,bqlj->nbqli', lidar2img[:, :, :3], outputs_centers)
        # N, B, Q, L, _ = uvd.shape

        # # uv: [num_cameras, batch, num_queries, num_layer, 2]
        # uv = uvd[..., :2] / (uvd[..., -1:] + 1e-8)
        # img_H, img_W, _ = img_metas[0]['img_shape'][0]

        # # normalize to [0, 1] -> [-1, 1]
        # uv = (uv / uv.new_tensor([img_W, img_H]).reshape(1, 1, 1, 1, 2)) * 2 - 1

        # Get the depth values from depth maps
        # [B, num_cameras, num_queries, num_layers, 2] -> [B * num_cameras, num_queries, num_layers, 2]
        normalized_uv = normalized_uv.flatten(0, 1)
        # depth_from_depth_map_values: [B * num_cameras, 1, num_queries, num_layers]
        depth_from_depth_map_values = F.grid_sample(
            depth_map_values,
            normalized_uv,
            mode='bilinear',
            align_corners=True,
        )
        # depth_from_depth_map_values: [B, num_cameras, num_queries, num_layers, 1]
        depth_from_depth_map_values = depth_from_depth_map_values.view(B, num_cameras, num_queries, num_layers, 1)
        depth_from_depth_map_values[~mask] = 0
        # # d: [num_cameras, B, num_queries, num_layer, 1]
        # d = uvd[..., 2:]

        # average_depth: [B, num_cameras, num_queries, num_layers, 1]
        average_depth = (depth_in_cameras + depth_from_depth_map_values) / 2
        # print(f'd_ave: {average_depth.shape}')

        # img2lidar: [B, num_cameras, 4, 4]
        img2lidar = torch.linalg.inv(lidar2img)
        # print(f'img2lidar: {img2lidar.shape}')

        # uvd_new: [batch, num_cameras, num_queries, num_layers, 3]
        uvd_new = torch.cat([uv, average_depth], dim=-1)

        # convert to homogeneous coordinate.
        # uvd_new: [batch, num_cameras, num_queries, num_layer, 4]
        uvd_new = convert_to_homogeneous(uvd_new)
        # print(f'uvd_new: {uvd_new.shape}')

        # [batch, num_cameras, num_queries, num_layer, 4]
        outputs_centers_new = torch.einsum('bnij,bnqlj->bnqli', img2lidar, uvd_new)
        # print(f'outputs_centers_new: {outputs_centers_new.shape}')
        # [batch, num_cameras, num_queries, num_layer, 3]
        outputs_centers_new = outputs_centers_new[..., :-1] / (outputs_centers_new[..., -1:] + 1e-5)
        outputs_centers_new[~mask] = 0
        # [batch, num_queries, num_layer, 3]
        outputs_centers_new = outputs_centers_new.sum(1) / (mask.sum(1).unsqueeze(-1) + 1e-5)
        # [num_layer, batch, num_queries, 3]
        outputs_centers_new = outputs_centers_new.permute(2, 0, 1, 3)

        all_bbox_preds_new = all_bbox_preds
        all_bbox_preds_new[..., :2] = outputs_centers_new[..., :2]
        all_bbox_preds_new[..., 4:5] = outputs_centers_new[..., -1:]

        return all_bbox_preds_new

    @force_fp32(apply_to=('preds_dicts'))
    def get_bboxes(self, preds_dicts, img_metas, rescale=False):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        preds_dicts = self.bbox_coder.decode(preds_dicts)
        num_samples = len(preds_dicts)

        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds['bboxes']
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
            bboxes = img_metas[i]['box_type_3d'](bboxes, bboxes.size(-1))
            scores = preds['scores']
            labels = preds['labels']
            ret_list.append([bboxes, scores, labels])
        return ret_list
