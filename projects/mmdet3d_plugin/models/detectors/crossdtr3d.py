# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------

import torch

from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask

import numpy as np


@DETECTORS.register_module()
class Crossdtr3D(MVXTwoStageDetector):
    """Crossdtr3D."""

    def __init__(
        self,
        use_grid_mask=False,
        pts_voxel_layer=None,
        pts_voxel_encoder=None,
        pts_middle_encoder=None,
        pts_fusion_layer=None,
        img_backbone=None,
        pts_backbone=None,
        img_neck=None,
        pts_neck=None,
        pts_bbox_head=None,
        img_roi_head=None,
        img_rpn_head=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
    ):
        super(Crossdtr3D, self).__init__(
            pts_voxel_layer,
            pts_voxel_encoder,
            pts_middle_encoder,
            pts_fusion_layer,
            img_backbone,
            pts_backbone,
            img_neck,
            pts_neck,
            pts_bbox_head,
            img_roi_head,
            img_rpn_head,
            train_cfg,
            test_cfg,
            pretrained,
        )
        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask

    @auto_fp16(apply_to=('img'), out_fp32=True)
    def extract_feat(
        self,
        img,
        img_metas,
    ):
        """Extract features from images and points."""
        img_feats = self.extract_img_feat(img, img_metas)
        return img_feats

    def extract_img_feat(
        self,
        img,
        img_metas,
    ):
        """Extract features of images."""
        # print(img[0].size())
        if isinstance(img, list):
            img = torch.stack(img, dim=0)

        B = img.size(0)
        if img is not None:
            input_shape = img.shape[-2:]
            # update real input shape of each single img
            for img_meta in img_metas:
                img_meta.update(input_shape=input_shape)
            if img.dim() == 5:
                if img.size(0) == 1 and img.size(1) != 1:
                    img.squeeze_()
                else:
                    B, N, C, H, W = img.size()
                    img = img.view(B * N, C, H, W)

            if self.use_grid_mask:
                img = self.grid_mask(img)
            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    @force_fp32(apply_to=('img', 'points'))
    def forward(
        self,
        return_loss=True,
        **kwargs,
    ):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

    def forward_train(
        self,
        points=None,
        img_metas=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        gt_labels=None,
        gt_bboxes=None,
        img=None,
        proposals=None,
        gt_bboxes_ignore=None,
        img_depth=None,
        img_mask=None,
    ):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """

        img_feats = self.extract_feat(img=img, img_metas=img_metas)

        losses = dict()
        losses_pts = self.forward_pts_train(
            img_feats,
            gt_bboxes_3d,
            gt_labels_3d,
            img_metas,
            gt_bboxes_ignore,
        )
        losses.update(losses_pts)
        return losses

    def forward_pts_train(
        self,
        pts_feats,
        gt_bboxes_3d,
        gt_labels_3d,
        img_metas,
        gt_bboxes_ignore=None,
    ):
        """Forward function for point cloud branch.
        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
        Returns:
            dict: Losses of each branch.
        """
        # outs = self.pts_bbox_head(pts_feats, img_metas)
        outs = self.pts_bbox_head(
            pts_feats,
            img_metas,
            gt_bboxes_3d,
        )
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, img_metas, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs)

        return losses

    def forward_test(
        self,
        img_metas,
        img=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        **kwargs,
    ):
        """
        Test function without augmentaiton.
        Args:
            x (list[torch.Tensor]): Features of multi-camera image
            img_metas (list[dict]): Meta information of samples.

            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sample
        Returns:
            bbox_list: dict of bboxes
        """
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))

        img = [img] if img is None else img
        gt_bboxes_3d = [gt_bboxes_3d] if gt_bboxes_3d is None else gt_bboxes_3d
        gt_labels_3d = [gt_labels_3d] if gt_labels_3d is None else gt_labels_3d

        return self.simple_test(
            img_metas=img_metas[0],
            img=img[0],
            gt_bboxes_3d=gt_bboxes_3d[0],
            gt_labels_3d=gt_labels_3d[0],
            **kwargs,
        )

    def simple_test(
        self,
        img_metas,
        img=None,
        rescale=False,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
    ):
        """
        Test function without augmentaiton.
        Args:
            x (list[torch.Tensor]): Features of multi-camera image
            img_metas (list[dict]): Meta information of samples.

            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
        Returns:
            bbox_list: dict of bboxes
        """
        img_feats = self.extract_feat(img=img, img_metas=img_metas)

        bbox_list = [dict() for i in range(len(img_metas))]
        bbox_pts = self.simple_test_pts(
            x=img_feats,
            img_metas=img_metas,
            rescale=rescale,
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
        )
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return bbox_list

    def simple_test_pts(
        self,
        x,
        img_metas,
        rescale=False,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
    ):
        """
        Test function of point cloud branch.
        Args:
            x (list[torch.Tensor]): Features of multi-camera image
            img_metas (list[dict]): Meta information of samples.

            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
        Returns:
            Bounding box results in cpu mode.
        """
        # outs = self.pts_bbox_head(x, img_metas)
        outs = self.pts_bbox_head(
            x,
            img_metas,
            gt_bboxes_3d,
        )

        bbox_list = self.pts_bbox_head.get_bboxes(
            outs,
            img_metas,
            rescale=rescale,
        )
        """
        Convert detection results to a list of numpy arrays.
        source: https://mmdetection3d.readthedocs.io/en/latest/_modules/mmdet3d/core/bbox/transforms.html#bbox3d2result
        """
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    def aug_test_pts(
        self,
        feats,
        img_metas,
        rescale=False,
    ):
        feats_list = []
        for j in range(len(feats[0])):
            feats_list_level = []
            for i in range(len(feats)):
                feats_list_level.append(feats[i][j])
            feats_list.append(torch.stack(feats_list_level, -1).mean(-1))
        outs = self.pts_bbox_head(feats_list, img_metas)
        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    def aug_test(
        self,
        img_metas,
        imgs=None,
        rescale=False,
    ):
        """Test function with augmentaiton."""
        img_feats = self.extract_feats(img_metas, imgs)
        img_metas = img_metas[0]
        bbox_list = [dict() for i in range(len(img_metas))]
        bbox_pts = self.aug_test_pts(img_feats, img_metas, rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return bbox_list

    def forward_dummy(self, points=None, img_metas=None, img_inputs=None, **kwargs):
        from mmdet3d.core.bbox.structures.box_3d_mode import LiDARInstance3DBoxes
        # pad_shape = input_shape + (3,)
        lidar2img = np.array([
            [[[667.5999, -11.3459, -1.8519, -891.7272],
              [38.3278, 67.5572, -559.0681, 760.1029],
              [0.5589, 0.8291, 0.0114, -1.1843],
              [0.5589, 0.8291, 0.0114, -1.1843]],

             [[412.9886, 506.6375, 1.8396, -668.0064],
              [-20.6782, 71.7123, -553.2657, 870.4648],
              [-0.3188, 0.9478, -0.0041, -0.1035],
              [0.5589, 0.8291, 0.0114, -1.1843]],

             [[-365.1992, 355.5753, 9.6372, -12.7062],
                [-78.5942, 2.7520, -354.6458, 559.4022],
                [-0.9998, -0.0013, 0.0186, -0.0372],
                [0.5589, 0.8291, 0.0114, -1.1843]],

             [[-643.1426, -139.6496, -12.1919, 556.9597],
              [-20.3761, -62.4205, -556.1431, 853.0998],
              [-0.3483, -0.9370, -0.0269, -0.0905],
              [0.5589, 0.8291, 0.0114, -1.1843]],

             [[-259.5370, -605.4091, -16.6692, 100.7252],
              [42.5486, -48.6904, -556.4778, 741.4697],
              [0.5614, -0.8272, -0.0248, -1.1877],
              [0.5589, 0.8291, 0.0114, -1.1843]],

             [[369.1537, -550.5783, -9.0176, -553.2021],
              [71.9614, 7.6353, -557.7432, 728.3124],
              [0.9998, 0.0181, -0.0075, -1.5520],
              [0.5589, 0.8291, 0.0114, -1.1843]]],
        ])
        img_shape = [[512, 1408, 3] for i in range(6)]

        # img_metas = [dict(box_type_3d=LiDARInstance3DBoxes, pad_shape=[[512, 1408, 3]], img_shape=[
        #                   [512, 1408, 3], [512, 1408, 3], [512, 1408, 3], [512, 1408, 3], [512, 1408, 3], [512, 1408, 3]])]

        img_metas = [
            dict(box_type_3d=LiDARInstance3DBoxes,
                 lidar2img=l2i,
                 pad_shape=img_shape,
                 img_shape=img_shape,
                 ) for l2i in lidar2img]

        img_feats = self.extract_feat(img=img_inputs, img_metas=img_metas)

        bbox_list = [dict() for _ in range(1)]
        assert self.with_pts_bbox
        bbox_pts = self.simple_test_pts(
            img_feats, img_metas, rescale=False)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return bbox_list
