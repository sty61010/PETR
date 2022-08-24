_base_ = [
    '../../../mmdetection3d/configs/_base_/datasets/nus-3d.py',
    '../../../mmdetection3d/configs/_base_/default_runtime.py'
]
backbone_norm_cfg = dict(type='LN', requires_grad=True)
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]

img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False,
)
embed_dims = 256
num_levels = 2
depth_maps_down_scale = 16
depth_emb_down_scale = 16
head_in_channels = 256
depth_start = 1e-3
depth_num = 64
position_range = [-61.2, -61.2, -10.0, 61.2, 61.2, 10.0]

model = dict(
    type='Depthr3D',
    use_grid_mask=True,
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(2, 3,),
        frozen_stages=-1,
        norm_cfg=dict(type='BN2d', requires_grad=False),
        norm_eval=True,
        style='caffe',
        with_cp=True,
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, False, True, True),
        pretrained='ckpts/resnet50_msra-5891d200.pth',
    ),
    img_neck=dict(
        type='CPFPN',
        in_channels=[1024, 2048],
        out_channels=head_in_channels,
        num_outs=2,
    ),
    pts_bbox_head=dict(
        type='DepthrHead',
        num_classes=10,
        in_channels=head_in_channels,
        num_query=900,
        LID=True,
        with_position=True,
        with_multiview=True,
        depth_num=depth_num,
        depth_start=depth_start,
        embed_dims=embed_dims,
        position_range=position_range,
        normedlinear=False,

        depth_predictor=dict(
            type='DepthPredictor',
            num_depth_bins=depth_num,
            depth_min=depth_start,
            depth_max=position_range[3],
            embed_dims=embed_dims,
            num_levels=num_levels,
            in_channels=embed_dims,
            depth_maps_down_scale=depth_maps_down_scale,
            depth_emb_down_scale=depth_emb_down_scale,
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=3,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=embed_dims,
                            num_heads=8,
                            dropout=0.1)
                    ],
                    feedforward_channels=256,
                    ffn_dropout=0.1,
                    operation_order=(
                        'self_attn', 'norm',
                        'ffn', 'norm',
                    )
                )
            ),
        ),
        only_cross_depth_attn=True,
        transformer=dict(
            type='DepthrTransformer',
            decoder=dict(
                type='DepthrTransformerDecoder',
                return_intermediate=True,
                num_layers=6,
                transformerlayers=dict(
                    type='MultiAttentionDecoderLayer',

                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),

                        # dict(
                        #     type='MultiheadAttention',
                        #     embed_dims=256,
                        #     num_heads=8,
                        #     dropout=0.1),

                        dict(
                            type='PETRMultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                    ],
                    feedforward_channels=2048,
                    ffn_dropout=0.1,
                    with_cp=True,
                    operation_order=(
                        'self_attn', 'norm',
                        # 'cross_depth_attn', 'norm',
                        'cross_view_attn', 'norm',
                        'ffn', 'norm',
                    )
                ),
            )),
        bbox_coder=dict(
            type='NMSFreeCoder',
            # type='NMSFreeClsCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            num_classes=10),
        positional_encoding=dict(
            type='SinePositionalEncoding3D', num_feats=128, normalize=True),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0,
        ),
        loss_bbox=dict(
            type='L1Loss',
            loss_weight=0.25,
        ),
        loss_iou=dict(
            type='GIoULoss',
            loss_weight=0.0,
        ),
        loss_ddn=dict(
            type='DDNLoss',
            alpha=0.25,
            gamma=2.0,
            fg_weight=13,
            bg_weight=1,
            downsample_factor=depth_maps_down_scale,
            loss_weight=1.0,
        ),
    ),
    # model training and testing settings
    train_cfg=dict(pts=dict(
        grid_size=[512, 512, 1],
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        out_size_factor=4,
        assigner=dict(
            type='HungarianAssigner3D',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            iou_cost=dict(type='IoUCost', weight=0.0),  # Fake cost. This is just to make it compatible with DETR head.
            pc_range=point_cloud_range))))

dataset_type = 'CustomNuScenesDataset'
data_root = 'data/nuscenes/'

file_client_args = dict(backend='disk')

db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'nuscenes_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(
            car=5,
            truck=5,
            bus=5,
            trailer=5,
            construction_vehicle=5,
            traffic_cone=5,
            barrier=5,
            motorcycle=5,
            bicycle=5,
            pedestrian=5)),
    classes=class_names,
    sample_groups=dict(
        car=2,
        truck=3,
        construction_vehicle=7,
        bus=4,
        trailer=6,
        barrier=2,
        motorcycle=6,
        bicycle=6,
        pedestrian=2,
        traffic_cone=2),
    points_loader=dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args))

ida_aug_conf = {
    "resize_lim": (0.8, 1.0),
    "final_dim": (512, 1408),
    "bot_pct_lim": (0.0, 0.0),
    "rot_lim": (0.0, 0.0),
    "H": 900,
    "W": 1600,
    "rand_flip": True,
}

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),

    dict(type='ResizeCropFlipImage', data_aug_conf=ida_aug_conf, training=True),
    dict(type='GlobalRotScaleTransImage',
         rot_range=[-0.3925, 0.3925],
         translation_std=[0, 0, 0],
         scale_ratio_range=[0.95, 1.05],
         reverse_angle=True,
         training=True
         ),

    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['gt_bboxes_3d', 'gt_labels_3d', 'img'])
]
test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),

    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),

    dict(type='ResizeCropFlipImage', data_aug_conf=ida_aug_conf, training=False),

    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(
                type='Collect3D',
                keys=['gt_bboxes_3d', 'gt_labels_3d', 'img'],
            ),
            # dict(type='Collect3D', keys=['img'])
        ])
]

data_length = 60000
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR',
        data_length=data_length,
    ),
    val=dict(
        type=dataset_type,
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality
    ),
    test=dict(
        type=dataset_type,
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality
    )
)

optimizer = dict(
    type='AdamW',
    lr=2e-4,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }),
    weight_decay=0.01)

optimizer_config = dict(type='Fp16OptimizerHook', loss_scale=512., grad_clip=dict(max_norm=35, norm_type=2))

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
    # by_epoch=False
)
total_epochs = 24
evaluation = dict(interval=1, pipeline=test_pipeline)
find_unused_parameters = False

runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
load_from = None
resume_from = None

# model_size: 29G
# 8 gpus bs=1 in TWCC
# mAP: 0.2813
# mATE: 0.8500
# mASE: 0.7144
# mAOE: 1.5397
# mAVE: 1.1223
# mAAE: 0.2873
# NDS: 0.2555
# Eval time: 205.0s

# Per-class results:
# Object Class    AP      ATE     ASE     AOE     AVE     AAE
# car     0.473   0.643   0.750   1.582   1.217   0.257
# truck   0.222   0.917   0.793   1.573   1.096   0.275
# bus     0.275   0.908   0.857   1.601   2.629   0.470
# trailer 0.058   1.172   0.852   1.587   0.409   0.053
# construction_vehicle    0.028   1.120   0.709   1.523   0.144   0.372
# pedestrian      0.389   0.746   0.334   1.512   0.940   0.482
# motorcycle      0.266   0.772   0.799   1.539   1.879   0.246
# bicycle 0.223   0.731   0.811   1.647   0.666   0.145
# traffic_cone    0.474   0.652   0.350   nan     nan     nan
# barrier 0.405   0.838   0.888   1.293   nan     nan
#                 ^M2022-08-21 09:30:51,201 - mmdet - INFO - Exp name: depthr_r50dcn_p4_512_1408_depth16_ddn16_w10_lid64_start1e-3_en3_de6_sd_bs1.py
