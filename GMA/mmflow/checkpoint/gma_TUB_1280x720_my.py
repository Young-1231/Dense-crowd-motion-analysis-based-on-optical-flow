model = dict(
    type='RAFT',
    num_levels=4,
    radius=4,
    cxt_channels=128,
    h_channels=128,
    encoder=dict(
        type='RAFTEncoder',
        in_channels=3,
        out_channels=256,
        net_type='Basic',
        norm_cfg=dict(type='IN'),
        init_cfg=[
            dict(
                type='Kaiming',
                layer=['Conv2d'],
                mode='fan_out',
                nonlinearity='relu'),
            dict(type='Constant', layer=['InstanceNorm2d'], val=1, bias=0)
        ]),
    cxt_encoder=dict(
        type='RAFTEncoder',
        in_channels=3,
        out_channels=256,
        net_type='Basic',
        norm_cfg=dict(type='SyncBN'),
        init_cfg=[
            dict(
                type='Kaiming',
                layer=['Conv2d'],
                mode='fan_out',
                nonlinearity='relu'),
            dict(type='Constant', layer=['SyncBatchNorm2d'], val=1, bias=0)
        ]),
    decoder=dict(
        type='GMADecoder',
        net_type='Basic',
        num_levels=4,
        radius=4,
        iters=12,
        corr_op_cfg=dict(type='CorrLookup', align_corners=True),
        gru_type='SeqConv',
        heads=1,
        motion_channels=128,
        position_only=False,
        max_pos_size=None,
        flow_loss=dict(type='SequenceLoss', gamma=0.85),
        act_cfg=dict(type='ReLU')),
    freeze_bn=False,
    train_cfg=dict(),
    test_cfg=dict(iters=32))
dataset_type = 'TUB'
data_root = 'E://mmflow//TUBCrowdFlow//TUBCrowdFlow'
img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_rgb=False)
crop_size = (1280, 720)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='ColorJitter',
        asymmetric_prob=0.0,
        brightness=0.4,
        contrast=0.4,
        saturation=0.4,
        hue=0.1592356687898089),
    dict(
        type='SpacialTransform',
        spacial_prob=0.8,
        stretch_prob=0.8,
        crop_size=(1280, 720),
        min_scale=-0.2,
        max_scale=0.4,
        max_stretch=0.2),
    dict(type='RandomCrop', crop_size=(1280, 720)),
    dict(
        type='Normalize',
        mean=[127.5, 127.5, 127.5],
        std=[127.5, 127.5, 127.5],
        to_rgb=False),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['imgs', 'flow_gt'],
        meta_keys=[
            'img_fields', 'ann_fields', 'filename1', 'filename2',
            'ori_filename1', 'ori_filename2', 'filename_flow',
            'ori_filename_flow', 'ori_shape', 'img_shape', 'img_norm_cfg'
        ])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='InputPad', exponent=3),
    dict(
        type='Normalize',
        mean=[127.5, 127.5, 127.5],
        std=[127.5, 127.5, 127.5],
        to_rgb=False),
    dict(type='TestFormatBundle'),
    dict(
        type='Collect',
        keys=['imgs'],
        meta_keys=[
            'flow_gt', 'filename1', 'filename2', 'ori_filename1',
            'ori_filename2', 'ori_shape', 'img_shape', 'img_norm_cfg',
            'scale_factor', 'pad_shape', 'pad'
        ])
]
data = dict(
    train_dataloader=dict(
        samples_per_gpu=2,
        workers_per_gpu=5,
        drop_last=True,
        shuffle=True,
        persistent_workers=True),
    val_dataloader=dict(
        samples_per_gpu=1,
        workers_per_gpu=5,
        shuffle=False,
        persistent_workers=True),
    test_dataloader=dict(samples_per_gpu=1, workers_per_gpu=2, shuffle=False),
    train=dict(
        type='TUB',
        data_root='E://mmflow//TUBCrowdFlow//TUBCrowdFlow',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(
                type='ColorJitter',
                asymmetric_prob=0.0,
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.1592356687898089),
            dict(
                type='SpacialTransform',
                spacial_prob=0.8,
                stretch_prob=0.8,
                crop_size=(1280, 720),
                min_scale=-0.2,
                max_scale=0.4,
                max_stretch=0.2),
            dict(type='RandomCrop', crop_size=(1280, 720)),
            dict(
                type='Normalize',
                mean=[127.5, 127.5, 127.5],
                std=[127.5, 127.5, 127.5],
                to_rgb=False),
            dict(type='DefaultFormatBundle'),
            dict(
                type='Collect',
                keys=['imgs', 'flow_gt'],
                meta_keys=[
                    'img_fields', 'ann_fields', 'filename1', 'filename2',
                    'ori_filename1', 'ori_filename2', 'filename_flow',
                    'ori_filename_flow', 'ori_shape', 'img_shape',
                    'img_norm_cfg'
                ])
        ],
        test_mode=False),
    val=dict(
        type='TUB',
        data_root='E://mmflow//TUBCrowdFlow//TUBCrowdFlow',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='InputPad', exponent=3),
            dict(
                type='Normalize',
                mean=[127.5, 127.5, 127.5],
                std=[127.5, 127.5, 127.5],
                to_rgb=False),
            dict(type='TestFormatBundle'),
            dict(
                type='Collect',
                keys=['imgs'],
                meta_keys=[
                    'flow_gt', 'filename1', 'filename2', 'ori_filename1',
                    'ori_filename2', 'ori_shape', 'img_shape', 'img_norm_cfg',
                    'scale_factor', 'pad_shape', 'pad'
                ])
        ],
        test_mode=True),
    test=dict(
        type='TUB',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='InputPad', exponent=3),
            dict(
                type='Normalize',
                mean=[127.5, 127.5, 127.5],
                std=[127.5, 127.5, 127.5],
                to_rgb=False),
            dict(type='TestFormatBundle'),
            dict(
                type='Collect',
                keys=['imgs'],
                meta_keys=[
                    'flow_gt', 'filename1', 'filename2', 'ori_filename1',
                    'ori_filename2', 'ori_shape', 'img_shape', 'img_norm_cfg',
                    'scale_factor', 'pad_shape', 'pad'
                ])
        ],
        data_root='E://mmflow//TUBCrowdFlow//TUBCrowdFlow',
        test_mode=True))
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = 'checkpoint/gma_8x2_120k_mixed_368x768.pth'
resume_from = None
workflow = [('train', 1)]
optimizer = dict(
    type='AdamW',
    lr=0.000125,
    betas=(0.9, 0.999),
    eps=1e-08,
    weight_decay=1e-05,
    amsgrad=False)
optimizer_config = dict(grad_clip=dict(max_norm=1.0))
lr_config = dict(
    policy='OneCycle',
    max_lr=0.000125,
    total_steps=50100,
    pct_start=0.05,
    anneal_strategy='linear')
runner = dict(type='IterBasedRunner', max_iters=50000)
checkpoint_config = dict(by_epoch=False, interval=5000)
evaluation = dict(interval=5000, metric='EPE')
work_dir = 'checkpoint'
auto_resume = False
gpu_ids = [0]
