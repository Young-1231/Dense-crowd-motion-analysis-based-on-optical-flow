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
        hue=0.5 / 3.14),
    dict(
        type='SpacialTransform',
        spacial_prob=0.8,
        stretch_prob=0.8,
        crop_size=crop_size,
        min_scale=-0.2,
        max_scale=0.4,
        max_stretch=0.2),
    dict(type='RandomCrop', crop_size=crop_size),
    dict(type='Normalize', **img_norm_cfg),
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
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='InputPad', exponent=3),
    dict(type='Normalize', **img_norm_cfg),
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
    train = dict(
        type=dataset_type,
        data_root=data_root,
        pipeline=train_pipeline,
        test_mode=False),
    val= dict(
        type=dataset_type,
        data_root=data_root,
        pipeline=test_pipeline,
        test_mode=True),
    test=dict(
        type=dataset_type,
        pipeline=test_pipeline,
        data_root=data_root,
        test_mode=True))