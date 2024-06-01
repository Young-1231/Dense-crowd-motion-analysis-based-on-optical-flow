dataset_type = 'TUB'
data_root = 'E://mmflow//TUBCrowdFlow//TUBCrowdFlow'

img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_rgb=False)

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
    test_dataloader=dict(samples_per_gpu=1, workers_per_gpu=2, shuffle=False),
    test=dict(
        type=dataset_type,
        pipeline=test_pipeline,
        data_root=data_root,
        test_mode=True))




"""img_norm_cfg = dict(mean=[0., 0., 0.], std=[255., 255., 255.], to_rgb=False)

kitti_test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', sparse=True),
    dict(type='InputResize', exponent=6),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='TestFormatBundle'),
    dict(
        type='Collect',
        keys=['imgs'],
        meta_keys=[
            'flow_gt', 'valid', 'filename1', 'filename2', 'ori_filename1',
            'ori_filename2', 'ori_shape', 'img_shape', 'img_norm_cfg',
            'scale_factor', 'pad_shape'
        ])
]

TUBCrowdFlow_test = dict(
    type=dataset_type,
    data_root=data_root,
    pipeline=kitti_test_pipeline,
    test_mode=True)

data = dict(
    test_dataloader=dict(samples_per_gpu=1, workers_per_gpu=2, shuffle=False),
    test=TUBCrowdFlow_test)     """