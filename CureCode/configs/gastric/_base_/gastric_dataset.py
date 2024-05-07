data_prefix = 'data/patches_captions/'
train_pipeline = [
    dict(type='PatchGastricPipeline'),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='PatchGastricPipeline', test_mode=False),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=128,
    workers_per_gpu=12,
    train=dict(
        type='PatchGastricCls3',
        data_prefix=data_prefix,
        ann_file='data/final/train_all_0.2.txt',
        #ann_file='data/final_0.7/train_all_0.3.txt',
        pipeline=train_pipeline),
    val=dict(
        type='PatchGastricCls3',
        data_prefix=data_prefix,
        ann_file='data/final/val_0.8_mapped.txt',
        #ann_file='data/final_0.7/val_0.7_mapped.txt',
        pipeline=test_pipeline),
    test=dict(
        type='PatchGastricCls3',
        data_prefix=data_prefix,
        ann_file='data/final/val_0.8_mapped.txt',
        #ann_file='data/final_0.7/val_0.7_mapped.txt',
        pipeline=test_pipeline))
