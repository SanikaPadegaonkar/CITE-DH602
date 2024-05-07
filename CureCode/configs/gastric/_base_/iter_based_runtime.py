# yapf:disable
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
    ])
# yapf:enable
checkpoint_config = dict(interval=3000, max_keep_ckpts=1)
evaluation = dict(by_epoch=False,
                  metric=['accuracy', 'class_accuracy', 'bag_accuracy', 'bag_class_accuracy'],
                  interval=3000)
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = '/data/nlp/shrey/CITE/CITE/work_dirs/final_0.7/clip-vitb16_biolinkbert-l_vpl1_pall-cb_bs128_i1000_lr0.03/iter_3000.pth'
workflow = [('train', 1)]
