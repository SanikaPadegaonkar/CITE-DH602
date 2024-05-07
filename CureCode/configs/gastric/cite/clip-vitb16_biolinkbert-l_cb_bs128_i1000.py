_base_ = [
    '../_base_/custom_imports.py',
    '../_base_/iter_based_runtime.py',
    '../_base_/gastric_dataset.py',
    '../_base_/sgd_i1000_lr0.001-cos.py'
]

lr = 0.03
n = 'all'
vpl = 1
run_name = f'clip-vitb16_biolinkbert-l_vpl{vpl}_p{n}-cb_bs128_i1000_lr{lr}'

arch = 'ViT-B/16'
# TEXTS = ['Well differentiated tubular adenocarcinoma',
#          'Moderately differentiated tubular adenocarcinoma',
#          'Poorly differentiated adenocarcinoma']
TEXTS = ['From the superficial epithelium to the muscularis mucosae, tumor tissue consisting of medium-sized and irregular glandular ducts infiltrating is observed. Well differentiated tubular adenocarcinoma', 'In the superficial epithelium, tumor tissue that invades by forming medium-sized to small, irregular ducts is observed. moderately differentiated tubular adenocarcinoma.', 'Tumor tissue consisting of cord-like or small, irregular glandular ducts fused and infiltrated is observed in the superficial epithelium. Poorly differentiated adenocarcinoma, non-solid type', 'In the superficial epithelium, tumor tissue that invades by forming medium-sized to small, irregular ducts is observed. Moderately differentiated adenocarcinoma.', 'The superficial epithelium shows a large sheet-like shape, and some tumor tissue infiltrates with small irregular ducts. Poorly differentiated adenocarcinoma, solid-type ', 'In the superficial epithelium, tumor tissue that invades by forming medium-sized to small, irregular ducts is observed. Well differentiated tubular adenocarcinoma.', 'Tumor tissue consisting of medium-sized and irregular ducts is observed infiltrating in the superficial epithelium. Tumor cells are highly columnar, with nuclei aligned basolaterally and polarized. Well differentiated tubular adenocarcinoma.', 'On the superficial epithelium, tumor tissue is found in which medium-sized irregular gland ducts fuse with each other and infiltrate. moderately differentiated tubular adenocarcinoma', 'Tumor tissue consisting of medium-sized, cord-like or irregular glandular infiltration is observed in the superficial epithelium. Poorly differentiated adenocarcinoma, non-solid type or moderately differentiated tubular adenocarcinoma', 'Tumor tissue in which medium to small irregular ducts infiltrate and proliferate in the submucosa can be seen in the epithelium. Well differentiated tubular adenocarcinoma', 'Large sheet-like or diffusely infiltrating tumor tissue is found in the superficial epithelium. Poorly differentiated adenocarcinoma, solid-type ', 'From the superficial epithelium to the muscularis mucosae, tumor tissue infiltrating by forming medium-sized irregular gland ducts is observed. Large tumor cells exhibit large nucleus and hyperchromatism. Well differentiated tubular adenocarcinoma', 'Group 5, Adenocarcinoma. Poorly differentiated adenocarcinoma, non-solid type']

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='PromptedCLIPImageBackbone',
        arch=arch,
        prompt_length=vpl,
        prompt_pos='prepend',
        prompt_init='token'),
    neck=dict(
        type='ProjectionNeck',
        in_features=512,
        out_features=1024,
        init='normal_identity',
        float16=True),
    head=dict(
        type='TextEmbeddingHead',
        texts=TEXTS,
        temperature=4.6052,
        float16=True,
        text_encoder=dict(
            type='BERT',
            model='michiyasunaga/BioLinkBERT-large')))

optimizer = dict(lr=lr)

data = dict(
    samples_per_gpu=64,  # use 2 gpus
    train=dict(
        # ann_file=f'data/final_0.7/train_{n}_0.3.txt',
        ann_file=f'data/final/train_{n}_0.2.txt',
        patch_balance=True))

work_dir = f'work_dirs/final/{run_name}'
