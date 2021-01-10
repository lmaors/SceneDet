import os.path as osp
experiment_name = "movie_bert_scene_seg"
experiment_description = "scene segmentation with all modality"
# overall confg
data_root = '../movie318'
shot_frm_path = data_root + "/shot_movie318"
shot_num = 4  # even
seq_len = 16  # even
sample_stride = 2 #seq_len / 2
gpus = "0"

# dataset settings
dataset = dict(
    name="all",
    mode=['place', 'cast', 'act', 'aud', 'sub'],
)
# model settings
model = dict(
    name='MTSD',
    sim_channel=512,  # dim of similarity vector
    place_feat_dim=2048,
    cast_feat_dim=512,
    act_feat_dim=512,
    aud_feat_dim=512,
    sub_feat_dim=768,
    aud=dict(cos_channel=512),
    
    ratio=[0.3, 0.1, 0.2, 0.2, 0.2],
    embed_size=512,
    transformer_seq_len=seq_len,
    transformer_n_layers=4,
    transformer_attn_heads=8,
    transformer_dropout=0.1,
    ncls=2,
    transformer_dim_feedforward = 2048, # 4 * embed_size
    transformer_activation="relu",
    mlp_dropout=0.1,
)


batch_size = 12
epochs = 20

# optimizer
optim = dict(name='Adam',
             setting=dict(lr=5e-4, weight_decay=5e-4))

# optim = dict(name='SGD',
#              setting=dict(lr=5e-5, weight_decay=5e-4, momentum=0.9))

stepper = dict(name='MultiStepLR',
               setting=dict(milestones=[20]))

stepper_cos_lr = dict(name='CosineAnnealingLR',
               setting=dict(T_max=epochs+20, eta_min=0,last_epoch=-1))

loss = dict(weight=[0.5, 5])

# runtime settings
# resume = 'logs/movie_bert_scene_seg/checkpoint.pth.tar'
# resume = 'logs/movie_bert_scene_seg/model_best.pth.tar'
resume = None
train_flag = 1
test_flag = 1

logger = dict(log_interval=200, logs_dir="logs/{}".format(experiment_name))
data_loader_kwargs = dict(num_workers=16, pin_memory=True, drop_last=True)
