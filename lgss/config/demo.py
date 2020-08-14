experiment_name = "image50"
experiment_description = "scene segmentation using images only"
# overall confg
data_root = '../data/demo'
shot_frm_path = data_root + "/shot_txt"  
# the correspondence file of shots and their frames
'''
video name is specially required in this setting,
please make sure there is no space in the name
'''
video_name = "demo"
shot_num = 4  # even
seq_len = 10  # even
gpus = "0,1,2,3,4,5,6,7"

# dataset settings
dataset = dict(
    name="demo",
    mode=['image'],
)
# model settings
model = dict(
    name='LGSS_image',
    backbone='resnet50',
    fix_resnet=False,
    sim_channel=512,  # dim of similarity vector
    bidirectional=True,
    lstm_hidden_size=512,
    )

# optimizer
optim = dict(name='SGD',
             setting=dict(lr=1e-2, weight_decay=5e-4))
stepper = dict(name='MultiStepLR',
               setting=dict(milestones=[15]))
loss = dict(weight=[0.5, 5])

# runtime settings
resume = None
trainFlag = False
testFlag = True
batch_size = 16
epochs = 30
logger = dict(log_interval=200, logs_dir="../run/{}".format(experiment_name))
data_loader_kwargs = dict(num_workers=32, pin_memory=True, drop_last=False)
