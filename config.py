# RRDB
nf = 3
gc = 32

# Super parameters
clamp = 2.0
channels_in = 3
log10_lr = -5.0
lr = 10 ** log10_lr  # 初始学习率
# lr3 = 10 ** -5.0
epochs = 50          # 迭代次数
weight_decay = 1e-5
init_scale = 0.01

device_ids = [0]

# Super loss
lamda_reconstruction_1 = 2  # revealing loss的系数 2
lamda_reconstruction_2 = 2
lamda_guide_1 = 1            # concealing loss的系数 1
lamda_guide_2 = 1

lamda_low_frequency_1 = 0      # wavelet loss的系数 1
lamda_low_frequency_2 = 0

use_imp_map = True
optim_step_1 = True  # 通过这三个参数控制三个网络是否同时更新
optim_step_2 = True
optim_step_3 = True

# Train: 1500 round
batchsize_train = 2   ## 3
cropsize_train = 128  # 训练图像的大小裁减为128*128
betas = (0.5, 0.999)
weight_step = 200
gamma = 0.98

# Val: 400 round
val_dataset = 'S1'
cropsize_val= 256
batchsize_val = 1
shuffle_val = False
val_freq = 100

# Dataset
PATH = '/home/u2080/RuonanY/train/'
TRAIN_PATH_cover = PATH + 'cover'
TRAIN_PATH_sec1 = PATH + 'sec1'
TRAIN_PATH_sec2 = PATH + 'sec2'
PATH1 = '/home/u2080/RuonanY/'+val_dataset+'/'
VAL_PATH_cover = PATH1 + 'cover'
VAL_PATH_sec1 = PATH1 + 'sec1'
VAL_PATH_sec2 = PATH1 + 'sec2'


# Display and logging:
loss_display_cutoff = 2.0  # cut off the loss so the plot isn't ruined
loss_names = ['L', 'lr']
silent = False
live_visualization = False
progress_bar = False

# Saving checkpoints:   # 保存模型的参数、优化器的参数，以及 Epoch, loss
MODEL_PATH = './model_save/'   # 保存模型  imp_map  A-E_map
checkpoint_on_error = True
SAVE_freq = 2


TEST_PATH = './test_results/'   # 保存测试结果  imp_map  A-E_map
TEST_PATH_cover = TEST_PATH + 'cover/'
TEST_PATH_secret_1 = TEST_PATH + 'secret_1/'
TEST_PATH_secret_2 = TEST_PATH + 'secret_2/'
TEST_PATH_steg_1 = TEST_PATH + 'steg_1/'
TEST_PATH_steg_2 = TEST_PATH + 'steg_2/'
TEST_PATH_secret_rev_1 = TEST_PATH + 'secret-rev_1/'
TEST_PATH_secret_rev_2 = TEST_PATH + 'secret-rev_2/'

TEST_PATH_A_map = TEST_PATH + 'A/'
TEST_PATH_E_map = TEST_PATH + 'E/'

TEST_PATH_resi_steg_1 = TEST_PATH + 'resi_steg_1/'
TEST_PATH_resi_steg_2 = TEST_PATH + 'resi_steg_2/'
TEST_PATH_resi_sec_1 = TEST_PATH + 'resi_sec_1/'
TEST_PATH_resi_sec_2 = TEST_PATH + 'resi_sec_2/'
# Load:
suffix_load = ''
train_next = False

trained_epoch = 194

pretrain = True
PRETRAIN_PATH = './model_save/'
suffix_pretrain = 'model_checkpoint_00194'
#PRETRAIN_PATH_3 = './model_save/'
#suffix_pretrain_3 = 'model_checkpoint_03000'


