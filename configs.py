"""
config file for directory and hyper parameter setting
"""
import os
import torch
from torchvision import transforms
import utils.joint_transforms as joint_transforms

print('loading configs.........')
# --------------------------------Data directory---------------------------------------------------
data_dir = '/home/jw7u18/LIDC/data'
dir_checkpoint = '/home/jw7u18/probabilistic_unet_output/training_ckpt'

# data_dir = 'D:\Datasets\LIDC\data'
# dir_checkpoint = 'D:\Probablistic-Unet-Pytorch-out\ckpt'

# -------------------------------------model dir----------------------------------------------------
model_eval = ''
resume_model = 'checkpoint_probUnet_epoch90_latenDim6_totalLoss38455.22733211517num_filters[32, 64, 128, 192]batch_size32_.pth.tar'
recon_dir = 'D:\\Probablistic-Unet-Pytorch-out\\reconstructions\\' + model_eval[:-8]


# check this setting to design to perform train or evaluation before you go
# input str type 'train' / 'eval'
train_or_eval = 'train'

# ------------------------------------------training setting---------------------------------------------------
save_ckpt = True
shuffle_indices = False
random = True
r_model = os.path.join(dir_checkpoint, resume_model)
# -------------------------------------evaluation setting------------------------------------------------------------
num_sample = [1, 4, 8, 16, 50, 100]
all_experts = False
eval_model = os.path.join(dir_checkpoint, model_eval)
# --------------------------------------------------------hyper para-----------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_filters = [32, 64, 128, 192]
batch_size = 32
lr = 1e-4
weight_decay = 1e-5
epochs = 600

# prob unet only
axis_aligned = True # axis-aligned gaussian or isotropic gaussian
partial_data = False
resume = True
latent_dim = 6
beta = 10
# isotropic = False

# kaiming_normal and orthogonal
initializers = {'w': 'kaiming_normal', 'b': 'normal'}
# initializers = {'w':None, 'b':None}

# Transforms
joint_transfm = joint_transforms.Compose([joint_transforms.RandomHorizontallyFlip(),
                                          joint_transforms.RandomSizedCrop(128),
                                          joint_transforms.RandomRotate(60)])
# joint_transfm = None
input_transfm = transforms.Compose([transforms.ToPILImage()])
target_transfm = transforms.Compose([transforms.ToTensor()])
# joint_transfm=None
# input_transfm=None
