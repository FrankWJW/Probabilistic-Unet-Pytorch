"""
config file for directory and hyper parameter setting
"""
import os
import torch
from torchvision import transforms
import utils.joint_transforms as joint_transforms


# --------------------------------Data directory---------------------------------------------------
# data_dir = '/home/jw7u18/LIDC/data'
# dir_checkpoint = '/home/jw7u18/probabilistic_unet_output/training_ckpt'

# data_dir = '/home/jw7u18/LIDC/data'
# dir_checkpoint = '/home/jw7u18/probabilistic_unet_output/training_ckpt'

data_dir = 'D:\Datasets\LIDC\data'
dir_checkpoint = 'D:\Probablistic-Unet-Pytorch-out\ckpt'

recon_dir = 'D:\\Probablistic-Unet-Pytorch-out\\reconstruction_latenDim_6'

# -------------------------------------model dir----------------------------------------------------
model_eval = 'checkpoint_probUnet_epoch510_latenDim6_totalLoss828750.686126709_total_reg_loss294305.4841308594_isotropic_False.pth.tar'
resume_model = ''

# ------------------------------------------training setting---------------------------------------------------
save_ckpt = True
r_model = os.path.join(dir_checkpoint, resume_model)
# -------------------------------------evaluation setting------------------------------------------------------------

eval_model = os.path.join(dir_checkpoint, model_eval)
# --------------------------------------------------------hyper para-----------------------------------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 32
lr = 1e-4
weight_decay = 1e-5
epochs = 600
partial_data = False
resume = False
latent_dim = 6
beta = 1.0
isotropic = False

random = False
small = False
num_sample = [1, 4, 8, 16, 50, 100]
all_experts = False
# kaiming_normal and orthogonal
initializers = {'w': 'kaiming_normal', 'b': 'normal'}

# Transforms
joint_transfm = joint_transforms.Compose([joint_transforms.RandomHorizontallyFlip(),
                                          joint_transforms.RandomSizedCrop(128),
                                          joint_transforms.RandomRotate(60)])
input_transfm = transforms.Compose([transforms.ToPILImage()])
target_transfm = transforms.Compose([transforms.ToTensor()])
# joint_transfm=None
# input_transfm=None
