import torch
from dataset.load_LIDC_data import LIDC_IDRI
from prob_unet.probabilistic_unet import ProbabilisticUnet
from utils.utils import l2_regularisation
from tqdm import tqdm
import os
import imageio
import numpy as np
from dataset.dataloader import Dataloader

from torchvision import transforms
import utils.joint_transforms as joint_transforms

# dirs
data_dir = 'D:\LIDC\data'
dir_checkpoint = 'D:\Probablistic-Unet-Pytorch-out\ckpt'

recon_dir = 'D:\\Probablistic-Unet-Pytorch-out\\reconstruction_isotropic'

# model for resume training and eval
model_eval = 'checkpoint_probUnet_epoch420_latenDim6_totalLoss563047.706817627_total_reg_loss262616.68255615234_isotropic_True.pth.tar'

# hyper para
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 32
beta = 10.0
latent_dim = 6
# kaiming_normal and orthogonal
initializers = {'w':'kaiming_normal', 'b':'normal'}

eval_model = os.path.join(dir_checkpoint, model_eval)

# Transforms
joint_transfm = None
input_transfm = None
target_transfm = transforms.Compose([transforms.ToTensor()])


def visualise_recon(data, num_sample=10):
    print(f'loading model to eval...{model_eval}')
    net = ProbabilisticUnet(input_channels=1, num_classes=1, num_filters=[32, 64, 128, 192], latent_dim=latent_dim,
                            no_convs_fcomb=4, beta=beta, initializers=initializers).to(device)
    resume_dict = torch.load(eval_model, map_location=device)
    net.load_state_dict(resume_dict['state_dict'])
    net.eval()
    with torch.no_grad():
        with tqdm(total=len(data.test_indices), unit='patch') as pbar:
            for step, (patch, mask, _) in enumerate(data.test_loader):
                reconstruction = []
                patch = patch.to(device)
                mask = mask.to(device)
                net.forward(patch, mask, training=False)
                for sample in range(num_sample):
                    reconstruction.append(net.visual_recon())
                for i in range(batch_size):
                    imageio.imwrite(os.path.join(recon_dir, str(step) + f'{i}_image.png'), patch[i].cpu().numpy().T)
                    imageio.imwrite(os.path.join(recon_dir, str(step) + f'{i}_mask.png'), mask[i].cpu().numpy().T)
                    for s in range(len(reconstruction)):
                        imageio.imwrite(os.path.join(recon_dir, str(step) + f'{i}_recon_{s}th_s.png'), -reconstruction[s][i].cpu().numpy().T.astype(np.uint8))
                break
            pbar.update(batch_size)


if __name__ == '__main__':
    dataset = LIDC_IDRI(dataset_location=data_dir, joint_transform=joint_transfm,
                        input_transform=input_transfm
                        , target_transform=target_transfm)
    dataloader = Dataloader(dataset, batch_size)
    visualise_recon(dataloader, num_sample=10)