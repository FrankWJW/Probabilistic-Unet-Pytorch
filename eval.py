import torch
from dataset.load_LIDC_data import LIDC_IDRI
from prob_unet.probabilistic_unet import ProbabilisticUnet
from utils.utils import l2_regularisation
from utils.utils import generalised_energy_distance
from tqdm import tqdm
import os
import imageio
import numpy as np
from dataset.dataloader import Dataloader

from torchvision import transforms
import utils.joint_transforms as joint_transforms

# dirs
data_dir = 'D:\Datasets\LIDC\data'
dir_checkpoint = 'D:\Probablistic-Unet-Pytorch-out\ckpt'

recon_dir = 'D:\\Probablistic-Unet-Pytorch-out\\reconstruction_bool'

# model for resume training and eval
model_eval = 'checkpoint_probUnet_epoch420_latenDim6_totalLoss563047.706817627_total_reg_loss262616.68255615234_isotropic_True.pth.tar'

# hyper para
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 32
beta = 10.0
latent_dim = 6
small = True
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
                patch = patch.to(device)
                mask = mask.to(device)
                net.forward(patch, mask, training=False)
                reconstruction = net.visual_recon(num_sample)
                for i in range(batch_size):
                    imageio.imwrite(os.path.join(recon_dir, str(step) + f'{i}_image.png'), patch[i].cpu().numpy().T)
                    imageio.imwrite(os.path.join(recon_dir, str(step) + f'{i}_mask.png'), mask[i].cpu().numpy().T)
                    for s in range(len(reconstruction)):
                        r = reconstruction[s][i].T >= 0
                        imageio.imwrite(os.path.join(recon_dir, str(step) + f'{i}_recon_{s}th_s.png'), r.astype(float))
                break
            pbar.update(batch_size)


def eval(data, num_sample=10):
    print('evaluation...')
    net = ProbabilisticUnet(input_channels=1, num_classes=1, num_filters=[32, 64, 128, 192], latent_dim=latent_dim,
                            no_convs_fcomb=4, beta=beta, initializers=initializers).to(device)
    resume_dict = torch.load(eval_model, map_location=device)
    net.load_state_dict(resume_dict['state_dict'])
    net.eval()
    with torch.no_grad():
        reconstruction = []
        masks = []
        for step, (patch, mask, _) in enumerate(data.test_loader):
            if mask.shape[0] is not batch_size:
                # discard last part
                break
            patch = patch.to(device)
            mask = mask.to(device)
            net.forward(patch, mask, training=False)
            temp = np.asarray(net.visual_recon(num_sample)) > 0
            reconstruction.append(temp.astype(float))

            temp_mask = mask.cpu().numpy() > 0
            masks.append(temp_mask.astype(float))

    masks = np.asarray(masks).reshape(-1,128,128)
    reconstruction = np.asarray(reconstruction).reshape(-1, 128, 128)

    # for x in range(masks.shape[0]):
    #     if np.sum(masks[x]) == 0.0 or np.sum(reconstruction[x]) == 0.0:
    #         masks = np.delete(masks, x, 0)
    #         reconstruction = np.delete(reconstruction, x, 0)

    ged = generalised_energy_distance(reconstruction, masks)
    print(ged)


if __name__ == '__main__':
    dataset = LIDC_IDRI(dataset_location=data_dir, joint_transform=joint_transfm,
                        input_transform=input_transfm
                        , target_transform=target_transfm)
    dataloader = Dataloader(dataset, batch_size, small=small)
    eval(dataloader, num_sample=10)
    # visualise_recon(dataloader)