import torch
from dataset.load_LIDC_data import LIDC_IDRI
from prob_unet.probabilistic_unet import ProbabilisticUnet
from utils.utils import l2_regularisation
from utils.utils import generalised_energy_distance
from tqdm import tqdm
import os
import imageio
import numpy as np
import statistics
from dataset.dataloader import Dataloader

from torchvision import transforms
import utils.joint_transforms as joint_transforms

# dirs
data_dir = 'D:\Datasets\LIDC\data'
dir_checkpoint = 'D:\Probablistic-Unet-Pytorch-out\ckpt'

recon_dir = 'D:\\Probablistic-Unet-Pytorch-out\\reconstruction_bool'

# model for resume training and eval
model_eval = 'checkpoint_probUnet_epoch240_latenDim6_totalLoss997976.3245849609_totalRecon187804.8690185547.pth.tar'

# hyper para
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 32
beta = 10.0
latent_dim = 6
small = False
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
                        r = reconstruction[s][i].T > 0
                        imageio.imwrite(os.path.join(recon_dir, str(step) + f'{i}_recon_{s}th_s.png'), r.astype(float))
                break
            pbar.update(batch_size)


def eval(data, num_sample=10):
    print('evaluation...')
    test_list = data.test_indices
    net = ProbabilisticUnet(input_channels=1, num_classes=1, num_filters=[32, 64, 128, 192], latent_dim=latent_dim,
                            no_convs_fcomb=4, beta=beta, initializers=initializers).to(device)
    resume_dict = torch.load(eval_model, map_location=device)
    net.load_state_dict(resume_dict['state_dict'])
    net.eval()
    with torch.no_grad():
        energy_dist = []
        with tqdm(total=len(data.test_indices), unit='step') as pbar:
            for step, (patch, _, _) in enumerate(data.test_loader):
                patch = patch.to(device)
                net.forward(patch, _, training=False)

                binary_recon = np.asarray(net.visual_recon(num_sample)) > 0
                binary_recon = binary_recon.astype(int)
                reconstruction = np.asarray(binary_recon).reshape(-1, 128, 128)

                mask = dataset.labels[test_list[step]]
                masks = np.asarray(mask).reshape(-1, 128, 128)

                energy_dist.append(generalised_energy_distance(reconstruction, masks))

                pbar.update(step)

        print(energy_dist)
        print(f'mean_energy_dist: {np.mean(energy_dist)}')




if __name__ == '__main__':
    dataset = LIDC_IDRI(dataset_location=data_dir, joint_transform=joint_transfm,
                        input_transform=input_transfm
                        , target_transform=target_transfm)
    dataloader = Dataloader(dataset, batch_size=1, small=small)
    eval(dataloader, num_sample=16)
    # visualise_recon(dataloader)