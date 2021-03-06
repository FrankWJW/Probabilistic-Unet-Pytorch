import torch
from dataset.load_LIDC_data import LIDC_IDRI
from prob_unet.probabilistic_unet import ProbabilisticUnet
from utils.utils import l2_regularisation
from utils.utils import generalised_energy_distance, variance_ncc_dist
from tqdm import tqdm
import os
import imageio
import numpy as np
import statistics
from dataset.dataloader import Dataloader
from configs import *
from torchvision import transforms
import utils.joint_transforms as joint_transforms
import matplotlib.pyplot as plt

def visualise_manifold(data, net):
    assert batch_size == 1
    print(f'loading model to eval...{model_eval}')
    resume_dict = torch.load(eval_model, map_location=device)
    net.load_state_dict(resume_dict['state_dict'])
    net.eval()
    with torch.no_grad():
        for step, (patch, mask, _) in enumerate(data.test_loader):
            patch = patch.to(device)
            net.forward(patch, _, training=False)
            canvas = net.output_predict_tensor(manifold_visualisation=True, patch=patch)
            canvas = (canvas.T > 0).astype(int)

            plt.figure(1, figsize=(256,256))
            plt.imshow(canvas, origin="upper", cmap="gray")
            plt.figure(2, figsize=(128,128))
            plt.imshow(mask[0,0,:].T, cmap="gray")
            plt.tight_layout()
            plt.show()

def dir_check(path):
    try:
        os.mkdir(path)
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s " % path)

def output_predict_img(data, net, num_sample=10):
    dir_check(recon_dir)
    print(f'loading model to eval...{model_eval}')
    resume_dict = torch.load(eval_model, map_location=device)
    net.load_state_dict(resume_dict['state_dict'])
    net.eval()
    with torch.no_grad():
        with tqdm(total=len(data.test_indices), unit='patch') as pbar:
            for step, (patch, mask, _) in enumerate(data.test_loader):
                patch = patch.to(device)
                mask = mask.to(device)
                net.forward(patch, mask, training=False)
                reconstruction = net.output_predict_tensor(num_sample, patch=patch)
                for i in range(batch_size):
                    imageio.imwrite(os.path.join(recon_dir, str(step) + f'{i}_image.png'), patch[i].cpu().numpy().T)
                    imageio.imwrite(os.path.join(recon_dir, str(step) + f'{i}_mask.png'), mask[i].cpu().numpy().T)
                    for s in range(len(reconstruction)):
                        r = reconstruction[s][i].T > 0
                        imageio.imwrite(os.path.join(recon_dir, str(step) + f'{i}_recon_{s}th_s.png'), r.astype(int))
                break
            pbar.update(batch_size)

def generalised_energy_distance(data, net, num_sample):
    print(f'evaluation, num_sample:{num_sample}, all_expert:{all_experts}')
    print(f'loading model to eval...{model_eval}')
    test_list = data.test_indices

    resume_dict = torch.load(eval_model, map_location=device)
    net.load_state_dict(resume_dict['state_dict'])
    net.eval()
    with torch.no_grad():
        energy_dist = []
        ncc = []
        with tqdm(total=len(data.test_indices), unit='step') as pbar:
            for step, (patch, _, _) in enumerate(data.test_loader):
                patch = patch.to(device)
                net.forward(patch, _, training=False)

                binary_recon = np.asarray(net.output_predict_tensor(num_sample, patch=patch)) > 0
                binary_recon = binary_recon.astype(int)
                reconstruction = np.asarray(binary_recon).reshape(-1, 128, 128)

                if not all_experts:
                    mask = dataset.labels[test_list[step]][np.random.randint(0,3)]
                else:
                    mask = dataset.labels[test_list[step]]
                masks = np.asarray(mask).reshape(-1, 128, 128)

                energy_dist.append(generalised_energy_distance(reconstruction, masks))
                # ncc.append(variance_ncc_dist(reconstruction, masks))

                pbar.update(step)

        # print(energy_dist)
        print(f'mean_energy_dist: {np.mean(energy_dist)}')


if __name__ == '__main__':
    dataset = LIDC_IDRI(dataset_location=data_dir, joint_transform=joint_transfm,
                        input_transform=input_transfm
                        , target_transform=target_transfm)
    dataloader = Dataloader(dataset, batch_size, small=partial_data)
    net = ProbabilisticUnet(input_channels=1, num_classes=1, num_filters_=num_filters, latent_dim=latent_dim,
                            no_convs_fcomb=4, beta=beta, initializers=initializers, device=device).to(device)
    # for s in num_sample:
    #     generalised_energy_distance(dataloader, net, s)
    output_predict_img(dataloader, net)
    # visualise_manifold(dataloader, net)