import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from load_LIDC_data import LIDC_IDRI
from probabilistic_unet import ProbabilisticUnet
from utils import l2_regularisation
from tqdm import tqdm
import os
import imageio

data_dir = 'D:\\LIDC\\data\\'
output_dir = 'D:\LIDC\LIDC-IDRI-out_final'
dir_checkpoint = 'D:\Probablistic-Unet-Pytorch-out\ckpt'
batch_size = 5
lr = 1e-4

def save_data_set(dataset):
    for k, np_img in enumerate(dataset.images):
        imageio.imwrite(os.path.join(output_dir, 'image_'+str(k)+'.png'), np_img)
        for k_l, np_label in enumerate(dataset.labels[k]):
            imageio.imwrite(os.path.join(output_dir, 'image_'+str(k)+'label_'+str(k_l)+'.png'), np_label*255)


def train(dataset):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(0.1 * dataset_size))
    np.random.shuffle(indices)
    train_indices, test_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)
    print("Number of training/test patches:", (len(train_indices),len(test_indices)))
    net = ProbabilisticUnet(input_channels=1, num_classes=1, num_filters=[32,64,128,192], latent_dim=2, no_convs_fcomb=4, beta=10.0)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=0)
    epochs = 10

    for epoch in range(epochs):
        with tqdm(total=len(train_indices), desc=f'Epoch {epoch + 1}/{epochs}', unit='patch') as pbar:
            for step, (patch, mask, _) in enumerate(train_loader):
                patch = patch.to(device)
                mask = mask.to(device)
                mask = torch.unsqueeze(mask,1)
                net.forward(patch, mask, training=True)
                elbo = net.elbo(mask)
                reg_loss = l2_regularisation(net.posterior) + l2_regularisation(net.prior) + l2_regularisation(net.fcomb.layers)
                loss = -elbo + 1e-5 * reg_loss

                pbar.set_postfix(**{'loss': loss.item(), 'reg_loss' : reg_loss})

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.update(batch_size)

        torch.save(net.state_dict(), os.path.join(dir_checkpoint, f'CKPT_epoch{epoch + 1}.pth'))

if __name__ == '__main__':
    dataset = LIDC_IDRI(dataset_location=data_dir)
    # save_data_set(dataset)
    train(dataset)