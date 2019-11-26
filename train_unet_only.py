import torch
from dataset.load_LIDC_data import LIDC_IDRI
from unet.unet import UNet
from utils import l2_regularisation
from tqdm import tqdm
import os
import imageio
import torch.nn as nn
from dataset.dataloader import Dataloader

# paras for training
data_dir = 'D:\\LIDC\\data\\'
output_dir = 'D:\LIDC\LIDC-IDRI-out_final'
unet_seg_outdir = 'D:\Probablistic-Unet-Pytorch-out\\unet_only_seg'
dir_checkpoint = 'D:\Probablistic-Unet-Pytorch-out\ckpt'

batch_size = 32
lr = 1e-4
weight_decay = 1e-5


def train(data):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = UNet(in_channels=1, n_classes=1, bilinear=True).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    epochs = 10

    for epoch in range(epochs):
        net.train()
        with tqdm(total=len(data.train_indices), desc=f'Epoch {epoch + 1}/{epochs}', unit='patch') as pbar:
            for step, (patch, mask, _) in enumerate(data.train_loader):
                patch = patch.to(device)
                mask = mask.to(device)
                mask = torch.unsqueeze(mask,1)
                patch_pred = net(patch)

                loss = criterion(patch_pred, mask)

                pbar.set_postfix(**{'loss': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.update(batch_size)

        torch.save(net.state_dict(), os.path.join(dir_checkpoint, f'CKPT_epoch_unet_{epoch + 1}.pth'))


if __name__ == '__main__':
    dataset = LIDC_IDRI(dataset_location=data_dir)
    dataloader = Dataloader(dataset, batch_size)
    train(dataloader)