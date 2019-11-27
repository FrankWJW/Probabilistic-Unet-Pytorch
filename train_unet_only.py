import torch
from dataset.load_LIDC_data import LIDC_IDRI
from unet.unet import UNet
from tqdm import tqdm
import os
import imageio
import torch.nn as nn
from dataset.dataloader import Dataloader
from torchvision import transforms


# dirs for training
data_dir = 'D:\\LIDC\\data\\'
output_dir = 'D:\LIDC\LIDC-IDRI-out_final'
unet_seg_outdir = 'D:\Probablistic-Unet-Pytorch-out\\unet_only_seg'
dir_checkpoint = 'D:\Probablistic-Unet-Pytorch-out\ckpt'


# dirs for eval and output
model_dir = 'D:\Probablistic-Unet-Pytorch-out\ckpt\CKPT_epoch293_unet_loss_2.8789007321101963.pth'
recon_dir = 'D:\\Probablistic-Unet-Pytorch-out\\segmentation'

# hyper para
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 32
lr = 1e-4
weight_decay = 1e-5
epochs = 300
partial_data = False
resume = False
resume_dir = 'D:\Probablistic-Unet-Pytorch-out\ckpt\CKPT_epoch293_unet_loss_2.8789007321101963.pth'

transfm = [None]
# TODO: transforms
#random elastic deformation, rotation, shearing, scaling and a randomly
#translated crop that results in a tile size of 128 × 128 pixels



def train(data):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = UNet(in_channels=1, n_classes=1, bilinear=True).to(device)
    if resume:
        print('loading checkpoint model to resume...')
        net.load_state_dict(torch.load(resume_dir))
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    milestones = list(range(0, epochs, int(epochs/4)))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.4)

    net.train()
    for epoch in range(epochs):

        total_loss = 0
        with tqdm(total=len(data.train_indices), desc=f'Epoch {epoch + 1}/{epochs}', unit='patch') as pbar:
            for step, (patch, mask, _) in enumerate(data.train_loader):
                patch = patch.to(device)
                mask = mask.to(device)
                mask = torch.unsqueeze(mask,1)
                patch_pred = net(patch)

                loss = criterion(patch_pred, mask)

                total_loss += loss.item()
                pbar.set_postfix(**{'loss_total': total_loss})

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                pbar.update(batch_size)

        torch.save(net.state_dict(), os.path.join(dir_checkpoint, f'CKPT_epoch{epoch + 1}_unet_loss_{total_loss}.pth'))


def eval(data):
    net = UNet(in_channels=1, n_classes=1, bilinear=True).to(device)
    net.load_state_dict(torch.load(model_dir))
    net.eval()
    with torch.no_grad():
        with tqdm (total=len(data.test_indices), unit='patch') as pbar:
            for step, (patch, mask, _) in enumerate(data.test_loader):
                patch = patch.to(device)
                mask = mask.to(device)
                mask = torch.unsqueeze(mask, 1)
                recon = net(patch)

                imageio.imwrite(os.path.join(recon_dir, str(step) + '_image.png'), patch[0].cpu().numpy().T)
                imageio.imwrite(os.path.join(recon_dir, str(step) + '_mask.png'), mask[0].cpu().numpy().T)
                imageio.imwrite(os.path.join(recon_dir, str(step)+'_recon.png'), recon[0].cpu().numpy().T)

                pbar.update(data.batch_size)


if __name__ == '__main__':
    dataset = LIDC_IDRI(dataset_location=data_dir, transform=transforms.Compose(transfm))
    dataloader = Dataloader(dataset, batch_size, small=partial_data)
    train(dataloader)
    # eval(dataloader)