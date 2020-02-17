import torch
from dataset.load_LIDC_data import LIDC_IDRI
from unet.unet import UNet
from tqdm import tqdm
import os
import imageio
import torch.nn as nn
from dataset.dataloader import Dataloader
import numpy as np
from configs import *

from torchvision import transforms
import utils.joint_transforms as joint_transforms


def train(data, UNet):
    print(f"initialisation: {initializers['w']}"
          f"\nsavingCKPT: {save_ckpt}\nlr_initial: {lr}\nbatchSize: {batch_size}\n")
    net = UNet
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    milestones = list(range(0, epochs, int(epochs / 4)))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.4)

    if resume:
        print('loading checkpoint model to resume...')
        resume_dict = torch.load(r_model)
        net.load_state_dict(resume_dict['state_dict'])
        optimizer.load_state_dict(resume_dict['optimizer'])
        scheduler.load_state_dict(resume_dict['scheduler'])
        epochs_trained = resume_dict['epoch']
    else:
        epochs_trained = 0

    net.train()
    for epoch in range(epochs_trained, epochs):
        total_loss = 0
        scheduler.step()
        with tqdm(total=len(data.train_indices), desc=f'Epoch {epoch + 1}/{epochs}', unit='patch') as pbar:
            for step, (patch, mask, _) in enumerate(data.train_loader):
                patch = patch.to(device)
                mask = mask.to(device)
                optimizer.zero_grad()

                patch_pred = net(patch)

                loss = criterion(patch_pred, mask)

                total_loss += loss.item()
                pbar.set_postfix(**{'loss_total': total_loss})

                loss.backward()
                optimizer.step()

                pbar.update(batch_size)

        if save_ckpt and epoch%10==0:
            print('Saving ckpt...')
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, dir_checkpoint, 'Unet_checkpoint_epoch{}_totalLoss_{}.pth.tar'.format(epoch,total_loss))


def eval(data):
    net = UNet(in_channels=1, n_classes=1, bilinear=True).to(device)
    load_dict = torch.load(eval_model, map_location='cuda:0')
    net.load_state_dict(load_dict['state_dict'])
    # net.load_state_dict(load_dict)
    net.eval()
    with torch.no_grad():
        with tqdm (total=len(data.test_indices), unit='patch') as pbar:
            for step, (patch, mask, _) in enumerate(data.test_loader):
                patch = patch.to(device)
                mask = mask.to(device)
                recon = net(patch)

                imageio.imwrite(os.path.join(recon_dir, str(step) + '_image.png'), patch[0].cpu().numpy().T)
                imageio.imwrite(os.path.join(recon_dir, str(step) + '_mask.png'), mask[0].cpu().numpy().T)
                imageio.imwrite(os.path.join(recon_dir, str(step)+'_recon.png'), -recon[0].cpu().numpy().T.astype(np.uint8))

                pbar.update(data.batch_size)


def save_checkpoint(state, save_path, filename):
        filename = os.path.join(save_path, filename)
        torch.save(state, filename)



if __name__ == '__main__':
    dataset = LIDC_IDRI(dataset_location=data_dir, joint_transform=None, input_transform=None
                        , target_transform=target_transfm, random=random)
    dataloader = Dataloader(dataset, batch_size=batch_size, small=partial_data, shuffle_indices=shuffle_indices)
    net = UNet(in_channels=1, n_classes=1, bilinear=True, num_filters=num_filters).to(device)
    if not random:
        print('always using first experts annotation')
    train(dataloader, net)