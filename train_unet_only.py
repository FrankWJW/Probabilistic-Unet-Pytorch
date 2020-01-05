import torch
from dataset.load_LIDC_data import LIDC_IDRI
from unet.unet import UNet
from tqdm import tqdm
import os
import imageio
import torch.nn as nn
from dataset.dataloader import Dataloader
import numpy as np

from torchvision import transforms
import utils.joint_transforms as joint_transforms

# if running on server, change dir to following:

# data_dir = '/home/jw7u18/LIDC/data'
# dir_checkpoint = '/home/jw7u18/probabilistic_unet_output/training_ckpt'

# dirs
data_dir = 'D:\Datasets\LIDC\data'
dir_checkpoint = 'D:\Probablistic-Unet-Pytorch-out\ckpt'
recon_dir = 'D:\\Probablistic-Unet-Pytorch-out\\segmentation1'
data_save_dir = 'D:\LIDC\LIDC-IDRI-out_final_transform'

# model for resume training and eval
model_eval = 'checkpoint_epoch280_totalLoss_6.8233585972338915.pth.tar'
resume_model = 'checkpoint_epoch0_totalLoss_178.5162927210331.pth.tar'

# hyper para
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 32
lr = 1e-4
weight_decay = 1e-5
epochs = 300
partial_data = False
resume = False
save_ckpt = False


eval_model = os.path.join(dir_checkpoint, model_eval)
r_model = os.path.join(dir_checkpoint, resume_model)

joint_transfm = joint_transforms.Compose([joint_transforms.RandomHorizontallyFlip(),
                                          joint_transforms.RandomSizedCrop(128),
                                          joint_transforms.RandomRotate(60)])
input_transfm = transforms.Compose([transforms.ToPILImage()])
target_transfm = transforms.Compose([transforms.ToTensor()])

# random elastic deformation, rotation, shearing, scaling and a randomly
# translated crop that results in a tile size of 128 × 128 pixels


def train(data):
    net = UNet(in_channels=1, n_classes=1, bilinear=True).to(device)
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
    for epoch in range(epochs - epochs_trained):

        total_loss = 0
        with tqdm(total=len(data.train_indices), desc=f'Epoch {epoch + 1}/{epochs}', unit='patch') as pbar:
            for step, (patch, mask, _) in enumerate(data.train_loader):
                patch = patch.to(device)
                mask = mask.to(device)

                patch_pred = net(patch)

                loss = criterion(patch_pred, mask)

                total_loss += loss.item()
                pbar.set_postfix(**{'loss_total': total_loss})

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                pbar.update(batch_size)

        if save_ckpt and epoch%10==0:
            print('Saving ckpt...')
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, dir_checkpoint, 'checkpoint_epoch{}_totalLoss_{}.pth.tar'.format(epoch,total_loss))


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


# TODO: save_transformed_data
def save_transformed_data():
    # imageio.imwrite(os.path.join(data_save_dir, f'epoch{epoch}_step{step}_image.png'), patch[0].squeeze().cpu().numpy())
    # imageio.imwrite(os.path.join(data_save_dir, f'epoch{epoch}_step{step}_mask.png'), mask[0].squeeze().cpu().numpy())
    return


if __name__ == '__main__':
    dataset = LIDC_IDRI(dataset_location=data_dir, joint_transform=None, input_transform=None
                        , target_transform=target_transfm)
    # dataset.save_data_set(data_save_dir)
    dataloader = Dataloader(dataset, batch_size=1, small=partial_data)
    train(dataloader)
    # eval(dataloader)