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

# if running on server, change dir to following:

data_dir = '/home/jw7u18/LIDC/data'
dir_checkpoint = '/home/jw7u18/probabilistic_unet_output/training_ckpt'

# dirs
# data_dir = 'D:\LIDC\data'
# dir_checkpoint = 'D:\Probablistic-Unet-Pytorch-out\ckpt'

recon_dir = 'D:\\Probablistic-Unet-Pytorch-out\\reconstruction3'
data_save_dir = 'D:\LIDC\LIDC-IDRI-out_final_transform'

# model for resume training and eval
model_eval = ''
resume_model = ''

# hyper para
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 32
lr = 1e-4
weight_decay = 1e-5
epochs = 600
partial_data = False
resume = False
latent_dim = 6
beta = 10.0
isotropic = True
save_ckpt = True
random = False
# kaiming_normal and orthogonal
initializers = {'w':'kaiming_normal', 'b':'normal'}

eval_model = os.path.join(dir_checkpoint, model_eval)
r_model = os.path.join(dir_checkpoint, resume_model)

# Transforms
# joint_transfm = joint_transforms.Compose([joint_transforms.RandomHorizontallyFlip(),
#                                           joint_transforms.RandomSizedCrop(128),
#                                           joint_transforms.RandomRotate(60)])
# input_transfm = transforms.Compose([transforms.ToPILImage()])
target_transfm = transforms.Compose([transforms.ToTensor()])
joint_transfm=None
input_transfm=None


def train(data):
    print(f"isotropic gaussian: {isotropic}\ninitialisation: {initializers['w']}"
          f"\nsavingCKPT: {save_ckpt}\nlr_initial: {lr}\nbatchSize: {batch_size}")
    net = ProbabilisticUnet(input_channels=1, num_classes=1, num_filters=[32,64,128,192],
                            latent_dim=latent_dim, no_convs_fcomb=4, beta=beta, initializers=initializers,
                            isotropic=isotropic).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
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

    for epoch in range(epochs - epochs_trained):
        total_loss, total_reg_loss = 0, 0
        with tqdm(total=len(data.train_indices), desc=f'Epoch {epoch + 1}/{epochs - epochs_trained}', unit='patch') as pbar:
            for step, (patch, mask, _) in enumerate(data.train_loader):
                patch = patch.to(device)
                mask = mask.to(device)
                net.forward(patch, mask, training=True)
                elbo = net.elbo(mask)
                # TODO: reg_loss not change and elbo loss too large?
                reg_loss = l2_regularisation(net.posterior) + l2_regularisation(net.prior) + l2_regularisation(net.fcomb.layers)
                loss = -elbo + 1e-5 * reg_loss

                total_loss += loss.item()
                total_reg_loss += reg_loss.item()
                pbar.set_postfix(**{'total_loss': total_loss, 'total_reg_loss' : total_reg_loss})

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                pbar.update(batch_size)

        if save_ckpt and epoch%30 == 0:
            print('saving ckpt...')
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, dir_checkpoint, f'checkpoint_probUnet_epoch{epoch}_latenDim{latent_dim}_totalLoss{total_loss}'
                               f'_total_reg_loss{total_reg_loss}_isotropic_{isotropic}.pth.tar')





def save_checkpoint(state, save_path, filename):
    filename = os.path.join(save_path, filename)
    torch.save(state, filename)


if __name__ == '__main__':
    dataset = LIDC_IDRI(dataset_location=data_dir, joint_transform=joint_transfm, input_transform=input_transfm
                        , target_transform=target_transfm)
    dataloader = Dataloader(dataset, batch_size, small=partial_data, random=random)
    train(dataloader)
    # visualise_recon(dataloader, num_sample=100)