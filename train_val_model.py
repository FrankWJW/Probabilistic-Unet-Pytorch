import torch
from dataset.load_LIDC_data import LIDC_IDRI
from prob_unet.probabilistic_unet import ProbabilisticUnet
from utils.utils import l2_regularisation
from tqdm import tqdm
import os
import imageio
from dataset.dataloader import Dataloader

from torchvision import transforms
import utils.joint_transforms as joint_transforms


# dirs
data_dir = 'D:\LIDC\data'
dir_checkpoint = 'D:\Probablistic-Unet-Pytorch-out\ckpt'
recon_dir = 'D:\\Probablistic-Unet-Pytorch-out\\segmentation'
data_save_dir = 'D:\LIDC\LIDC-IDRI-out_final_transform'

# model for resume training and eval
model_eval = 'CKPT_epoch168_unet_loss_12.697673916816711.pth'
resume_model = 'checkpoint_epoch0_totalLoss_178.5162927210331.pth.tar'

# hyper para
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 32
lr = 1e-4
weight_decay = 1e-5
epochs = 300
partial_data = False
resume = False
latent_dim = 6
beta = 10.0
save_ckpt = False

eval_model = os.path.join(dir_checkpoint, model_eval)
r_model = os.path.join(dir_checkpoint, resume_model)

joint_transfm = joint_transforms.Compose([joint_transforms.RandomHorizontallyFlip(),
                                          joint_transforms.RandomSizedCrop(128),
                                          joint_transforms.RandomRotate(60)])
input_transfm = transforms.Compose([transforms.ToPILImage()])
target_transfm = transforms.Compose([transforms.ToTensor()])


def train(data):
    net = ProbabilisticUnet(input_channels=1, num_classes=1, num_filters=[32,64,128,192], latent_dim=latent_dim, no_convs_fcomb=4, beta=beta).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    milestones = list(range(0, epochs, int(epochs / 4)))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.4)

    for epoch in range(epochs):
        total_loss, total_reg_loss = 0, 0
        with tqdm(total=len(data.train_indices), desc=f'Epoch {epoch + 1}/{epochs}', unit='patch') as pbar:
            for step, (patch, mask, _) in enumerate(data.train_loader):
                patch = patch.to(device)
                mask = mask.to(device)
                net.forward(patch, mask, training=True)
                elbo = net.elbo(mask)
                reg_loss = l2_regularisation(net.posterior) + l2_regularisation(net.prior) + l2_regularisation(net.fcomb.layers)
                loss = -elbo + 1e-5 * reg_loss

                total_loss += loss
                total_reg_loss += reg_loss
                pbar.set_postfix(**{'total_loss': total_loss.item(), 'total_reg_loss' : total_reg_loss.item()})

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                pbar.update(batch_size)

        if save_ckpt and epoch%20 == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, dir_checkpoint, 'checkpoint_epoch{}_totalLoss{}_totalRecon{}.pth.tar'.format(epoch, total_loss, total_reg_loss))


def save_checkpoint(state, save_path, filename):
    filename = os.path.join(save_path, filename)
    torch.save(state, filename)


if __name__ == '__main__':
    dataset = LIDC_IDRI(dataset_location=data_dir, joint_transform=joint_transfm, input_transform=input_transfm
                        , target_transform=target_transfm)
    dataloader = Dataloader(dataset, batch_size, small=partial_data)
    train(dataloader)