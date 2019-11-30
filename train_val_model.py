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

eval_model = os.path.join(dir_checkpoint, model_eval)
r_model = os.path.join(dir_checkpoint, resume_model)

joint_transfm = joint_transforms.Compose([joint_transforms.RandomHorizontallyFlip(),
                                          joint_transforms.RandomSizedCrop(128),
                                          joint_transforms.RandomRotate(60)])
input_transfm = transforms.Compose([transforms.ToPILImage()])
target_transfm = transforms.Compose([transforms.ToTensor()])


def train(data):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = ProbabilisticUnet(input_channels=1, num_classes=1, num_filters=[32,64,128,192], latent_dim=latent_dim, no_convs_fcomb=4, beta=beta)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    epochs = 10

    for epoch in range(epochs):
        with tqdm(total=len(data.train_indices), desc=f'Epoch {epoch + 1}/{epochs}', unit='patch') as pbar:
            for step, (patch, mask, _) in enumerate(data.train_loader):
                patch = patch.to(device)
                mask = mask.to(device)
                net.forward(patch, mask, training=True)
                elbo = net.elbo(mask)
                reg_loss = l2_regularisation(net.posterior) + l2_regularisation(net.prior) + l2_regularisation(net.fcomb.layers)
                loss = -elbo + 1e-5 * reg_loss

                pbar.set_postfix(**{'loss': loss.item(), 'reg_loss' : reg_loss})

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.update(batch_size)


def eval(data):
    net = ProbabilisticUnet(input_channels=1, num_classes=1, num_filters=[32,64,128,192], latent_dim=latent_dim, no_convs_fcomb=4, beta=beta).to(device)
    net.load_state_dict(torch.load(model_dir))
    net.eval()
    with torch.no_grad():
        with tqdm (total=len(data.test_indices), unit='patch') as pbar:
            for step, (patch, mask, _) in enumerate(data.test_loader):
                patch = patch.to(device)
                mask = mask.to(device)
                mask = torch.unsqueeze(mask, 1)
                net.forward(patch, mask, training=True)

                elbo = net.elbo(mask)
                reg_loss = l2_regularisation(net.posterior) + l2_regularisation(net.prior) + l2_regularisation(
                    net.fcomb.layers)
                loss = -elbo + 1e-5 * reg_loss

                recon = net.reconstruction

                imageio.imwrite(os.path.join(recon_dir, str(step) + '_image.png'), patch[0].cpu().numpy().T)
                imageio.imwrite(os.path.join(recon_dir, str(step) + '_mask.png'), mask[0].cpu().numpy().T)
                imageio.imwrite(os.path.join(recon_dir, str(step)+'_recon.png'), recon[0].cpu().numpy().T)

                pbar.set_postfix(**{'loss': loss.item(), 'reg_loss': reg_loss})

                pbar.update(data.batch_size)

if __name__ == '__main__':
    dataset = LIDC_IDRI(dataset_location=data_dir, joint_transform=joint_transfm, input_transform=input_transfm
                        , target_transform=target_transfm)
    # dataset.save_data_set(data_save_dir)
    dataloader = Dataloader(dataset, batch_size, small=partial_data)
    train(dataloader)
    # eval(dataloader)