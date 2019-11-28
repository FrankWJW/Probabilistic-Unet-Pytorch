import torch
from dataset.load_LIDC_data import LIDC_IDRI
from prob_unet.probabilistic_unet import ProbabilisticUnet
from utils.utils import l2_regularisation
from tqdm import tqdm
import os
import imageio
from dataset.dataloader import Dataloader

# paras for training
data_dir = 'D:\\LIDC\\data\\'
output_dir = 'D:\LIDC\LIDC-IDRI-out_final'
dir_checkpoint = 'D:\Probablistic-Unet-Pytorch-out\ckpt'
batch_size = 32
lr = 1e-4
latent_dim = 6
beta = 1.0
weight_decay = 1e-5

# paras for eval
model_dir = 'D:\Probablistic-Unet-Pytorch-out\ckpt\CKPT_epoch1.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
recon_dir = 'D:\\Probablistic-Unet-Pytorch-out\\reconstruction'




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
    dataset = LIDC_IDRI(dataset_location=data_dir)
    loaded_data = Dataloader(dataset, batch_size, small=True)
    train(loaded_data)
    # eval(loaded_data)