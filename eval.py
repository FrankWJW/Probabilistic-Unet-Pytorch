
from probabilistic_unet import ProbabilisticUnet
import imageio
from tqdm import tqdm
from utils import l2_regularisation
import torch

model_dir = 'D:\Probablistic-Unet-Pytorch-out\ckpt\CKPT_epoch6.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
datadir = 'D:\LIDC\LIDC-IDRI-out_final\image_15095.png'

def eval(data):
    net = ProbabilisticUnet(input_channels=1, num_classes=1, num_filters=[32,64,128,192], latent_dim=2, no_convs_fcomb=4, beta=10.0).to(device)
    net.load_state_dict(torch.load(model_dir))
    net.eval()
    with torch.no_grad():
        with tqdm (total=len(data.test_indices), unit='patch') as pbar:
            for step, (patch, mask, _) in enumerate(data.test_loader):
                patch = patch.to(device)
                mask = mask.to(device)
                mask = torch.unsqueeze(mask, 1)
                net.forward(patch, mask, training=False)
                elbo = net.elbo(mask)
                reg_loss = l2_regularisation(net.posterior) + l2_regularisation(net.prior) + l2_regularisation(
                    net.fcomb.layers)
                loss = -elbo + 1e-5 * reg_loss

                pbar.set_postfix(**{'loss': loss.item(), 'reg_loss': reg_loss})

                pbar.update(data.batch_size)
