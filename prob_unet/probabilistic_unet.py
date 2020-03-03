# This code is based on: https://github.com/SimonKohl/probabilistic_unet

from unet.unet_blocks import *
from torch.distributions import Normal, Independent, kl
from unet.unet import UNet
from prob_unet.ConvGaussian import IsotropicGaussian, AxisAlignedGaussian
from prob_unet.Fcomb import Fcomb
from prob_unet.Encoders import Encoder
from scipy.stats import norm
import numpy as np



class ProbabilisticUnet(nn.Module):
    """
    A probabilistic UNet (https://arxiv.org/abs/1806.05034) implementation.
    input_channels: the number of channels in the image (1 for greyscale and 3 for RGB)
    num_classes: the number of classes to predict
    num_filters: is a list consisint of the amount of filters layer
    latent_dim: dimension of the latent space
    no_cons_per_block: no convs per block in the (convolutional) encoder of prior and posterior
    """

    def __init__(self, input_channels=1, num_classes=1, num_filters=[32,64,128,192], latent_dim=6, no_convs_fcomb=4, beta=10.0, initializers=None, isotropic=False,
                 device='cuda'):
        super(ProbabilisticUnet, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.latent_dim = latent_dim
        self.no_convs_per_block = 3
        self.no_convs_fcomb = no_convs_fcomb
        self.initializers = initializers
        self.beta = beta
        self.z_prior_sample = 0
        self.isotropic = isotropic

        self.unet = UNet(self.input_channels, self.num_classes, self.num_filters, if_last_layer=False).to(device)
        self.prior = AxisAlignedGaussian(self.input_channels, self.num_filters, self.no_convs_per_block, self.latent_dim, self.initializers, isotropic=isotropic).to(device)
        self.posterior = AxisAlignedGaussian(self.input_channels, self.num_filters, self.no_convs_per_block, self.latent_dim, self.initializers, isotropic=isotropic, posterior=True).to(device)
        self.fcomb = Fcomb(self.num_filters, self.latent_dim, self.input_channels,
                           self.num_classes, self.no_convs_fcomb, self.initializers, use_tile=True, device=device).to(device)

    def forward(self, patch, segm, training=False):
        """
        Construct prior latent space for patch and run patch through UNet,
        in case training is True also construct posterior latent space
        """
        if training:
            self.posterior_z = self.posterior.forward(patch, segm)
        self.prior_z = self.prior.forward(patch)
        self.unet_features = self.unet.forward(patch)

    def sample_(self, testing=False):
        """
        Sample a segmentation by reconstructing from a prior sample
        and combining this with UNet features
        """
        if testing == False:
            z_prior = self.prior_z.rsample()
            self.z_prior_sample = z_prior
        else:
            #You can choose whether you mean a sample or the mean here. For the GED it is important to take a sample.
            #z_prior = self.prior_z.base_dist.loc 
            z_prior = self.prior_z.sample()
            self.z_prior_sample = z_prior
        return self.fcomb.forward(self.unet_features, self.z_prior_sample)

    def reconstruct(self, use_posterior_mean=False, calculate_posterior=False, z_posterior=None):
        """
        Reconstruct a segmentation from a posterior sample (decoding a posterior sample) and UNet feature map
        use_posterior_mean: use posterior_mean instead of sampling z_q
        calculate_posterior: use a provided sample or sample from posterior latent space
        """
        assert use_posterior_mean == use_posterior_mean
        if use_posterior_mean:
            z_posterior = self.posterior_z.loc
        else:
            if calculate_posterior:
                z_posterior = self.posterior_z.rsample()
        return self.fcomb.forward(self.unet_features, z_posterior)


    def elbo(self, segm, reconstruct_posterior_mean=False, patch=None):
        assert type(patch) != None
        """
        Calculate the evidence lower bound of the log-likelihood of P(Y|X)
        """

        criterion = nn.BCEWithLogitsLoss(reduction='none')
        self.kl = torch.mean(kl.kl_divergence(self.posterior_z, self.prior_z))
        #Here we use the posterior sample sampled above
        self.reconstruction = self.reconstruct(use_posterior_mean=reconstruct_posterior_mean, calculate_posterior=True)
        
        reconstruction_loss = criterion(input=self.reconstruction, target=segm)
        self.reconstruction_loss = torch.sum(reconstruction_loss)
        self.mean_reconstruction_loss = torch.mean(reconstruction_loss)

        return self.reconstruction_loss + self.beta * self.kl

    def output_predict_manifold(self, patch=None):
        assert type(patch) != None
        assert self.latent_dim == 2
        nx = ny = 10
        x_values = np.linspace(.05, .95, nx)
        y_values = np.linspace(.05, .95, ny)

        canvas = np.empty((128 * ny, 128 * nx))
        for i, yi in enumerate(x_values):
            for j, xi in enumerate(y_values):
                z_posterior = np.array([norm.ppf(xi), norm.ppf(yi)]).astype('float32').T
                z_posterior = torch.unsqueeze(torch.tensor(z_posterior), dim=0).cuda()
                canvas[(nx-i-1)*128:(nx-i)*128, j*128:(j+1)*128] = self.reconstruct(z_posterior=z_posterior).cpu().numpy().reshape(128, 128)
        return canvas

    def output_predict_tensor(self, num_sample=10, patch=None):
        assert type(patch) != None
        r = []
        for samp in range(num_sample):
            # z_posterior = self.prior.sample_()
            reconstruction = self.sample_(testing=True).cpu().numpy()
            r.append(reconstruction)
        return r