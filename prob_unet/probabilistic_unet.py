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
                 device='cuda', axis_aligned=True):
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
        if not axis_aligned:
            self.prior = IsotropicGaussian(self.input_channels, self.num_filters, self.no_convs_per_block, self.latent_dim, self.initializers, isotropic=isotropic).to(device)
            self.posterior = IsotropicGaussian(self.input_channels, self.num_filters, self.no_convs_per_block, self.latent_dim, self.initializers, isotropic=isotropic, posterior=True).to(device)
        else:
            self.prior = AxisAlignedGaussian(self.input_channels, self.num_filters, self.no_convs_per_block, self.latent_dim,
                                           self.initializers, isotropic=isotropic).to(device)
            self.posterior = AxisAlignedGaussian(self.input_channels, self.num_filters, self.no_convs_per_block,
                                                self.latent_dim, self.initializers, isotropic=isotropic, posterior=True).to(device)
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

    def sample(self, testing=False):
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
        return self.fcomb.forward(self.unet_features,self.z_prior_sample)

    def reconstruct(self, use_posterior_mean=False, calculate_posterior=False, z_posterior=None):
        """
        Reconstruct a segmentation from a posterior sample (decoding a posterior sample) and UNet feature map
        use_posterior_mean: use posterior_mean instead of sampling z_q
        calculate_posterior: use a provided sample or sample from posterior latent space
        """
        if use_posterior_mean:
            z_posterior = self.posterior_z.loc
        else:
            if calculate_posterior:
                z_posterior = self.posterior_z.rsample()
        return self.fcomb.forward(self.unet_features, z_posterior)

    def kl_divergence(self, analytic=True, calculate_posterior=False, z_posterior=None):
        """
        Calculate the KL divergence between the posterior and prior KL(Q||P)
        analytic: calculate KL analytically or via sampling from the posterior
        calculate_posterior: if we use samapling to approximate KL we can sample here or supply a sample
        """
        if analytic:
            #Neeed to add this to torch source code, see: https://github.com/pytorch/pytorch/issues/13545
            kl_div = kl.kl_divergence(self.posterior_z, self.prior_z)
        else:
            if calculate_posterior:
                z_posterior = self.posterior_z.rsample()
            log_posterior_prob = self.posterior_z.log_prob(z_posterior)
            log_prior_prob = self.prior_z.log_prob(z_posterior)
            kl_div = log_posterior_prob - log_prior_prob
        return kl_div


    def elbo(self, segm, analytic_kl=True, reconstruct_posterior_mean=False, patch=None):
        assert type(patch) != None
        """
        Calculate the evidence lower bound of the log-likelihood of P(Y|X)
        """

        # criterion = nn.BCEWithLogitsLoss(size_average = False, reduce=False, reduction=None)
        criterion = nn.BCEWithLogitsLoss(reduction='sum')
        z_posterior = self.posterior_z
        log_var = z_posterior[:,0]
        mean = z_posterior[:,1]
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        self.kl = KLD
        # self.kl = torch.mean(self.kl_divergence(analytic=analytic_kl, calculate_posterior=False, z_posterior=z_posterior))

        #Here we use the posterior sample sampled above
        self.reconstruction = self.reconstruct(use_posterior_mean=reconstruct_posterior_mean, calculate_posterior=False, z_posterior=z_posterior)
        
        reconstruction_loss = criterion(input=self.reconstruction, target=segm)
        self.reconstruction_loss = torch.sum(reconstruction_loss)
        self.mean_reconstruction_loss = torch.mean(reconstruction_loss)

        return (self.reconstruction_loss + self.beta * self.kl)/patch.size(0)

    def output_predict_tensor(self, num_sample=10, manifold_visualisation=False, patch=None):
        assert type(patch) != None
        r = []
        if manifold_visualisation == True:
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
        else:
            for samp in range(num_sample):
                # z_posterior = self.prior_z.rsample()
                z_posterior = self.prior.forward(patch)
                reconstruction = self.reconstruct(z_posterior=z_posterior).cpu().numpy()
                r.append(reconstruction)
            return r