import torch
from dataset.load_LIDC_data import LIDC_IDRI
from prob_unet.probabilistic_unet import ProbabilisticUnet
from utils.utils import l2_regularisation
from tqdm import tqdm
import os
import imageio
import numpy as np
from dataset.dataloader import Dataloader
from configs import *
from torchvision import transforms
import utils.joint_transforms as joint_transforms
from eval import visualise_manifold, dir_check, output_predict_img, generalised_energy_distance
from train_model import train, dir_check


if __name__ == '__main__':
    dataset = LIDC_IDRI(dataset_location=data_dir, joint_transform=joint_transfm, input_transform=input_transfm
                        , target_transform=target_transfm, random=random)
    dataloader = Dataloader(dataset, batch_size, small=partial_data, shuffle_indices=shuffle_indices)

    assert train_or_eval is not 'train' or 'eval'
    if train_or_eval == 'eval':
        print('evaluating.....')
        net = ProbabilisticUnet(input_channels=1, num_classes=1, num_filters_=num_filters, latent_dim=latent_dim,
                                beta=beta, initializers=initializers, device=device).to(device)
        # for s in num_sample:
        #     generalised_energy_distance(dataloader, net, s)
        output_predict_img(dataloader, net)
        # visualise_manifold(dataloader, net)
    elif train_or_eval == 'train':
        print('training.....')
        train(dataloader)
