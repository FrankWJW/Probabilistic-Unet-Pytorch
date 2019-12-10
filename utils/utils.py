import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt


def truncated_normal_(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)


def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        # nn.init.normal_(m.weight, std=0.001)
        # nn.init.normal_(m.bias, std=0.001)
        truncated_normal_(m.bias, mean=0, std=0.001)


def init_weights_orthogonal_normal(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.orthogonal_(m.weight)
        truncated_normal_(m.bias, mean=0, std=0.001)
        # nn.init.normal_(m.bias, std=0.001)


def l2_regularisation(m):
    l2_reg = None

    for W in m.parameters():
        if l2_reg is None:
            l2_reg = W.norm(2)
        else:
            l2_reg = l2_reg + W.norm(2)
    return l2_reg


def save_mask_prediction_example(mask, pred, iter):
    plt.imshow(pred[0, :, :], cmap='Greys')
    plt.savefig('images/' + str(iter) + "_prediction.png")
    plt.imshow(mask[0, :, :], cmap='Greys')
    plt.savefig('images/' + str(iter) + "_mask.png")


import numpy as np
from medpy.metric import jc
from tqdm import tqdm


def dist_fct(m1, m2, n_labels=1):
    per_label_iou = []
    for lbl in range(n_labels):

        # assert not lbl == 0  # tmp check
        m1_bin = (m1 != lbl) * 1
        m2_bin = (m2 != lbl) * 1

        if np.sum(m1_bin) == 0 and np.sum(m2_bin) == 0:
            per_label_iou.append(1)
        elif np.sum(m1_bin) > 0 and np.sum(m2_bin) == 0 or np.sum(m1_bin) == 0 and np.sum(m2_bin) > 0:
            per_label_iou.append(0)
        else:
            per_label_iou.append(jc(m1_bin, m2_bin))

    # print(1-(sum(per_label_iou) / n_labels))

    return 1 - (sum(per_label_iou) / n_labels)


def generalised_energy_distance(sample_arr, gt_arr):
    """
    :param sample_arr: expected shape N x X x Y
    :param gt_arr: M x X x Y
    :return:
    """

    N = sample_arr.shape[0]
    M = gt_arr.shape[0]

    d_sy = []
    d_ss = []
    d_yy = []

    for i in range(N):
        for j in range(M):
            # print(dist_fct(sample_arr[i,...], gt_arr[j,...]))
            d_sy.append(dist_fct(sample_arr[i, ...], gt_arr[j, ...]))

    for i in range(N):
        for j in range(N):
            # print(dist_fct(sample_arr[i,...], sample_arr[j,...]))
            d_ss.append(dist_fct(sample_arr[i, ...], sample_arr[j, ...]))

    for i in range(M):
        for j in range(M):
            # print(dist_fct(gt_arr[i,...], gt_arr[j,...]))
            d_yy.append(dist_fct(gt_arr[i, ...], gt_arr[j, ...]))

    return (2. / (N * M)) * sum(d_sy) - (1. / N ** 2) * sum(d_ss) - (1. / M ** 2) * sum(d_yy)


def ncc(a, v, zero_norm=True):
    a = a.flatten()
    v = v.flatten()

    if zero_norm:

        a = (a - np.mean(a)) / (np.std(a) * len(a))
        v = (v - np.mean(v)) / np.std(v)

    else:

        a = (a) / (np.std(a) * len(a))
        v = (v) / np.std(v)

    return np.correlate(a, v)


def variance_ncc_dist(sample_arr, gt_arr):
    def pixel_wise_xent(m_samp, m_gt, eps=1e-8):

        log_samples = np.log(m_samp + eps)

        return -1.0 * np.sum(m_gt * log_samples, axis=-1)

    """
    :param sample_arr: expected shape N x X x Y 
    :param gt_arr: M x X x Y
    :return: 
    """

    mean_seg = np.mean(sample_arr, axis=0)

    N = sample_arr.shape[0]
    M = gt_arr.shape[0]

    sX = sample_arr.shape[1]
    sY = sample_arr.shape[2]

    E_ss_arr = np.zeros((N, sX, sY))
    for i in range(N):
        E_ss_arr[i, ...] = pixel_wise_xent(sample_arr[i, ...], mean_seg)
        # print('pixel wise xent')
        # plt.imshow( E_ss_arr[i,...])
        # plt.show()

    E_ss = np.mean(E_ss_arr, axis=0)

    E_sy_arr = np.zeros((M, N, sX, sY))
    for j in range(M):
        for i in range(N):
            E_sy_arr[j, i, ...] = pixel_wise_xent(sample_arr[i, ...], gt_arr[j, ...])

    E_sy = np.mean(E_sy_arr, axis=1)

    ncc_list = []
    for j in range(M):
        ncc_list.append(ncc(E_ss, E_sy[j, ...]))

    return (1 / M) * sum(ncc_list)
