import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
import os
import random
import pickle
import matplotlib.pyplot as plt
import imageio
from PIL import Image


class LIDC_IDRI(Dataset):
    images = []
    labels = []
    series_uid = []

    def __init__(self, dataset_location, joint_transform=None, input_transform=None, target_transform=None):
        self.input_transform = input_transform
        self.joint_transform = joint_transform
        self.target_transform = target_transform
        max_bytes = 2 ** 31 - 1
        data = {}
        for file in os.listdir(dataset_location):
            filename = os.fsdecode(file)
            if '.pickle' in filename:
                print("Loading file", filename)
                file_path = os.path.join(dataset_location, filename)
                bytes_in = bytearray(0)
                input_size = os.path.getsize(file_path)
                with open(file_path, 'rb') as f_in:
                    for _ in range(0, input_size, max_bytes):
                        bytes_in += f_in.read(max_bytes)
                new_data = pickle.loads(bytes_in)
                data.update(new_data)

        for key, value in data.items():
            self.images.append(value['image'].astype(float))
            self.labels.append(value['masks'])
            self.series_uid.append(value['series_uid'])

        assert (len(self.images) == len(self.labels) == len(self.series_uid))

        for img in self.images:
            assert np.max(img) <= 1 and np.min(img) >= 0
        for label in self.labels:
            assert np.max(label) <= 1 and np.min(label) >= 0

        del new_data
        del data

    def __getitem__(self, index):

        image = self.images[index]

        # Randomly select one of the four labels for this image
        label = self.labels[index][random.randint(0, 3)].astype(float)
        if self.input_transform is not None:
            image = np.uint8(image*255)
            label = np.uint8(label*255)
            image = self.input_transform(image)
            label = self.input_transform(label)
        if self.joint_transform is not None:
            image, label = self.joint_transform(image, label)
        if self.target_transform is not None:
            image = self.target_transform(image)
            label = self.target_transform(label)
        series_uid = self.series_uid[index]

        # Convert uint8 to float tensors
        image = image.type(torch.FloatTensor)
        label = label.type(torch.FloatTensor)

        return image, label, series_uid

    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.images)

    def save_data_set(dataset, output_dir):
        print('saving dataset...')
        for k, np_img in enumerate(dataset.images):
            imageio.imwrite(os.path.join(output_dir, 'image_' + str(k) + '.png'), np_img)
            for k_l, np_label in enumerate(dataset.labels[k]):
                imageio.imwrite(os.path.join(output_dir, 'image_' + str(k) + 'label_' + str(k_l) + '.png'),
                                np_label * 255)



