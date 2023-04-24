import os
import pickle

import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image


def load_data(file):
    with open(file, 'rb') as fo:
        return pickle.load(fo)


class TieredImageNet(data.Dataset):
    def __init__(self, set_name, args, augment=False):
        assert (set_name == 'train' or set_name == 'val' or set_name == 'test')

        npz_path = './data/tiered-imagenet'

        file_path = {
            'train': [os.path.join(npz_path, 'train_images.npz'), os.path.join(npz_path, 'train_labels.pkl')],
            'val': [os.path.join(npz_path, 'val_images.npz'), os.path.join(npz_path, 'val_labels.pkl')],
            'test': [os.path.join(npz_path, 'test_images.npz'), os.path.join(npz_path, 'test_labels.pkl')]}

        image_path = file_path[set_name][0]
        label_path = file_path[set_name][1]

        data_train = load_data(label_path)
        labels = data_train['labels']
        self.data = np.load(image_path)['images']
        label = []
        lb = -1
        self.label_ids = []
        for label_id in labels:
            if label_id not in self.label_ids:
                self.label_ids.append(label_id)
                lb += 1
            label.append(lb)

        self.label = label
        self.num_class = len(set(label))

        # Transformation
        if args.backbone_class in ['ConvNet', 'Res18', 'WRN']:
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
        elif args.backbone_class == 'ResNet':
            mean = np.array([x / 255.0 for x in [125.3, 123.0, 113.9]])
            std = np.array([x / 255.0 for x in [63.0, 62.1, 66.7]])
        elif args.backbone_class == 'Res12':
            mean = np.array([x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]])
            std = np.array([x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]])
        else:
            raise ValueError('Non-supported Network Types. Please Revise Data Pre-Processing Scripts.')

        if augment and set_name == 'train':
            transforms_list = [
                transforms.RandomCrop(84, padding=8),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]
        else:
            transforms_list = [
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]

        self.transform = transforms.Compose(transforms_list)

    def __getitem__(self, index):
        img, label = self.data[index], self.label[index]
        img = self.transform(Image.fromarray(img))
        return img, label

    def __len__(self):
        return len(self.data)
