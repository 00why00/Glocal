import os
import pickle

import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class CifarFs(Dataset):
    def __init__(self, set_name, args, augment=False):
        image_path = './data/CIFAR-FS'
        with open(os.path.join(image_path, 'CIFAR_FS_{}.pickle'.format(set_name)), 'rb') as f:
            pack = pickle.load(f, encoding='latin1')
            self.images = pack['data']
            self.label = pack['labels']

        self.num_class = len(set(self.label))
        self.label_trans = {j: i for i, j in enumerate(set(self.label))}
        self.label = [self.label_trans[self.label[i]] for i in range(len(self.label))]

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
                transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]
        else:
            transforms_list = [
                transforms.ToPILImage(),
                transforms.Resize(40),
                transforms.CenterCrop(32),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]

        self.transform = transforms.Compose(transforms_list)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        image = self.images[i]
        label = self.label[i]
        image = self.transform(image)

        return image, label
