import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

data_path = "./data/CUB_200_2011"


class Cub(Dataset):
    def __init__(self, set_name, args, augment=False):

        with open(os.path.join(data_path, 'images.txt'), 'r') as f:
            image_list = f.readlines()

        image_index = []
        image_path = []
        for data in image_list:
            index, path = data.split(' ')
            image_index.append(int(index))
            image_path.append(os.path.join(data_path, 'images', path[:-1]))

        self.image_path = image_path

        train_flag = np.loadtxt(os.path.join(data_path, 'train_test_split.txt'), delimiter=' ', dtype=np.int32)
        labels = np.loadtxt(os.path.join(data_path, 'image_class_labels.txt'), delimiter=' ', dtype=np.int32)
        labels = labels[:, 1]

        # use first 100 classes
        targets = np.where(labels < 101)[0]
        self.labels = labels
        self.indices = targets
        self.label = list(self.labels[self.indices] - 1)
        self.num_classes = self.num_class = 100

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
                transforms.RandomResizedCrop(84),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]
        else:
            transforms_list = [
                transforms.Resize(92),
                transforms.CenterCrop(84),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]

        self.transform = transforms.Compose(transforms_list)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        index = self.indices[i]
        path = self.image_path[index]
        label = self.labels[index]
        image = self.transform(Image.open(path).convert('RGB'))

        return image, label
