import os.path as osp

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm


class MiniImageNet(Dataset):
    def __init__(self, set_name, args, augment=False):

        self.IMAGE_PATH1 = osp.join('./data/miniimagenet', 'images')
        self.SPLIT_PATH = osp.join('./data/miniimagenet', 'split')

        self.label_ids = []

        if set_name != all:
            csv_path = osp.join(self.SPLIT_PATH, set_name + '.csv')
            self.data, self.label = self.parse_csv(csv_path, set_name)
        else:
            self.data, self.label = self.parse_csv(None, set_name)

        self.num_class = len(set(self.label))

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

    def parse_csv(self, csv_path, set_name):
        if set_name != "all":
            lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]
        else:
            lines = []
            lines.extend([x.strip() for x in open(osp.join(self.SPLIT_PATH, 'train.csv'), 'r').readlines()][1:])
            lines.extend([x.strip() for x in open(osp.join(self.SPLIT_PATH, 'val.csv'), 'r').readlines()][1:])
            lines.extend([x.strip() for x in open(osp.join(self.SPLIT_PATH, 'test.csv'), 'r').readlines()][1:])

        data = []
        label = []
        lb = -1

        for line in tqdm(lines, ncols=64):
            name, label_id = line.split(',')
            path = osp.join(self.IMAGE_PATH1, name)
            if label_id not in self.label_ids:
                self.label_ids.append(label_id)
                lb += 1
            data.append(path)
            label.append(lb)

        return data, label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        data, label = self.data[i], self.label[i]
        image = self.transform(Image.open(data).convert('RGB'))

        return image, data
