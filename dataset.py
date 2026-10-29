import torchvision.transforms.functional as F
import numpy as np
import random
import os
from PIL import Image
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch


class ToTensor(object):

    def __call__(self, data):
        image, label = data['image'], data['label']
        return {'image': F.to_tensor(image), 'label': F.to_tensor(label)}


class Resize(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, data):
        image, label = data['image'], data['label']

        return {'image': F.resize(image, self.size), 'label': F.resize(label, self.size, interpolation=InterpolationMode.BICUBIC)}


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        image, label = data['image'], data['label']

        if random.random() < self.p:
            return {'image': F.hflip(image), 'label': F.hflip(label)}

        return {'image': image, 'label': label}


class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        image, label = data['image'], data['label']

        if random.random() < self.p:
            return {'image': F.vflip(image), 'label': F.vflip(label)}

        return {'image': image, 'label': label}


class Normalize(object):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = F.normalize(image, self.mean, self.std)
        return {'image': image, 'label': label}


class FullDataset(Dataset):
    def __init__(self, image_root, gt_root, size, mode):
        self.images = [os.path.join(image_root, f) for f in os.listdir(image_root) if f.endswith(('.jpg', '.png', '.jpeg'))]
        self.gts = [os.path.join(gt_root, f) for f in os.listdir(gt_root) if f.endswith(('.jpg', '.png', '.jpeg'))]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        if len(self.images) != len(self.gts):
            raise ValueError(
                f"Image/mask count mismatch: {len(self.images)} images in {image_root}, "
                f"{len(self.gts)} masks in {gt_root}"
            )
        for image, mask in zip(self.images, self.gts):
            if os.path.splitext(os.path.basename(image))[0] != os.path.splitext(os.path.basename(mask))[0]:
                raise ValueError(f'Image/mask stem mismatch: {image}, {mask}')
        if mode == 'train':
            self.transform = transforms.Compose([
                Resize((size, size)),
                RandomHorizontalFlip(p=0.5),
                RandomVerticalFlip(p=0.5),
                ToTensor(),
                Normalize()
            ])
        else:
            self.transform = transforms.Compose([
                Resize((size, size)),
                ToTensor(),
                Normalize()
            ])

    def __getitem__(self, idx):
        image = self.rgb_loader(self.images[idx])
        label = self.binary_loader(self.gts[idx])
        data = {'image': image, 'label': label}
        data = self.transform(data)
        data['sample_id'] = os.path.abspath(self.images[idx])
        parent = os.path.dirname(os.path.normpath(os.path.dirname(self.images[idx])))
        if os.path.basename(parent).lower() in ('train', 'val', 'validation', 'test'):
            parent = os.path.dirname(parent)
        data['dataset'] = os.path.basename(parent)
        return data

    def __len__(self):
        return len(self.images)

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')


class TestDataset:
    def __init__(self, image_root, gt_root, size):
        self.images = [os.path.join(image_root, f) for f in os.listdir(image_root) if
                       f.endswith(('.jpg', '.png', '.jpeg'))]
        self.gts = [os.path.join(gt_root, f) for f in os.listdir(gt_root) if
                    f.endswith(('.jpg', '.png', '.jpeg'))]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        if len(self.images) != len(self.gts):
            raise ValueError(
                f"Image/mask count mismatch: {len(self.images)} images in {image_root}, "
                f"{len(self.gts)} masks in {gt_root}"
            )
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if idx >= self.size:
            raise IndexError("Index out of range")

        current_index = self.index
        self.index = idx

        image, gt, name = self.load_data()

        self.index = current_index

        return {
            'image': image.squeeze(0),
            'label': torch.from_numpy(gt).unsqueeze(0),
            'name': name
        }

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)

        gt = self.binary_loader(self.gts[self.index])
        gt = np.array(gt)

        name = os.path.basename(self.images[self.index])

        self.index += 1
        return image, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
