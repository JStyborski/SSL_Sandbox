import random
from PIL import ImageFilter
import numpy as np
import torch
import torchvision.transforms as transforms

def t_pretrain(cropSize):
    t = transforms.Compose([
        transforms.RandomResizedCrop(cropSize, scale=(0.2, 1.)),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return t

def t_randcrop(cropSize):
    t = transforms.Compose([
        transforms.RandomResizedCrop(cropSize, scale=(0.2, 1.)),
        transforms.ToTensor(),
    ])
    return t

def t_toPIL_pretrain():
    t = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return t

def t_finetune(cropSize):
    t = transforms.Compose([
        transforms.RandomResizedCrop(cropSize),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return t

def t_test(cropSize):
    t = transforms.Compose([
        transforms.Resize(int(round(1.1428 * cropSize))),  # CIFAR: 1.1428 * 28 = 32, IN: 1.1428 * 224 = 256
        transforms.CenterCrop(cropSize),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return t


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class RandomFilter(object):

    def __init__(self, nFilters=1, maxMag=1, blurMag=0.06, kernelSize=9):
        # CIFAR: 1, 1, 0.3, 3 | ImageNet: 1, 1, 0.06, 9
        self.nFilters = nFilters
        self.maxMag = maxMag
        self.blurMag = blurMag
        self.kernelSize = kernelSize

    def __call__(self, x):
        for i in range(self.nFilters):
            kernel = np.random.uniform(low=0, high=self.blurMag, size=(1, 1, self.kernelSize, self.kernelSize))
            if self.maxMag is not None:
                kernel[0, 0, np.random.randint(kernel.shape[2]), np.random.randint(kernel.shape[3])] = self.maxMag
            kernel = np.repeat(kernel, 3, axis=0)
            x = torch.nn.functional.conv2d(x, torch.from_numpy(kernel).float(), stride=1, groups=3, padding='same')
            x /= x.max()
        return x


class GaussianNoise(object):

    def __init__(self, sigma=0.5):
        self.sigma = sigma

    def __call__(self, x):
        x = x + torch.randn_like(x) * self.sigma ** 2
        x = torch.clamp(x, 0.0, 1.0)
        return x


class RandomMask(object):

    def __init__(self, cropSize, patchSize, maskProb):
        self.cropSize = cropSize
        self.patchSize = patchSize
        self.maskProb = maskProb

    def __call__(self, x):
        mask = np.random.choice([0, 1], replace=True, p=[self.maskProb, 1 - self.maskProb],
                                size=(1, int(self.cropSize / self.patchSize), int(self.cropSize / self.patchSize)))
        mask = np.repeat(np.repeat(mask, self.patchSize, axis=1), self.patchSize, axis=2)
        x = x * mask
        return x


class NTimesTransform:
    """Take n random crops of one image as the query and key."""

    def __init__(self, n_views, base_transform):
        self.n_views = n_views
        self.base_transform = base_transform

    def __call__(self, x):
        tList = [self.base_transform(x) for _ in range(self.n_views)]
        return tList


class NTimesTransform_2Parts:
    """Take n random crops of one image as the query and key."""

    def __init__(self, n_views, crop_transform, main_transform):
        self.n_views = n_views
        self.crop_transform = crop_transform
        self.main_transform = main_transform

    def __call__(self, x):
        cList = [self.crop_transform(x) for _ in range(self.n_views)]
        tList = [self.main_transform(crop) for crop in cList]
        return cList, tList
