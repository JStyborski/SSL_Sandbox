import random
from PIL import ImageFilter
import numpy as np
import torch
import torchvision.transforms as transforms

class RandomCrop1D(object):
    def __init__(self, cropSize, device):
        """
        Initialize RandomCrop1D parameters
        :param cropSize: [int] [1] - Output size of the crop
        :param device: [string] [1] - Device to store cropping mask tensor
        """
        self.cropSize = cropSize
        self.device = device

    def __call__(self, x):
        """
        Generate random crop masks for each input sample and apply to input
        :param x: [tensor] [m x n] - Input tensor
        :return xa: [tensor] [m x cropSize] - Randomly cropped output tensor
        """
        startElem = np.random.choice(range(x.size(1) - self.cropSize + 1), size=x.size(0))
        cropMask = np.zeros(x.size())
        for i in range(x.size(0)):
            cropMask[i, startElem[i]:startElem[i] + self.cropSize] = 1
        cropMaskTens = torch.tensor(cropMask).to(torch.bool).to(self.device)
        xa = x[cropMaskTens].view(x.size(0), self.cropSize)
        return xa


class CenterCrop1D(object):
    def __init__(self, cropSize, device):
        """
        Initialize CenterCrop1D parameters
        :param cropSize: [int] [1] - Output size of the crop
        :param device: [string] [1] - Device to store cropping mask tensor
        """
        self.cropSize = cropSize
        self.device = device

    def __call__(self, x):
        """
        Generate center-crop masks for each input sample and apply to input
        :param x: [tensor] [m x n] - Input tensor
        :return xa: [tensor] [m x cropSize] - Center cropped output tensor
        """
        startElem = round(x.size(1) / 2) - round(self.cropSize / 2)
        cropMask = np.zeros(x.size())
        cropMask[:, startElem:startElem + self.cropSize] = 1
        cropMaskTens = torch.tensor(cropMask).to(torch.bool).to(self.device)
        xa = x[cropMaskTens].view(x.size(0), self.cropSize)
        return xa


class RandomScale1D(object):
    def __init__(self, scaleMag, device):
        """
        Initialize RandomScale1D parameters
        :param scaleMag: [float] [1] - Maximum magnitude of random scaling for each element
        :param device: [string] [1] - Device to store cropping mask tensor
        """
        self.scaleMag = scaleMag
        self.device = device

    def __call__(self, x):
        """
        Generate a random scaling tensor (with magnitudes between [1 - scaleMag, 1 + scaleMag] and apply to input
        :param x: [tensor] [m x n] - Input tensor
        :return xa: [tensor] [m x n] - Randomly scaled output tensor
        """
        scaleVal = self.scaleMag * 2 * (np.random.rand(x.size(0), x.size(1)) - 0.5) + 1
        scaleVal = torch.tensor(scaleVal, dtype=torch.float32).to(self.device)
        xa = torch.mul(x, scaleVal)
        return xa


def MoCoV2Transform(mode, cropSize):

    if mode == 'train':
        t = transforms.Compose([
            transforms.RandomResizedCrop(cropSize, scale=(0.2, 1.)),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    elif mode == 'test':
        t = transforms.Compose([
            transforms.RandomResizedCrop(cropSize),
            transforms.RandomHorizontalFlip(),
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


class TwoTimesTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        t1 = self.base_transform(x)
        t2 = self.base_transform(x)
        return [t1, t2]
