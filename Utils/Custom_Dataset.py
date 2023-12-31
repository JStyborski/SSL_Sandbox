import numpy as np
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.datasets as datasets

class No_Labels_Images(Dataset):
    def __init__(self, rootDir, transform):
        super(No_Labels_Images, self).__init__()
        self.rootDir = rootDir
        self.transform = transform
        self.filesList = [f for f in os.listdir(self.rootDir)]

    def __len__(self):
        return len(self.filesList)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.rootDir, self.filesList[index]))
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        target = 0

        return img, target


class No_Labels(Dataset):
    def __init__(self, rootDir, transform, loader):
        super(No_Labels, self).__init__()
        self.rootDir = rootDir
        self.transform = transform
        self.loader = loader
        self.filesList = [f for f in os.listdir(self.rootDir)]

    def __len__(self):
        return len(self.filesList)

    def __getitem__(self, index):
        sample = self.loader(os.path.join(self.rootDir, self.filesList[index]))
        if self.transform is not None:
            sample = self.transform(sample)
        target = 0

        return sample, target


class No_Labels_Plus_Path(No_Labels):
    # Same as original, but additionally returns the corresponding path

    def __getitem__(self, index):
        path = os.path.join(self.rootDir, self.filesList[index])
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        target = 0

        return sample, target, path


class DatasetFolder_Plus_Path(datasets.DatasetFolder):
    # Same as original, but additionally returns the corresponding path

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, path


class No_Labels_Images_Plus_Poison(No_Labels_Images):
    # Same as original but additionally returns the corresponding poison delta

    def __init__(self, rootDir, transform, deltaDataset):
        super().__init__(rootDir, transform)
        self.deltaDataset = deltaDataset

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.rootDir, self.filesList[index]))
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        target = 0
        delta, _, deltaPath = self.deltaDataset.__getitem__(index)

        return img, target, delta.type(torch.float32), deltaPath


class ImageFolder_Plus_Poison(datasets.ImageFolder):
    # Same as original but additionally returns the corresponding poison delta

    def __init__(self, rootDir, transform, deltaDataset):
        super().__init__(rootDir, transform)
        self.deltaDataset = deltaDataset

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        delta, _, deltaPath = self.deltaDataset.__getitem__(index)

        return sample, target, delta.type(torch.float32), deltaPath


def create_shadow_tensors(sourceFolder, shadowFolder, trainLabels, randInit=True, precision=torch.float16):
    """
    Given a dataset of images, initialize a folder of tensors that shadow the image names and shapes
    :param imageDataset:
    :return:
    """

    # If labels exist, then images are sorted into class folders
    if trainLabels:

        # Get class folders and for each, create an identical folder in the shadow directory
        classFolders = os.listdir(sourceFolder)
        for classFolder in classFolders:
            classPath = os.path.join(shadowFolder, str(classFolder))
            if not os.path.exists(classPath):
                os.mkdir(classPath)

            # Get the images in the original class folder and for each, create a tensor of the same name in the shadow folder
            imgFiles = os.listdir(os.path.join(sourceFolder, classFolder))
            for imgFile in imgFiles:
                if not os.path.exists(os.path.join(shadowFolder, classFolder, imgFile.split('.')[0])):
                    imgSize = Image.open(os.path.join(sourceFolder, classFolder, imgFile)).size
                    if randInit:
                        tens = torch.rand(3, imgSize[1], imgSize[0], dtype=precision).mul(2.).sub(1.)
                    else:
                        tens = torch.zeros(3, imgSize[1], imgSize[0], dtype=precision)
                    torch.save(tens, os.path.join(shadowFolder, classFolder, imgFile.split('.')[0]))

    # If no labels exist, then do the same as above without worrying about class folders
    else:
        imgFiles = os.listdir(sourceFolder)
        for imgFile in imgFiles:
            if not os.path.exists(os.path.join(shadowFolder, imgFile.split('.')[0])):
                imgSize = Image.open(os.path.join(sourceFolder, imgFile)).size
                if randInit:
                    tens = torch.rand(3, imgSize[1], imgSize[0], dtype=precision).mul(2.).sub(1.)
                else:
                    tens = torch.zeros(3, imgSize[1], imgSize[0], dtype=precision)
                torch.save(tens, os.path.join(shadowFolder, imgFile.split('.')[0]))


def combine_shadow_tensors(sourceFolder, shadowFolder, poisonFolder, trainLabels, advEps):

    # If labels exist, then images are sorted into class folders
    if trainLabels:

        # Get class folders and for each, create an identical folder in the shadow directory
        classFolders = os.listdir(sourceFolder)
        for classFolder in classFolders:
            classPath = os.path.join(poisonFolder, str(classFolder))
            if not os.path.exists(classPath):
                os.mkdir(classPath)

            # Get the images in the original class folder and for each, create a tensor of the same name in the shadow folder
            imgFiles = os.listdir(os.path.join(sourceFolder, classFolder))
            for imgFile in imgFiles:
                sourceImg = np.asarray(Image.open(os.path.join(sourceFolder, classFolder, imgFile)))
                delta = torch.load(os.path.join(shadowFolder, classFolder, imgFile.split('.')[0])).numpy()
                poisonImg = Image.fromarray(np.clip(sourceImg + 255. * advEps * np.transpose(delta, (1, 2, 0)), a_min=0., a_max=255.).astype('uint8'))
                poisonImg.save(os.path.join(poisonFolder, classFolder, imgFile))

    # If no labels exist, then do the same as above without worrying about class folders
    else:
        imgFiles = os.listdir(sourceFolder)
        for imgFile in imgFiles:
            sourceImg = np.asarray(Image.open(os.path.join(sourceFolder, imgFile)))
            delta = torch.load(os.path.join(shadowFolder, imgFile.split('.')[0])).numpy()
            delta = np.transpose(delta, (2, 1, 0))
            poisonImg = Image.fromarray(np.clip(sourceImg + 255. * advEps * np.transpose(delta, (1, 2, 0)), a_min=0., a_max=255.).astype('uint8'))
            poisonImg.save(os.path.join(poisonFolder, imgFile))
