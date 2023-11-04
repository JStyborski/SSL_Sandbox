import os
from PIL import Image

from torch.utils.data import Dataset

class no_label_dataset(Dataset):
    def __init__(self, rootDir, transform):
        super(no_label_dataset, self).__init__()
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

        imgClass = 0

        return img, imgClass
