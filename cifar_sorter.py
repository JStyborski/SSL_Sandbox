import os
import torchvision.datasets as datasets

fakePoisonRoot = r'D:/CIFAR-10/Poisoned/ehh'
truePoisonRoot = r'D:/CIFAR-10/Poisoned/TAP_untargeted'

allLabels = datasets.CIFAR10(root=fakePoisonRoot, train=True, download=True).targets
unqLabels = list(set(allLabels))

def make_folders(allClasses, imgDir):
    for fileClass in allClasses:
        classPath = os.path.join(imgDir, str(fileClass))
        if not os.path.exists(classPath):
            os.mkdir(classPath)

def sort_imgs(labelList):
    for i, label in enumerate(labelList):
        img = str(i) + '.png'
        os.rename(truePoisonRoot + '/' + img, truePoisonRoot + '/' + str(label) + '/' + img)

make_folders(unqLabels, truePoisonRoot)
sort_imgs(allLabels)
