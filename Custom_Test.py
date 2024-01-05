import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import torch
from torch import nn
import torchvision.datasets as datasets
import torchvision.transforms as T

import Utils.Misc_Functions as MF
import Utils.Custom_Transforms as CT
import SSL_Model

###############
# User Inputs #
###############

device = torch.device('cuda')

# Load data
dataRoot = r'D:/ImageNet100/val'

# Model data
ptFile = r'Saved_Models/ImageNet-100/SimCLR_He2023/UE/UE_0p2T_pt_0200.pth.tar'
ftFile = r'Saved_Models/ImageNet-100/SimCLR_He2023/UE/UE_0p2T_pt_0200_UE_lp_0100.pth.tar'
cropSize = 224

batchSize = 256 # 256 for CIFAR, 4096 for IN1k
epochBatches = 1e6

##############
# Data Setup #
##############

dataDataset = datasets.ImageFolder(dataRoot, CT.t_resize_normalize(cropSize))
dataLoader = torch.utils.data.DataLoader(dataDataset, batch_size=batchSize, shuffle=False)

nClasses = len(dataDataset.classes)

###############
# Model Setup #
###############

# Load saved state
ptStateDict = torch.load(ptFile, map_location='cuda:{}'.format(0))

# Create model from state dict
#model = SSL_Model.Base_Model(ptStateDict['encArch'], ptStateDict['rnCifarMod'], ptStateDict['vitPPFreeze'], ptStateDict['prjArch'],
#                             ptStateDict['prjHidDim'], ptStateDict['prjBotDim'], ptStateDict['prjOutDim'], ptStateDict['prdHidDim'],
#                             None, 0.3, 0.5, 0.0, True)
model = SSL_Model.Base_Model(ptStateDict['encArch'], ptStateDict['cifarMod'], ptStateDict['vitPPFreeze'], 'moco',
                             ptStateDict['prjHidDim'], 256, ptStateDict['prjOutDim'], ptStateDict['prdDim'],
                             None, 0.3, 0.5, 0.0, True)

# Replace the projector with identity and the predictor with linear classifier
encDim = model.encoder.inplanes if 'resnet' in ptStateDict['encArch'] else model.encoder.num_features
model.projector = nn.Identity()
model.predictor = nn.Linear(encDim, nClasses)

# Load finetune file
ftStateDict = torch.load(ftFile, map_location='cuda:{}'.format(0))

# If a stateDict key has "module" in (from running parallel), create a new dictionary with the right names
for key in list(ftStateDict.keys()):
    if key.startswith('module.'):
        ftStateDict[key[7:]] = ftStateDict[key]
        del ftStateDict[key]

# Load finetune parameters and freeze all
model.load_state_dict(ftStateDict, strict=True)
for param in model.parameters(): param.requires_grad = False

model = model.to(device)
model.eval()

##
#
##

#nCorrectNoFilter = 0
#nTotal = 0

#powerList = [round(item, 2) for item in list(np.arange(-0.5, 0.52, 0.02))]
#nCorrectFilter = [0] * len(powerList)

for batchI, batch in enumerate(dataLoader):

    # CenterCrop raw data
    augTensOrig = batch[0].to(device)
    truthTens = batch[1].to(device)

    # Further augmentation
    for i in range(0, 225, 56):
        for j in range(0, 225, 56):
            augTens = T.functional.crop(augTensOrig, i, j, cropSize, cropSize)

            # Show sample image:
            #img = Image.fromarray(np.transpose(((augTens[0].detach().cpu().numpy() * 0.229 + 0.45) * 255), (1, 2, 0)).astype(np.uint8))
            #img.show()

            # Run augmented data through model with linear classifier
            p, _, _, _ = model(augTens)

            # Keep running sum of accuracyCustom_Test.py
            nCorrect = torch.sum(torch.argmax(p.detach(), dim=1) == truthTens).cpu().numpy()
            nTotal = augTens.size(0)

            print('Down: {}, Right: {}, Accuracy: {}'.format(i, j, nCorrect / nTotal))

    # Run augmented data through model with linear classifier
    #p, _, _, _ = model(augTens)

    # Keep running sum of accuracy
    #nCorrectNoFilter += torch.sum(torch.argmax(p.detach(), dim=1) == truthTens).cpu().numpy()
    #nTotal += augTens.size(0)

    # Spectral filter applied to z
    #for i, power in enumerate(powerList):
    #    with torch.no_grad():
    #        zSpec = MF.spectral_filter(z, power=power, cutoff=None)
    #        pSpec = model.predictor(zSpec)
    #    nCorrectFilter[i] += torch.sum(torch.argmax(pSpec.detach(), dim=1) == truthTens).cpu().numpy()

    if batchI + 1 >= epochBatches:
        break

#noFilterAcc = nCorrectNoFilter / nTotal
#print(noFilterAcc)

#filterAcc = [item / nTotal for item in nCorrectFilter]
#bestFilterAcc = max(filterAcc)
#bestFilterPow = powerList[filterAcc.index(bestFilterAcc)]

#print(noFilterAcc, bestFilterAcc, bestFilterPow)
#print(filterAcc)
#plt.plot(powerList, filterAcc)
#plt.show()
