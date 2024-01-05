import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import torch
from torch import nn
import torchvision.datasets as datasets

import Utils.IG_Visualization_Library as IGVis
import Utils.Custom_Transforms as CT
import Utils.Custom_Visualization as CV
import SSL_Model

###############
# User Inputs #
###############

device = torch.device('cuda')

# Load data
dataRoot = r'D:/ImageNet100/val'

# Model data
ptFile = r'Saved_Models/ImageNet-100/SimCLR_He2023/Clean/Clean_0p2T_pt_0200.pth.tar'
ftFile = r'Saved_Models/ImageNet-100/SimCLR_He2023/Clean/Clean_0p2T_pt_0200_Clean_lp_0100.pth.tar'
cropSize = 224

batchSize = 256 # 256 for CIFAR, 4096 for IN1k
epochBatches = 1e6

desiredClass = 20

##############
# Data Setup #
##############

dataDataset = datasets.ImageFolder(dataRoot, CT.t_test(cropSize))
dataLoader = torch.utils.data.DataLoader(dataDataset, batch_size=batchSize, shuffle=True)

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

##########################
# Run Visualization Loop #
##########################

for batchI, batch in enumerate(dataLoader):

    # CenterCrop raw data
    augTens = batch[0].to(device)
    truthTens = batch[1].to(device)

    if desiredClass is not None:
        # Get indices of desired class and select first instance
        desiredClassIdx = torch.nonzero(truthTens == desiredClass)[0]
        augTens = augTens[desiredClassIdx]
        truthTens = truthTens[desiredClassIdx]
    else:
        # Select first instance
        augTens = augTens[[0]]
        truthTens = truthTens[[0]]

    # Use Integrated Gradients to determine input attributions to class label
    attributions = CV.multi_baseline_IG(augTens, model, truthTens, 50, 10)

    # Convert attributions to HxWx3 numpy array, convert input image to 0-255 HxWx3 numpy array
    attArr = np.transpose(attributions.cpu().numpy(), (1, 2, 0))
    augArr = np.transpose(np.clip(augTens[0].cpu().numpy() * 0.226 + 0.45, 0, 1) * 255., (1, 2, 0))

    # Get visualizable attributions and plot
    # polarity can be ['positive', 'negative', 'both'] to visualize positive and/or negative attributions
    # overlay combines the attributions with source image
    attArr = IGVis.Visualize(attArr, augArr, polarity='both', overlay=False)
    augImg = Image.fromarray(augArr.astype(np.uint8))
    augImg.show()
    attImg = Image.fromarray(attArr.astype(np.uint8))
    attImg.show()

    if batchI + 1 >= epochBatches:
        break

