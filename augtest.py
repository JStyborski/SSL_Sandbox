import os
import numpy as np
import torch
from torch import nn
import torchvision.datasets as datasets

import SSL_Transforms
import SSL_Model
from Adversarial import FGSM_PGD, Local_Lip
import SSL_Probes

###############
# User Inputs #
###############

device = torch.device('cuda')

# Load data
dataRoot = r'D:/CIFAR-10/Poisoned/TAP_untargeted'

# Model data
ptFile = r'Saved_Models/CIFAR10/TAP_untargeted/SimSiam_1000_512_LR0p5_WD1e-4/SimSiam-TAP-1e-4_pt_1000.pth.tar'
ftFile = r'Saved_Models/CIFAR10/TAP_untargeted/SimCLR_WD1e-4_INCE-Fix_He_Params/He-SimCLR-TAP_pt_1000_TAP_He_lp_0100.pth.tar'
cropSize = 28

batchSize = 256 # 256 for CIFAR, 4096 for IN1k
epochBatches = 1e6

##############
# Data Setup #
##############

dataDataset = datasets.ImageFolder(dataRoot, SSL_Transforms.MoCoV2Transform('test', cropSize))
dataLoader = torch.utils.data.DataLoader(dataDataset, batch_size=batchSize, shuffle=False)

nClasses = len(dataDataset.classes)

###############
# Model Setup #
###############

# Load saved state
ptStateDict = torch.load(ptFile, map_location='cuda:{}'.format(0))

# Create model and load model weights
model = SSL_Model.Base_Model(ptStateDict['encArch'], ptStateDict['cifarMod'], ptStateDict['encDim'],
                             ptStateDict['prjHidDim'], ptStateDict['prjOutDim'], ptStateDict['prdDim'], None, 0.3, 0.5, 0.0)

# Replace the projector with identity and the predictor with linear classifier
model.projector = nn.Identity(ptStateDict['encDim'])
model.predictor = nn.Linear(ptStateDict['encDim'], nClasses)

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

crossEnt = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-5)

for batchI, batch in enumerate(dataLoader):

    # Get input tensor and truth labels
    augTens = batch[0].to(device)
    truthTens = batch[1].to(device)

    augTens.requires_grad = True

    # Run augmented data through SimSiam with linear classifier
    p, _, _, _ = model(augTens)

    # Calculate loss
    lossVal = crossEnt(p, truthTens)

    # Backpropagate
    optimizer.zero_grad()
    lossVal.backward()
    optimizer.step()
