import matplotlib.pyplot as plt
import numpy as np

import torch
from torch import nn
import torchvision.datasets as datasets

import SSL_Transforms
import SSL_Model

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
dataLoader = torch.utils.data.DataLoader(dataDataset, batch_size=batchSize, shuffle=True)

nClasses = len(dataDataset.classes)

###############
# Model Setup #
###############

# Load saved state
ptStateDict = torch.load(ptFile, map_location='cuda:{}'.format(0))

# Create model and load model weights
model = SSL_Model.Base_Model(ptStateDict['encArch'], ptStateDict['cifarMod'], ptStateDict['encDim'],
                             ptStateDict['prjHidDim'], ptStateDict['prjOutDim'], ptStateDict['prdDim'], None, 0.3, 0.5, 0.0, True)

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

#############################
# Define Setup and Training #
#############################

def spectral_filter(z, power=0.0, cutoff=None):
    # For z (n x d) and Cz (d x d), z = U Sigz V.T, Cz = 1/n z.T z = Q Lamz Q.T, with Q = V and Lamz = 1/n Sigz^2
    # Spectral filter g(Lam) adjusts eigenvalues and then applies W = V g(Lamz) V.T on z, p = z @ W
    # This affects output correlation: Cp = V g(Lamz)^2 Lam V.T, such that Lamp = g(Lamz)^2 Lamz
    # Low pass filter emphasizes large eigvals and diminishes low eigvals - high pass filter vice versa
    # In this function we specifically apply g(Lamz) = Lamz.pow(power)
    # power should be between -0.5 and +1.0 - [-0.5, 0] gives high pass filter, [0, 1.0] gives low pass filter
    # Power examples: -0.5 -> Lamp = I, 0 -> Lamp = Lamz, 0.5 -> Lamp = Lamz^2, 1.0 -> Lamp = Lamz^3
    U, Sigz, VT = torch.linalg.svd(z, full_matrices=False)
    Lamz = 1 / z.size(0) * Sigz.clamp(0).pow(2)
    Lamp = Lamz
    if power is not None:
        Lamp = Lamz.pow(1 + 2 * power)
    if cutoff is not None:
        Lamp[cutoff:] = 0
    Sigp = Lamp.sqrt() * z.size(0) ** 0.5
    specZ = U @ torch.diag(Sigp) @ VT
    return specZ

##
#
##

nCorrectNoFilter = 0
nTotal = 0

#powerList = [round(item, 2) for item in list(np.arange(-0.5, 0.52, 0.02))]
#nCorrectFilter = [0] * len(powerList)

for batchI, batch in enumerate(dataLoader):

    # CenterCrop raw data
    augTens = batch[0].to(device)
    truthTens = batch[1].to(device)

    # Run augmented data through SimSiam with linear classifier
    p, _, _, _ = model(augTens)

    # Keep running sum of accuracy
    nCorrectNoFilter += torch.sum(torch.argmax(p.detach(), dim=1) == truthTens).cpu().numpy()
    nTotal += batch[0].size(0)

    # Spectral filter applied to z
    #for i, power in enumerate(powerList):
    #    with torch.no_grad():
    #        zSpec = spectral_filter(z, power=power, cutoff=None)
    #        pSpec = model.predictor(zSpec)
    #    nCorrectFilter[i] += torch.sum(torch.argmax(pSpec.detach(), dim=1) == truthTens).cpu().numpy()

    if batchI + 1 >= epochBatches:
        break

noFilterAcc = nCorrectNoFilter / nTotal
print(noFilterAcc)

#filterAcc = [item / nTotal for item in nCorrectFilter]
#bestFilterAcc = max(filterAcc)
#bestFilterPow = powerList[filterAcc.index(bestFilterAcc)]

#print(noFilterAcc, bestFilterAcc, bestFilterPow)
#print(filterAcc)
#plt.plot(powerList, filterAcc)
#plt.show()
