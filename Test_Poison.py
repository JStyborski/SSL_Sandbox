import matplotlib.pyplot as plt
import numpy as np

import torch
from torch import nn
import torchvision.datasets as datasets

import SSL_Dataset
import SSL_Transforms
import SSL_Model

###############
# User Inputs #
###############

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
dataRoot = r'D:/CIFAR-10/Poisoned/OPS'
dataLabels = True

# Model data
ptFile = r'Saved_Models/SimSiam_CIFAR10-OPS_pt0400.pth.tar'
ftFile = r'Saved_Models/SimSiam_CIFAR10-OPS_pt0400_OPS_ft0100.pth.tar'
freezeEnc = True
cropSize = 28
nClasses = 10

batchSize = 1024 # 256 for CIFAR, 4096 for IN1k
epochBatches = 10

##############
# Data Setup #
##############

if dataLabels:
    dataDataset = datasets.ImageFolder(dataRoot, SSL_Transforms.MoCoV2Transform('test', cropSize))
else:
    dataDataset = SSL_Dataset.no_label_dataset(dataRoot, SSL_Transforms.MoCoV2Transform('test', cropSize))
dataLoader = torch.utils.data.DataLoader(dataDataset, batch_size=batchSize, shuffle=True)

###############
# Model Setup #
###############

# Load saved state
stateDict = torch.load(ptFile)

# If a stateDict key has "module" in (from running parallel), create a new dictionary with the right names
for key in list(stateDict['stateDict'].keys()):
    if key.startswith('module.'):
        stateDict['stateDict'][key[7:]] = stateDict['stateDict'][key]
        del stateDict['stateDict'][key]

# Create model and load model weights
model = SSL_Model.Base_Model(stateDict['encArch'], stateDict['cifarMod'], stateDict['encDim'],
                             stateDict['prjHidDim'], stateDict['prjOutDim'], stateDict['prdDim'], None, 0.3, 0.5, 0.0)
model.load_state_dict(stateDict['stateDict'], strict=True)

# Freeze all layers (though predictor will later be replaced and trainable)
if freezeEnc:
    for param in model.parameters(): param.requires_grad = False

# Replace the projector with identity and the predictor with linear classifier
model.projector = nn.Identity(stateDict['encDim'])
model.predictor = nn.Linear(stateDict['encDim'], nClasses)

# If using a finetune file, load it, freeze all
if ftFile is not None:
    model.load_state_dict(torch.load(ftFile), strict=True)
    for param in model.parameters(): param.requires_grad = False

model = model.to(device)

#############################
# Define Setup and Training #
#############################

def spectral_filter(z, power=1.0, cutoff=None):
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

model.eval()

nCorrectNoFilter = 0
nTotal = 0

powerList = [round(item, 2) for item in list(np.arange(-0.5, 0.52, 0.02))]
nCorrectFilter = [0] * len(powerList)

for batchI, batch in enumerate(dataLoader):

    # CenterCrop raw data
    augTens = batch[0].to(device)
    truthTens = batch[1].to(device)

    # Run augmented data through SimSiam with linear classifier
    p, z, r, _ = model(augTens)

    # Keep running sum of accuracy
    nCorrectNoFilter += torch.sum(torch.argmax(p.detach(), dim=1) == truthTens).cpu().numpy()
    nTotal += batch[0].size(0)

    # Spectral filter applied to z
    for i, power in enumerate(powerList):
        with torch.no_grad():
            zSpec = spectral_filter(z, power=power, cutoff=None)
            pSpec = model.predictor(zSpec)

        nCorrectFilter[i] += torch.sum(torch.argmax(pSpec.detach(), dim=1) == truthTens).cpu().numpy()

    if batchI + 1 >= epochBatches:
        break

noFilterAcc = nCorrectNoFilter / nTotal
filterAcc = [item / nTotal for item in nCorrectFilter]
bestFilterAcc = max(filterAcc)
bestFilterPow = powerList[filterAcc.index(bestFilterAcc)]

print(noFilterAcc, bestFilterAcc, bestFilterPow)
print(filterAcc)

plt.plot(powerList, filterAcc)
plt.show()
