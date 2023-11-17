import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.manifold import MDS, TSNE

import torch
from torch import nn
import torchvision.datasets as datasets

import Utils.Custom_Transforms as CT
import SSLAE_Model

###############
# User Inputs #
###############

device = torch.device('cuda')

# Data information
data1Root = r'D:/ImageNet-10/train'
rep1Idx = 2
rep1Dim = 512
label1 = 'Clean'
data2Root = r'D:/ImageNet-10/Poisoned/CUDA_10/train'
rep2Idx = 2
rep2Dim = 512
label2 = 'CUDA'

# Model data
ptFile = r'Saved_Models/CIFAR-10//CUDA_pt_0400.pth.tar'
ftFile = r'Saved_Models/ImageNet-100/SSL_He2023_400ep/CUDA/CUDA_pt_0400_CUDA_lp_0100.pth.tar'
useFinetune = True

cropSize = 224

batchSize = 256
bankBatches = 5

useClsList = list(range(4))

##################
# User Functions #
##################

def make_rep_bank(model, loader, batchSize, nBankBatches, repIdx, repDim, useClsList):

    inpBank = torch.zeros(nBankBatches * batchSize, 3 * cropSize * cropSize)
    repBank = torch.zeros(nBankBatches * batchSize, repDim)
    labelBank = torch.zeros(nBankBatches * batchSize)

    for batchI, batch in enumerate(loader):

        # Augment data and get labels
        augTens = batch[0].to(device)
        truthTens = batch[1].to(device)

        # Get input encodings and write encodings + labels to bank
        inpBank[batchI * batchSize:(batchI + 1) * batchSize, :] = torch.flatten(augTens, start_dim=1)
        r = model(augTens)[repIdx]
        repBank[batchI * batchSize:(batchI + 1) * batchSize, :] = r.detach()
        labelBank[batchI * batchSize:(batchI + 1) * batchSize] = truthTens

        if batchI + 1 >= nBankBatches:
            break

    if useClsList is not None:
        keepBool = [label in useClsList for label in labelBank]
        inpBank = inpBank[keepBool, :]
        repBank = repBank[keepBool, :]
        labelBank = labelBank[keepBool]

    return inpBank, repBank, labelBank

##############
# Data Setup #
##############

data1Dataset = datasets.ImageFolder(data1Root, CT.t_test(cropSize))
data2Dataset = datasets.ImageFolder(data2Root, CT.t_test(cropSize))

nClasses = len(data1Dataset.classes)

data1Loader = torch.utils.data.DataLoader(data1Dataset, batch_size=batchSize, shuffle=True)
data2Loader = torch.utils.data.DataLoader(data2Dataset, batch_size=batchSize, shuffle=True)

###############
# Model Setup #
###############

# Load saved state
ptStateDict = torch.load(ptFile, map_location='cuda:{}'.format(0))

# Create model and load model weights
model = SSLAE_Model.Base_Model(ptStateDict['encArch'], ptStateDict['cifarMod'], ptStateDict['encDim'],
                             ptStateDict['prjHidDim'], ptStateDict['prjOutDim'], ptStateDict['prdDim'], None, 0.3, 0.5, 0.0, True, None)

# If a stateDict key has "module" in (from running parallel), create a new dictionary with the right names
for key in list(ptStateDict['stateDict'].keys()):
    if key.startswith('module.'):
        ptStateDict['stateDict'][key[7:]] = ptStateDict['stateDict'][key]
        del ptStateDict['stateDict'][key]

# Load weights
model.load_state_dict(ptStateDict['stateDict'], strict=True)

if useFinetune:

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

# Freeze model parameters (mostly to reduce computation - grads for Lolip and AdvAtk can still propagate through)
for param in model.parameters(): param.requires_grad = False

# Set model to eval for assessment
model = model.to(device)
model.eval()

#############################
# Define Setup and Training #
#############################

inpBank1, repBank1, labelBank1 = make_rep_bank(model, data1Loader, batchSize, bankBatches, rep1Idx, rep1Dim, useClsList)
inpBank2, repBank2, labelBank2 = make_rep_bank(model, data1Loader, batchSize, bankBatches, rep2Idx, rep2Dim, useClsList)
inpBank1 = inpBank1.cpu().numpy()
repBank1 = repBank1.cpu().numpy()
labelBank1 = labelBank1.cpu().numpy()
inpBank2 = inpBank2.cpu().numpy()
repBank2 = repBank2.cpu().numpy()
labelBank2 = labelBank2.cpu().numpy()

tsne = TSNE(n_components=2)
inpTSNE = tsne.fit_transform(np.concatenate((inpBank1, inpBank2), axis=0))
inpTSNE1 = inpTSNE[:len(labelBank1)]
inpTSNE2 = inpTSNE[len(labelBank1):]
repTSNE = tsne.fit_transform(np.concatenate((repBank1, repBank2), axis=0))
repTSNE1 = repTSNE[:len(labelBank1)]
repTSNE2 = repTSNE[len(labelBank1):]

colorList = ['c', 'm', 'y', 'b', 'g', 'r',  'k']
for clsI, cls in enumerate(useClsList):

    clsBool1 = labelBank1 == cls
    clsBool2 = labelBank2 == cls

    plt.scatter(inpTSNE1[clsBool1, 0], inpTSNE1[clsBool1, 1], s=10, marker='o', facecolors='none', edgecolors=colorList[clsI])
    plt.scatter(inpTSNE2[clsBool2, 0], inpTSNE2[clsBool2, 1], s=10, c=colorList[clsI], marker='x')

plt.title('Input Space Classes | o = {}, x = {}'.format(label1, label2))
plt.show()


for clsI, cls in enumerate(useClsList):

    clsBool1 = labelBank1 == cls
    clsBool2 = labelBank2 == cls

    plt.scatter(repTSNE1[clsBool1, 0], repTSNE1[clsBool1, 1], s=10, marker='o', facecolors='none', edgecolors=colorList[clsI])
    plt.scatter(repTSNE2[clsBool2, 0], repTSNE2[clsBool2, 1], s=10, c=colorList[clsI], marker='x')

plt.title('Rep Space Classes | o = {}, x = {}'.format(label1, label2))
plt.show()

