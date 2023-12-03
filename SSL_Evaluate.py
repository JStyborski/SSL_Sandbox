import os
import numpy as np
import torch
from torch import nn
import torchvision.datasets as datasets

import SSL_Model
import Utils.Custom_Transforms as CT
import Utils.Custom_Probes as CP
from Adversarial import FGSM_PGD, Local_Lip

###############
# User Inputs #
###############

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
trainRoot = r'D:/ImageNet-100/train'
testRoot = r'D:/ImageNet-100/val'

# Find pretrained models
dirName = 'Trained_Models'
fileList = os.listdir(dirName)
ptList = sorted([dirName + '/' + stateFile for stateFile in fileList
                 if ('_pt_' in stateFile and '_lp_' not in stateFile and '_ft_' not in stateFile)])
#ptList = [dirName + '/LSP_pt_0200.pth.tar']
ftType = 'lp' # 'lp' or 'ft'

# Data parameters
cropSize = 224

# Nominal batch size for evaluation
batchSize = 512

# Representation eigenvalues parameters
repBankBatches = 20

# KNN parameters
knnBankBatches = repBankBatches
knnTestBatches = 5
k = 20

evalFinetune = True

# LinCls parameters
trainAccBatches = 1e6
testAccBatches = 1e6

# Adversarial parameters
atkSamples = 1024
atkBatchSize = 32
advAlpha = 0.6/255
advEps = 4/255
advRestarts = 5
advSteps = 10
randInit = True
xMin = None # 0.
xMax = None # 1.

##############
# Data Setup #
##############

# Create datasets and dataloaders
trainDataset = datasets.ImageFolder(trainRoot, CT.t_finetune(cropSize))
testDataset = datasets.ImageFolder(testRoot, CT.t_test(cropSize))

knnTrainLoader = torch.utils.data.DataLoader(trainDataset, batch_size=batchSize, shuffle=True)
knnTestLoader = torch.utils.data.DataLoader(testDataset, batch_size=batchSize, shuffle=True)
linTrainLoader = torch.utils.data.DataLoader(trainDataset, batch_size=batchSize, shuffle=True)
linTestLoader = torch.utils.data.DataLoader(testDataset, batch_size=batchSize, shuffle=True)
atkLoader = torch.utils.data.DataLoader(testDataset, batch_size=atkSamples, shuffle=True)

nClasses = len(trainDataset.classes)

#############################
# Define Setup and Training #
#############################

def make_rep_bank(loader, batchSize, model, encDim, nBankBatches):

    repBank = torch.zeros(nBankBatches * batchSize, encDim)
    labelBank = torch.zeros(nBankBatches * batchSize)

    for batchI, batch in enumerate(loader):

        # Augment data and get labels
        augTens = batch[0].to(device)
        truthTens = batch[1].to(device)

        # Get input encodings and write encodings + labels to bank
        _, _, r, _ = model(augTens)
        repBank[batchI * batchSize:(batchI + 1) * batchSize, :] = r.detach()
        labelBank[batchI * batchSize:(batchI + 1) * batchSize] = truthTens

        if batchI + 1 >= nBankBatches:
            break

    return repBank, labelBank


def calculate_smoothness(loader, model):

    # Get new batch and reset model grads
    batch = next(iter(loader))
    augTens = batch[0].to(device)

    # Reset model gradients before adv attack
    model.zero_grad()

    # Calculate local smoothness
    avgRepLolip, _ = Local_Lip.maximize_local_lip(model, augTens, 1/255, 8/255, 1, np.inf, atkBatchSize, 5, 10, outIdx=2)

    return avgRepLolip


def knn_vote(loader, knnBank, labelBank, k, knnTestBatches):

    simFcn = nn.CosineSimilarity(dim=1)

    accCount = 0
    totalCount = 0
    for batchI, batch in enumerate(loader):

        # Get data and labels
        augTens = batch[0].to(device)
        truthTens = batch[1].to(device).detach().cpu()

        # Run augmented data through model, get output before linear classifier
        _, _, r, _ = model(augTens)
        r = r.detach().cpu()

        batchSize = r.size(0)
        totalCount += batchSize

        # Loop through each sample in batch size and test KNN
        for i in range(batchSize):

            # Get the count of labels corresponding to the k nearest training vectors
            nearestKIdx = simFcn(r[i, :], knnBank).argsort()[-k:]
            uniqLabels, counts = np.unique(labelBank[nearestKIdx].numpy(), return_counts=True)

            # Check for ties between nearest labels:
            nModalLabels = len(np.where(counts == np.max(counts))[0])
            if nModalLabels == 1:
                modalLabel = uniqLabels[np.argsort(counts)[-1]]
            else:
                modalLabel = uniqLabels[np.argsort(counts)[-1 * (1 + np.random.choice(nModalLabels))]]

            # Check if KNN label is correct
            if modalLabel == truthTens[i].numpy():
                accCount += 1

        if batchI + 1 >= knnTestBatches:
            break

    return accCount / totalCount


def train_test_acc(loader, model, nBatches):

    nCorrect = 0
    nTotal = 0

    for batchI, batch in enumerate(loader):

        # CenterCrop raw data
        augTens = batch[0].to(device)
        truthTens = batch[1].to(device)

        # Run augmented data through SimSiam with linear classifier
        p, _, _, _ = model(augTens)

        # Keep running sum of loss
        nCorrect += torch.sum(torch.argmax(p.detach(), dim=1) == truthTens).cpu().numpy()
        nTotal += batch[0].size(0)

        if batchI + 1 >= nBatches:
            break

    clnAcc = nCorrect / nTotal

    return clnAcc


def adv_attack(loader, model, lossfn):

    # Get new batch
    batch = next(iter(loader))
    augTens = batch[0].to(device)
    truthTens = batch[1].to(device)

    # Reset model gradients before adv attack
    model.zero_grad()

    # Attack batch of images with FGSM or PGD and calculate accuracy
    avgAdvLoss, advTens = FGSM_PGD.sl_pgd(model, lossfn, augTens, truthTens, advAlpha, advEps, np.inf, advRestarts,
                                          advSteps, atkBatchSize, 0, False, randInit, None, xMin, xMax)
    advAcc = torch.sum(torch.argmax(model(advTens)[0].detach(), dim=1) == truthTens).cpu().numpy() / advTens.size(0)

    return advAcc


#######################
# Finetune Evaluation #
#######################

probes = CP.Finetune_Probes()

for stateFile in ptList:

    print('Evaluating ' + stateFile)

    # Load saved state
    ptStateDict = torch.load(stateFile, map_location='cuda:{}'.format(0))

    # Create model and load model weights
    model = SSL_Model.Base_Model(ptStateDict['encArch'], ptStateDict['rnCifarMod'], ptStateDict['vitPPFreeze'], ptStateDict['prjArch'],
                                   ptStateDict['prjHidDim'], ptStateDict['prjBotDim'], ptStateDict['prjOutDim'], ptStateDict['prdHidDim'],
                                   None, 0.3, 0.5, 0.0, True)

    # If a stateDict key has "module" in (from running parallel), create a new dictionary with the right names
    for key in list(ptStateDict['stateDict'].keys()):
        if key.startswith('module.'):
            ptStateDict['stateDict'][key[7:]] = ptStateDict['stateDict'][key]
            del ptStateDict['stateDict'][key]

    # Load weights
    model.load_state_dict(ptStateDict['stateDict'], strict=True)

    # Replace the projector with identity and the predictor with linear classifier
    encDim = model.encoder.inplanes if 'resnet' in ptStateDict['encArch'] else model.encoder.num_features
    model.projector = nn.Identity()
    model.predictor = nn.Linear(encDim, nClasses)

    # Freeze model parameters (mostly to reduce computation - grads for Lolip and AdvAtk can still propagate through)
    for param in model.parameters(): param.requires_grad = False

    # Set model to eval for assessment
    model = model.to(device)
    model.eval()

    # Make a representation bank (useful for KNN and for probe metrics)
    repBank, labelBank = make_rep_bank(knnTrainLoader, batchSize, model, encDim, repBankBatches)

    # Evaluate representation smoothness
    avgRepLolip = calculate_smoothness(atkLoader, model)

    # Evaluate KNN accuracy of the model
    # Can reuse the representation bank already assembled
    if knnBankBatches == repBankBatches and ftType == 'lp':
        knnBank = repBank
        knnLabelBank = labelBank
    else:
        knnBank, knnLabelBank = make_rep_bank(knnTrainLoader, batchSize, model, encDim, knnBankBatches)
    knnAcc = knn_vote(knnTestLoader, knnBank, knnLabelBank, k, knnTestBatches)
    print('KNN Accuracy: {:0.4f}'.format(knnAcc))

    # Declare finetune file name and check if it already exists
    ptPrefix = (stateFile[:-8] + '_').split('/')[-1]
    ftPrefix = '_' + ftType + '_'
    ftFiles = sorted([dirName + '/' + file for file in fileList if (ptPrefix in file and ftPrefix in file)])
    if evalFinetune and len(ftFiles) > 0:

        for ftFile in ftFiles:

            print('Evaluating ' + ftFile)

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

            # Clean dataset accuracy
            clnTrainAcc = train_test_acc(linTrainLoader, model, trainAccBatches)
            print('Train Accuracy: {:0.4f}'.format(clnTrainAcc))
            clnTestAcc = train_test_acc(linTestLoader, model, testAccBatches)
            print('Test Accuracy: {:0.4f}'.format(clnTestAcc))

            # Evaluate adversarial accuracy
            advAcc = adv_attack(atkLoader, model, nn.NLLLoss(reduction='none'))
            print('Adv Accuracy: {:0.4f}'.format(advAcc))

            # Update probes for each ftFile
            probes.update_probes(ptStateDict['epoch'], repBank, avgRepLolip, knnAcc, clnTrainAcc, clnTestAcc, advAcc)

    else:
        clnTrainAcc = None
        clnTestAcc = None
        advAcc = None

        # Update probes even if no FT files were run
        probes.update_probes(ptStateDict['epoch'], repBank, avgRepLolip, knnAcc, clnTrainAcc, clnTestAcc, advAcc)

##################
# Postprocessing #
##################

#probes.plot_probes()

import csv
writer = csv.writer(open('Evaluate_Output.csv', 'w', newline=''))
writer.writerow(['R1 Eig'] + np.log(probes.repEigProbe.storeList[-1]).tolist())
writer.writerow(['R1 E-Rank'] + probes.repEigERankProbe.storeList)
writer.writerow(['R1 Local Lip'] + probes.repLolipProbe.storeList)
writer.writerow(['KNN Acc'] + probes.knnAccProbe.storeList)
writer.writerow(['Train Acc'] + probes.clnTrainAccProbe.storeList)
writer.writerow(['Clean Test Acc'] + probes.clnTestAccProbe.storeList)
writer.writerow(['Adv Acc'] + probes.advAccProbe.storeList)

