import time
import math
import os
import re
import numpy as np
import torch
from torch import nn
import torchvision.datasets as datasets

import SSL_Transforms
import SSL_Model
from Adversarial import FGSM_PGD
import Custom_Probes

###############
# User Inputs #
###############

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
trainRoot = r'D:/CIFAR-10/Poisoned/LSP'
testRoot = r'D:/CIFAR-10/test'

# Find pretrained models
ptDirName = 'Trained_Models'
stateList = os.listdir(ptDirName)
stateList = sorted([ptDirName + '/' + stateFile for stateFile in stateList if 'pt' in stateFile])
freezeEnc = False

# Data parameters
cropSize = 28
nClasses = 10

# LinCls parameters
runLinCls = True
nEpochs = 100
batchSize = 512 # 256 for CIFAR, 4096 for IN1k
linClsTrainBatches = 1e6
linClsTestBatches = 10
momentum = 0.9
weightDecay = 0
initLR = 30 # 30 for CIFAR, 0.1 for IN1k
initLR = initLR * batchSize / 256
decayLR = 'stepdn' # None,'stepdn' (CIFAR), 'cosdn' (IN1k)
decaySteps = [60, 80] # Epochs at which to cut LR - only used with 'stepdn' decay

# KNN parameters
runKNNCls = True
knnBankBatches = 20
knnTestBatches = 5
k = 20

# Adversarial parameters
runAdvCls = True
atkSamples = 1024
atkBatchSize = 32
advAlpha = 1/255
advEps = 5/255
advRestarts = 5
advSteps = 10
randInit = True

# Representation eigenvalues parameters
repBankBatches = knnBankBatches

##############
# Data Setup #
##############

# Create datasets and dataloaders
trainDataset = datasets.ImageFolder(trainRoot, SSL_Transforms.MoCoV2Transform('lincls_train', cropSize))
testDataset = datasets.ImageFolder(testRoot, SSL_Transforms.MoCoV2Transform('test', cropSize))

linTrainLoader = torch.utils.data.DataLoader(trainDataset, batch_size=batchSize, shuffle=True)
linTestLoader = torch.utils.data.DataLoader(testDataset, batch_size=batchSize, shuffle=True)
knnTrainLoader = torch.utils.data.DataLoader(trainDataset, batch_size=batchSize, shuffle=True)
knnTestLoader = torch.utils.data.DataLoader(testDataset, batch_size=batchSize, shuffle=True)
atkLoader = torch.utils.data.DataLoader(testDataset, batch_size=atkSamples, shuffle=True)

#############################
# Define Setup and Training #
#############################


def lincls_train(loader, model, lossFn, optimizer, linClsTrainBatches):

    # Reset sum of losses for each epoch
    sumLoss = 0.0
    nTrainCorrect = 0
    nTrainTotal = 0

    for batchI, batch in enumerate(loader):

        # CenterCrop raw data
        augTens = batch[0].to(device)
        truthTens = batch[1].to(device)

        # Run augmented data through SimSiam with linear classifier
        p, _, _, _ = model(augTens)

        # Calculate loss
        loss = lossFn(p, truthTens)

        # Backpropagate
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Keep running sum of loss
        sumLoss += loss.detach().cpu().numpy()
        nTrainCorrect += torch.sum(torch.argmax(p.detach(), dim=1) == truthTens).cpu().numpy()
        nTrainTotal += batch[0].size(0)

        if batchI + 1 >= linClsTrainBatches:
            break

    lossVal = sumLoss / (batchI + 1)
    clnTrainAcc = nTrainCorrect / nTrainTotal

    return lossVal, clnTrainAcc


def lincls_test(loader, model, linClsTestBatches):

    nTestCorrect = 0
    nTestTotal = 0

    for batchI, batch in enumerate(loader):

        # CenterCrop raw data
        augTens = batch[0].to(device)
        truthTens = batch[1].to(device)

        # Run augmented data through SimSiam with linear classifier
        p, _, _, _ = model(augTens)

        # Keep running sum of loss
        nTestCorrect += torch.sum(torch.argmax(p.detach(), dim=1) == truthTens).cpu().numpy()
        nTestTotal += batch[0].size(0)

        if batchI + 1 >= linClsTestBatches:
            break

    clnTrainAcc = nTestCorrect / nTestTotal

    return lossVal, clnTrainAcc


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


def PGD_attack(loader, model, lossfn):

    batch = next(iter(loader))

    # Get truth class and raw data for batch
    augTens = batch[0].to(device)
    truthTens = batch[1].to(device)

    # Attack batch of images with FGSM or PGD and calculate accuracy
    avgAdvLoss, perturbTens, advTens = FGSM_PGD.pgd(model, lossfn, augTens, truthTens, advAlpha, advEps, np.inf,
                                                    atkBatchSize, advRestarts, advSteps, 0, False, randInit)
    advAcc = torch.sum(torch.argmax(model(advTens)[0].detach(), dim=1) == truthTens).cpu().numpy() / advTens.size(0)

    return advAcc, avgAdvLoss, perturbTens, advTens


##############
# Finetuning #
##############

probes = Custom_Probes.Finetune_Probes()

for stateFile in stateList:

    print('Evaluating ' + stateFile)

    # Load saved state
    stateDict = torch.load(stateFile)

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

    model = model.to(device)

    if runLinCls:

        # Instantiate loss criterion, and optimizer
        crossEnt = nn.CrossEntropyLoss()
        optim = torch.optim.SGD(params=model.parameters(), lr=initLR, momentum=momentum, weight_decay=weightDecay)
        optim.zero_grad()

        # Start timer
        timeStart = time.time()

        # Finetune model - if loaded frozen finetune model, "train" for 1 epoch to get acc values
        for epoch in range(1, nEpochs + 1):

            # If desired, adjust learning rate for the current epoch
            if decayLR == 'stepdn':
                if epoch in decaySteps:
                    curLR = initLR / 10 ** (decaySteps.index(epoch) + 1)
                    for param_group in optim.param_groups:
                        param_group['lr'] = curLR
            elif decayLR == 'cosdn':
                curLR = initLR * 0.5 * (1. + math.cos(math.pi * (epoch - 1) / nEpochs))
                for param_group in optim.param_groups:
                    param_group['lr'] = curLR

            # Train lincls model for 1 epoch - if training linear probe, keep whole model frozen (no BN updates)
            model.eval() if freezeEnc else model.train()
            lossVal, clnTrainAcc = lincls_train(linTrainLoader, model, crossEnt, optim, linClsTrainBatches)

            # Test accuracy of model
            model.eval()
            with torch.no_grad():
                clnTestAcc = lincls_test(linTestLoader, model, linClsTestBatches)

            if epoch == 1 or epoch % 10 == 0:
                print('Epoch: {} / {} | Elapsed Time: {:0.2f} | Avg Loss: {:0.4f} | Avg Trn Acc: {:0.4f} | Avg Test Acc: {:0.4f}'
                      .format(epoch, nEpochs, time.time() - timeStart, lossVal, clnTrainAcc, clnTestAcc))

        # Save out finetune model
        finetuneName = ptDirName + '/finetune_' + re.findall(r'\d+', stateFile)[0] + '.pth.tar'
        #finetuneName = stateFile + 'finetune.pth.tar'
        torch.save(model.state_dict(), finetuneName)
        print('Saved Model {}'.format(finetuneName))

    else:
        clnTrainAcc = None
        clnTestAcc = None

    # Set model to eval for evaluating accuracies outside training
    model.eval()

    # Make a representation bank (useful for KNN and for probe metrics)
    repBank, labelBank = make_rep_bank(knnTrainLoader, batchSize, model, stateDict['encDim'], repBankBatches)

    # Evaluate KNN accuracy of the model
    if runKNNCls:
        # Can reuse the representation bank already assembled
        if knnBankBatches == repBankBatches:
            knnBank = repBank
            knnLabelBank = labelBank
        else:
            knnBank, knnLabelBank = make_rep_bank(knnTrainLoader, batchSize, model, stateDict['encDim'], knnBankBatches)
        knnAcc = knn_vote(knnTestLoader, knnBank, knnLabelBank, k, knnTestBatches)
        print('KNN Accuracy: {:0.4f}'.format(knnAcc))
    else:
        knnAcc = None

    # Evaluate linear probe adversarial accuracy
    if runAdvCls:
        #advAcc, _, perturbTens, _ = PGD_attack(atkLoader, model, nn.CrossEntropyLoss(reduction='none'))
        advAcc, _, _, _ = PGD_attack(atkLoader, model, nn.NLLLoss(reduction='none'))
        print('Adv Accuracy: {:0.4f}'.format(advAcc))
    else:
        advAcc = None

    # Update probes
    probes.update_probes(int(re.findall(r'\d+', stateFile)[0]), clnTrainAcc, clnTestAcc, knnAcc, advAcc, repBank)


##################
# Postprocessing #
##################

#probes.plot_probes()

print(np.log(probes.repEigProbe.storeList[-1]).tolist())
print(probes.repEigERankProbe.storeList)
print(probes.clnTrainAccProbe.storeList)
print(probes.clnTestAccProbe.storeList)
print(probes.knnAccProbe.storeList)
print(probes.advAccProbe.storeList)
