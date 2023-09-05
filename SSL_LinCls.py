import time
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
trainRoot = r'D:\CIFAR-10\train'
testRoot = r'D:\CIFAR-10\test'

# Find pretrained models
ptDirName = 'Trained_Models'
stateList = os.listdir(ptDirName)
stateList = sorted([ptDirName + '/' + stateFile for stateFile in stateList if 'pretrain' in stateFile])

# Data parameters
cropSize = 28
nClasses = 10

# Training parameters
nEpochs = 100
batchSize = 512
epochBatches = 100000
momentum = 0.9
weightDecay = 1e-4
initLR = 0.1
lr = initLR * batchSize / 256
freezeEnc = True

# KNN parameters
knnBankBatches = 20
k = 20

# Attack parameters
atkSamples = 512
atkBatchSize = 32

##############
# Data Setup #
##############

# Create datasets and dataloaders
trainDataset = datasets.ImageFolder(trainRoot, SSL_Transforms.MoCoV2Transform('test', cropSize))
trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size=batchSize, shuffle=True)
testDataset = datasets.ImageFolder(testRoot, SSL_Transforms.MoCoV2Transform('test', cropSize))
testLoader = torch.utils.data.DataLoader(testDataset, batch_size=batchSize, shuffle=True)
atkDataset = datasets.ImageFolder(testRoot, SSL_Transforms.MoCoV2Transform('test', cropSize))
atkLoader = torch.utils.data.DataLoader(atkDataset, batch_size=atkSamples, shuffle=True)

#############################
# Define Setup and Training #
#############################


def model_setup(stateDict, freezeEnc):
    """
    Create SimSiam model using the input stateDict and set up linear classifier
    :param stateDict: [dict] [3] - Contains cropSize, outDim, and SimSiam pretrain state dict
    :param freezeEnc: [Bool] [1] - Whether to freeze the encoder during finetuning
    :return model: [Pytorch model] [1] - Pretrained model with untrained linear classifier
    """

    # Create model and load model weights
    model = SSL_Model.SmallSimSiam(stateDict['encArch'], stateDict['encDim'], stateDict['prjDim'], stateDict['prdDim'])
    model.load_state_dict(stateDict['stateDict'], strict=False)

    # Freeze all layers (though predictor will later be replaced and trainable)
    if freezeEnc:
        for param in model.parameters(): param.requires_grad = False

    # Replace the projector with identity and the predictor with linear classifier
    model.projector = nn.Identity(stateDict['encDim'])
    model.predictor = nn.Linear(stateDict['encDim'], nClasses)
    model = model.to(device)

    return model


def make_knn_bank(loader, batchSize, model, encDim, knnBankBatches):

    knnBank = torch.zeros(knnBankBatches * batchSize, encDim)
    labelBank = torch.zeros(knnBankBatches * batchSize)

    for batchI, batch in enumerate(loader):

        # Augment data and get labels
        augTens = batch[0].to(device)
        truthTens = batch[1].to(device)

        # Get input encodings and write encodings + labels to bank
        _, z, _, _ = model(augTens)
        knnBank[batchI * batchSize:(batchI + 1) * batchSize, :] = z.detach()
        labelBank[batchI * batchSize:(batchI + 1) * batchSize] = truthTens

        if batchI + 1 >= knnBankBatches:
            break

    return knnBank, labelBank


def knn_vote(batch, labels, knnBank, labelBank, k):

    batchSize = batch.size(0)
    simFcn = nn.CosineSimilarity(dim=1)

    accCount = 0
    for i in range(batchSize):

        # Get the count of labels corresponding to the k nearest training vectors
        nearestKIdx = simFcn(batch[i, :], knnBank).argsort()[-k:]
        uniqLabels, counts = np.unique(labelBank[nearestKIdx].numpy(), return_counts=True)

        # Check for ties between nearest labels:
        nModalLabels = len(np.where(counts == np.max(counts))[0])
        if nModalLabels == 1:
            modalLabel = uniqLabels[np.argsort(counts)[-1]]
        else:
            modalLabel = uniqLabels[np.argsort(counts)[-1 * (1 + np.random.choice(nModalLabels))]]

        # Check if KNN label is correct
        if modalLabel == labels[i].numpy():
            accCount += 1

    return accCount / batchSize


def PGD_attack(loader, model, lossfn):

    batch = next(iter(loader))

    # Get truth class and raw data for batch
    augTens = batch[0].to(device)
    truthTens = batch[1].to(device)

    # Attack batch of images with FGSM or PGD and calculate accuracy
    avgAdvLoss, perturbTens, advTens = FGSM_PGD.pgd(model, lossfn, augTens, truthTens, 0.005, 0.01, np.inf,
                                                    atkBatchSize, 1, 1, 0, False, False)
    advAcc = torch.sum(torch.argmax(model(advTens)[1].detach(), dim=1) == truthTens).cpu().numpy() / advTens.size(0)

    return advAcc, avgAdvLoss, perturbTens, advTens


##############
# Finetuning #
##############

probes = Custom_Probes.Finetune_Probes()

lastAccList = []
for stateFile in stateList:

    print('Finetuning ' + stateFile)

    # Load saved state
    stateDict = torch.load(stateFile)

    # Set up model
    model = model_setup(stateDict, freezeEnc)

    # Instantiate loss criterion, and optimizer
    crossEnt = nn.CrossEntropyLoss()
    optim = torch.optim.SGD(params=model.parameters(), lr=initLR, momentum=momentum, weight_decay=weightDecay)

    # Set model to train, and start timer
    model.train()
    timeStart = time.time()

    # Finetune model across epochs
    for epoch in range(1, nEpochs + 1):

        # Reset sum of losses for each epoch
        sumLoss = 0.0
        nCorrect = 0
        nTotal = 0

        for batchI, batch in enumerate(testLoader):

            # CenterCrop raw data
            augTens = batch[0].to(device)
            truthTens = batch[1].to(device)

            # Run augmented data through SimSiam with linear classifier
            p, z, _, _ = model(augTens)

            # Calculate cross-entropy loss
            loss = crossEnt(p, truthTens)

            # Backpropagate
            optim.zero_grad()
            loss.backward()
            optim.step()

            # Keep running sum of loss and accuracy
            sumLoss += loss.detach().cpu().numpy()
            nCorrect += torch.sum(torch.argmax(p.detach(), dim=1) == truthTens).cpu().numpy()
            nTotal += batch[0].size(0)

            if batchI + 1 >= epochBatches:
                break

        lossVal = sumLoss / (batchI + 1)
        clnAcc = nCorrect / nTotal

        if epoch == 1 or epoch % 10 == 0:
            print('Epoch: {} / {} | Elapsed Time: {:0.2f} | Avg Loss: {:0.4f} | Avg Acc: {:0.4f}'
                  .format(epoch, nEpochs, time.time() - timeStart, lossVal, clnAcc))

    lastAccList.append(clnAcc)

    # Set model to eval for evaluating accuracies outside training
    model.eval()

    # Evaluate KNN accuracy of the model (uses encoder output so is unaffected by linear probe training)
    knnBank, labelBank = make_knn_bank(trainLoader, batchSize, model, stateDict['encDim'], knnBankBatches)
    knnAcc = knn_vote(z.detach().cpu(), truthTens.detach().cpu(), knnBank, labelBank, k)
    print('KNN Accuracy: {:0.4f}'.format(knnAcc))

    # Evaluate linear probe adversarial accuracy
    advAcc, _, perturbTens, _ = PGD_attack(atkLoader, model, nn.CrossEntropyLoss(reduction='none'))
    print('Adv Accuracy: {:0.4f}'.format(advAcc))

    # Update probes and reset model/optimizer
    probes.update_probes(int(re.findall(r'\d+', stateFile)[0]), clnAcc, knnAcc, advAcc, perturbTens.detach().flatten(1, -1))
    model.train()
    optim.zero_grad()

    # Save out finetune model
    finetuneName = ptDirName + '/finetune_' + re.findall(r'\d+', stateFile)[0] + '.pth.tar'
    torch.save(model.state_dict(), finetuneName)
    print('Saved Model {}'.format(finetuneName))


##################
# Postprocessing #
##################

#probes.plot_probes()

print(probes.clnAccProbe.storeList)
print(probes.knnAccProbe.storeList)
print(probes.advAccProbe.storeList)
