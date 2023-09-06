import time
import math
import numpy as np
import torch
import torchvision.datasets as datasets

import SSL_Transforms
import SSL_Model
import Custom_Probes

###############
# User Inputs #
###############

# SimSiam Settings: prdDim not None, momEncBeta = 0.0, nceBeta = 0.0, lossFn = Weighted_InfoNCE, applySG = True
# BYOL Settings: prdDim not None, momEncBeta > 0.0, nceBeta = 0.0, lossFn = Weighted_InfoNCE, applySG = True
# SimCLR Settings: prdDim = None, momEncBeta = 0.0, nceBeta > 0.0, lossFn = Weighted_InfoNCE, applySG = False
# DINO Settings: prdDim = None, momEncBeta > 0.0, lossFn = DINO_CrossEnt, applySG = True
# Barlow Twins Settings: prdDim = None, momEncBeta = 0.0, lossFn = BT_CrossCorr, applySG = False (symmetrizeLoss = True is redundant)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data parameters
trainRoot = r'D:\CIFAR-10\train'
cropSize = 28

# Model parameters
encArch = 'resnet18' # 'vit_b_16' # vit requires 224 input size
encDim = 512
prjDim = 2048
prdDim = int(prjDim / 4) # Set prdDim = None to not use a predictor (sets predictor as Identity)
#prdDim = None

# Momentum encoder settings
momEncBeta = 0.0 # Setting as 0 means no momentum and the momentum encoder is bypassed entirely to save compute

# Define loss function
# Setting nceBeta = 0.0 is equivalent to non-contrastive loss, downSamples=None uses full batch size for contrastive
lossFn = SSL_Model.Weighted_InfoNCE(nceBeta=0.0, nceBetaScheme=None, usePrd4CL=True, nceTau=0.1, downSamples=None,
                                    mecReg=False, mecED2=0.06, mecTay=2)
#lossFn = SSL_Model.MEC(ed2=0.06, taylorTerms=2)
# 2021 Tsai paper on HSIC-SSL recommends 1/d for off-diagonal covariance coefficient lambda
# lossForm = 'bt' for original BT loss, lossForm = 'hsic' for HSIC-SSL loss
#lossFn = SSS_Model.BT_CrossCorr(btLam=1/prjDim, lossForm='bt')
# Note DINO seems unstable, requiring sharp teacher, low LR, and warmup
# I got moderately stable training for momEncBeta=0.99, centerMom=0.9, studentTau=0.1, teacherTau=0.04, initLR=0.01
#lossFn = SSS_Model.DINO_CrossEnt(centerInit='zero', centerMom=0.9, studentTau=0.1, teacherTau=0.04)
applySG = True
symmetrizeLoss = True

# Training parameters
nEpochs = 100
batchSize = 512
epochBatches = 10000000
momentum = 0.9
weightDecay = 1e-5
initLR = 0.5
initLR = initLR * batchSize / 256
decayEncLR = True
decayPrdLR = False
lrWarmupEp = 10 # Set as 0 for no warmup

########################
# Data and Model Setup #
########################

# Create dataset and dataloader from fake training data
trainDataset = datasets.ImageFolder(trainRoot, SSL_Transforms.TwoTimesTransform(SSL_Transforms.MoCoV2Transform('train', cropSize)))
trainDataLoader = torch.utils.data.DataLoader(trainDataset, batch_size=batchSize, shuffle=True, drop_last=True)

# Instantiate model and (custom) optimizer
model = SSL_Model.Base_Model(encArch, encDim, prjDim, prdDim, momEncBeta).to(device)
optimParams = [{'params': model.encoder.parameters(), 'decayLR': decayEncLR},
               {'params': model.projector.parameters(), 'decayLR': decayEncLR},
               {'params': model.predictor.parameters(), 'decayLR': decayPrdLR}]
optimizer = torch.optim.SGD(params=optimParams, lr=initLR, momentum=momentum, weight_decay=weightDecay)

############
# Training #
############

# Initialize probes for training metrics, set model to train, and start timer
probes = Custom_Probes.Pretrain_Probes()
model.train()

# Save untrained model for later LinCls
torch.save({'encArch': encArch, 'encDim': encDim, 'prjDim': prjDim, 'prdDim': prdDim, 'momEncBeta': momEncBeta,
            'stateDict': model.state_dict()}, 'Trained_Models/pretrain_{:04d}.pth.tar'.format(0))

timeStart = time.time()

for epoch in range(1, nEpochs + 1):

    # If desired, adjust learning rate for the current epoch using cosine annealing
    if decayEncLR or decayPrdLR:
        curLR = initLR * 0.5 * (1. + math.cos(math.pi * (epoch - 1) / nEpochs))
        for param_group in optimizer.param_groups:
            if param_group['decayLR']:
                param_group['lr'] = curLR

    # If using LR warmup, the warmup LR is linear, applies to all layers, and overwrites any decay
    if epoch < lrWarmupEp:
        curLR = epoch / lrWarmupEp * initLR
        for param_group in optimizer.param_groups:
            param_group['lr'] = curLR

    if type(lossFn) is SSL_Model.Weighted_InfoNCE:
        if lossFn.nceBetaScheme is not None:
            lossFn.update_nceBeta(epoch, nEpochs)

    # Reset sum of losses for each epoch
    sumLoss = 0.0

    for batchI, batch in enumerate(trainDataLoader):

        aug1Tens = batch[0][0].to(device)
        aug2Tens = batch[0][1].to(device)

        # Run each augmented batch through encoder, projector, predictor, and momentum encoder/projector
        p1, z1, r1, mz1 = model(aug1Tens)
        p2, z2, r2, mz2 = model(aug2Tens)

        # Apply stop-gradient
        if applySG:
            mz1 = mz1.detach()
            mz2 = mz2.detach()

        # Calculate loss
        if symmetrizeLoss:
            lossVal = 0.5 * (lossFn.forward(p1, z1, mz2) + lossFn.forward(p2, z2, mz1))
        else:
            lossVal = lossFn.forward(p1, z1, mz2)

        # Backpropagate
        optimizer.zero_grad()
        lossVal.backward()
        optimizer.step()

        # Update momentum encoder
        if momEncBeta > 0.0:
            model.update_momentum_network()

        # Update DINO loss center
        if type(lossFn) is SSL_Model.DINO_CrossEnt:
            lossFn.update_center(torch.cat((mz1, mz2), dim=0)) if symmetrizeLoss else lossFn.update_center(mz2)

        # Keep running sum of loss
        sumLoss += lossVal.detach()

        if batchI + 1 >= epochBatches:
            break

    print('Epoch: {} / {} | Elapsed Time: {:0.2f} | Avg Loss: {:0.4f} | encLR: {:0.4f} | prdLR {:0.4f}'
          .format(epoch, nEpochs, time.time() - timeStart, sumLoss / (batchI + 1),
                  optimizer.param_groups[0]['lr'], optimizer.param_groups[2]['lr']))

    # Update probes
    model.eval()
    # Note that p1, z1, and mz2 are L2 normd, as SimSiam, BYOL, InfoNCE, and MEC use L2 normalized encodings
    # This is taken care of in loss functions, but I have to do it explicitly here
    # This probe update is inaccurate for softmax-normalized encs (DINO, SwAV) or batch normalized encs (Barlow Twins)
    probes.update_probes(epoch, model, lossVal.detach(),
                         (p1 / torch.linalg.vector_norm(p1, dim=-1, keepdim=True)).detach(),
                         (z1 / torch.linalg.vector_norm(z1, dim=-1, keepdim=True)).detach(), r1.detach(), r2.detach(),
                         (mz2 / torch.linalg.vector_norm(mz2, dim=-1, keepdim=True)).detach(), aug1Tens.detach())
    model.train()
    optimizer.zero_grad()

    if epoch == 1 or epoch == 5 or (epoch <= 200 and epoch % 10 == 0) or epoch % 100 == 0:
        torch.save({'encArch': encArch, 'encDim': encDim, 'prjDim': prjDim, 'prdDim': prdDim, 'momEncBeta': momEncBeta,
                    'stateDict': model.state_dict()}, 'Trained_Models/pretrain_{:04d}.pth.tar'.format(epoch))

##################
# Postprocessing #
##################

#probes.plot_probes()

epochList = list(range(1, nEpochs + 1))

print([probes.lossProbe.storeList[epIdx - 1] for epIdx in epochList])
print([probes.r1r2AugSimProbe.storeList[epIdx - 1] for epIdx in epochList])
print([probes.r1AugSimProbe.storeList[epIdx - 1] for epIdx in epochList])
print([probes.p1EntropyProbe.storeList[epIdx - 1] for epIdx in epochList])
print([probes.mz2EntropyProbe.storeList[epIdx - 1] for epIdx in epochList])
print([probes.mz2p1KLDivProbe.storeList[epIdx - 1] for epIdx in epochList])
print([probes.p1EigERankProbe.storeList[epIdx - 1] for epIdx in epochList])
print([probes.z1EigERankProbe.storeList[epIdx - 1] for epIdx in epochList])
#print([probes.r1EigERankProbe.storeList[epIdx - 1] for epIdx in epochList])
print([probes.mz2EigERankProbe.storeList[epIdx - 1] for epIdx in epochList])
print([probes.p1EigAUCProbe.storeList[epIdx - 1] for epIdx in epochList])
print([probes.z1EigAUCProbe.storeList[epIdx - 1] for epIdx in epochList])
#print([probes.r1EigAUCProbe.storeList[epIdx - 1] for epIdx in epochList])
print([probes.mz2EigAUCProbe.storeList[epIdx - 1] for epIdx in epochList])
print([probes.p1z1EigAlignProbe.storeList[epIdx - 1] for epIdx in epochList])
print([probes.p1mz2EigAlignProbe.storeList[epIdx - 1] for epIdx in epochList])
#print([probes.p1LolipProbe.storeList[epIdx - 1] for epIdx in epochList])
#print([probes.z1LolipProbe.storeList[epIdx - 1] for epIdx in epochList])
print([probes.r1LolipProbe.storeList[epIdx - 1] for epIdx in epochList])

print(np.log(probes.p1EigProbe.storeList[-1]).tolist())
print(np.log(probes.mz2EigProbe.storeList[-1]).tolist())
