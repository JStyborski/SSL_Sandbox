import time
import torch
import torchvision.datasets as datasets

import SSL_Transforms
import SSL_Model
from Model_Analysis import Compute_Analysis_Utils as CAU

##
#
##


class Checkpoint_Metrics:
    def __init__(self, opt='usage'):
        self.opt = opt
        self.gpuList = []
        self.cpuList = []
        self.timeList = []

    def update(self):
        if self.opt == 'usage':
            self.gpuList.append(CAU.gpu_mem_usage())
            self.cpuList.append(CAU.cpu_mem_usage()[0])
        elif self.opt == 'time':
            self.timeList.append(time.time())

cpm = Checkpoint_Metrics('usage')
cpm.update()

###############
# User Inputs #
###############

# Load data
root = r'D:\CIFAR-10\train'

# Find pretrained models
stateFile = r'Trained_Models/pretrain_0000.pth.tar'

# Data parameters
cropSize = 28

lossFn = SSL_Model.Weighted_InfoNCE(nceBeta=0.0, nceTau=0.1, downSamples=None)
#lossFn = SSS_Model.BT_CrossCorr(prjDim, device, btLam=1/prjDim, lossForm='bt')
#lossFn = SSS_Model.DINO_CrossEnt(prjDim, device, centerMom=0.9, studentTau=0.1, teacherTau=0.04)
symmetrizeLoss = True
applySG = True

# Training parameters
batchSize = 1024
momentum = 0.9
weightDecay = 1e-4
initLR = 0.1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')

############
cpm.update()
############

################################
# Data, Model, Optimizer Setup #
################################

# Create datasets and dataloaders
dataset = datasets.ImageFolder(root, SSL_Transforms.TwoTimesTransform(SSL_Transforms.MoCoV2Transform('test', cropSize)))
loader = torch.utils.data.DataLoader(dataset, batch_size=batchSize, shuffle=True)

############
cpm.update()
############

# Create model and load model weights
stateDict = torch.load(stateFile)
model = SSL_Model.SmallSimSiam(stateDict['encArch'], stateDict['encDim'], stateDict['prjDim'], stateDict['prdDim'], stateDict['momEncBeta'])
model.load_state_dict(stateDict['stateDict'], strict=False)
model = model.to(device)

############
cpm.update()
############

# Instantiate model and (custom) optimizer
optimParams = [{'params': model.encoder.parameters()},
               {'params': model.projector.parameters()},
               {'params': model.predictor.parameters()}]
optimizer = torch.optim.SGD(params=optimParams, lr=initLR, momentum=momentum, weight_decay=weightDecay)

############
cpm.update()
############

########################
# Run Model for 1 Iter #
########################

batch = next(iter(loader))

aug1Tens = batch[0][0].to(device)
aug2Tens = batch[0][1].to(device)

############
cpm.update()
############

# Run each augmented batch through encoder, projector, predictor, and momentum encoder/projector
p1, z1, r1, mz1 = model(aug1Tens)
p2, z2, r2, mz2 = model(aug2Tens)

############
cpm.update()
############

# Apply stop-gradient
if applySG:
    mz1 = mz1.detach()
    mz2 = mz2.detach()

# Calculate loss
if symmetrizeLoss:
    lossVal = 0.5 * (lossFn.forward(p1, mz2) + lossFn.forward(p2, mz1))
else:
    lossVal = lossFn.forward(p1, mz2)

############
cpm.update()
############

# Backpropagate
optimizer.zero_grad()
lossVal.backward()
optimizer.step()

############
cpm.update()
############

# Update momentum encoder
if stateDict['momEncBeta'] > 0.0:
    model.update_momentum_network()

# Update DINO loss center
if type(lossFn) is SSL_Model.DINO_CrossEnt:
    lossFn.update_center(torch.cat((mz1, mz2), dim=0)) if symmetrizeLoss else lossFn.update_center(mz2)

############
cpm.update()
############
trainableParams, allParams = CAU.count_params(model)
activCt = CAU.activation_count(model, (aug1Tens))
flopCt = CAU.flop_count(model, (aug1Tens))
#paramCt2 = CAU.parameter_count(model)

print(cpm.gpuList)
print(cpm.cpuList)
print(cpm.timeList)
print(trainableParams)
print(allParams)
print([activCt[0]['conv'], activCt[0]['linear']])
print([flopCt[0]['conv'], flopCt[0]['linear'], flopCt[0]['batch_norm'], flopCt[0]['adaptive_avg_pool2d']])
#print(paramCt2)
