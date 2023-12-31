import os
import numpy as np
import json
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torchvision.models import resnet18, ResNet18_Weights
import torchvision.transforms as tvtran

from FGSM_PGD import *

##################
# User Functions #
##################

# From 3D tensor input, convert
def show_image(imgTens, titleText):

    # Convert tensor to array and transpose to HxWxC
    imgArr = imgTens.cpu().detach().numpy()
    imgArr = np.transpose(imgArr, (1, 2, 0))

    # Plot images
    plt.imshow(imgArr)
    plt.title(titleText)
    plt.show()

###############
# User Inputs #
###############

imgDir = 'Source_Images'
imgList = ['giant_panda.JPEG', 'mashed_potato.JPEG']

norm = np.inf
eps = 8/255 # Maximum eta norm
alpha = 1/255 # Eta multiplier when added to image
n_restarts = 1
n_steps = 10 # Number of PGD iters
rand_init = False

targeted = False
targLabel = 'bucket'

#testTens = torch.tensor(np.random.rand(2,2,3,3))
#print(testTens)

# scale_tens works well
#scaleTens = scale_tens(testTens, useDims, norm, eps)
#print(scaleTens)
#print(torch.norm(testTens, dim=useDims, keepdim=True, p=norm))
#print(torch.norm(scaleTens, dim=useDims, keepdim=True, p=norm))

# clip_tens works well
#clipTens = clip_tens(testTens, useDims, norm, eps)
#print(clipTens)
#print(torch.norm(testTens, dim=useDims, keepdim=True, p=norm))
#print(torch.norm(clipTens, dim=useDims, keepdim=True, p=norm))

# optimize_linear works well
#eta = optimize_linear(testTens, useDims, norm)
#print(eta)
#print(torch.norm(eta, dim=useDims, keepdim=True, p=norm))

################
# Prepare Data #
################

# CPU runs just as fast for only 1 image and small model
device = torch.device('cpu')

# Load the ImageNet labels JSON file and create related dictionaries
imageNetLabelsDict = json.load(open('imagenet_class_index.json'))
id2label = {int(k): v[1] for k, v in imageNetLabelsDict.items()}
label2id = {v: k for k, v in id2label.items()}

# Define the pytorch image transform to resize and convert to tensor
transform = tvtran.Compose([
    tvtran.Resize((224, 224)),
    #tvtran.RandomHorizontalFlip(),
    tvtran.ToTensor(),
    #tvtran.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

imgTensList = []
trueLabelList = []
labelTensList = []
trueIdxList = []
targIdxList = []
for imgName in imgList:

    # Import the original image and push to a pytorch tensor
    imgPIL = Image.open(os.path.join(imgDir, imgName))
    imgTens = transform(imgPIL).unsqueeze(0).to(device)

    # Get true label from image name and corresponding index
    trueLabel = imgName.split('.')[0]
    trueIdx = label2id[trueLabel]

    # Prepare the target label
    # If targeted, use target label to get target index - FGM will step toward target label
    # If not targeted, use true index as target index - FGM will step away from truth label
    if targeted:
        targIdx = label2id[targLabel]
    else:
        targIdx = trueIdx
    labelTens = torch.Tensor(1, 1000).fill_(0).to(device)
    labelTens[0, targIdx] = 1
    #labelTens = torch.LongTensor(1).fill_(targIdx).to(device)

    imgTensList.append(imgTens)
    trueLabelList.append(trueLabel)
    labelTensList.append(labelTens)
    trueIdxList.append(trueIdx)
    targIdxList.append(targIdx)

imgTens = torch.cat(imgTensList, dim=0)
labelTens = torch.cat(labelTensList, dim=0)

################
# Define Model #
################

# Create model and set in eval mode
model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.to(device)
model.eval()

# Define loss function as CE
loss = torch.nn.CrossEntropyLoss(reduction='none')

###############
# Run FGM/PGD #
###############

batch_size = 2
avgLoss, x_adv = sl_pgd(model, loss, imgTens, labelTens, alpha, eps, norm, n_restarts, n_steps, batch_size,
                               outIdx=None, targeted=targeted, rand_init=rand_init, noise_mag=None, x_min=0., x_max=1.)
import torchattacks
attacker = torchattacks.PGD
#atk = attacker(model, eps=eps, alpha=alpha, steps=n_steps, random_start=rand_init)
#x_adv = atk(imgTens, labelTens)
#avgLoss = 1000

###############
# Get Outputs #
###############

with torch.no_grad():
    trueOutputLogitsTens = model(imgTens)
    advOutputLogitsTens = model(x_adv)

for i in range(len(imgTens)):

    trueProb = trueOutputLogitsTens[i, trueIdxList[i]].cpu().detach().tolist()
    predIdx = torch.topk(advOutputLogitsTens[i, :], k=1)[1].cpu().tolist()[0]
    predLabel = id2label[predIdx]
    predProb = advOutputLogitsTens[i, predIdx].cpu().detach().tolist()
    advTrueProb = advOutputLogitsTens[i, trueIdxList[i]].cpu().detach().tolist()

    # Run the plotter function
    show_image(imgTens[i], 'Orig Img: ' + str(trueIdxList[i]) + ' ' + trueLabelList[i] + ' ' + '{0:.4f}'.format(trueProb))
    show_image(torch.clamp((x_adv[i] - imgTens[i]).sign(), 0, 1), 'Adversarial Perturbation, Step = ' + str(alpha))
    show_image(x_adv[i], 'Adv Img: ' + str(predIdx) + ' ' + predLabel + ' ' + '{0:.4f}'.format(predProb)
               + '\n' + 'Orig Class: ' + str(trueIdxList[i]) + ' ' + trueLabelList[i] + ' ' + '{0:.4f}'.format(advTrueProb))
