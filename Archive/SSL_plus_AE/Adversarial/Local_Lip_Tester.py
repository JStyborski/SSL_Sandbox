import os
import numpy as np
import json
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torchvision.models import resnet18, ResNet18_Weights
import torchvision.transforms as tvtran

from Local_Lip import *

###############
# User Inputs #
###############

imgDir = 'Source_Images'
imgName = 'giant_panda.JPEG'

top_norm = 1
bot_norm = np.inf
alpha = 0.02
eps = 0.1
perturb_steps = 10


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
    tvtran.ToTensor()
])

# Import the original image and push to a pytorch tensor
imgPIL = Image.open(os.path.join(imgDir, imgName))
imgTens = transform(imgPIL).unsqueeze(0).to(device)

# Create perturbed image
noise_minmax = eps
noise = torch.zeros_like(imgTens).uniform_(-noise_minmax, noise_minmax)
imgTensP = imgTens + noise

################
# Define Model #
################

# Create model and set in eval mode
model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.to(device)
model.eval()

#################
# Run Local Lip #
#################

#lolip = calc_local_lip(model, imgTens, imgTensP, top_norm, bot_norm, reduction='mean')
#print(lolip)

avgLip, advTens = maximize_local_lip(model, imgTens, top_norm, bot_norm, batch_size=len(imgTens),
                                     perturb_steps=perturb_steps, alpha=alpha, eps=eps, device=device)

print(avgLip)

###############
# Plot Images #
###############

# From 4D tensor input, convert
def show_image(imgTens, titleText):

    # Convert tensor to array and transpose to HxWxC
    imgArr = imgTens.cpu().detach().numpy()
    imgArr = np.transpose(imgArr[0], (1, 2, 0))

    # Plot images
    plt.imshow(imgArr)
    plt.title(titleText)
    plt.show()

# Run the plotter function
show_image(imgTens, 'Orig Img')
show_image(advTens, 'Adv Img')