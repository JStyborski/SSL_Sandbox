import os
import json
import numpy as np
import sklearn.decomposition
import matplotlib.pyplot as plt
import pandas as pd

import torch
import torchvision.transforms

from Analysis_Utils import *

###############
# User Inputs #
###############

architecture = 'resnet18'
outDim = 512
lastLayer = 'avgpool' # avgpool, fc
plotPCA = False
runFDA = False

imgsPath = r'D:\ImageNet1k\ILSVRC\Data\CLS-LOC\val_sorted'
#imgsPath = r'D:\MSCOCO\val2017'
annotationsPath = r'D:\MSCOCO\annotations\instances_val2017.json'
nImgsPerClass = 50

checkpointPath = r'C:\Users\jeremy\Python_Projects\simsiam\checkpoints\SimSiam_Loc-1_Imagenette_100ep_RN18'
checkpointFiles = [file for file in os.listdir(checkpointPath) if 'checkpoint' in file]

outCSVFile = 'out_data.csv'

###############
# Load Images #
###############

# For MSCOCO images
# From the JSON annotations list, find image IDs with the correct category ID
# Then from the JSON images list, find image filenames with the correct image ID
def get_n_MSCOCO_imgs(jsonDict, imgsPath, categID, nImgs):

    annotationsList = jsonDict['annotations']
    imgsList = jsonDict['images']

    imgIDsList = []
    for item in annotationsList:
        if item['category_id'] == categID:
            imgIDsList.append(item['image_id'])

    imgFilesList = []
    for item in imgsList:
        if item['id'] in imgIDsList:
            imgFilesList.append(os.path.join(imgsPath, item['file_name']))
        if len(imgFilesList) >= nImgs:
            break

    return imgFilesList

# For ImageNet images
def get_n_ImageNet_imgs(imgsPath, categID, nImgs):

    imgsList = os.listdir(os.path.join(imgsPath, categID))

    imgFilesList = []
    for imgFile in imgsList:
        imgFilesList.append(os.path.join(imgsPath, categID, imgFile))
        if len(imgFilesList) >= nImgs:
            break

    return imgFilesList

if 'mscoco' in imgsPath.lower():

    # Load the annotations JSON file
    cocoJSON = json.load(open(annotationsPath))
    cls1ImgID = 17 # 17 is cat
    cls2ImgID = 18 # 18 is dog
    cls3ImgID = 67 # 67 is dining table

    cls1Imgs = get_n_MSCOCO_imgs(cocoJSON, imgsPath, cls1ImgID, nImgsPerClass)
    cls2Imgs = get_n_MSCOCO_imgs(cocoJSON, imgsPath, cls2ImgID, nImgsPerClass)
    cls3Imgs = get_n_MSCOCO_imgs(cocoJSON, imgsPath, cls3ImgID, nImgsPerClass)

elif 'imagenet' in imgsPath.lower():

    #cls1ImgID = 'n02127052' # Lynx (in IN100)
    cls1ImgID = 'n01440764' # Tench (in Imagenette)
    cls2ImgID = 'n02106662' # German Shephard (in IN100)
    #cls3ImgID = 'n02114367' # Timber Wolf (not in IN100)
    cls3ImgID = 'n03000684' # Chainsaw (in Imagenette)
    #cls4ImgID = 'n03956157' # Planetarium (in IN100)
    cls4ImgID = 'n03028079' # Church (in Imagenette)
    cls5ImgID = 'n03788195' # Mosque (not in IN100)

    cls1Imgs = get_n_ImageNet_imgs(imgsPath, cls1ImgID, nImgsPerClass)
    cls2Imgs = get_n_ImageNet_imgs(imgsPath, cls2ImgID, nImgsPerClass)
    cls3Imgs = get_n_ImageNet_imgs(imgsPath, cls3ImgID, nImgsPerClass)
    cls4Imgs = get_n_ImageNet_imgs(imgsPath, cls4ImgID, nImgsPerClass)
    cls5Imgs = get_n_ImageNet_imgs(imgsPath, cls5ImgID, nImgsPerClass)

################
# Create Model #
################

print("=> Creating model '{}'".format(architecture))
encoder = torchvision.models.__dict__[architecture](num_classes=outDim)
encoder.eval()

if lastLayer == 'fc':
    # Replace last fc layer to match with pretraining model
    prevDim = encoder.fc.weight.shape[1]
    encoder.fc = torch.nn.Linear(prevDim, outDim, bias=False)
elif lastLayer == 'avgpool':
    # Replace last fc layer with identity to see AvgPool output
    encoder.fc = torch.nn.Identity()

##############################
# Define Image Augmentations #
##############################

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

# Define the training set augmentation procedure
train_augmentation = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
    torchvision.transforms.RandomApply([torchvision.transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    torchvision.transforms.RandomGrayscale(p=0.2),
    torchvision.transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define the validation set augmentation procedure
val_augmentation = torchvision.transforms.Compose([
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

###########################
# Analyze Each Checkpoint #
###########################

for ii, checkpointFile in enumerate(checkpointFiles):

    print('\nRunning {}'.format(checkpointFile))

    ##############
    # Load Model #
    ##############

    # Load state dict
    checkpoint = torch.load(os.path.join(checkpointPath, checkpointFile), map_location='cpu')

    # Remake the encoder keys, without "encoder." and delete all original keys
    for k in list(checkpoint['state_dict'].keys()):
        if k.startswith('module.encoder'):
            checkpoint['state_dict'][k[len('module.encoder.'):]] = checkpoint['state_dict'][k]
        elif k.startswith('encoder'):
            checkpoint['state_dict'][k[len('encoder.'):]] = checkpoint['state_dict'][k]
        del checkpoint['state_dict'][k]

    # To use load fc in SimSiam, need to rename the extra fc
    archFromCPPath = checkpointPath.split('\\')
    if lastLayer == 'fc' and 'simsiam' in archFromCPPath[-1].lower():
        checkpoint['state_dict']['fc.weight'] = checkpoint['state_dict']['fc.0.weight']
        del checkpoint['state_dict']['fc.0.weight']

    # Load parameters into encoder
    msg = encoder.load_state_dict(checkpoint['state_dict'], strict=False)
    print(msg)

    #######################
    # Get Image Encodings #
    #######################

    cls1EncArr = get_encodings(encoder, outDim, val_augmentation, cls1Imgs)
    cls1AugEncArr = get_encodings(encoder, outDim, train_augmentation, cls1Imgs)
    cls2EncArr = get_encodings(encoder, outDim, val_augmentation, cls2Imgs)
    cls3EncArr = get_encodings(encoder, outDim, val_augmentation, cls3Imgs)
    cls4EncArr = get_encodings(encoder, outDim, val_augmentation, cls4Imgs)
    cls5EncArr = get_encodings(encoder, outDim, val_augmentation, cls5Imgs)

    #####################
    # Analyze Encodings #
    #####################

    def one_cls_stats(clsEncArr, clsID, outDict):
        clsEncAvg = np.mean(clsEncArr, axis=1)
        outDict['{}avg 2Norm'.format(clsID)] = np.linalg.norm(clsEncAvg)
        outDict['{}avg Mean'.format(clsID)] = np.mean(clsEncAvg)
        outDict['{}avg Min'.format(clsID)] = np.min(clsEncAvg)
        outDict['{}avg Max'.format(clsID)] = np.max(clsEncAvg)
        outDict['{}arr Var Mean'.format(clsID)] = np.mean(np.var(clsEncArr, axis=1))
        outDict['{}arr Var Min'.format(clsID)] = np.min(np.var(clsEncArr, axis=1))
        outDict['{}arr Var Max'.format(clsID)] = np.max(np.var(clsEncArr, axis=1))
        outDict['{}arr Cov FNorm'.format(clsID)] = np.linalg.norm(cross_cov(clsEncArr, clsEncArr))
        outDict['{}arr Mean EucDist to Avg'.format(clsID)] = avg_euc_dist_to_one_vec(clsEncArr, clsEncAvg)
        outDict['{}arr Mean EucDist bt Arr'.format(clsID)] = avg_euc_dist_bt_array(clsEncArr)
        outDict['{}arr Mean CosSim to Avg'.format(clsID)] = avg_cos_sim_to_one_vec(clsEncArr, clsEncAvg)
        outDict['{}arr Mean CosSim bt Arr'.format(clsID)] = avg_cos_sim_bt_array(clsEncArr)
        outDict['{}avg Sparsity'.format(clsID)] = sum(one_vec_sparsity(clsEncAvg))
        outDict['{}arr Mean Sparsity'.format(clsID)] = array_avg_sparsity(clsEncArr)
        outDict['{}avg Near0'.format(clsID)] = sum(elem_near_zero(clsEncAvg))
        outDict['{}arr Mean Near0'.format(clsID)] = array_avg_near_zero(clsEncArr)
        return outDict

    def two_cls_stats(clsEncArr1, clsEncArr2, clsID1, clsID2, outDict):
        clsEncAvg1 = np.mean(clsEncArr1, axis=1)
        clsEncAvg2 = np.mean(clsEncArr2, axis=1)
        outDict['{}arr-{}arr CrossCov FNorm'.format(clsID1, clsID2)] = np.linalg.norm(cross_cov(clsEncArr1, clsEncArr2))
        outDict['{}avg-{}avg EucDist'.format(clsID1, clsID2)] = euc_dist_bt_vecs(clsEncAvg1, clsEncAvg2)
        outDict['{}arr-{}avg EucDist'.format(clsID1, clsID2)] = avg_euc_dist_to_one_vec(clsEncArr1, clsEncAvg2)
        outDict['{}avg-{}avg CosSim'.format(clsID1, clsID2)] = cos_sim_bt_vecs(clsEncAvg1, clsEncAvg2)
        outDict['{}arr-{}avg CosSim'.format(clsID1, clsID2)] = avg_cos_sim_to_one_vec(clsEncArr1, clsEncAvg2)
        nonZeroBool1 = one_vec_sparsity(clsEncAvg1)
        nonZeroBool2 = one_vec_sparsity(clsEncAvg2)
        outDict['{}avg-{}avg Intersect Sparsity'.format(clsID1, clsID2)] = sum(np.logical_and(nonZeroBool1, nonZeroBool2))
        nearZeroBool1 = elem_near_zero(clsEncAvg1)
        nearZeroBool2 = elem_near_zero(clsEncAvg2)
        outDict['{}avg-{}avg Intersect Near0'.format(clsID1, clsID2)] = sum(np.logical_and(nearZeroBool1, nearZeroBool2))
        return outDict

    outDict = {}

    # Run metadata
    outDict['Epoch'] = int(checkpointFile.split('_')[1][:4]) + 1
    outDict['Checkpoint'] = checkpointFile
    outDict['Arch'] = architecture
    outDict['Last Layer'] = lastLayer
    outDict['Out Dim'] = cls1EncArr.shape[1]

    # Image classes
    outDict['Class 1 Idx'] = cls1ImgID
    outDict['Class 2 Idx'] = cls2ImgID
    outDict['Class 3 Idx'] = cls3ImgID
    outDict['Class 4 Idx'] = cls4ImgID
    outDict['Class 5 Idx'] = cls5ImgID

    # 1 class stats
    outDict = one_cls_stats(cls1EncArr, '1', outDict)
    outDict = one_cls_stats(cls1AugEncArr, '1aug', outDict)
    outDict = one_cls_stats(cls2EncArr, '2', outDict)
    outDict = one_cls_stats(cls3EncArr, '3', outDict)
    outDict = one_cls_stats(cls4EncArr, '4', outDict)
    outDict = one_cls_stats(cls5EncArr, '5', outDict)

    # 2 class stats
    outDict = two_cls_stats(cls1EncArr, cls1AugEncArr, '1', '1aug', outDict)
    outDict = two_cls_stats(cls1EncArr, cls2EncArr, '1', '2', outDict)
    outDict = two_cls_stats(cls1EncArr, cls3EncArr, '1', '3', outDict)
    outDict = two_cls_stats(cls1EncArr, cls4EncArr, '1', '4', outDict)
    outDict = two_cls_stats(cls1EncArr, cls5EncArr, '1', '5', outDict)
    outDict = two_cls_stats(cls2EncArr, cls3EncArr, '2', '3', outDict)
    outDict = two_cls_stats(cls4EncArr, cls5EncArr, '4', '5', outDict)

    ##################
    # FDA Projection #
    ##################

    # The formula is correct (pretty sure) but the totalScatter matrix is singular (why). Cannot invert.
    # np.linalg.cond(totalScatter) = inf, np.linalg.det(totalScatter) = 0
    # Digging deeper, the encodings of the vectors are a lot of zeros, so the scatter matrices have entire columns of zeros

    if lastLayer == 'fc' and runFDA:
        W12 = fisher_W(cls1EncArr, cls2EncArr)
        W13 = fisher_W(cls1EncArr, cls3EncArr)
        W23 = fisher_W(cls2EncArr, cls3EncArr)

        cls1OnW12 = np.dot(cls1EncArr, W12)
        cls2OnW12 = np.dot(cls2EncArr, W12)
        cls1OnW13 = np.dot(cls1EncArr, W13)
        cls3OnW13 = np.dot(cls3EncArr, W13)
        cls2OnW23 = np.dot(cls2EncArr, W23)
        cls3OnW23 = np.dot(cls3EncArr, W23)

        #print('\nFisher Discriminant Mappings')
        #print('Class 1 on W12: ', cls1OnW12[0:4])
        #print('Class 2 on W12: ', cls2OnW12[0:4])
        #print('Class 1 on W13: ', cls1OnW13[0:4])
        #print('Class 3 on W13: ', cls3OnW13[0:4])
        #print('Class 2 on W23: ', cls2OnW23[0:4])
        #print('Class 3 on W23: ', cls3OnW23[0:4])

        # Fisher Discriminant Mappings
        outDict['1-W12 FDA Avg'] = np.mean(cls1OnW12)
        outDict['2-W12 FDA Avg'] = np.mean(cls2OnW12)
        outDict['1-W13 FDA Avg'] = np.mean(cls1OnW13)
        outDict['3-W13 FDA Avg'] = np.mean(cls3OnW13)
        outDict['2-W23 FDA Avg'] = np.mean(cls2OnW23)
        outDict['3-W23 FDA Avg'] = np.mean(cls3OnW23)

    #######################
    # Plot PCA Projection #
    #######################

    if plotPCA:
        allEnc = np.concatenate((cls1EncArr, cls2EncArr, cls3EncArr), axis=0)
        pca = sklearn.decomposition.PCA(n_components=3)
        pcaResult = pca.fit_transform(allEnc)

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        for i in range(3):
            xVals = pcaResult[i*nImgsPerClass:(i+1)*nImgsPerClass+1, 0].tolist()
            yVals = pcaResult[i*nImgsPerClass:(i+1)*nImgsPerClass+1, 1].tolist()
            zVals = pcaResult[i*nImgsPerClass:(i+1)*nImgsPerClass+1, 2].tolist()
            ax.scatter(xVals, yVals, zVals, label=i, alpha=1)
            ax.legend()
        plt.show()

    #################
    # Output Pandas #
    #################

    if ii == 0:
        outDF = pd.Series(outDict, index=outDict.keys())
    else:
        outDF = pd.concat((outDF, pd.Series(outDict, index=outDict.keys())), axis=1)

#######################
# Output and Plotting #
#######################

outDF.to_csv(os.path.join(checkpointPath, outCSVFile))

def make_plot(xDS, yDF, logy):
    plt.plot(xDS, yDF)
    plt.legend(yDF.columns)
    if logy:
        plt.yscale('log')
    plt.show()

xDS = outDF.loc['Epoch']
make_plot(xDS, outDF.loc[['1avg 2Norm', '2avg 2Norm', '3avg 2Norm', '4avg 2Norm', '5avg 2Norm'], :].T, logy=True)
make_plot(xDS, outDF.loc[['1avg Mean', '2avg Mean', '3avg Mean', '4avg Mean', '5avg Mean'], :].T, logy=False)
make_plot(xDS, outDF.loc[['1avg Min', '2avg Min', '3avg Min', '4avg Min', '5avg Min'], :].T, logy=False)
make_plot(xDS, outDF.loc[['1avg Max', '2avg Max', '3avg Max', '4avg Max', '5avg Max'], :].T, logy=False)
make_plot(xDS, outDF.loc[['1arr Var Mean', '2arr Var Mean', '3arr Var Mean', '4arr Var Mean', '5arr Var Mean'], :].T, logy=True)
make_plot(xDS, outDF.loc[['1avg-1augavg EucDist', '1avg-2avg EucDist', '1avg-3avg EucDist', '1avg-4avg EucDist',
                          '1avg-5avg EucDist', '2avg-3avg EucDist', '4avg-5avg EucDist'], :].T, logy=True)
#make_plot(xDS, outDF.loc['1avg-1augavg EucDist':'4avg-5avg EucDist'].T, logy=True)
make_plot(xDS, outDF.loc[['1arr-1augavg EucDist', '1arr-2avg EucDist', '1arr-3avg EucDist', '1arr-4avg EucDist',
                          '1arr-5avg EucDist'], :].T, logy=True)
make_plot(xDS, outDF.loc[['1arr Mean EucDist to Avg', '2arr Mean EucDist to Avg', '3arr Mean EucDist to Avg',
                          '4arr Mean EucDist to Avg', '5arr Mean EucDist to Avg'], :].T, logy=True)
make_plot(xDS, outDF.loc[['1avg-1augavg CosSim', '1avg-2avg CosSim', '1avg-3avg CosSim', '1avg-4avg CosSim',
                          '1avg-5avg CosSim'], :].T, logy=False)
#make_plot(xDS, outDF.loc['1avg-1augavg CosAng':'4avg-5avg CosAng'].T, logy=False)
make_plot(xDS, outDF.loc[['1arr-1augavg CosSim', '1arr-2avg CosSim', '1arr-3avg CosSim', '1arr-4avg CosSim',
                          '1arr-5avg CosSim'], :].T, logy=False)
make_plot(xDS, outDF.loc[['1arr Mean CosSim to Avg', '2arr Mean CosSim to Avg', '3arr Mean CosSim to Avg',
                          '4arr Mean CosSim to Avg', '5arr Mean CosSim to Avg'], :].T, logy=False)
make_plot(xDS, outDF.loc[['1arr Mean CosSim bt Arr', '2arr Mean CosSim bt Arr', '3arr Mean CosSim bt Arr',
                          '4arr Mean CosSim bt Arr', '5arr Mean CosSim bt Arr'], :].T, logy=False)
make_plot(xDS, outDF.loc[['1avg Sparsity', '2avg Sparsity', '3avg Sparsity', '4avg Sparsity', '5avg Sparsity'], :].T, logy=False)
make_plot(xDS, outDF.loc[['1arr Mean Sparsity', '2arr Mean Sparsity', '3arr Mean Sparsity', '4arr Mean Sparsity',
                          '5arr Mean Sparsity'], :].T, logy=False)
#make_plot(xDS, outDF.loc['1avg Spars':'Intersect 45avg Spars'].T, logy=False)
make_plot(xDS, outDF.loc[['1avg Near0', '2avg Near0', '3avg Near0', '4avg Near0', '5avg Near0'], :].T, logy=False)
make_plot(xDS, outDF.loc[['1arr Mean Near0', '2arr Mean Near0', '3arr Mean Near0', '4arr Mean Near0',
                          '5arr Mean Near0'], :].T, logy=False)
#make_plot(xDS, outDF.loc['1avg Near0':'Intersect 45avg Near0'].T, logy=False)
