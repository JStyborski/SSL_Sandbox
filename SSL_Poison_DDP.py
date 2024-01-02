import argparse
import builtins
from distutils.util import strtobool
import math
import numpy as np
import os
import random
import time
import warnings

import torch
import torch.backends.cudnn as cudnn

import SSL_Model
import SSL_Loss
import Utils.Custom_Dataset as CD
import Utils.Custom_Transforms as CT
import Utils.Misc_Functions as MF
from Adversarial import FGSM_PGD

# CUDNN automatically searches for the best algorithm for processing a given model/optimizer/dataset
cudnn.benchmark = True

###############
# User Inputs #
###############

parser = argparse.ArgumentParser()

# Run processing parameters
parser.add_argument('--useDDP', default=False, type=lambda x:bool(strtobool(x)), help='Strategy to launch on single/multiple GPU')
parser.add_argument('--gatherTensors', default=True, type=lambda x:bool(strtobool(x)), help='Collect tensors across multiple GPU for loss and backpropagation')
parser.add_argument('--gpu', default=0, type=int, help='GPU ID to use for training (single GPU)')
parser.add_argument('--nodeCount', default=1, type=int, help='Number of nodes/servers to use for distributed training')
parser.add_argument('--nodeRank', default=0, type=int, help='Global rank of nodes/servers')
parser.add_argument('--distURL', default='tcp://127.0.0.1:2345', type=str, help='URL for distributed training setup')
parser.add_argument('--distBackend', default='nccl', type=str, help='Distributed backend method')
parser.add_argument('--workers', default=4, type=int, help='Total number of data-loading workers')
parser.add_argument('--randSeed', default=None, type=int, help='RNG initial set point')

# Dataset parameters
parser.add_argument('--trainRoot', default='', type=str, help='Training dataset root directory')
parser.add_argument('--deltaRoot', default='', type=str, help='Poison delta tensor root directory')
parser.add_argument('--poisonRoot', default='', type=str, help='Poison dataset output root directory')
parser.add_argument('--trainLabels', default=True, type=lambda x:bool(strtobool(x)), help='Boolean if the training data is in label folders')
parser.add_argument('--ptPrefix', default='', type=str, help='Prefix to add to pretrained file name')
parser.add_argument('--cropSize', default=224, type=int, help='Crop size to use for input images')
parser.add_argument('--nAugs', default=2, type=int, help='Number of augmentations to apply to each batch')

# Training parameters
parser.add_argument('--nEpochs', default=100, type=int, help='Number of epochs to run')
parser.add_argument('--startEpoch', default=1, type=int, help='Epoch at which to start')
parser.add_argument('--batchSize', default=128, type=int, help='Data loader batch size')
parser.add_argument('--nBatches', default=1e10, type=int, help='Maximum number of batches to run per epoch')
parser.add_argument('--modelSteps', default=1e10, type=int, help='Number of model training steps to run per epoch')
parser.add_argument('--poisonSteps', default=1e10, type=int, help='Number of poison training steps to run per epoch')
parser.add_argument('--momentum', default=0.9, type=float, help='SGD momentum value')
parser.add_argument('--weightDecay', default=1e-4, type=float, help='SGD weight decay value')
parser.add_argument('--initLR', default=0.5, type=float, help='SGD initial learning rate')
parser.add_argument('--useLARS', default=False, type=lambda x:bool(strtobool(x)), help='Boolean to apply LARS optimizer')
parser.add_argument('--decayEncLR', default=True, type=lambda x:bool(strtobool(x)), help='Boolean to apply cosine decay to encoder/projector learning rate')
parser.add_argument('--decayPrdLR', default=False, type=lambda x:bool(strtobool(x)), help='Boolean to apply cosine decay to predictor learning rate')
parser.add_argument('--lrWarmupEp', default=10, type=int, help='Number of linear warmup steps to apply on learning rate - set as 0 for no warmup')
parser.add_argument('--loadChkPt', default=None, type=str, help='File name of checkpoint from which to resume')
parser.add_argument('--runProbes', default=True, type=lambda x:bool(strtobool(x)), help='Boolean to run metric probes')

# Model parameters
parser.add_argument('--encArch', default='resnet18', type=str, choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'vit_tiny', 'vit_small', 'vit_base', 'vit_large'], help='Encoder network (backbone) type')
parser.add_argument('--rnCifarMod', default=False, type=lambda x:bool(strtobool(x)), help='Boolean to apply CIFAR modification to ResNets')
parser.add_argument('--vitPPFreeze', default=True, type=lambda x:bool(strtobool(x)), help='Boolean to freeze ViT patch projection for training stability')
parser.add_argument('--prjArch', default='moco', type=str, choices=['simsiam', 'simclr', 'mec', 'moco', 'byol', 'barlow_twins', 'vicreg', 'dino_cnn', 'dino_vit'], help='Projector network type')
parser.add_argument('--prjHidDim', default=2048, type=int, help='Projector hidden dimension')
parser.add_argument('--prjBotDim', default=256, type=int, help='Projector bottleneck dimension (only used with DINO projector)')
parser.add_argument('--prjOutDim', default=2048, type=int, help='Projector output dimension')
parser.add_argument('--prdHidDim', default=512, type=int, help='Predictor hidden dimension - set as 0 for no predictor')
parser.add_argument('--prdAlpha', default=None, type=float, help='Optimal predictor correlation exponent - set as None for no optimal predictor')
parser.add_argument('--prdEps', default=0.3, type=float, help='Optimal predictor regularization coefficient')
parser.add_argument('--prdBeta', default=0.5, type=float, help='Optimal predictor correlation update momentum')
parser.add_argument('--momEncBeta', default=0.0, type=float, help='Momentum encoder update momentum - set as 0.0 for no momentum encoder')
parser.add_argument('--applySG', default=True, type=lambda x:bool(strtobool(x)), help='Boolean to apply stop-gradient to one branch')

# Loss parameters
parser.add_argument('--symmetrizeLoss', default=True, type=lambda x:bool(strtobool(x)), help='Boolean to apply loss function equally on both augmentation batches')
parser.add_argument('--lossType', default='wince', type=str, choices=['wince', 'bt', 'vicreg', 'mec', 'dino'], help='SSL loss type to apply')
parser.add_argument('--winceBeta', default=0.0, type=float, help='Contrastive term coefficient in InfoNCE loss - set as 0.0 for no contrastive term')
parser.add_argument('--winceTau', default=0.2, type=float, help='Contrastive loss temperature factor')
parser.add_argument('--winceEps', default=0.0, type=float, help='Similarity perturbation constant for disentanglement - 0.0 applies no modification')
parser.add_argument('--btLam', default=0.005, type=float, help='Coefficient to apply to off-diagonal terms of BT loss')
parser.add_argument('--btLossType', default='bt', type=str, help='Method of calculating loss for off-diagonal terms')
parser.add_argument('--btNormType', default='bn', type=str, help='Method of normalizing encoding data')
parser.add_argument('--vicAlpha', default=25.0, type=float, help='Coefficient on variance loss term')
parser.add_argument('--vicBeta', default=25.0, type=float, help='Coefficient on invariance loss term')
parser.add_argument('--vicGamma', default=1.0, type=float, help='Coefficient on covariance loss term')
parser.add_argument('--mecEd2', default=0.06, type=float, help='Related to the coefficient applied to correlation matrix')
parser.add_argument('--mecTaylorTerms', default=2, type=int, help='Number of Taylor expansion terms to include in matrix logarithm approximation')
parser.add_argument('--dinoCentMom', default=0.9, type=float, help='Momentum coefficient for teacher center vector')
parser.add_argument('--dinoTauS', default=0.1, type=float, help='Temperature for student network (online) softmax')
parser.add_argument('--dinoTauT', default=0.05, type=float, help='Temperature for teacher network (target) softmax')

# Adversarial training parameters
parser.add_argument('--advAlpha', default=0.1, type=float, help='PGD step size')
parser.add_argument('--advEps', default=8./255., type=float, help='PGD attack radius limit, measured in specified norm')
parser.add_argument('--advNorm', default=float('inf'), type=float, help='Norm type for measuring perturbation radius')
parser.add_argument('--advDirection', default=-1., type=float, choices=[-1., 1.], help='Perturbation step direction: -1 is unlearnable, +1 is adversarial')
parser.add_argument('--advRestarts', default=1, type=int, help='Number of PGD restarts to search for best attack')
parser.add_argument('--advSteps', default=5, type=int, help='Number of PGD steps to take')
parser.add_argument('--advNoise', default=True, type=lambda x:bool(strtobool(x)), help='Boolean to use random initialization')
parser.add_argument('--advClipMin', default=0., type=int, help='Minimium value to clip adversarial inputs')
parser.add_argument('--advClipMax', default=1., type=int, help='Maximum value to clip adversarial inputs')
parser.add_argument('--advSavePrecision', default=torch.float16, type=torch.dtype, help='Precision for saving poison deltas')

##################
# Misc Functions #
##################

def collate_list(list_items):
    # Incoming list has shape batchSize x nOutputs, reshape to nOutputs x batchSize
    return [[sample[i] for sample in list_items] for i in range(len(list_items[0]))]

def save_chkpt(prefix, epoch, encArch, rnCifarMod, vitPPFreeze, prjArch, prjHidDim, prjBotDim, prjOutDim, prdHidDim, prdAlpha, prdEps, prdBeta, momEncBeta, applySG, model, optimizer):
    torch.save({'epoch': epoch, 'encArch': encArch, 'rnCifarMod': rnCifarMod, 'vitPPFreeze': vitPPFreeze, 'prjArch': prjArch, 'prjHidDim': prjHidDim,
                'prjBotDim': prjBotDim, 'prjOutDim': prjOutDim, 'prdHidDim': prdHidDim, 'prdAlpha': prdAlpha, 'prdEps': prdEps, 'prdBeta': prdBeta,
                'momEncBeta': momEncBeta, 'applySG': applySG, 'stateDict': model.state_dict(),
                'optimStateDict': optimizer.state_dict()}, 'Trained_Models/{}_pt_{:04d}.pth.tar'.format(prefix, epoch))

###################
# Setup Functions #
###################

def main():

    args = parser.parse_args()

    #args.trainRoot = r'D:/ImageNet10/train'
    #args.ptPrefix = 'test'
    #args.deltaRoot = r'D:/Poisoned_ImageNet/CP_100/deltas'
    #args.poisonRoot = r'D:/Poisoned_ImageNet/CP_100/train'
    #args.batchSize = 64
    #args.nEpochs = 2
    #args.modelSteps = 5
    #args.poisonSteps = 5
    #args.advSteps = 1

    if args.randSeed is not None:
        random.seed(args.randSeed)
        np.random.seed(args.randSeed)
        torch.manual_seed(args.randSeed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training'
                      'This will turn on the CUDNN deterministic setting, which can slow down your training'
                      'You may see unexpected behavior when restarting from checkpoints')

    if not args.useDDP:
        warnings.warn('You have disabled DDP - the model will train on 1 GPU without data parallelism')

    if not os.path.exists('Trained_Models'):
        os.mkdir('Trained_Models')

    # Initialize poison data directory (this will may a while, depending on trainRoot size)
    CD.create_shadow_tensors(args.trainRoot, args.deltaRoot, args.trainLabels, args.advNoise, args.advSavePrecision)

    # Infer learning rate
    args.initLR = args.initLR * args.batchSize / 256

    # Determine process settings
    args.nProcPerNode = torch.cuda.device_count()
    args.nProcs = args.nProcPerNode * args.nodeCount
    args.gatherTensors = (args.useDDP and args.nProcs > 1 and args.gatherTensors)

    # Launch multiple (or single) distributed processes for main_worker function - will automatically assign GPUs
    if args.useDDP or args.nProcs > 1:
        args.useDDP = True  # Ensure this flag is True, as multiple loops rely on it
        torch.multiprocessing.spawn(main_worker, nprocs=args.nProcs, args=(args,))
    # Launch one process for main_worker function
    else:
        main_worker(args.gpu, args)

    CD.combine_shadow_tensors(args.trainRoot, args.deltaRoot, args.poisonRoot, args.trainLabels, args.advEps)

######################
# Execution Function #
######################

def main_worker(gpu, args):

    # Replace initial GPU index with assigned one
    args.gpu = gpu
    print('- GPU {} online'.format(args.gpu))

    # Suppress printing if not master (gpu 0)
    if args.useDDP and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass
    print('- Using GPU {} as master process'.format(args.gpu))

    # Set up distributed training on backend
    # For multiprocessing distributed training, rank needs to be the global rank among all the processes
    if args.useDDP:
        args.procRank = args.nodeRank * args.nProcPerNode + args.gpu
        torch.distributed.init_process_group(backend=args.distBackend, init_method=args.distURL, world_size=args.nProcs, rank=args.procRank)
        torch.distributed.barrier()

    print('- Defining dataset and loader')
    if args.trainLabels:
        deltaDataset = CD.DatasetFolder_Plus_Path(args.deltaRoot, transform=None, loader=torch.load, extensions='')
        trainDataset = CD.ImageFolder_Plus_Poison(args.trainRoot, CT.t_tensor(), deltaDataset)
    else:
        deltaDataset = CD.No_Labels_Plus_Path(args.deltaRoot, transform=None, loader=torch.load)
        trainDataset = CD.No_Labels_Images_Plus_Poison(args.trainRoot, CT.t_tensor(), deltaDataset)
    if args.useDDP:
        trainSampler = torch.utils.data.distributed.DistributedSampler(trainDataset)
        # When using single GPU per process and per DDP, need to divide batch size and workers based on nGPUs
        args.batchSize = int(args.batchSize / args.nProcs)
        args.workers = int(args.workers / args.nProcs)
    else:
        trainSampler = None
    # Note that DistributedSampler automatically shuffles dataset given the set_epoch() function during training
    trainDataLoader = torch.utils.data.DataLoader(trainDataset, batch_size=args.batchSize, shuffle=(trainSampler is None), collate_fn=collate_list,
                                                  num_workers=args.workers, pin_memory=True, sampler=trainSampler, drop_last=True)
    loaderSteps = len(trainDataLoader)

    # Define multiaug transform for after regular augmentation in dataset
    nAugTransform = CT.NTimesTransform(args.nAugs, CT.t_tensor_aug(args.cropSize))

    print('- Instantiating new model with {} backbone'.format(args.encArch))
    model = SSL_Model.Base_Model(args.encArch, args.rnCifarMod, args.vitPPFreeze, args.prjArch, args.prjHidDim, args.prjBotDim, args.prjOutDim,
                                 args.prdHidDim, args.prdAlpha, args.prdEps, args.prdBeta, args.momEncBeta, args.applySG)

    # Set up model on parallel or single GPU
    if args.useDDP:
        # Convert BN to SyncBN
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # For multiprocess distributed, DDP constructor should set the single device scope - otherwises uses all available
        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu)
        # broadcast_buffers (default=True) lets SyncBN sync running mean and variance for BN
        # There is a bug in the specific case of applying DDP to 1 GPU:
        # https://github.com/pytorch/pytorch/issues/73332, https://github.com/pytorch/pytorch/issues/66504
        # Workaround is to set broadcast_buffers = False when using 1 GPU
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], broadcast_buffers=(args.nProcs > 1))
    # Single GPU training
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        raise NotImplementedError('CPU training not supported')

    print('- Instantiating loss functions')
    if args.lossType == 'wince':
        lossFn = SSL_Loss.Weighted_InfoNCE_Loss(args.symmetrizeLoss, args.winceBeta, args.winceTau, args.winceEps)
    elif args.lossType == 'bt':
        lossFn = SSL_Loss.Barlow_Twins_Loss(args.symmetrizeLoss, args.btLam, args.btLossType, args.btNormType)
    elif args.lossType == 'vicreg':
        lossFn = SSL_Loss.VICReg_Loss(args.symmetrizeLoss, args.vicAlpha, args.vicBeta, args.vicGamma)
    elif args.lossType == 'mec':
        lossFn = SSL_Loss.MEC_Loss(args.symmetrizeLoss, args.mecEd2, args.mecTaylorTerms)
    elif args.lossType == 'dino':
        lossFn = SSL_Loss.DINO_Loss(args.symmetrizeLoss, args.dinoCentMom, args.dinoTauS, args.dinoTauT)

    # Instantiate custom optimizer that skips momentum encoder and applies decay
    print('- Instantiating optimizer')
    if args.useDDP:
        optimParams = [{'params': model.module.encoder.parameters(), 'decayLR': args.decayEncLR},
                       {'params': model.module.projector.parameters(), 'decayLR': args.decayEncLR},
                       {'params': model.module.predictor.parameters(), 'decayLR': args.decayPrdLR}]
    else:
        optimParams = [{'params': model.encoder.parameters(), 'decayLR': args.decayEncLR},
                       {'params': model.projector.parameters(), 'decayLR': args.decayEncLR},
                       {'params': model.predictor.parameters(), 'decayLR': args.decayPrdLR}]
    if 'vit' in args.encArch.lower():
        optimizer = torch.optim.AdamW(params=optimParams, lr=args.initLR, weight_decay=args.weightDecay)
    else:
        optimizer = torch.optim.SGD(params=optimParams, lr=args.initLR, momentum=args.momentum, weight_decay=args.weightDecay)
    if args.useLARS:
        print("- Using LARS optimizer.")
        from Utils.Apex_LARC import LARC
        optimizer = LARC(optimizer=optimizer, trust_coefficient=0.001, clip=False)

    # Optionally resume model/optimizer from checkpoint
    if args.loadChkPt is not None:
        print('- Loading checkpoint file {}'.format(args.loadChkPt))
        # Map model to be loaded to specified single GPU
        chkPt = torch.load(args.loadChkPt, map_location='cuda:{}'.format(args.gpu))
        args.startEpoch = chkPt['epoch'] + 1
        model.load_state_dict(chkPt['stateDict'], strict=True)
        optimizer.load_state_dict(chkPt['optimStateDict'])
        del chkPt  # to save space

    # Checkpoint initial model, and start timer
    save_chkpt(args.ptPrefix, 0, args.encArch, args.rnCifarMod, args.vitPPFreeze, args.prjArch, args.prjHidDim, args.prjBotDim, args.prjOutDim,
               args.prdHidDim, args.prdAlpha, args.prdEps, args.prdBeta, args.momEncBeta, args.applySG, model, optimizer)
    timeStart = time.time()

    print('- Beginning training')
    for epoch in range(args.startEpoch, args.nEpochs + 1):

        # Update sampler with current epoch - required to ensure shuffling works across devices
        if args.useDDP:
            trainSampler.set_epoch(epoch)

        # If using LR warmup, the warmup LR is linear, applies to all layers, and takes priority over decay
        if epoch <= args.lrWarmupEp:
            curLR = epoch / args.lrWarmupEp * args.initLR
            for param_group in optimizer.param_groups:
                param_group['lr'] = curLR

        # If desired, adjust learning rate for the current epoch using cosine annealing
        if (args.decayEncLR or args.decayPrdLR) and epoch >= args.lrWarmupEp:
            curLR = args.initLR * 0.5 * (1. + math.cos(math.pi * (epoch - 1) / args.nEpochs))
            for param_group in optimizer.param_groups:
                if param_group['decayLR']:
                    param_group['lr'] = curLR

        # Setup for model training
        sumModelLoss = 0.
        trainPoison = False
        model.train()
        if args.useDDP and args.nProcs > 1:
            model.broadcast_buffers = True
        trainIterator1 = iter(trainDataLoader)

        # Loop through specified number of steps in loader
        for modelI in range(min(args.modelSteps, loaderSteps)):

            # Get next batch of inputs
            images, _, deltas, _ = next(trainIterator1)

            # Combine images and poisons and augment
            augList = [ [] for _ in range(args.nAugs) ]
            for i in range(len(deltas)):
                if trainPoison:
                    deltas[i].requires_grad = True
                poisonAugs = nAugTransform(torch.clamp(images[i] + args.advEps * deltas[i], min=args.advClipMin, max=args.advClipMax))
                for j in range(args.nAugs):
                    augList[j].append(poisonAugs[j])

            # Collect augmentations as torch tensors
            for i in range(len(augList)):
                augList[i] = torch.stack(augList[i], dim=0).cuda(args.gpu, non_blocking=True)

            # Loop through each of the augs and create a list of results
            outList = []
            for augTens in augList:

                # Get input tensor, push through model
                p, z, r, mz = model(augTens)

                # Gather tensors across GPUs (required for accurate loss calculations)
                if args.gatherTensors:
                    torch.distributed.barrier()  # Sync processes before gathering inputs
                    p = torch.cat(MF.FullGatherLayer.apply(p.contiguous()), dim=0)
                    z = torch.cat(MF.FullGatherLayer.apply(z.contiguous()), dim=0)
                    r = torch.cat(MF.FullGatherLayer.apply(r.contiguous()), dim=0)
                    mz = torch.cat(MF.FullGatherLayer.apply(mz.contiguous()), dim=0)

                # Append to lists for loss calculation
                outList.append([p, z, r, mz])

            # Calculate loss and backpropagate
            lossVal = lossFn(outList)
            model.zero_grad()  # momenc and optPrd not included in optimizer, but they don't use grads - this should be redundant
            optimizer.zero_grad()
            lossVal.backward()
            optimizer.step()

            # Update momentum encoder
            if args.momEncBeta > 0.0:
                if args.useDDP:
                    model.module.update_momentum_network()
                else:
                    model.update_momentum_network()

            # Keep running sum of loss
            sumModelLoss += lossVal.detach()

            if modelI + 1 >= args.nBatches:
                break

        print('Epoch: {} / {} | Time: {:0.2f} | Avg Model Loss: {:0.4f}'.format(epoch, args.nEpochs, time.time() - timeStart, sumModelLoss / (modelI + 1)))

        # Setup for poison training
        sumPoisonLoss = 0.
        trainPoison = True
        model.eval()
        if args.useDDP and args.nProcs > 1:
            model.broadcast_buffers = False
        trainIterator2 = iter(trainDataLoader)

        # Loop through specified number of steps in loader
        for poisonI in range(min(args.poisonSteps, loaderSteps)):

            # Get next batch of inputs
            images, _, deltas, deltaPaths = next(trainIterator2)

            for _ in range(args.advSteps):

                # Combine images and poisons and augment
                augList = [[] for _ in range(args.nAugs)]
                for i in range(len(deltas)):
                    if trainPoison:
                        deltas[i].requires_grad = True
                    poisonAugs = nAugTransform(torch.clamp(images[i] + args.advEps * deltas[i], min=args.advClipMin, max=args.advClipMax))
                    for j in range(args.nAugs):
                        augList[j].append(poisonAugs[j])

                # Collect augmentations as torch tensors
                for i in range(len(augList)):
                    augList[i] = torch.stack(augList[i], dim=0).cuda(args.gpu, non_blocking=True)

                # Loop through each of the augs and create a list of results
                outList = []
                for augTens in augList:

                    # Get input tensor, push through model
                    p, z, r, mz = model(augTens)

                    # Gather tensors across GPUs (required for accurate loss calculations)
                    if args.gatherTensors:
                        torch.distributed.barrier() # Sync processes before gathering inputs
                        p = torch.cat(MF.FullGatherLayer.apply(p.contiguous()), dim=0)
                        z = torch.cat(MF.FullGatherLayer.apply(z.contiguous()), dim=0)
                        r = torch.cat(MF.FullGatherLayer.apply(r.contiguous()), dim=0)
                        mz = torch.cat(MF.FullGatherLayer.apply(mz.contiguous()), dim=0)

                    # Append to lists for loss calculation
                    outList.append([p, z, r, mz])

                # Calculate loss and backpropagate
                lossVal = lossFn(outList)
                model.zero_grad()  # momenc and optPrd not included in optimizer, but they don't use grads - this should be redundant
                optimizer.zero_grad()
                lossVal.backward()

                # Apply PGD attack
                for i in range(len(deltas)):
                    eta = FGSM_PGD.optimize_linear(deltas[i].grad, None, norm=args.advNorm)
                    deltas[i] = torch.clamp(deltas[i].data + args.advDirection * args.advAlpha * eta, -1., 1.)

            # Save updated perturbations
            for i in range(len(deltas)):
                torch.save(deltas[i].cpu().type(args.advSavePrecision), deltaPaths[i])

            # Keep running sum of loss
            sumPoisonLoss += lossVal.detach()

            if poisonI + 1 >= args.nBatches:
                break

        print('Epoch: {} / {} | Time: {:0.2f} | Avg Poison Loss: {:0.4f}'.format(epoch, args.nEpochs, time.time() - timeStart, sumPoisonLoss / (poisonI + 1)))

        # Checkpoint model
        if epoch in [1, 10] or (epoch <= 200 and epoch % 20 == 0) or (epoch > 200 and epoch % 50 == 0):
            save_chkpt(args.ptPrefix, epoch, args.encArch, args.rnCifarMod, args.vitPPFreeze, args.prjArch, args.prjHidDim, args.prjBotDim, args.prjOutDim,
                       args.prdHidDim, args.prdAlpha, args.prdEps, args.prdBeta, args.momEncBeta, args.applySG, model, optimizer)

if __name__ == '__main__':
    main()
