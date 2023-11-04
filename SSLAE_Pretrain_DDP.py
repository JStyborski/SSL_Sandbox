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
import torchvision.datasets as datasets

import SSLAE_Model
import SSLAE_Loss
import Utils.Custom_Dataset as CD
import Utils.Custom_Transforms as CT
import Utils.Custom_Probes as CP
from Adversarial import FGSM_PGD

# CUDNN automatically searches for the best algorithm for processing a given model/optimizer/dataset
cudnn.benchmark = True

###############
# User Inputs #
###############

parser = argparse.ArgumentParser()

# Run processing parameters
parser.add_argument('--multiprocDistrib', default=False, type=lambda x:bool(strtobool(x)), help='Strategy to launch on single/multiple GPU')
parser.add_argument('--gpu', default=0, type=int, help='GPU ID to use for training (single GPU)')
parser.add_argument('--nodeCount', default=1, type=int, help='Number of nodes/servers to use for distributed training')
parser.add_argument('--nodeRank', default=0, type=int, help='Global rank of nodes/servers')
parser.add_argument('--distURL', default='tcp://127.0.0.1:2345', type=str, help='URL for distributed training setup')
parser.add_argument('--distBackend', default='nccl', type=str, help='Distributed backend method')
parser.add_argument('--workers', default=4, type=int, help='Total number of data-loading workers')
parser.add_argument('--randSeed', default=None, type=int, help='RNG initial set point')

# Dataset parameters
parser.add_argument('--trainRoot', default='', type=str, help='Training dataset root directory')
parser.add_argument('--trainLabels', default=True, type=lambda x:bool(strtobool(x)), help='Boolean if the training data is in label folders')
parser.add_argument('--ptPrefix', default='', type=str, help='Prefix to add to pretrained file name')
parser.add_argument('--cropSize', default=224, type=int, help='Crop size to use for input images')
parser.add_argument('--nAugs', default=2, type=int, help='Number of augmentations to apply to each batch')

# Training parameters
parser.add_argument('--nEpochs', default=100, type=int, help='Number of epochs to run')
parser.add_argument('--startEpoch', default=1, type=int, help='Epoch at which to start')
parser.add_argument('--batchSize', default=512, type=int, help='Data loader batch size')
parser.add_argument('--momentum', default=0.9, type=float, help='SGD momentum value')
parser.add_argument('--weightDecay', default=1e-4, type=float, help='SGD weight decay value')
parser.add_argument('--initLR', default=0.05, type=float, help='SGD initial learning rate')
parser.add_argument('--useLARS', default=False, type=lambda x:bool(strtobool(x)), help='Boolean to apply LARS optimizer')
parser.add_argument('--decayEncLR', default=True, type=lambda x:bool(strtobool(x)), help='Boolean to apply cosine decay to encoder/projector learning rate')
parser.add_argument('--decayPrdLR', default=False, type=lambda x:bool(strtobool(x)), help='Boolean to apply cosine decay to predictor learning rate')
parser.add_argument('--lrWarmupEp', default=10, type=int, help='Number of linear warmup steps to apply on learning rate - set as 0 for no warmup')
parser.add_argument('--loadChkPt', default=None, type=str, help='File name of checkpoint from which to resume')
parser.add_argument('--runProbes', default=True, type=lambda x:bool(strtobool(x)), help='Boolean to run metric probes')

# Model parameters
parser.add_argument('--encArch', default='resnet18', type=str, help='Encoder network (backbone) type')
parser.add_argument('--cifarMod', default=False, type=lambda x:bool(strtobool(x)), help='Boolean to apply CIFAR modification to ResNets')
parser.add_argument('--encDim', default=512, type=int, help='Encoder output dimension')
parser.add_argument('--prjHidDim', default=2048, type=int, help='Projector hidden dimension')
parser.add_argument('--prjOutDim', default=2048, type=int, help='Projector output dimension')
parser.add_argument('--prdDim', default=512, type=int, help='Predictor hidden dimension - set as 0 for no predictor')
parser.add_argument('--prdAlpha', default=None, type=float, help='Optimal predictor correlation exponent - set as None for no optimal predictor')
parser.add_argument('--prdEps', default=0.3, type=float, help='Optimal predictor regularization coefficient')
parser.add_argument('--prdBeta', default=0.5, type=float, help='Optimal predictor correlation update momentum')
parser.add_argument('--momEncBeta', default=0.0, type=float, help='Momentum encoder update momentum - set as 0.0 for no momentum encoder')
parser.add_argument('--applySG', default=True, type=lambda x:bool(strtobool(x)), help='Boolean to apply stop-gradient to one branch')
parser.add_argument('--decArch', default=None, type=str, help='Decoder network type (ResNets only)')

# Loss parameters
parser.add_argument('--symmetrizeLoss', default=True, type=lambda x:bool(strtobool(x)), help='Boolean to apply loss function equally on both augmentation batches')
parser.add_argument('--sslLossType', default='wince', type=str, help='SSL loss type to apply')
parser.add_argument('--winceBeta', default=0.0, type=float, help='Contrastive term coefficient in InfoNCE loss - set as 0.0 for no contrastive term')
parser.add_argument('--winceTau', default=0.1, type=float, help='Contrastive loss temperature factor')
parser.add_argument('--btLam', default=0.005, type=float, help='Coefficient to apply to off-diagonal terms of BT loss')
parser.add_argument('--btLossType', default='bt', type=str, help='Method of calculating loss for off-diagonal terms')
parser.add_argument('--vicAlpha', default=25.0, type=float, help='Coefficient on variance loss term')
parser.add_argument('--vicBeta', default=25.0, type=float, help='Coefficient on invariance loss term')
parser.add_argument('--vicGamma', default=1.0, type=float, help='Coefficient on covariance loss term')
parser.add_argument('--mecEd2', default=0.06, type=float, help='Related to the coefficient applied to correlation matrix')
parser.add_argument('--mecTaylorTerms', default=2, type=int, help='Number of Taylor expansion terms to include in matrix logarithm approximation')
parser.add_argument('--aeLossType', default='mse', type=str, help='AE loss type to apply')
parser.add_argument('--aeLossFactor', default=0.25, type=float, help='Factor to apply to AE MSE loss -> (1 - factor) applied to SSL loss')

# Adversarial training parameters
parser.add_argument('--useAdvList', default=[False, False], type=list, help='List of Booleans to apply adversarial training for each view')
parser.add_argument('--keepStd', default=False, type=lambda x:bool(strtobool(x)), help='Boolean to train with the adversarial plus original images - increases batch size')
parser.add_argument('--advBatchSize', default=512, type=int, help='Batch size to use for adversarial training loader')
parser.add_argument('--advAlpha', default=1/255, type=float, help='PGD step size')
parser.add_argument('--advEps', default=8/255, type=float, help='PGD attack radius limit, measured in specified norm')
parser.add_argument('--advNorm', default=float('inf'), type=float, help='Norm type for measuring perturbation radius')
parser.add_argument('--advSteps', default=10, type=int, help='Number of PGD steps to take')
parser.add_argument('--advNoise', default=False, type=lambda x:bool(strtobool(x)), help='Boolean to use random initialization')
parser.add_argument('--advRestarts', default=1, type=int, help='Number of PGD restarts to search for best attack')

##################
# Misc Functions #
##################

def save_chkpt(prefix, epoch, encArch, cifarMod, encDim, prjHidDim, prjOutDim, prdDim, prdAlpha, prdEps, prdBeta, momEncBeta, applySG, decArch, model, optimizer):
    torch.save({'epoch': epoch, 'encArch': encArch, 'cifarMod': cifarMod, 'encDim': encDim, 'prjHidDim': prjHidDim, 'prjOutDim': prjOutDim,
                'prdDim': prdDim, 'prdAlpha': prdAlpha, 'prdEps': prdEps, 'prdBeta': prdBeta, 'momEncBeta': momEncBeta, 'applySG': applySG,
                'decArch': decArch, 'stateDict': model.state_dict(), 'optimStateDict': optimizer.state_dict()},
               'Trained_Models/{}_pt_{:04d}.pth.tar'.format(prefix, epoch))

def gather_tensors(outLen, tens):
    outTens = torch.zeros(outLen, tens.size(1), device=tens.get_device())
    torch.distributed.all_gather_into_tensor(outTens, tens)
    return outTens

###################
# Setup Functions #
###################

def main():

    args = parser.parse_args()

    #args.trainRoot = r'D:/ImageNet-100/Poisoned/TAP_100/train'
    #args.ptPrefix = 'TAP'
    #args.nEpochs = 200
    #args.batchSize = 32
    #args.initLR = 0.5
    #args.decArch = args.encArch

    if args.randSeed is not None:
        random.seed(args.randSeed)
        np.random.seed(args.randSeed)
        torch.manual_seed(args.randSeed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training'
                      'This will turn on the CUDNN deterministic setting, which can slow down your training'
                      'You may see unexpected behavior when restarting from checkpoints')

    if not args.multiprocDistrib:
        warnings.warn('You have disabled DDP - the model will train on 1 GPU without data parallelism')

    if args.multiprocDistrib and args.winceBeta > 0.0:
        warnings.warn('InfoNCE loss is not suited for DDP training on multiple GPU')
        # DDP splits data across multiple GPU and later averages the gradients across GPU
        # This behavior gives an incorrect estimate of InfoNCE loss due to its reliance on negative pair similarities
        # 1GPU: batch of 512 -> 2 augmented batches of 1024 -> each of 1024 samples has 1023 negatives
        # 4GPU: batch of 512, split 128/GPU -> 4 * 2 augmented batches of 256 -> each of 256 samples has 255 negatives
        # SimCLR disallows multi-GPU training entirely, only allowing parallelizing on TPUs
        # CLIP accepts that the batch size is split across GPUs and doesn't call their loss "InfoNCE"

    if args.decArch is not None and any(args.useAdvList):
        warnings.warn('Adversarial SSL does not incorporate AE architecture - may get bugs or strange results')

    assert args.sslLossType in ['wince', 'bt', 'vicreg', 'mec']
    assert args.aeLossType in ['mse']

    if not os.path.exists('Trained_Models'):
        os.mkdir('Trained_Models')

    # Infer learning rate
    args.initLR = args.initLR * args.batchSize / 256

    # Launch multiple (or single) distributed processes for main_worker function - will automatically assign GPUs
    if args.multiprocDistrib:
        args.nProcPerNode = torch.cuda.device_count()
        args.nProcs = args.nProcPerNode * args.nodeCount
        torch.multiprocessing.spawn(main_worker, nprocs=args.nProcs, args=(args,))
    # Launch one process for main_worker function
    else:
        main_worker(args.gpu, args)

######################
# Execution Function #
######################

def main_worker(gpu, args):

    # Replace initial GPU index with assigned one
    args.gpu = gpu
    print('- GPU {} online'.format(args.gpu))

    # Suppress printing if not master (gpu 0)
    if args.multiprocDistrib and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    print('- Using GPU {} as master process'.format(args.gpu))

    # Set up distributed training on backend
    # For multiprocessing distributed training, rank needs to be the global rank among all the processes
    if args.multiprocDistrib:
        args.procRank = args.nodeRank * args.nProcPerNode + args.gpu
        torch.distributed.init_process_group(backend=args.distBackend, init_method=args.distURL, world_size=args.nProcs, rank=args.procRank)
        torch.distributed.barrier()

    print('- Defining dataset and loader')
    if args.trainLabels:
        if args.decArch is None:
            trainDataset = datasets.ImageFolder(args.trainRoot, CT.NTimesTransform(args.nAugs, CT.t_pretrain(args.cropSize)))
        else:
            trainDataset = datasets.ImageFolder(args.trainRoot, CT.NTimesTransform_2Parts(args.nAugs, CT.t_randcrop(args.cropSize), CT.t_toPIL_pretrain()))
    else:
        trainDataset = CD.no_label_dataset(args.trainRoot, CT.NTimesTransform(args.nAugs, CT.t_pretrain(args.cropSize)))
    if args.multiprocDistrib:
        trainSampler = torch.utils.data.distributed.DistributedSampler(trainDataset)
        # When using single GPU per process and per DDP, need to divide batch size and workers based on nGPUs
        args.batchSize = int(args.batchSize / args.nProcs)
        args.advBatchSize = int(args.advBatchSize / args.nProcs)
        args.workers = int(args.workers / args.nProcs)
    else:
        trainSampler = None
    # Note that DistributedSampler automatically shuffles dataset given the set_epoch() function during training
    trainDataLoader = torch.utils.data.DataLoader(trainDataset, batch_size=args.batchSize, shuffle=(trainSampler is None),
                                                  num_workers=args.workers, pin_memory=True, sampler=trainSampler, drop_last=True)

    print('- Instantiating new model with {} backbone'.format(args.encArch))
    model = SSLAE_Model.Base_Model(args.encArch, args.cifarMod, args.encDim, args.prjHidDim, args.prjOutDim, args.prdDim,
                                 args.prdAlpha, args.prdEps, args.prdBeta, args.momEncBeta, args.applySG, args.decArch)

    print('- Setting up model on single/multiple devices')
    if args.multiprocDistrib:
        # Convert BN to SyncBN
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # For multiproc distributed, DDP constructor should set the single device scope - otherwises uses all available
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
    if args.sslLossType == 'wince':
        sslLossFn = SSLAE_Loss.Weighted_InfoNCE_Loss(args.symmetrizeLoss, args.winceBeta, args.winceTau)
    elif args.sslLossType == 'bt':
        sslLossFn = SSLAE_Loss.Barlow_Twins_Loss(args.symmetrizeLoss, args.btLam, args.btLossType)
    elif args.sslLossType == 'vicreg':
        sslLossFn = SSLAE_Loss.VICReg_Loss(args.symmetrizeLoss, args.vicAlpha, args.vicBeta, args.vicGamma)
    #elif args.sslLossType == 'mec':
    #    sslLossFn = SSLAE_Model.MEC_Loss(args.symmetrizeLoss, args.mecEd2, args.mecTaylorTerms)
    if args.decArch is not None and args.aeLossType == 'mse':
        aeLossFn = SSLAE_Loss.MSE_Loss(args.symmetrizeLoss)

    # Instantiate custom optimizer that skips momentum encoder and applies decay
    print('- Instantiating optimizer')
    if args.multiprocDistrib:
        optimParams = [{'params': model.module.encoder.parameters(), 'decayLR': args.decayEncLR},
                       {'params': model.module.projector.parameters(), 'decayLR': args.decayEncLR},
                       {'params': model.module.predictor.parameters(), 'decayLR': args.decayPrdLR}]
        if args.decArch is not None:
            optimParams.append({'params': model.module.decoder.parameters(), 'decayLR': args.decayEncLR})
    else:
        optimParams = [{'params': model.encoder.parameters(), 'decayLR': args.decayEncLR},
                       {'params': model.projector.parameters(), 'decayLR': args.decayEncLR},
                       {'params': model.predictor.parameters(), 'decayLR': args.decayPrdLR}]
        if args.decArch is not None:
            optimParams.append({'params': model.decoder.parameters(), 'decayLR': args.decayEncLR})
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

    # Initialize probes for training metrics, checkpoint initial model, and start timer
    if args.runProbes:
        probes = CP.Pretrain_Probes()
    save_chkpt(args.ptPrefix, 0, args.encArch, args.cifarMod, args.encDim, args.prjHidDim, args.prjOutDim, args.prdDim,
              args.prdAlpha, args.prdEps, args.prdBeta, args.momEncBeta, args.applySG, args.decArch, model, optimizer)
    timeStart = time.time()

    print('- Beginning training')
    for epoch in range(args.startEpoch, args.nEpochs + 1):

        # Update sampler with current epoch - required to ensure shuffling works across devices
        if args.multiprocDistrib:
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

        # Set model to train and reset sum of losses for each epoch
        model.train()
        sumLoss = 0.0
        sumSSLLoss = 0.0
        sumAELoss = 0.0

        for batchI, batch in enumerate(trainDataLoader):

            srcList = []
            for srcTens in batch[0][0]:
                srcList.append(srcTens.cuda(args.gpu, non_blocking=True))

            augList = []
            for augTens in batch[0][1]:
                augList.append(augTens.cuda(args.gpu, non_blocking=True))

            if any(args.useAdvList):

                _, advTensList = FGSM_PGD.ssl_pgd(model, sslLossFn, augList, args.useAdvList, args.advAlpha, args.advEps, args.advNorm,
                                                  args.advRestarts, args.advSteps, args.advBatchSize, targeted=False, rand_init=args.advNoise)

                for i in range(len(advTensList)):
                    if args.keepStd:
                        augList[i] = torch.cat((augList[i], advTensList[i].detach()), dim=0).cuda(args.gpu, non_blocking=True)
                    else:
                        augList[i] = advTensList[i].detach()

            # Reset outputs lists
            outList = []
            recList = []

            # Loop through each of the augs and create a list of results
            for augTens in augList:

                # Get input tensor, push through model, and append to output list
                p, z, r, mz, xd = model(augTens)
                outList.append([p, z, r, mz])
                recList.append(xd)

            # Calculate loss
            sslLossVal = sslLossFn.forward(outList)
            if args.decArch is not None:
                aeLossVal = aeLossFn.forward(recList, srcList)
                lossVal = (1 - args.aeLossFactor) * sslLossVal + args.aeLossFactor * aeLossVal
            else:
                aeLossVal = torch.tensor([0.])
                lossVal = sslLossVal

            # Backpropagate
            model.zero_grad()  # momenc and optPrd not included in optimizer, but they don't use grads - this should be redundant
            optimizer.zero_grad()
            lossVal.backward()
            optimizer.step()

            # Update momentum encoder
            if args.momEncBeta > 0.0:
                if args.multiprocDistrib:
                    model.module.update_momentum_network()
                else:
                    model.update_momentum_network()

            # Keep running sum of loss
            sumLoss += lossVal.detach()
            sumSSLLoss += sslLossVal.detach()
            sumAELoss += aeLossVal.detach()

        print('Epoch: {} / {} | Time: {:0.2f} | Avg Loss: {:0.4f} | Avg SSL Loss: {:0.4f} | Avg AE Loss: {:0.4f}'
              .format(epoch, args.nEpochs, time.time() - timeStart, sumLoss / (batchI + 1), sumSSLLoss / (batchI + 1), sumAELoss / (batchI + 1)))

        # Track record metrics while running
        if args.runProbes:

            # Ensures that BN and momentum BN running stats don't update
            model.eval()

            # If data was split out for DDP, then combine output tensors for analysis
            if args.multiprocDistrib and args.nProcs > 1:
                outLen = args.batchSize * args.nProcs
                p1 = gather_tensors(outLen, outList[0][0].detach())
                z1 = gather_tensors(outLen, outList[0][1].detach())
                r1 = gather_tensors(outLen, outList[0][2].detach())
                r2 = gather_tensors(outLen, outList[1][2].detach())
                mz2 = gather_tensors(outLen, outList[1][3].detach())
            else:
                p1 = outList[0][0].detach()
                z1 = outList[0][1].detach()
                r1 = outList[0][2].detach()
                r2 = outList[1][2].detach()
                mz2 = outList[1][3].detach()

            # Note that p1, z1, and mz2 are L2 normd, as SimSiam, BYOL, InfoNCE, and MEC use L2 normalized encodings
            # This is taken care of in loss functions, but I have to do it explicitly here
            # This probe update is inaccurate for softmax-normalized encs (DINO, SwAV) or batch normalized encs (Barlow Twins)
            probes.update_probes(epoch, lossVal.detach(), sslLossVal.detach(), aeLossVal.detach(),
                                 (p1 / torch.linalg.vector_norm(p1, dim=-1, keepdim=True)).detach(),
                                 (z1 / torch.linalg.vector_norm(z1, dim=-1, keepdim=True)).detach(),
                                 r1.detach(), r2.detach(),
                                 (mz2 / torch.linalg.vector_norm(mz2, dim=-1, keepdim=True)).detach())

        # Checkpoint model
        if epoch in [1, 10] or (epoch <= 200 and epoch % 20 == 0) or (epoch > 200 and epoch % 50 == 0):
            save_chkpt(args.ptPrefix, epoch, args.encArch, args.cifarMod, args.encDim, args.prjHidDim, args.prjOutDim,
                       args.prdDim, args.prdAlpha, args.prdEps, args.prdBeta, args.momEncBeta, args.applySG, args.decArch, model, optimizer)

    # Postprocessing and outputs

    if args.runProbes and (args.gpu == 0 or not args.multiprocDistrib):

        #probes.plot_probes()

        epochList = list(range(1, args.nEpochs + 1))

        import csv
        writer = csv.writer(open('Pretrain_Output.csv', 'w', newline=''))
        writer.writerow(['Epoch'] + epochList)
        writer.writerow(['Loss'] + [probes.lossProbe.storeList[epIdx - 1] for epIdx in epochList])
        writer.writerow(['SSL Loss'] + [probes.sslLossProbe.storeList[epIdx - 1] for epIdx in epochList])
        writer.writerow(['AE Loss'] + [probes.aeLossProbe.storeList[epIdx - 1] for epIdx in epochList])
        writer.writerow(['R1-R2 Sim'] + [probes.r1r2AugSimProbe.storeList[epIdx - 1] for epIdx in epochList])
        writer.writerow(['R1 SelfSim'] + [probes.r1AugSimProbe.storeList[epIdx - 1] for epIdx in epochList])
        writer.writerow(['R1 Var'] + [probes.r1VarProbe.storeList[epIdx - 1] for epIdx in epochList])
        writer.writerow(['R1 Corr Str'] + [probes.r1CorrStrProbe.storeList[epIdx - 1] for epIdx in epochList])
        writer.writerow(['R1 E-Rank'] + [probes.r1EigERankProbe.storeList[epIdx - 1] for epIdx in epochList])
        writer.writerow(['P1 Entropy'] + [probes.p1EntropyProbe.storeList[epIdx - 1] for epIdx in epochList])
        writer.writerow(['MZ2 Entropy'] + [probes.mz2EntropyProbe.storeList[epIdx - 1] for epIdx in epochList])
        writer.writerow(['MZ2-P1 KL Div'] + [probes.mz2p1KLDivProbe.storeList[epIdx - 1] for epIdx in epochList])
        writer.writerow(['P1 E-Rank'] + [probes.p1EigERankProbe.storeList[epIdx - 1] for epIdx in epochList])
        writer.writerow(['Z1 E-Rank'] + [probes.z1EigERankProbe.storeList[epIdx - 1] for epIdx in epochList])
        writer.writerow(['MZ2 E-Rank'] + [probes.mz2EigERankProbe.storeList[epIdx - 1] for epIdx in epochList])
        writer.writerow(['P1-Z1 Eig Align'] + [probes.p1z1EigAlignProbe.storeList[epIdx - 1] for epIdx in epochList])
        writer.writerow(['P1-MZ2 Eig Align'] + [probes.p1mz2EigAlignProbe.storeList[epIdx - 1] for epIdx in epochList])
        writer.writerow(['P1 Eig'] + np.log(probes.p1EigProbe.storeList[-1]).tolist())
        writer.writerow(['MZ2 Eig'] + np.log(probes.mz2EigProbe.storeList[-1]).tolist())
        writer.writerow(['R1 Eig'] + np.log(probes.r1EigProbe.storeList[-1]).tolist())

if __name__ == '__main__':
    main()