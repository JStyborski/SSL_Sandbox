import argparse
import builtins
from distutils.util import strtobool
import math
import os
import random
import time
import warnings

import torch
from torch import nn
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets

import SSL_Model
import Utils.Custom_Transforms as CT
from Adversarial import FGSM_PGD

# CUDNN automatically searches for the best algorithm for processing a given model/optimizer/dataset
cudnn.benchmark = True

###############
# User Inputs #
###############

parser = argparse.ArgumentParser()

# Run processing parameters
parser.add_argument('--useDDP', default=False, type=lambda x:bool(strtobool(x)), help='Strategy to launch on single/multiple GPU')
parser.add_argument('--gpu', default=0, type=int, help='GPU ID to use for training (single GPU)')
parser.add_argument('--nodeCount', default=1, type=int, help='Number of nodes/servers to use for distributed training')
parser.add_argument('--nodeRank', default=0, type=int, help='Global rank of nodes/servers')
parser.add_argument('--distURL', default='tcp://127.0.0.1:2345', type=str, help='URL for distributed training setup')
parser.add_argument('--distBackend', default='nccl', type=str, help='Distributed backend method')
parser.add_argument('--workers', default=4, type=int, help='Total number of data-loading workers')
parser.add_argument('--randSeed', default=None, type=int, help='RNG initial set point')

# Dataset parameters
parser.add_argument('--ptFile', default='', type=str, help='Single pretrained model to train - if empty, finetunes all in ptDir')
parser.add_argument('--ptDir', default='Trained_Models', type=str, help='Folder containing pretrained models')
parser.add_argument('--trainRoot', default='', type=str, help='Training dataset root directory')
parser.add_argument('--ftPrefix', default='', type=str, help='Prefix to add to finetuned file name')
parser.add_argument('--ftType', default='lp', type=str, help='Type of finetuning to apply - linear probe or finetune')
parser.add_argument('--cropSize', default=224, type=int, help='Crop size to use for input images')
parser.add_argument('--nAugs', default=1, type=int, help='Number of augmentations to apply to each batch')

# Training parameters
parser.add_argument('--nEpochs', default=100, type=int, help='Number of epochs to run')
parser.add_argument('--batchSize', default=512, type=int, help='Data loader batch size')
parser.add_argument('--nBatches', default=1e10, type=int, help='Maximum number of batches to run per epoch')
parser.add_argument('--momentum', default=0.9, type=float, help='SGD momentum value')
parser.add_argument('--weightDecay', default=0, type=float, help='SGD weight decay value')
parser.add_argument('--initLR', default=5.0, type=float, help='SGD initial learning rate')
parser.add_argument('--useLARS', default=False, type=lambda x:bool(strtobool(x)), help='Boolean to apply LARS optimizer')
parser.add_argument('--decayLR', default='stepdn', type=str, help='Learning rate decay method')
parser.add_argument('--decaySteps', default=[60, 75, 90], type=list, help='Steps at which to apply stepdn decay')
parser.add_argument('--decayFactor', default=0.2, type=float, help='Factor by which to multiply LR at step down')

# Adversarial training parameters
parser.add_argument('--useAdv', default=False, type=lambda x:bool(strtobool(x)), help='Boolean to apply adversarial training')
parser.add_argument('--keepStd', default=False, type=lambda x:bool(strtobool(x)), help='Boolean to train with the adversarial plus original images - increases batch size')
parser.add_argument('--advAlpha', default=1/255, type=float, help='PGD step size')
parser.add_argument('--advEps', default=8/255, type=float, help='PGD attack radius limit, measured in specified norm')
parser.add_argument('--advNorm', default=float('inf'), type=float, help='Norm type for measuring perturbation radius')
parser.add_argument('--advRestarts', default=1, type=int, help='Number of PGD restarts to search for best attack')
parser.add_argument('--advSteps', default=10, type=int, help='Number of PGD steps to take')
parser.add_argument('--advBatchSize', default=512, type=int, help='Batch size to use for adversarial training loader')
parser.add_argument('--advNoise', default=False, type=lambda x:bool(strtobool(x)), help='Boolean to use random initialization')
parser.add_argument('--advNoiseMag', default=None, type=float, help='Magnitude of noise to add to random start attack')
parser.add_argument('--advClipMin', default=None, type=int, help='Minimium value to clip adversarial inputs')
parser.add_argument('--advClipMax', default=None, type=int, help='Maximum value to clip adversarial inputs')

###################
# Setup Functions #
###################

def main():

    args = parser.parse_args()

    # HY Adversarial Training Overwrites
    #args.ptDir = 'Trained_Models'
    #args.trainRoot = r'D:/ImageNet-100/Poisoned/CUDA_100/train'
    #args.ftPrefix = 'CUDA'
    #args.ftType = 'ft'
    #args.batchSize = 128
    #args.weightDecay = 0.0005
    #args.initLR = 0.2
    #args.decayLR = 'cosdn'
    #args.decaySteps = [40, 80]
    #args.decayFactor = 0.1
    #args.useAdv = True
    #args.advBatchSize = 128
    #args.advAlpha = 0.6/255
    #args.advEps = 4/255
    #args.advNoise = True
    #args.advClipMin = 0.0
    #args.advClipMax = 1.0

    if args.randSeed is not None:
        random.seed(args.randSeed)
        torch.manual_seed(args.randSeed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training'
                      'This will turn on the CUDNN deterministic setting, which can slow down your training'
                      'You may see unexpected behavior when restarting from checkpoints')

    if not args.useDDP:
        warnings.warn('You have disabled DDP - the model will train on 1 GPU without data parallelism')

    if args.ptFile == '':
        # List of pretrained models as any model with _pt_ in the name
        args.ptList = sorted([args.ptDir + '/' + stateFile for stateFile in os.listdir(args.ptDir)
                         if ('_pt_' in stateFile and '_lp_' not in stateFile and '_ft_' not in stateFile)])
    else:
        # Single file
        args.ptList = [args.ptDir + '/' + args.ptFile]

    # Infer learning rate
    args.initLR = args.initLR * args.batchSize / 256

    # Launch multiple (or single) distributed processes for main_worker function - will automatically assign GPUs
    if args.useDDP:
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
    print('GPU {} online'.format(args.gpu))

    # Suppress printing if not master (gpu 0)
    if args.useDDP and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    print('Using GPU {} as master process'.format(args.gpu))

    # Set up distributed training on backend
    # For multiprocessing distributed training, rank needs to be the global rank among all the processes
    if args.useDDP:
        args.procRank = args.nodeRank * args.nProcPerNode + args.gpu
        torch.distributed.init_process_group(backend=args.distBackend, init_method=args.distURL, world_size=args.nProcs, rank=args.procRank)
        torch.distributed.barrier()

    print('Defining dataset and loader')
    trainDataset = datasets.ImageFolder(args.trainRoot, CT.NTimesTransform(args.nAugs, CT.t_finetune(args.cropSize)))
    if args.useDDP:
        trainSampler = torch.utils.data.distributed.DistributedSampler(trainDataset)
        # When using single GPU per process and per DDP, need to divide batch size and workers based on nGPUs
        args.batchSize = int(args.batchSize / args.nProcs)
        args.advBatchSize = int(args.advBatchSize / args.nProcs)
        args.workers = int(args.workers / args.nProcs)
    else:
        trainSampler = None
    # Note that DistributedSampler automatically shuffles dataset given the set_epoch() function during training
    trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size=args.batchSize, shuffle=(trainSampler is None),
                                              num_workers=args.workers, pin_memory=True, sampler=trainSampler, drop_last=True)

    args.nClasses = len(trainDataset.classes)

    for stateFile in args.ptList:

        print('\nFinetuning ' + stateFile)

        # Get finetune file name and check if it already exists
        ftFile = stateFile[:-8] + '_' + args.ftPrefix + '_' + args.ftType + '_{:04d}'.format(args.nEpochs) + '.pth.tar'
        if os.path.exists(ftFile):
            print('- {} already exists, skipping'.format(ftFile))
            continue

        # Load saved state
        stateDict = torch.load(stateFile, map_location='cuda:{}'.format(args.gpu))

        print('- Instantiating new model with {} backbone'.format(stateDict['encArch']))
        model = SSL_Model.Base_Model(stateDict['encArch'], stateDict['rnCifarMod'], stateDict['vitPPFreeze'], stateDict['prjArch'],
                                       stateDict['prjHidDim'], stateDict['prjBotDim'], stateDict['prjOutDim'], stateDict['prdHidDim'],
                                       None, 0.3, 0.5, 0.0, True)

        # If a stateDict key has "module" in (from running parallel), create a new dictionary with the right names
        for key in list(stateDict['stateDict'].keys()):
            if key.startswith('module.'):
                stateDict['stateDict'][key[7:]] = stateDict['stateDict'][key]
                del stateDict['stateDict'][key]

        print('- Loading model weights from {}'.format(stateFile))
        model.load_state_dict(stateDict['stateDict'], strict=False)

        # Freeze all layers (though predictor will later be replaced and trainable)
        if args.ftType == 'lp':
            for param in model.parameters(): param.requires_grad = False

        # Replace the projector with identity and the predictor with linear classifier
        encDim = model.encoder.inplanes if 'resnet' in stateDict['encArch'] else model.encoder.num_features
        model.projector = nn.Identity()
        model.predictor = nn.Linear(encDim, args.nClasses)

        print('- Setting up model on single/multiple devices')
        if args.useDDP:
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

        # If linear probe, set model to eval mode (no BN updates). If finetuning, allow model to update
        model.eval() if args.ftType == 'lp' else model.train()

        print('- Instantiating loss and optimizer')
        crossEnt = nn.CrossEntropyLoss()
        if 'vit' in stateDict['encArch'].lower() and args.useDDP:
            optimizer = torch.optim.AdamW(params=model.module.parameters(), lr=args.initLR, weight_decay=args.weightDecay)
        elif 'vit' in stateDict['encArch'].lower():
            optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.initLR, weight_decay=args.weightDecay)
        elif args.useDDP:
            optimizer = torch.optim.SGD(params=model.module.parameters(), lr=args.initLR, momentum=args.momentum, weight_decay=args.weightDecay)
        else:
            optimizer = torch.optim.SGD(params=model.parameters(), lr=args.initLR, momentum=args.momentum, weight_decay=args.weightDecay)
        if args.useLARS:
            print("- Using LARS optimizer.")
            from Utils.Apex_LARC import LARC
            optimizer = LARC(optimizer=optimizer, trust_coefficient=0.001, clip=False)

        # Start timer
        timeStart = time.time()

        print('- Beginning training')
        for epoch in range(1, args.nEpochs + 1):

            # Update sampler with current epoch - required to ensure shuffling works across devices
            if args.useDDP:
                trainSampler.set_epoch(epoch)

            # If desired, adjust learning rate for the current epoch
            if args.decayLR == 'stepdn':
                if epoch in args.decaySteps:
                    curLR = args.initLR * args.decayFactor ** (args.decaySteps.index(epoch) + 1)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = curLR
            elif args.decayLR == 'cosdn':
                curLR = args.initLR * 0.5 * (1. + math.cos(math.pi * (epoch - 1) / args.nEpochs))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = curLR

            # Reset sum of losses for each epoch
            sumLossVal = 0.0
            nTrainCorrect = 0
            nTrainTotal = 0

            for batchI, batch in enumerate(trainLoader):

                # Get truth labels and reset predictions list
                pList = []
                truthTens = batch[1].cuda(args.gpu, non_blocking=True)

                # Loop through each of the augs and create a list of results
                for aug in batch[0]:

                    # Get input tensor
                    augTens = aug.cuda(args.gpu, non_blocking=True)

                    # Calculate (untargeted) adversarial samples from inputs and use for training
                    if args.useAdv:
                        _, advTens = FGSM_PGD.sl_pgd(model, nn.CrossEntropyLoss(reduction='none'), augTens, truthTens, args.advAlpha, args.advEps,
                                                     args.advNorm, args.advRestarts, args.advSteps, args.advBatchSize, outIdx=0, targeted=False,
                                                     randInit=args.advNoise, noiseMag=args.advNoiseMag, xMin=args.advClipMin, xMax=args.advClipMax)
                        if args.keepStd:
                            augTens = torch.cat((augTens, advTens.detach()), dim=0).cuda(args.gpu, non_blocking=True)
                        else:
                            augTens = advTens.detach()

                    # Run augmented data through SimSiam with linear classifier
                    p, _, _, _ = model(augTens)
                    pList.append(p)

                p = torch.concatenate(pList, dim=0).cuda(args.gpu, non_blocking=True)
                truthTens = truthTens.repeat(int(len(p) / len(truthTens)))

                # Calculate loss
                lossVal = crossEnt(p, truthTens)

                # Backpropagate
                optimizer.zero_grad()
                lossVal.backward()
                optimizer.step()

                # Keep running sum of loss
                sumLossVal += lossVal.detach().cpu().numpy()
                nTrainCorrect += torch.sum(torch.argmax(p.detach(), dim=1) == truthTens).cpu().numpy()
                nTrainTotal += len(truthTens)

                if batchI + 1 >= args.nBatches:
                    break

            avgLossVal = sumLossVal / (batchI + 1)
            clnTrainAcc = nTrainCorrect / nTrainTotal

            if epoch == 1 or epoch % 10 == 0:
                print('Epoch: {} / {} | Elapsed Time: {:0.2f} | Avg Loss: {:0.4f} | Avg Trn Acc: {:0.4f}'
                      .format(epoch, args.nEpochs, time.time() - timeStart, avgLossVal, clnTrainAcc))

        # Save out finetune model
        torch.save(model.state_dict(), ftFile)
        print('Saved Model {}'.format(ftFile))

if __name__ == '__main__':
    main()
