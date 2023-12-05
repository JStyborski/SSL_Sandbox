import torch
from torch import nn
from torchmetrics.functional.regression import cosine_similarity
from torchmetrics.functional.pairwise import pairwise_cosine_similarity


class Weighted_InfoNCE_Loss:
    # Implements weighted InfoNCE loss as in https://arxiv.org/abs/2006.07733 and based on https://arxiv.org/abs/2002.05709
    # Data is automatically L2 normalized in the encoding dimension in the cosine_similarity and pairwise_cosine_similarity functions
    # This algorithm (plus the SSL_Model) is pretty faithful for BYOL, SimSiam, MoCoV3, and SimCLR. Differences are noted below:
    # - BYOL actually uses MSE loss. However for normalized outputs, minimizing MSE loss minimizes a scaled cosine similarity
    # - MoCoV3 actually uses CE loss, though I think their formulation is equivalent to InfoNCE, since InfoNCE is technically a CE loss

    def __init__(self, symmetrizeLoss=True, winceBeta=0.0, winceTau=0.1):
        """
        :param symmetrizeLoss: [Bool] - Boolean to symmetrize loss
        :param winceBeta: [float] - Coefficient weight for contrastive loss term (0 gives SimSiam loss, 1 gives InfoNCE)
        :param winceTau: [float] - Temperature term used in InfoNCE loss
        """
        self.symmetrizeLoss = symmetrizeLoss
        self.winceBeta = winceBeta
        self.winceTau = winceTau

    def __call__(self, outList):

        # Track loss across all views
        totalLoss = 0

        # Loop through each view
        for i in range(len(outList)):

            # Initialize loss for ith view
            ithLoss = 0.

            # Calculate positive pairwise similarity loss for the ith view
            for j in range(len(outList)):
                if i == j: continue
                ithLoss += -1.0 * cosine_similarity(outList[i][0], outList[j][-1], reduction='mean')

            # Calculate negative similarity loss (InfoNCE denominator) - This formulation is best seen in the BYOL paper
            if self.winceBeta > 0.0:

                # Calculate the pairwise cosine similarity matrices for same-view similarity
                nss = pairwise_cosine_similarity(outList[i][0], outList[i][0])
                nss.fill_diagonal_(0.0)  # Self-similarity within view skips itself (similarity = 1 otherwise)

                # Concatenate a tensor for all differing views and calculate pairwise similarity to ith view
                ndsList = []
                for j in range(len(outList)):
                    if i == j: continue
                    ndsList.append(outList[j][-1])
                ndsTens = torch.concatenate(ndsList, dim=0).to(nss.device)
                nds = pairwise_cosine_similarity(outList[i][0], ndsTens)

                # To implement InfoNCE contrastive terms correctly, first reweight the positive similarity loss by Tau
                # Then add the contrastive loss terms for the 2 pairwise similarity tensors
                ithLoss /= self.winceTau
                ithLoss += self.winceBeta * (torch.exp(nss / self.winceTau).sum(dim=-1) +
                                             torch.exp(nds / self.winceTau).sum(dim=-1)).log().mean()

            totalLoss += ithLoss

            # If not symmetrizing loss, break after calculating loss for first component
            if not self.symmetrizeLoss:
                break

        totalLoss /= (i + 1)

        return totalLoss


class Barlow_Twins_Loss:
    # Implements Barlow Twins loss as in https://arxiv.org/abs/2103.03230
    # Similar, but not identical, formulation in https://arxiv.org/abs/2205.11508
    # Also includes an optional loss modification as in https://arxiv.org/abs/2104.13712
    # Note that there are multiple methods proposed for normalizing encodings before BT loss - I include options 1 and 3
    # 1) Batch normalization across batch dimension, as in original BT paper
    #    - This tends to be redundant, since the final projector layer typically uses non-affine batch norm anyway
    # 2) L2 normalization across batch dimension, and then modified cross-correlation (supposedly equivalent to BT, but didn't match when I checked)
    # 3) L2 normalization across encoding dimension, as in https://arxiv.org/abs/2112.05141 (admittedly different but found no performance loss)

    def __init__(self, symmetrizeLoss=False, btLam=0.005, btLossType='bt', btNormType='bn'):
        """
        :param symmetrizeLoss: [Bool] - Boolean to symmetrize loss
        :param btLam: [float] - Coefficient for off-diagonal loss terms - Tsai 2021 paper on BT + HSIC recommends 1/d
        :param btLossType: ['bt' or 'hsic'] - Method of calculating loss, particularly differs for off-diagonal terms
        :param btNormType: ['bn' or 'l2'] - Method of normalizing data
        """
        self.symmetrizeLoss = symmetrizeLoss
        self.btLam = btLam  # Note that the Tsai 2021 paper on HSIC + BT recommends setting btLam to 1/d
        self.btLossType = btLossType
        self.btNormType = btNormType
        assert self.btLossType in ['bt', 'hsic']
        assert self.btNormType in ['bn', 'l2']

    def __call__(self, outList):

        # Barlow Twins loss is based on cross-correlation between 2 views only
        # If you want to use more views, there are ways to hack BT loss by inserting other views into the 2 views - See 2022 Balestriero paper
        assert len(outList) == 2

        # Track loss across all views
        totalLoss = 0

        # Loop through each view
        for i in range(len(outList)):

            # Since only 2 views allowed, if i=1, then j=0, and vice-versa
            j = 1 - i

            # Batch normalize each batch
            if self.btNormType == 'bn':
                batch1 = (outList[i][0] - outList[i][0].mean(dim=0, keepdim=True)) / outList[i][0].std(dim=0, unbiased=True, keepdim=True)
                batch2 = (outList[j][-1] - outList[j][-1].mean(dim=0, keepdim=True)) / outList[j][-1].std(dim=0, unbiased=True, keepdim=True)
            else:
                batch1 = torch.nn.functional.normalize(outList[i][0], dim=1)
                batch2 = torch.nn.functional.normalize(outList[j][-1], dim=1)

            # Calculate cross-correlation
            crossCorr = 1.0 / outList[i][0].size(0) * torch.tensordot(batch1, batch2, dims=([0], [0]))

            # Calculate Barlow Twins cross-correlation loss
            if self.btLossType == 'bt':
                totalLoss += (torch.diag(crossCorr) - 1).square().sum() + 2 * self.btLam * torch.triu(crossCorr, 1).square().sum()
            else:
                totalLoss += (torch.diag(crossCorr) - 1).square().sum() + 2 * self.btLam * (torch.triu(crossCorr, 1) + 1).square().sum()

            # If not symmetrizing loss, break after calculating loss for first component
            if not self.symmetrizeLoss:
                break

        totalLoss /= (i + 1)

        return totalLoss


class VICReg_Loss:
    # Implements VICReg loss from https://arxiv.org/abs/2105.04906 and https://arxiv.org/abs/2205.11508
    # The variance loss hinge value is technically a variable, though in the VICReg paper and here, it is set to 1
    # If the projector outputs are batch normalized and the variance loss hinge value is 1, then variance loss is 0
    # - i.e., Cannot use batchnorm on projector outputs if you want the variance loss to do anything
    # https://arxiv.org/abs/2206.02574 claims that VICReg uses centering and batchnorm, and considers L2 normalization on the embedding dimension
    # https://arxiv.org/abs/2112.05141 also proposes L2 normalization on the embedding dimension
    # Looking at the code in the original VICReg paper and the Balestriero paper:
    # - Centering is only used to calculate the covariance matrix, and is not used in the invariance loss
    # - BatchNorm is not applied in VICReg by default, the variance loss is used to regularize the variance values
    # - The original VICReg paper (Appendix) tests L2 normalization on embeddings and finds worse performance - therefore I don't implement it

    def __init__(self, symmetrizeLoss=False, vicAlpha=25.0, vicBeta=25.0, vicGamma=1.0):
        """
        :param symmetrizeLoss: [Bool] - Boolean to symmetrize loss
        :param vicAlpha: [float] - Coefficient on variance loss
        :param vicBeta: [float] - Coefficient on invariance loss
        :param vicGamma: [float] - Coefficient on covariance loss
        """
        self.symmetrizeLoss = symmetrizeLoss
        self.vicAlpha = vicAlpha
        self.vicBeta = vicBeta
        self.vicGamma = vicGamma

    def __call__(self, outList):

        # Initialize total loss across views and determine projection dimension
        totalLoss = 0
        d = outList[0][0].size(1)

        # Loop through each view
        for i in range(len(outList)):

            # Calculate ith view covariance and then the covariance and variance losses for it
            covI = torch.cov(outList[i][0].t())
            totalLoss += self.vicGamma * 2 / d * torch.triu(covI, 1).square().sum()
            totalLoss += self.vicAlpha * (1 - torch.diag(covI).clamp(1e-6).sqrt()).clamp(0).mean()

            for j in range(len(outList)):
                if i == j: continue

                # Calculate jth view covariance and then the covariance and variance losses for it
                covJ = torch.cov(outList[j][-1].t())
                totalLoss += self.vicGamma * 2 / d * torch.triu(covJ, 1).square().sum()
                totalLoss += self.vicAlpha * (1 - torch.diag(covJ).clamp(1e-6).sqrt()).clamp(0).mean()

                # Invariance loss between ith and jth views
                totalLoss += self.vicBeta * torch.linalg.vector_norm(outList[i][0] - outList[j][-1], ord=2, dim=1).square().mean()

            # If not symmetrizing loss, break after calculating loss for first component
            if not self.symmetrizeLoss:
                break

        totalLoss /= (i + 1)

        return totalLoss


class MEC_Loss:
    # Implements Maximum Entropy Coding loss as in https://arxiv.org/abs/2210.11464
    # MEC loss can be formulated either batch-wise or feature-wise
    # This is due to the relationship between the Gram Matrix and covariance matrix, also explored here: https://arxiv.org/abs/2206.02574

    def __init__(self, symmetrizeLoss=False, mecEd2=0.06, mecTaylorTerms=2):
        """
        :param symmetrizeLoss: [Bool] - Boolean to symmetrize loss
        :param mecEd2: [float] - lam = d / (b * eps^2) = 1 / (b * ed2), so ed2 = eps^2 / d. Authors use ed2 = [0.01, 0.12]
        :param mecTaylorTerms: [int] - Number of terms to use in Taylor expansion of the matrix logarithm
        """
        self.symmetrizeLoss = symmetrizeLoss
        self.mecEd2 = mecEd2
        self.mecTaylorTerms = mecTaylorTerms

    def __call__(self, outList):

        # MEC loss is based on Gram matrix or cross-correlation between 2 views only
        assert len(outList) == 2

        # Initialize total loss across views and determine projection dimension
        totalLoss = 0
        b, d = outList[0][0].size()

        # Calculate mu and lam coefficients
        mu = (b + d) / 2
        lam = 1 / (b * self.mecEd2)

        # Loop through each view
        for i in range(len(outList)):

            # Since only 2 views allowed, if i=1, then j=0, and vice-versa
            j = 1 - i

            # MEC assumes batches are L2 normalized along feature dimension
            batch1 = torch.nn.functional.normalize(outList[i][0], dim=1)
            batch2 = torch.nn.functional.normalize(outList[j][-1], dim=1)

            # Calculate correlation matrix and initialize the correlation exponential and Taylor sum
            corr = lam * torch.tensordot(batch1, batch2, dims=([-1], [-1]))  # [m x m] batch-wise correlation matrix
            #corr = lam * torch.tensordot(batch1, batch2, dims=([0], [0]))  # [d x d] feature-wise correlation matrix
            powerCorr = torch.eye(corr.size(0), device=torch.device(corr.get_device()))
            sumCorr = torch.zeros_like(corr)

            # Loop through Taylor terms and cumulatively add powers of the correlation matrix
            for k in range(self.mecTaylorTerms):
                powerCorr = torch.tensordot(powerCorr, corr, dims=([-1], [0]))
                sumCorr += (-1) ** k / (k + 1) * powerCorr

            # Calculate final loss value
            totalLoss += -1.0 * mu * torch.trace(sumCorr)

            # If not symmetrizing loss, break after calculating loss for first component
            if not self.symmetrizeLoss:
                break

        totalLoss /= (i + 1)

        return totalLoss


class DINO_Loss:
    # Implements DINO cross-entropy loss as in https://arxiv.org/abs/2104.14294
    # Due to the teacher network's reliance on the center vector, it expects that the projector output is not batch-normalized
    # In fact, in the ViT implementation, DINO uses no BatchNorm layers at all
    # The authors of the original DINO paper also find that L2 normalization in the projector head is crucial for stability (see appendix)

    def __init__(self, symmetrizeLoss=False, centerMom=0.99, studentTau=0.1, teacherTau=0.05):
        """
        :param symmetrizeLoss: [Bool] - Boolean to symmetrize loss
        :param centerMom: [float] - Momentum coefficient for teacher center vector
        :param studentTau: [float] - Temperature for student network (online) softmax
        :param teacherTau: [float] - Temperature for teacher network (target) softmax - teacher temperature should be lower than the student
        """
        self.symmetrizeLoss = symmetrizeLoss
        self.center = None
        self.centerMom = centerMom
        self.studentTau = studentTau
        self.teacherTau = teacherTau

    def __call__(self, outList):

        # Define centering tensor if previously undefined - original paper uses zeros initialization
        if self.center is None:
            self.center = torch.zeros(1, outList[0][0].size(-1), device=torch.device(outList[0][0].get_device()))

        # Initialize total loss across views
        totalLoss = 0

        # Loop through each view
        for i in range(len(outList)):

            # Calculate the student output probabilities, student temperature applied
            studentSM = torch.nn.functional.log_softmax(outList[i][0] / self.studentTau, dim=1)

            for j in range(len(outList)):
                if i == j: continue

                # Calculate the teacher output probabilities and corresponding CE loss - teacher is centered and has temperature applied
                teacherSM = torch.nn.functional.softmax((outList[j][-1] - self.center) / self.teacherTau, dim=1)
                totalLoss += -1.0 * (teacherSM * studentSM).sum(dim=-1).mean()

            # If not symmetrizing loss, break after calculating loss for first component
            if not self.symmetrizeLoss:
                break

        totalLoss /= (i + 1)

        # Update center using momentum
        teacherOut = torch.cat([outList[i][-1] for i in range(len(outList))], dim=0)
        self.center = self.centerMom * self.center + (1 - self.centerMom) * torch.mean(teacherOut, dim=0, keepdim=True)

        return totalLoss


class MSE_Loss:
    # Implements

    def __init__(self, symmetrizeLoss):
        self.symmetrizeLoss = symmetrizeLoss
        self.mse = nn.MSELoss()

    def __call__(self, recList, srcList):

        # Track loss across all views
        totalLoss = 0

        # Loop through each view
        for i in range(len(recList)):

            totalLoss += self.mse(recList[i], srcList[i])

            # If not symmetrizing loss, break after calculating loss for first component
            if not self.symmetrizeLoss:
                break

        totalLoss /= (i + 1)

        return totalLoss


class SeLa_Clusters:
    # Need to review and test. I don't even think the SK iter is right
    def __init__(self, skTau, skIter, ceTau):
        self.skTau = skTau
        self.skIter = skIter
        self.ceTau = ceTau
        self.softmax = nn.Softmax(dim=1)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, batch1Prd, batch1Prj, batch2Prj):
        P = torch.cat((batch1Prd, batch2Prj), dim=0)
        Q = P.detach()
        Q = torch.exp(Q / self.skTau)

        # Do Sinkhorn-Knopp iteration (taken from alg in DINO paper)
        for _ in range(self.skIter):
            # Divide by column sum (total weight per dimension (or cluster), across samples)
            Q /= torch.sum(Q, dim=0, keepdim=True)
            # Divide by row sum (total weight per sample (or assignment), across dimensions)
            Q /= torch.sum(Q, dim=1, keepdim=True)

        # Calculate the Q (labels) and P (model estimates) softmaxes
        #QSM = self.softmax(Q / self.ceTau)
        #PSM = self.logsoftmax(P / self.ceTau)

        # Calculate the cross entropy between Q and P
        #lossVal = -1.0 * (QSM * PSM).sum(dim=-1).mean()
        lossVal = -1.0 * (Q * torch.log(P)).sum(dim=-1).mean()

        return lossVal


class MCE:
    # Need to review and test out - unsure about the whole thing
    def __init__(self, alignLam=1):
        self.alignLam = alignLam

    def compute_cov(self, Z1, Z2, device):

        batchSize = Z1.size(0)
        cB = (torch.eye(batchSize) - 1 / batchSize * torch.ones(batchSize, batchSize)).to(device)
        cov = 1 / (batchSize - 1) * torch.tensordot(torch.tensordot(Z1, cB, dims=([0], [0])), Z2, dims=([-1], [0]))

        return cov

    def forward(self, batch1Prd, batch1Prj, batch2Prj):

        device = torch.device(batch1Prd.get_device()) if batch1Prd.get_device() > -1 else torch.device('cpu')

        iii = 1 / batch1Prd.size(1) * torch.eye(batch1Prd.size(1), device=device)

        xCov = self.compute_cov(batch1Prd, batch2Prj, device)
        unifL = -1.0 * torch.trace(torch.tensordot(iii, torch.log(xCov), dims=([-1], [0])))

        cov1 = self.compute_cov(batch1Prd, batch1Prd, device)
        cov2 = self.compute_cov(batch2Prj, batch2Prj, device)
        alignL = -1.0 * torch.trace(torch.tensordot(cov1, torch.log(cov2), dims=([-1], [0])))

        lossVal = unifL + self.alignLam * alignL

        return lossVal
