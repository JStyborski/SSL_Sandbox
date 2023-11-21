import torch
from torch import nn
from torchmetrics.functional.regression import cosine_similarity
from torchmetrics.functional.pairwise import pairwise_cosine_similarity


class Weighted_InfoNCE_Loss:
    # Implements weighted InfoNCE loss as in https://arxiv.org/abs/2006.07733 and based on https://arxiv.org/abs/2002.05709

    def __init__(self, symmetrizeLoss=True, winceBeta=0.0, winceTau=0.1):
        """
        :param symmetrizeLoss: [Bool] - Boolean to symmetrize loss
        :param winceBeta: [float] - Coefficient weight for contrastive loss term (0 gives SimSiam loss, 1 gives InfoNCE)
        :param winceTau: [float] - Temperature term used in InfoNCE loss
        """
        self.symmetrizeLoss = symmetrizeLoss
        self.winceBeta = winceBeta
        self.winceTau = winceTau

    def forward(self, outList):

        # Track loss across all views
        totalLoss = 0

        # Loop through each view
        for i in range(len(outList)):

            # Calculate positive pairwise similarity loss for the ith view
            for j in range(len(outList)):
                if i == j: continue
                ithLoss = -1.0 * cosine_similarity(outList[i][0], outList[j][-1], reduction='mean')

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

        totalLoss /= ((i + 1) * (len(outList) - 1))

        return totalLoss


class Barlow_Twins_Loss:
    # Implements Barlow Twins loss as in https://arxiv.org/abs/2103.03230 and https://arxiv.org/abs/2205.11508
    # Also includes an optional loss modification as in https://arxiv.org/abs/2104.13712
    # Should gather tensors across GPU for DDP?

    def __init__(self, symmetrizeLoss=False, btLam=0.005, btLossType='bt'):
        """
        :param symmetrizeLoss: [Bool] - Boolean to symmetrize loss
        :param btLam: [float] - Coefficient for off-diagonal loss terms - Tsai 2021 paper on BT + HSIC recommends 1/d
        :param btLossType: ['bt' or 'hsic'] - Method of calculating loss, particularly differs for off-diagonal terms
        """
        self.symmetrizeLoss = symmetrizeLoss
        self.btLam = btLam # Note that the Tsai 2021 paper on HSIC + BT recommends setting btLam to 1/d
        self.btLossType = btLossType
        assert self.btLossType in ['bt', 'hsic']

    def forward(self, outList):

        # Barlow Twins loss is based on cross-correlation between 2 views only
        # If you want to use more views, there are ways to hack BT loss by inserting other views into the 2 views
        # See 2022 Balestriero paper for an example
        assert len(outList) == 2

        # Track loss across all views
        totalLoss = 0

        # Loop through each view
        for i in range(len(outList)):

            # If i=1, then j=0, and vice-versa
            j = 1 - i

            # Batch normalize each batch
            # Note that the Balestriero 2023 implementation of BT doesn't center the batches
            outList[i][0] = (outList[i][0] - outList[i][0].mean(dim=0, keepdim=True)) / outList[i][0].std(dim=0, unbiased=True, keepdim=True)
            outList[j][-1] = (outList[j][-1] - outList[j][-1].mean(dim=0, keepdim=True)) / outList[j][-1].std(dim=0, unbiased=True, keepdim=True)

            # Calculate cross-correlation
            # Note that the Balestriero 2023 implementation of BT doesn't use the 1/N factor
            crossCorr = 1.0 / outList[i][0].size(0) * torch.tensordot(outList[i][0], outList[j][-1], dims=([0], [0]))

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
    # Need to double-check this implementation - compare with official, which uses centered encodings and gathered tensors
    # https://github.com/facebookresearch/vicreg

    def __init__(self, symmetrizeLoss, vicAlpha=25.0, vicBeta=25.0, vicGamma=1.0):
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

    def forward(self, outList):

        # Track loss across all views
        totalLoss = 0

        # Loop through each view
        for i in range(len(outList)):

            # Calculate ith view covariance and then the covariance and variance losses for it
            covI = torch.cov(outList[i][0])
            ithLoss = self.vicGamma * 2 / outList[i][0].size(1) * torch.triu(covI, 1).square().sum()
            ithLoss += self.vicAlpha * 1 / outList[i][0].size(1) * (1 - torch.diag(covI).clamp(1e-6).sqrt()).clamp(0).sum()

            for j in range(len(outList)):
                if i == j: continue

                # Calculate jth view covariance and then the covariance and variance losses for it
                covJ = torch.cov(outList[j][-1])
                ithLoss += self.vicGamma * 2 / outList[j][-1].size(1) * torch.triu(covJ, 1).square().sum()
                ithLoss += self.vicAlpha * 1 / outList[j][-1].size(1) * (1 - torch.diag(covJ).clamp(1e-6).sqrt()).clamp(0).sum()

                # Invariance loss between ith and jth views
                ithLoss += self.vicBeta * 1 / outList[i][0].size(0) \
                           * torch.linalg.vector_norm(outList[i][0] - outList[j][-1], ord=2, dim=1).square().sum()

            totalLoss += ithLoss

            # If not symmetrizing loss, break after calculating loss for first component
            if not self.symmetrizeLoss:
                break

        totalLoss /= (i + 1)

        return totalLoss


class MEC_Loss:
    # Implements Maximum Entropy Coding loss as in https://arxiv.org/abs/2210.11464

    def __init__(self, mecEd2=0.06, mecTaylorTerms=2):
        """
        :param mecEd2: [float] - lam = d / (m * eps^2) = 1 / (m * ed2), so ed2 = eps^2 / d. Authors use ed2 = [0.01, 0.12]
        :param mecTaylorTerms: [int] - Number of terms to use in Taylor expansion of the matrix logarithm
        """
        self.mecEd2 = mecEd2
        self.mecTaylorTerms = mecTaylorTerms

    def forward(self, batch1Prd, batch2Prj):

        # Ensure batches are L2 normalized along feature dimension
        batch1 = batch1Prd / torch.norm(batch1Prd, p=2, dim=1, keepdim=True)
        batch2 = batch2Prj / torch.norm(batch2Prj, p=2, dim=1, keepdim=True)

        # Calculate mu and lam coefficients
        mu = (batch1.size(0) + batch1.size(1)) / 2
        lam = 1 / (batch1.size(0) * self.mecEd2)

        # Calculate correlation matrix and initialize the correlation exponential and Taylor sum
        corr = lam * torch.tensordot(batch1, batch2, dims=([-1], [-1])) # [m x m] batch-wise correlation matrix
        #corr = lam * torch.tensordot(batch1, batch2, dims=([0], [0])) # [d x d] feature-wise correlation matrix
        powerCorr = torch.eye(corr.size(0), device=torch.device(batch1Prd.get_device()))
        sumCorr = torch.zeros_like(corr)

        # Loop through Taylor terms and cumulatively add powers of the correlation matrix
        for k in range(self.mecTaylorTerms):
            powerCorr = torch.tensordot(powerCorr, corr, dims=([-1], [0]))
            sumCorr += (-1) ** k / (k + 1) * powerCorr

        # Calculate final loss value
        lossVal = -1.0 * mu * torch.trace(sumCorr)

        return lossVal


class MSE_Loss:
    # Implements

    def __init__(self, symmetrizeLoss):
        self.symmetrizeLoss = symmetrizeLoss
        self.mse = nn.MSELoss()

    def forward(self, recList, srcList):

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


class DINO_CrossEnt:
    # Need to review this whole thing. It doesn't work. Are the batches supposed to be normalized?
    def __init__(self, centerInit='zero', centerMom=0.99, studentTau=0.1, teacherTau=0.05):
        self.centerInit = centerInit
        self.center = None
        self.centerMom = centerMom
        self.studentTau = studentTau
        self.teacherTau = teacherTau
        self.softmax = nn.Softmax(dim=1)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, batch1Prd, batch1Prj, batch2Prj):

        # Define centering tensor if previously undefined
        # Original paper uses zeros, but initializing as teacherBatch may be more stable
        if self.center is None:
            device = torch.device(batch1Prd.get_device()) if batch1Prd.get_device() > -1 else torch.device('cpu')
            if self.centerInit == 'zero':
                self.center = torch.zeros(1, batch2Prj.size(-1)).to(device)
            elif self.centerInit == 'teacher':
                self.center = batch2Prj.clone().to(device)

        # Calculate the teacher and student output probabilities, teacher is centered, both have temperature applied
        teacherSM = self.softmax((batch2Prj - self.center) / self.teacherTau)
        studentSM = self.logsoftmax(batch1Prd / self.studentTau)

        # Calculate the cross entropy between student and teacher
        lossVal = -1.0 * (teacherSM * studentSM).sum(dim=-1).mean()

        return lossVal

    def update_center(self, batchData):

        # Update center using momentum
        self.center = self.centerMom * self.center + (1 - self.centerMom) * torch.mean(batchData, dim=0, keepdim=True)


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
