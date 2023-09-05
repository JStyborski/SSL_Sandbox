import copy
import math
import torch
from torch import nn
from torchmetrics.functional.pairwise import pairwise_cosine_similarity
import torchvision.models as models


# Function to reinitialize all resettable weights of a given model
def reset_model_weights(layer):
    if hasattr(layer, 'reset_parameters'):
        layer.reset_parameters()
    elif hasattr(layer, 'children'):
        for child in layer.children():
            reset_model_weights(child)


class ResNetBlock(nn.Module):
    def __init__(self, inCh, outCh, stride, expansion):
        super(ResNetBlock, self).__init__()

        if stride == 1:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(inCh, outCh * expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outCh * expansion)
            )
        self.conv1 = nn.Conv2d(inCh, outCh, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(outCh)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(outCh, outCh * expansion, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outCh)

    def forward(self, x):

        if self.downsample is None:
            xSelf = x
        else:
            xSelf = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += xSelf
        out = self.relu(out)

        return out


class SimSiam(nn.Module):
    def __init__(self, encArch=None, encDim=256, prjDim=256, prdDim=128, momEncBeta=0):
        """
        Create SimSiam model with encoder and predictor
        :param inDim: [int] [1] - Input data dimensionality
        :param encDim: [int] [1] - Encoder output dimensionality
        :param prdDim: [int] [1] - Predictor hidden dimensionality
        """
        super(SimSiam, self).__init__()
        self.momEncBeta = momEncBeta

        self.encoder = models.__dict__[encArch](num_classes=encDim, zero_init_residual=True)
        self.encoder.fc = nn.Identity(encDim)

        # CIFAR ResNet mod (used in SoloLearn)
        self.encoder.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
        self.encoder.maxpool = nn.Identity()

        self.projector = nn.Sequential(
            nn.Linear(encDim, encDim, bias=True),
            nn.BatchNorm1d(encDim),
            nn.ReLU(inplace=True),
            nn.Linear(encDim, encDim, bias=True),
            nn.BatchNorm1d(prjDim)
        )

        if prdDim is not None:
            self.predictor = nn.Sequential(
                nn.Linear(prjDim, prdDim, bias=True),
                nn.BatchNorm1d(prdDim),
                nn.ReLU(inplace=True),
                nn.Linear(prdDim, prjDim, bias=True),
            )
        else:
            self.predictor = nn.Identity(prjDim)

        if self.momEncBeta > 0.0:

            self.momentum_encoder = copy.deepcopy(self.encoder)
            self.momentum_encoder.apply(fn=reset_model_weights)
            for param in self.momentum_encoder.parameters(): param.requires_grad = False

            self.momentum_projector = copy.deepcopy(self.projector)
            self.momentum_projector.apply(fn=reset_model_weights)
            for param in self.momentum_projector.parameters(): param.requires_grad = False

    def forward(self, x):
        """
        Propagate input through encoder and decoder
        :param x: [tensor] [m x n] - Input tensor
        :return p: [tensor] [m x outDim] - Predictor output
        :return z: [tensor] [m x outDim] - Encoder output
        """
        r = self.encoder(x)
        z = self.projector(r)
        p = self.predictor(z)

        if self.momEncBeta > 0.0:
            mz = self.momentum_projector(self.momentum_encoder(x))
        else:
            mz = z

        return p, z, r, mz

    def update_momentum_network(self):
        if self.momEncBeta > 0.0:
            for enc_params, mom_enc_params in zip(self.encoder.parameters(), self.momentum_encoder.parameters()):
                mom_enc_params.data = self.momEncBeta * mom_enc_params.data + (1 - self.momEncBeta) * enc_params.data
            for prj_params, mom_prj_params in zip(self.projector.parameters(), self.momentum_projector.parameters()):
                mom_prj_params.data = self.momEncBeta * mom_prj_params.data + (1 - self.momEncBeta) * prj_params.data


class Weighted_InfoNCE:
    def __init__(self, nceBeta=0.0, nceBetaScheme=None, usePrd4CL=True, nceTau=0.1, downSamples=None,
                 mecReg=False, mecED2=0.06, mecTay=2):
        self.nceBeta = nceBeta
        self.nceBetaScheme = nceBetaScheme
        if self.nceBetaScheme is not None:
            self.nceBetaOrig = self.nceBeta
        self.usePrd4CL = usePrd4CL
        self.nceTau = nceTau
        self.downSamples = downSamples
        self.cossim = nn.CosineSimilarity(dim=1)
        self.mecReg = mecReg
        if self.mecReg:
            self.mecLoss = MEC(mecED2, mecTay)

    def forward(self, batch1Prd, batch1Prj, batch2Prj):

        # Positive similarity loss (InfoNCE numerator)
        lossVal = -1.0 * self.cossim(batch1Prd, batch2Prj).mean()

        # Negative similarity loss (InfoNCE denominator) - This formulation is best seen in the BYOL paper
        if self.nceBeta > 0.0:

            batch1 = batch1Prd if self.usePrd4CL else batch1Prj

            # Use a subset of the contrastive samples for loss calc (if specified)
            lossSamples = self.downSamples if self.downSamples is not None else batch1.size(0)

            # Calculate the pairwise cossim matrices for b1 x b2 and b1 x b1 (using matrix mult on normalized matrices)
            nds = pairwise_cosine_similarity(batch1[:lossSamples, :], batch2Prj[:lossSamples, :])
            nss = pairwise_cosine_similarity(batch1[:lossSamples, :], batch1[:lossSamples, :])
            nss.fill_diagonal_(0.0)  # Self-similarity within batch skips itself (similarity = 1 otherwise)

            # Calculate the log-sum-exp of the pairwise similarities, averaged over the batch
            lossVal += self.nceBeta * self.nceTau * \
                       (torch.exp(nds / self.nceTau) + torch.exp(nss / self.nceTau)).sum(dim=-1).log().mean()

        # Apply MEC loss as regularizer (if specified)
        if self.mecReg:
            lossVal += self.mecLoss.forward(batch1Prd, batch1Prj, batch2Prj)

        return lossVal

    def update_nceBeta(self, epoch, nEpochs):
        # Currently assumes that the final nceBeta value is either 0 (for dn) or 1 (for up)
        # Note cosine decay slightly modified from LR decay formula to hit exactly 0 for last epoch
        if self.nceBetaScheme is not None:
            if self.nceBetaScheme == 'stepdn':
                self.nceBeta = 0.0 if epoch > nEpochs / 2 else self.nceBetaOrig
            elif self.nceBetaScheme == 'stepup':
                self.nceBeta = 1.0 if epoch > nEpochs / 2 else self.nceBetaOrig
            elif self.nceBetaScheme == 'lindn':
                self.nceBeta = self.nceBetaOrig * (nEpochs - epoch) / (nEpochs - 1)
            elif self.nceBetaScheme == 'linup':
                self.nceBeta = self.nceBetaOrig + (1 - self.nceBetaOrig) * (epoch - 1) / (nEpochs - 1)
            elif self.nceBetaScheme == 'cosdn':
                self.nceBeta = self.nceBetaOrig * 0.5 * (1. + math.cos(math.pi * (epoch - 1) / (nEpochs - 1)))
            elif self.nceBetaScheme == 'cosup':
                self.nceBeta = self.nceBetaOrig + (1 - self.nceBetaOrig) * 0.5 * (1. - math.cos(math.pi * (epoch - 1) / (nEpochs - 1)))


class MEC:
    """
    Implements Maximum Entropy Coding loss from https://arxiv.org/abs/2210.11464
    """

    def __init__(self, ed2=0.06, taylorTerms=2):
        """
        :param ed2 [float]: lam = d / (m * eps^2) = 1 / (m * ed2), so ed2 = eps^2 / d. Authors use ed2 = [0.01, 0.12]
        :param taylorTerms [int]: Number of terms to use in Taylor expansion of the matrix logarithm
        """
        self.ed2 = ed2
        self.taylorTerms = taylorTerms

    def forward(self, batch1Prd, batch1Prj, batch2Prj):
        """
        :param batch1Prd: [tensor] [m x d] - Batch of online predictor outputs
        :param batch1Prj: [tensor] [m x d] - Batch of online projector outputs
        :param batch2Prj: [tensor] [m x d] - Batch of target projector outputs
        :return lossVal: [float] - MEC loss
        """

        # Get device
        device = torch.device(batch1Prd.get_device()) if batch1Prd.get_device() > -1 else torch.device('cpu')

        # Ensure batches are L2 normalized along feature dimension
        batch1 = batch1Prd / torch.norm(batch1Prd, p=2, dim=1, keepdim=True)
        batch2 = batch2Prj / torch.norm(batch2Prj, p=2, dim=1, keepdim=True)

        # Calculate mu and lam coefficients
        mu = (batch1.size(0) + batch1.size(1)) / 2
        lam = 1 / (batch1.size(0) * self.ed2)

        # Calculate correlation matrix and initialize the correlation exponential and Taylor sum
        corr = lam * torch.tensordot(batch1, batch2, dims=([-1], [-1])) # [m x m] batch-wise correlation matrix
        #corr = lam * torch.tensordot(batch1, batch2, dims=([0], [0])) # [d x d] feature-wise correlation matrix
        powerCorr = torch.eye(corr.size(0), device=device)
        sumCorr = torch.zeros_like(corr)

        # Loop through Taylor terms and cumulatively add powers of the correlation matrix
        for k in range(self.taylorTerms):
            powerCorr = torch.tensordot(powerCorr, corr, dims=([-1], [0]))
            sumCorr += (-1) ** k / (k + 1) * powerCorr

        # Calculate final loss value
        lossVal = -1.0 * mu * torch.trace(sumCorr)

        return lossVal


class BT_CrossCorr:
    # Review for correctness - do batches need to be normalized in feature dim? double check multiplication
    # Doublecheck hsic application
    def __init__(self, btLam, lossForm='bt'):
        self.btLam = btLam
        self.lossForm = lossForm

    def forward(self, batch1Prd, batch1Prj, batch2Prj):

        device = torch.device(batch1Prd.get_device()) if batch1Prd.get_device() > -1 else torch.device('cpu')
        prjDim = batch1Prd.size(-1)

        # Batch normalize each batch
        normBatch1 = (batch1Prd - batch1Prd.mean(dim=0, keepdim=True)) / batch1Prd.std(dim=0, unbiased=True, keepdim=True)
        normBatch2 = (batch2Prj - batch2Prj.mean(dim=0, keepdim=True)) / batch2Prj.std(dim=0, unbiased=True, keepdim=True)

        if self.lossForm == 'bt':
            onesMat = -1.0 * torch.eye(prjDim, device=device)
        elif self.lossForm == 'hsic':
            onesMat = (torch.ones(prjDim, prjDim) - 2.0 * torch.eye(prjDim)).to(device)
        offDiagMult = (self.btLam * torch.ones(prjDim, prjDim) + (1 - self.btLam) * torch.eye(prjDim)).to(device)

        # Calculate cross-correlation and Barlow Twins cross-correlation loss
        crossCorr = 1.0 / batch1Prd.size(0) * torch.tensordot(normBatch1, normBatch2, dims=([0], [0]))
        lossVal = ((crossCorr + onesMat).pow(2) * offDiagMult).sum()

        return lossVal


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