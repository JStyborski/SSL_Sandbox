import copy
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


class Base_Model(nn.Module):
    def __init__(self, encArch=None, cifarMod=False, encDim=512, prjHidDim=2048, prjOutDim=2048, prdDim=512, prdAlpha=None, prdEps=0.3, prdBeta=0.5, momEncBeta=0):
        super(Base_Model, self).__init__()
        self.prdAlpha = prdAlpha
        self.prdEps = prdEps
        self.prdBeta = prdBeta
        self.momZCor = None  # Initialize momentum correlation matrix as None (overwritten later)
        self.momEncBeta = momEncBeta

        self.encoder = models.__dict__[encArch](num_classes=encDim, zero_init_residual=True)
        self.encoder.fc = nn.Identity(encDim)

        # CIFAR ResNet mod
        if 'resnet' in encArch and cifarMod:
            self.encoder.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
            self.encoder.maxpool = nn.Identity()

        self.projector = nn.Sequential(
            nn.Linear(encDim, prjHidDim, bias=False),
            nn.BatchNorm1d(prjHidDim),
            nn.ReLU(inplace=True),
            nn.Linear(prjHidDim, prjHidDim, bias=False),
            nn.BatchNorm1d(prjHidDim),
            nn.ReLU(inplace=True),
            nn.Linear(prjHidDim, prjOutDim, bias=False),
            nn.BatchNorm1d(prjOutDim, affine=False)
        )

        if prdDim > 0 and self.prdAlpha is None:
            self.predictor = nn.Sequential(
                nn.Linear(prjOutDim, prdDim, bias=False),
                nn.BatchNorm1d(prdDim),
                nn.ReLU(inplace=True),
                nn.Linear(prdDim, prjOutDim, bias=True),
            )
        else:
            self.predictor = nn.Identity(prjOutDim)

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
        :param x: [tensor] [m x d] - Input tensor
        :return p: [tensor] [m x prjOutDim] - Predictor output
        :return z: [tensor] [m x prjOutDim] - Projector output
        :return r: [tensor] [m x encDim] - Encoder output
        :return mz: [tensor] [m x prjOutDim] - Momentum encoder/projector output
        """

        r = self.encoder(x)
        z = self.projector(r)

        # Run predictor or optimal predictor
        if self.prdAlpha is None:
            p = self.predictor(z)
        else:
            Wp = self.calculate_optimal_predictor(z)
            p = z @ Wp

        # Run momentum encoder or treat momEnc as regular encoder
        if self.momEncBeta > 0.0:
            mz = self.momentum_projector(self.momentum_encoder(x))
        else:
            mz = z

        return p, z, r, mz

    def calculate_optimal_predictor(self, z):
        """
        Calculate the spectral filter of z covariance to apply in place of a predictor
        :param z: [tensor] [m x d] - Input tensor, output of projector
        :return Wp: [tensor] [d x d] - Spectrally filtered, normalized, and regularized correlation matrix
        """

        # Use stop-gradient on calculating optimal weights matrix (I think)
        with torch.no_grad():

            # Calculate projector output correlation matrix
            zCor = 1 / z.size(0) * torch.tensordot(z, z, dims=([0], [0]))

            # Momentum update (or initialize) correlation matrix
            if self.momZCor is not None:
                self.momZCor = self.prdBeta * self.momZCor + (1 - self.prdBeta) * zCor
            else:
                self.momZCor = zCor

            # Calculate exponential of correlation matrix and then optimal predictor weight matrix
            # Note that I've tested multiple values of alpha. alpha=0.5 works well, alpha=1.0 causes complete collapse
            # The DirectSet paper mentions that for alpha=1.0, fAlpha or Wp needs regularization and normalization
            # I thought this referred to the prdEps term and matrix norm of fAlpha, but apparently that's not enough
            corEigvals, corEigvecs = torch.linalg.eigh(self.momZCor)
            corEigvals = torch.clamp(torch.real(corEigvals), 0)
            corEigvecs = torch.real(corEigvecs)
            fAlpha = corEigvecs @ torch.diag(torch.pow(corEigvals, self.prdAlpha)) @ torch.transpose(corEigvecs, 0, 1)
            # Shortcut: ||fAlpha||_spec = torch.linalg.matrix_norm(fAlpha, ord=2) = corEigval[-1].pow(self.prdAlpha)
            Wp = fAlpha / corEigvals[-1].pow(self.prdAlpha) + self.prdEps * torch.eye(z.size(1), device=torch.device(z.get_device()))

        return Wp

    def update_momentum_network(self):
        if self.momEncBeta > 0.0:
            for enc_params, mom_enc_params in zip(self.encoder.parameters(), self.momentum_encoder.parameters()):
                mom_enc_params.data = self.momEncBeta * mom_enc_params.data + (1 - self.momEncBeta) * enc_params.data
            for prj_params, mom_prj_params in zip(self.projector.parameters(), self.momentum_projector.parameters()):
                mom_prj_params.data = self.momEncBeta * mom_prj_params.data + (1 - self.momEncBeta) * prj_params.data


class Weighted_InfoNCE:
    def __init__(self, nceBeta=0.0, usePrd4CL=True, nceTau=0.1, downSamples=None):
        self.nceBeta = nceBeta
        self.usePrd4CL = usePrd4CL
        self.nceTau = nceTau
        self.downSamples = downSamples
        self.cossim = nn.CosineSimilarity(dim=1)

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

        return lossVal


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


class Barlow_Twins_Loss:
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

        # Redo the following but elegantly to use triu - saves time and device allocation
        if self.lossForm == 'bt':
            onesMat = -1.0 * torch.eye(prjDim, device=device)
        elif self.lossForm == 'hsic':
            onesMat = (torch.ones(prjDim, prjDim) - 2.0 * torch.eye(prjDim)).to(device)
        offDiagMult = (self.btLam * torch.ones(prjDim, prjDim) + (1 - self.btLam) * torch.eye(prjDim)).to(device)

        # Calculate cross-correlation and Barlow Twins cross-correlation loss
        crossCorr = 1.0 / batch1Prd.size(0) * torch.tensordot(normBatch1, normBatch2, dims=([0], [0]))
        lossVal = ((crossCorr + onesMat).pow(2) * offDiagMult).sum()

        return lossVal


class VICReg_Loss:

    def __init__(self, alpha, beta, gamma):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, batch1Prd, batch1Prj, batch2Prj):

        # Calculate covariance and covariance losses of batches 1 and 2
        b1Cov = torch.cov(batch1Prd.T)
        b2Cov = torch.cov(batch2Prj.T)
        cov1Loss = 2 / batch1Prd.size(1) * torch.triu(b1Cov, 1).square().sum()
        cov2Loss = 2 / batch2Prj.size(1) * torch.triu(b2Cov, 1).square().sum()

        # Get variances from covariance matrices and calculate variance losses
        var1Loss = 1 / batch1Prd.size(1) * (1 - torch.diag(batch1Prd).clamp(1e-6).sqrt()).clamp(0).sum()
        var2Loss = 1 / batch2Prj.size(1) * (1 - torch.diag(batch2Prj).clamp(1e-6).sqrt()).clamp(0).sum()

        # Calculate L2 distance loss between encodings
        invLoss = 1 / batch1Prd.size(0) * torch.linalg.vector_norm(batch1Prd - batch2Prj, ord=2, dim=1).square().sum()

        lossVal = self.alpha * (var1Loss + var2Loss) + self.beta * invLoss + self.gamma * (cov1Loss + cov2Loss)

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
