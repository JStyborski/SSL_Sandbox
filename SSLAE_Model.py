import copy
import torch
from torch import nn
import torchvision.models as models

import Utils.ResNet_AutoEncoder as RNAE

# Function to reinitialize all resettable weights of a given model
def reset_model_weights(layer):
    if hasattr(layer, 'reset_parameters'):
        layer.reset_parameters()
    elif hasattr(layer, 'children'):
        for child in layer.children():
            reset_model_weights(child)

# Function to hook intermediate network outputs
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output
    return hook

class Base_Model(nn.Module):
    def __init__(self, encArch=None, cifarMod=False, encDim=512, prjHidDim=2048, prjOutDim=2048, prdDim=512,
                 prdAlpha=None, prdEps=0.3, prdBeta=0.5, momEncBeta=0, applySG=True, decArch=None):
        super(Base_Model, self).__init__()
        self.prdAlpha = prdAlpha
        self.prdEps = prdEps
        self.prdBeta = prdBeta
        self.momZCor = None  # Initialize momentum correlation matrix as None (overwritten later)
        self.momEncBeta = momEncBeta
        self.applySG = applySG
        self.decArch = decArch

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

            # Copy the encoder, reset weights, and freeze weights
            self.momentum_encoder = copy.deepcopy(self.encoder)
            self.momentum_encoder.apply(fn=reset_model_weights)
            for param in self.momentum_encoder.parameters(): param.requires_grad = False

            # Copy the projector, reset weights, and freeze weights
            self.momentum_projector = copy.deepcopy(self.projector)
            self.momentum_projector.apply(fn=reset_model_weights)
            for param in self.momentum_projector.parameters(): param.requires_grad = False

        if self.decArch is not None:

            # Register a hook to get the output before the adaptive average pool (ResNet decoder expects 512x7x7 size "encodings")
            self.encoder.layer4.register_forward_hook(get_activation('layer4'))

            # Find decoder config and instantiate
            configs, bottleneck = RNAE.get_configs(encArch)
            self.decoder = RNAE.ResNetDecoder(configs, bottleneck)

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

        # Apply stop-gradient to second branch
        if self.applySG:
            mz = mz.detach()

        # Get decoder input and push through decoder
        if self.decArch is not None:
            r_pre = activation['layer4']
            xd = self.decoder(r_pre)
        else:
            xd = None

        return p, z, r, mz, xd

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