import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class Local_Lip_Loss:

    def __init__(self, top_norm=1, bot_norm=np.inf, reduction='mean'):
        """
        Initialize norms and reduction methods
        :param top_norm: [float] [1] - Norm to use on the numerator ||f(x) - f(xp)||
        :param bot_norm: [float] [1] - Norm to use on the denominator ||x - xp||
        :param reduction: [string] [1] - Type of reduction to apply across all batch samples
        """
        self.top_norm = top_norm
        self.bot_norm = bot_norm
        self.reduction = reduction

        if self.top_norm not in [1, 2, np.inf, 'kl']:
            raise ValueError(f'Unsupported norm {self.top_norm}')
        if self.bot_norm not in [1, 2, np.inf]:
            raise ValueError(f'Unsupported norm {self.bot_norm}')

    def forward(self, x, xp, xOut, xpOut):
        """
        Calculate local Lipschitz value, ||f(x) - f(xp)||_p / ||x - xp||_q
        :param x: [Pytorch tensor] [m x n] - Nominal input tensor
        :param xp: [Pytorch tensor] [m x n] - Adversarial input tensor of x
        :param xOut: [Pytorch tensor] [m x n2] - Encoding output of nominal input
        :param xpOut: [Pytorch tensor] [m x n2] - Encoding output of adversarial input
        :return:
        """

        # Calculate difference between input samples
        # Convert all tensor dimensions after batchsize dimension into a vector - N,C,H,W to N,C*H*W
        bot = torch.flatten(x - xp, start_dim=1)

        # Use KL divergence to calculate the difference between model outputs, then calculate Lipschitz
        # PyTorch KLDivLoss calculates reduction(ytrue*log(ytrue/ypred)) where reduction is some method of aggregating the results (sum, mean)
        # yt*log(yt/yp) = yt*(log(yt)-log(yp)) --> PyTorch expects yp to be in logspace already, such that you input log(yp) and yt
        if self.top_norm == 'kl':
            criterion_kl = nn.KLDivLoss(reduction='none')
            top = criterion_kl(F.log_softmax(xpOut, dim=1), F.softmax(xOut, dim=1))
            lolip = torch.sum(top, dim=1) / torch.norm(bot + 1e-6, dim=1, p=self.bot_norm)

        # Calculate Lipschitz constant using regular norms - the top just uses output logits (no softmax)
        else:
            top = torch.flatten(xOut, start_dim=1) - torch.flatten(xpOut, start_dim=1)
            lolip = torch.norm(top, dim=1, p=self.top_norm) / torch.norm(bot + 1e-6, dim=1, p=self.bot_norm)

        if self.reduction == 'mean':
            return torch.mean(lolip)
        elif self.reduction == 'sum':
            return torch.sum(lolip)
        elif self.reduction == 'none':
            return lolip


def maximize_local_lip(model, X, alpha=0.003, eps=0.01, top_norm=1, bot_norm=np.inf, batch_size=16, n_restarts=10,
                       n_steps=10, outIdx=None, rand_init=True, noise_mag=None, x_min=None, x_max=None):
    """
    Iteratively search for input that maximizes local Lipschitz within a specified radius
    This function is similar to the FGSM_PGD script in the same folder, but modified to accommodate Lipschitz as loss
    This function assumes that model and X are on same device
    :param model: [Pytorch Model] [] - Callable object that takes an input and returns an output encoding
    :param X: [Pytorch tensor] [m x n] - Nominal input tensor
    :param alpha: [float] [1] - Adversarial sample step size
    :param eps: [float] [1] - Norm constraint bound for infinity norm of adversarial sample
    :param top_norm: [float] [1] - Norm type to apply to ||f(x) - f(xp)||
    :param bot_norm: [float] [1] - Norm type to apply to ||x - xp||
    :param batch_size: [int] [1] - Number of samples to load from the dataloader each iteration
    :param n_restarts: [int] [1] - Number of PGD restarts to try
    :param n_steps: [int] [1] - Number of PGD steps to run for maximizing local Lipschitz
    :param outIdx: [int] [1] - Index corresponding to the desired output (set as None for only 1 output)
    :param rand_init: [Bool] [1] - Whether to start adversarial search with random offset
    :param noise_mag: [float] [1] - Maximum random initial perturbation magnitude
    :param x_min: [float] [1] - Minimum value of x_adv (useful for ensuring images are useable)
    :param x_max: [float] [1] - Maximum value of x_adv (useful for ensuring images are useable)
    :return avgLolip: [float] [1] - Average final local Lipschitz values across input samples
    :return advTens: [Pytorch tensor] [m x n] - Final adversarial samples tensor
    """
    
    # Make a loader using the X tensor input
    loader = torch.utils.data.DataLoader(X, batch_size=batch_size, shuffle=False)

    # Ensure model doesn't update batchnorm or anything
    model.eval()

    # Instantiate Local_Lip_Loss
    lll = Local_Lip_Loss(top_norm, bot_norm, 'none')
    
    # Initialize the running variables, total lolip and list of adversarial samples (for concatenation later)
    total_lolip = 0.
    adv_samples_list = []

    for x in loader:

        # Get the outputs of the model for x and detach (don't need gradients for x or xOut)
        xOut = model(x) if outIdx is None else model(x)[outIdx]
        xOut = xOut.detach()

        for i in range(n_restarts):

            # Apply random initial perturbation to input (or don't)
            # Madry 2017 apply random perturbations over many runs to see if adv results were very different - they were not
            if rand_init:
                if noise_mag is None:
                    noise_mag = eps
                noise = torch.zeros_like(x).uniform_(-noise_mag, noise_mag)
                # Clip noise to ensure it does not violate x_adv norm constraint and then apply to x
                noise = torch.clamp(noise, -eps, eps)
                x_adv = x + noise.to(x.device)
            else:
                x_adv = x

            # Ensure x_adv elements within appropriate bounds
            if x_min is not None or x_max is not None:
                x_adv = torch.clamp(x_adv, x_min, x_max)

            for _ in range(n_steps):

                # Reset the gradients of the adversarial input and model
                x_adv = x_adv.detach().requires_grad_(True)
                model.zero_grad()

                # Get outputs of the model for x_adv
                modelOut = model(x_adv)
                xadvOut = modelOut if outIdx is None else modelOut[outIdx]

                # Calculate the local lipschitz constant using x and x_adv, then backpropagate to get gradients
                lolip = lll.forward(x, x_adv, xOut, xadvOut)
                sumlolip = torch.sum(lolip)
                sumlolip.backward()

                # Calculate the new adversarial example given the new step - gradient ascent towards higher Lipschitz
                # x_adv is detached, since x_adv.data and x_adv.grad do not carry autograd
                x_adv = x_adv.data + alpha * x_adv.grad.sign()

                # Clip total perturbation (measured from center x) to norm ball associated with x_adv limit
                eta = torch.clamp(x_adv - x, -eps, eps)
                x_adv = x + eta

                # Ensure x_adv elements within appropriate bounds
                if x_min is not None or x_max is not None:
                    x_adv = torch.clamp(x_adv, x_min, x_max)

            # Compare max lolip so far for each sample with lolip for this restart and take max
            if i == 0:
                max_lolip = lolip.detach()
                x_adv_final = x_adv.detach()
            else:
                betterLolipMask = lolip > max_lolip
                max_lolip[betterLolipMask] = lolip[betterLolipMask].detach()
                x_adv_final[betterLolipMask, :] = x_adv[betterLolipMask, :].detach()

        # Calculate lolip for each sample and sum batch and append the adversarial tensor to list for concat later
        total_lolip += torch.sum(max_lolip).item()
        adv_samples_list.append(x_adv_final.detach().cpu())

    # Concatenate the adversarial output tensor
    advTens = torch.concatenate(adv_samples_list, dim=0).to(X.device)

    return total_lolip / len(X), advTens