"""The Projected Gradient Descent attack."""

import numpy as np
import torch

def scale_tens(tens, tens_idx, norm, eps):
    # If ||tens||_p > eps, scales all tens values such that ||tens||_p = eps
    # This function follows the torch.nn.utils.clip_grad implementation of norm scaling
    # :tens (tensor): the input tensor to modify based on its norm
    # :tens_idx (list): the list of tens dimension indices along which to calculate norms - dimensions not included will have a separate norm value for each element
    # :norm (float, int, or 'inf'): the type of norm to apply to tens
    # :eps (float or int): the maximum allowable norm value of tens
    # :returns: scaled_tens
    
    # Get the norm of tens across the specified indices
    tens_norm = torch.norm(tens, dim=tens_idx, keepdim=True, p=norm)
    
    # If eps > eta_norm in certain elements, don't upscale them
    scale_coef = torch.clamp(eps / (tens_norm + 1e-6), max=1.0)
    scaled_tens = tens * scale_coef
    
    return scaled_tens

def clip_tens(tens, tens_idx, norm, eps):
    # If ||tens||_inf > eps, projects tens tensor to the nearest point where ||tens||_inf = eps
    # :tens (Tensor): the input tensor to modify based on its norm
    # :tens_idx (list): the list of tens dimension indices along which to calculate norms - dimensions not included will have a separate norm value for each element
    # :norm (float, int, or 'inf'): the type of norm to apply to tens
    # :eps (float): the maximum allowable inf norm value of tens
    # :returns clipped_tens (Tensor): 
    
    # Inf norm of tens cannot exceed eps - corresponds to clipping all values of tens beyond +/- eps
    if norm == np.inf:
        clipped_tens = torch.clamp(tens, min=-eps, max=eps)
    
    # Scaling isn't the same idea as clipping, but in practice, people will refer to scaling as clipping
    # I keep this part to ensure the function still works if people want to "clip" for other norms
    # Note that in 2-norm, clipping and scaling are identical processes
    else:
        clipped_tens = scale_tens(tens, tens_idx, norm, eps)
    
    return clipped_tens

def optimize_linear(grad, grad_idx, norm=np.inf):
    # Solves for the optimal input to a linear function under a norm constraint.
    # Optimal_perturbation = argmax_eta,||eta||_p<=1(dot(eta, grad))
    # i.e., Find eta s.t. pth norm of eta <= 1 such that dot(eta, grad) is maximized
    # :grad (Tensor): batch of gradients
    # :grad_idx (list): the list of grad dimension indices along which to calculate norms - dimensions not included will have a separate norm value for each element
    # :norm (number): np.inf, 1, or 2. Order of norm constraint.
    # :returns eta (Tensor): optimal perturbation, the eta where ||eta||_p <= 1
    
    # dot(eta, grad) with ||eta||inf = max(abs(eta)) <= 1 is maximized when eta=sign(grad)
    # Optimal inf-norm constrained perturbation direction is the max magnitude grad value in every dimension
    if norm == np.inf:
        eta = torch.sign(grad)
    
    # dot(eta, grad) with ||eta||1 = sum(abs(eta_i)) <= 1 is maximized when eta is a +/- 1-hot corresponding to the maximum magnitude value of grad
    # Optimal 1-norm constrained perturbation direction is the max magnitude pixel value in any dimension
    elif norm == 1:
    
        # Absolute value and sign tensors of gradient, used later
        abs_grad = torch.abs(grad)

        # Get the maximum values of the tensor as well as their locations
        # Max is executed across all dimensions except the first (batch dim), such that one max value is found for each batch sample
        max_abs_grad = torch.amax(abs_grad, dim=grad_idx, keepdim=True)
        max_mask = abs_grad.eq(max_abs_grad).to(torch.float)
        
        # Count the number of tied values at maximum for each batch sample
        num_ties = torch.sum(max_mask, dim=grad_idx, keepdim=True)
        
        # Optimal perturbation for 1-norm is only along the maximum magnitude dimensions
        # Hack to apply smaller perturbation in >1 dimensions if num_ties > 1
        eta = torch.sign(grad) * max_mask / num_ties
    
    # dot(eta, grad) with ||eta||2 = sqrt(sum(eta_i^2)) <= 1 is maximized when eta is equal to the normalized gradient vector
    # Optimal 2-norm constrained perturbation direction is Euclidean scaling along the gradient vector
    elif norm == 2:
    
        # Get the 2 norm of the gradient vector for each batch sample and normalize
        eta = grad / torch.norm(grad, dim=grad_idx, keepdim=True, p=2)
        
    else:
        raise NotImplementedError('Only L-inf, L1 and L2 norms are currently implemented.')

    return eta


def pgd(model, loss_fn, X, Y, alpha, eps, norm, batch_size, n_restarts, n_steps, outIdx=None, targeted=False,
        rand_init=True, noise_mag=None, x_min=None, x_max=None):
    """
    Implementation the Kurakin 2016 Basic Iterative Method (rand_init=False) or Madry 2017 PGD method (rand_init=True)
    This function assumes that model and x are on same device, and that model is in desired mode
    Equivalent to FGSM if eps is high such that no limit is applied, rand_init = False, and n_restarts = perturb_steps = 1
    :param model: [function] [1] - Callable function that takes an input tensor and returns the model logits
    :param loss_fn: [function] [1] - Callable function for calculating loss values per input - the loss function
        should be initialized already with reduction='none' so that each input gets a loss
        - CrossEntropyLoss - Typical loss used for FGSM/PGD. The main issue is when logit magnitudes are large (e.g., 500)
        such that the softmax(logits) exponential is dominated by 1 term and the output distribution is a one-hot.
        If the one-hot is correct, then CE loss is 0, then the gradient is 0, then eta is 0, the x_adv = x.
        Typical remedies are to L2 normalize logits or use NLLLoss (below)
        - NLLLoss - Not typically used for FGSM/PGD. This essentially returns the negative of the correct logit as loss.
    :param X: [Pytorch tensor] [m x n] - Nominal input tensor
    :param Y: [tensor] [m] - Tensor with truth labels
    :param alpha: [float] [1] - Input variation parameter, see https://arxiv.org/abs/1412.6572
    :param eps: [float] [1] - Norm constraint bound for adversarial example
    :param norm: [float] [1] - Order of the norm (mimics NumPy)
    :param batch_size: [int] [1] - Number of samples to calculate loss for at once
    :param n_restarts: [int] [1] - Number of PGD restarts
    :param n_steps: [int] [1] - Number of PGD steps
    :param outIdx: [int] [1] - Index corresponding to the desired output (set as None for only 1 output)
    :param targeted: [Bool] [1] - Whether to direct the adversarial attack towards a specific label/target
    :param rand_init: [Bool] [1] - Whether to start adversarial search with random offset
    :param noise_mag: [float] [1] - Maximum random initial perturbation magnitude
    :param x_min: [float] [1] - Minimum value of x_adv (useful for ensuring images are useable)
    :param x_max: [float] [1] - Maximum value of x_adv (useful for ensuring images are useable)
    :return: a tensor for the adversarial example
    """

    if norm not in [np.inf, 1, 2]:
        raise ValueError(f'Unsupported norm {norm}')
    if norm == 1:
        print('Warning: FGM may not be a good inner loop step, because norm=1 FGM only changes 1 pixel at a time')

    # Make a dataset and loader using the X and Y tensor inputs
    dataset = torch.utils.data.TensorDataset(X, Y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Initialize the running variables, total loss and list of adversarial samples (for concatenation later)
    total_loss = 0.
    adv_samples_list = []

    for x, y in loader:

        for i in range(n_restarts):

            # Apply random initial perturbation to input (or don't)
            # Madry 2017 apply random perturbations over many runs to see if adv results were very different - they were not
            if rand_init:
                if noise_mag is None:
                    noise_mag = eps
                noise = torch.zeros_like(x).uniform_(-noise_mag, noise_mag)
                # Clip noise to ensure it does not violate x_adv norm constraint and then apply to x
                noise = clip_tens(noise, list(range(1, len(noise.size()))), norm, eps)
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

                # Calculate loss and get loss gradient
                # If attack is targeted, define loss such that gradient dL/dx will point towards target class
                # else, define loss such that gradient dL/dx will point away from correct class
                loss = loss_fn(model(x_adv), y) if outIdx is None else loss_fn(model(x_adv)[outIdx], y)
                if targeted:
                    loss = -loss
                sumloss = torch.sum(loss)
                sumloss.backward()

                # eta is the norm-constrained direction that maximizes dot(perturbation, x.grad)
                # eta is detached, since x_adv.grad does not carry autograd
                eta = optimize_linear(x_adv.grad, list(range(1, len(x_adv.grad.size()))), norm)

                # Add perturbation to original example to step away from correct label and obtain adversarial example
                # x_adv is detached, since x_adv.data does not carry autograd
                x_adv = x_adv.data + alpha * eta

                # Clip total perturbation (measured from center x) to norm ball associated with x_adv limit
                eta = clip_tens(x_adv - x, list(range(1, len(x.size()))), norm, eps)
                x_adv = x + eta

                # Ensure x_adv elements within appropriate bounds
                if x_min is not None or x_max is not None:
                    x_adv = torch.clamp(x_adv, x_min, x_max)

            # Compare the best loss so far to loss for this restart and take the better adversarial sample
            if i == 0:
                best_loss = loss.detach()
                x_adv_final = x_adv.detach()
            else:
                if targeted:
                    betterLossMask = loss.detach() < best_loss
                else:
                    betterLossMask = loss.detach() > best_loss
                best_loss[betterLossMask] = loss[betterLossMask].detach()
                x_adv_final[betterLossMask, :] = x_adv[betterLossMask, :].detach()

        # Calculate loss for each sample and sum batch and append the adversarial tensor to list for concat later
        total_loss += torch.sum(best_loss).item()
        adv_samples_list.append(x_adv_final.detach().cpu())

    # Calculate the average loss across all samples and concatenate the adversarial output tensor
    avgLoss = total_loss / len(X)
    advTens = torch.concatenate(adv_samples_list, dim=0).to(X.device)

    return avgLoss, (advTens - X), advTens

