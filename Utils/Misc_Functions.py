import torch

def gather_tensors(outLen, tens):
    outTens = torch.zeros(outLen, tens.size(1), device=tens.get_device())
    torch.distributed.all_gather_into_tensor(outTens, tens)
    return outTens


class FullGatherLayer(torch.autograd.Function):
    """
    Gather tensors from all process and support backward propagation for the gradients across processes.
    This function is taken directly from the official VICReg code: https://github.com/facebookresearch/vicreg/blob/main/main_vicreg.py
    I checked through the results already - using 4GPU with gather gives the same losses and gradients as using 1GPU with/without gather.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        torch.distributed.all_reduce(all_gradients)
        return all_gradients[torch.distributed.get_rank()]


def spectral_filter(z, power=0.0, cutoff=None):
    # For z (n x d) and Cz (d x d), z = U Sigz V.T, Cz = 1/n z.T z = Q Lamz Q.T, with Q = V and Lamz = 1/n Sigz^2
    # Spectral filter g(Lam) adjusts eigenvalues and then applies W = V g(Lamz) V.T on z, p = z @ W
    # This affects output correlation: Cp = V g(Lamz)^2 Lam V.T, such that Lamp = g(Lamz)^2 Lamz
    # Low pass filter emphasizes large eigvals and diminishes low eigvals - high pass filter vice versa
    # In this function we specifically apply g(Lamz) = Lamz.pow(power)
    # power should be between -0.5 and +1.0 - [-0.5, 0] gives high pass filter, [0, 1.0] gives low pass filter
    # Power examples: -0.5 -> Lamp = I, 0 -> Lamp = Lamz, 0.5 -> Lamp = Lamz^2, 1.0 -> Lamp = Lamz^3
    U, Sigz, VT = torch.linalg.svd(z, full_matrices=False)
    Lamz = 1 / z.size(0) * Sigz.clamp(0).pow(2)
    Lamp = Lamz
    if power is not None:
        Lamp = Lamz.pow(1 + 2 * power)
    if cutoff is not None:
        Lamp[cutoff:] = 0
    Sigp = Lamp.sqrt() * z.size(0) ** 0.5
    specZ = U @ torch.diag(Sigp) @ VT
    return specZ
