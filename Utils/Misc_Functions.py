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
