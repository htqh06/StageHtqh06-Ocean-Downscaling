import numpy as np 
import torch

###############################################################################################################
#                                                                                                             #
#    This file handles the operations related to the observation operator in the guided deoising process,     #
#    with  upsampling,  downsampling,  and gradient  computations. The  parameters  size and  blur_factor     #
#    are the image size and downsampling factor, and the masking is for performing guidance only on some      #
#    part of the image                                                                                        #
#                                                                                                             #
###############################################################################################################


size = 64  # Define the size variable
blur_factor = 4  # Define the blur factor

masking = False  # Dwether to use masking or not
mask = torch.ones(size, size)
#mask[25:48, 8:32] = 0  # Mask out a central square

def downsample_to_mean(A, block_size=blur_factor):
    """
    Downsamples a size x size image tensor A to a 
    (size // blur_factor) x (size // blur_factor) image tensor B,
    where each value in B is the mean of blur_factor x blur_factor corresponding values in A.

    Input: tensor oA of shape (size, size).

    Output: Downsampled tensor B of shape (size // blur_factor, size // blur_factor).
    """
    if A.shape != (size, size):
        raise ValueError(f"Input tensor A must have shape ({size}, {size})")

    B = A.view(size // block_size, block_size, size // block_size, block_size).mean(dim=(1, 3))

    return B


def upsample_to_original(B, block_size=blur_factor):
    """
    Upsamples the first channel of a (size // block_size) x (size // block_size) image tensor B 
    to a size x size x 2 image tensor A, where each value in B is repeated into a 
    blur_fablock_sizector x block_size block in A. The mask is applied to set masked parts of the image to 0.

    Input : tensor B of shape (1, size // block_size, size // block_size).

    Output: Upsampled tensor A of shape (2, size, size), with masked parts set to 0.
    """
    if B.shape != (size // block_size, size // block_size):
        raise ValueError(f"Input tensor B must have shape ({size // block_size}, {size // block_size})")

    # Repeat each value in B into a block_size x block_size block
    upsampled_first_channel = B.repeat_interleave(block_size, dim=0).repeat_interleave(block_size, dim=1)

    # Apply the mask to the first channel
    if masking:
        upsampled_first_channel = upsampled_first_channel * mask

    """# Create a tensor with 2 channels: the first channel is the upsampled data,
    # and the second channel is left as zeros (or can be filled with other data if needed)
    A = torch.zeros(2, size, size)
    A[0, :, :] = upsampled_first_channel"""

    return upsampled_first_channel

def get_difference(Y_sss, Y_sst, Y_ssh, X, scaling_factor=1): 
    """
    Inputs:
        Y (torch.Tensor): Input tensor of shape (size // blur_factor, size // blur_factor).
        X (torch.Tensor): Input tensor of shape (size, size).
        sc is just a test factor, keep at one

    Output:
        torch.Tensor: Difference tensor of shape (size, size)
    """

    # Compute the difference
    diff_sss = upsample_to_original(Y_sss) - upsample_to_original(downsample_to_mean(X[0,:,:]))
    diff_sst = Y_sst - X[1,:,:]
    diff_ssh = Y_ssh - X[2,:,:]
    g=1.5 # empirical factor, if too high the guidance is unstable, too low and it is not constraining enough
    diff = torch.zeros(3, size, size)
    diff[0, :, :] = diff_sss*g*scaling_factor 
    diff[1, :, :] = diff_sst*g
    diff[2, :, :] = diff_ssh*g
    return diff

def get_grad(Y_sss, Y_sst, Y_ssh, X, x, scaling_factor=1):
    """
    Computes the gradient of the difference between the upsampled Y and downsampled X,
    after scaling both tensors to the range [-1, 1].

    Input:
        Y_sss (torch.Tensor): Input tensor of shape (size // blur_factor, size // blur_factor).
        Y_sst (torch.Tensor): Input tensor of shape (size, size).
        X (torch.Tensor): Input tensor of shape (2, size, size).
        x (torch.Tensor): Tensor with respect to which the gradient is computed.

    Output:
        torch.Tensor: Gradient of the difference with respect to X.
    """
    if Y_sss.shape != (size // blur_factor, size // blur_factor):
        raise ValueError(f"Input tensor Y_sss must have shape ({size // blur_factor}, {size // blur_factor})")
    if X.shape != (3, size, size):
        raise ValueError(f"Input tensor X must have shape (3, {size}, {size})")
    if Y_sst.shape != (size, size):
        raise ValueError(f"Input tensor Y_sst must have shape ({size}, {size})")
    if Y_ssh.shape != (size, size):
        raise ValueError(f"Input tensor Y_ssh must have shape ({size}, {size})")

    difference = get_difference(Y_sss, Y_sst, Y_ssh, X, scaling_factor)
    difference = difference.requires_grad_()

    # Ensure difference is on the same device as x
    difference = difference.to(x.device)

    # Compute the gradient
    gradient = torch.autograd.grad(outputs=X.sum(), inputs=x)[0]
    gradient = difference * gradient
    return gradient


def get_grad_batch(Y_sss, Y_sst, Y_ssh, X_batch, x_batch, scaling_factor=1.0):
    """
    Vectorized version of get_grad over a batch.

    Args:
        Y_sss: Tensor of shape (B, 16, 16)
        Y_sst: Tensor of shape (B, 64, 64)
        Y_ssh: Tensor of shape (B, 64, 64)
        X_batch: Tensor of shape (B, 3, 64, 64) — output of model (x0_pred), must retain gradient path
        x_batch: Tensor of shape (B, 3, 64, 64) — input to the model with requires_grad=True
        sc : see get_grad

    Returns:
        Tensor of shape (B, 3, 64, 64) — gradient of the guidance loss with respect to x_batch
    """
    B, C, H, W = x_batch.shape
    device = x_batch.device

    # === Vectorized get_difference ===
    # Downsample X_batch[:, 0] → shape (B, 16, 16)
    x_sss_down = x_batch[:, 0].unfold(1, 4, 4).unfold(2, 4, 4)  # shape (B, 16, 16, 4, 4)
    x_sss_down = x_sss_down.contiguous().view(B, 16, 16, -1).mean(dim=-1)  # → (B, 16, 16)

    # Upsample Y_sss and x_sss_down back to (64, 64)
    Y_sss_up = Y_sss.repeat_interleave(4, dim=1).repeat_interleave(4, dim=2)  # (B, 64, 64)
    x_sss_up = x_sss_down.repeat_interleave(4, dim=1).repeat_interleave(4, dim=2)  # (B, 64, 64)
    diff_sss = (Y_sss_up - x_sss_up) * scaling_factor * 1.5  # (B, 64, 64)

    # SST + SSH (already at 64x64)
    diff_sst = (Y_sst - x_batch[:, 1]) * 1.5
    diff_ssh = (Y_ssh - x_batch[:, 2]) * 1.5

    # Stack shape (B, 3, 64, 64)
    difference = torch.stack([diff_sss, diff_sst, diff_ssh], dim=1).to(device).requires_grad_()

    # === Vectorized loss ===
    # Here we compute a simple dot-product guidance: (difference * X_pred).sum() for all samples
    loss = (X_batch * difference).view(B, -1).sum(dim=1).sum()  # Scalar

    # === Gradient with respect to input x_batch ===
    grad = torch.autograd.grad(loss, x_batch, retain_graph=True)[0]  # shape (B, 3, 64, 64)

    return grad