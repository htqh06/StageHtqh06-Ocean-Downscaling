import math
import torch
import diffusers
import numpy as np
import os
from diff_3var_fast64 import DiffusionModel_3var
from obs_operator_3var import *
from tqdm import tqdm


"""def extract_into_tensor(a, t, x_shape): #old functions, no use anymore
    b = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def predict_start_from_noise(x_t, t, noise, alphas_cumprod):
    return (
        extract_into_tensor(1 / alphas_cumprod, t, x_t.shape) * x_t -
        extract_into_tensor(1 / alphas_cumprod - 1, t, x_t.shape) * noise
    )"""

################### functions for the pseudo-inverse method, see the related article ################
def get_c1(alph_t, alph_s):
    numerator = (1 - alph_t / alph_s) * (1 - alph_s)
    denominator = (1 - alph_t)
    return torch.sqrt(numerator / denominator)


def get_c2(alph_t, alph_s):
   return torch.sqrt(1 - alph_s - get_c1(alph_t, alph_s) ** 2)
#####################################################################################################

################### functions for grad enhancement ##################################################
def enhance_grad(x0):
    x0_mean = upsample_to_original(downsample_to_mean(x0[0], block_size=2), block_size=2)
    x0[0] -= x0_mean
    c_std = torch.max(torch.abs(x0[0]))
    x0[0] *= 1.35-0.31*torch.pow(torch.abs(x0[0])/c_std, 0.5)
    x0[0] += x0_mean
    return x0
#####################################################################################################

########### function for performing guided sampling following the pseudo-inverse method #############

#Inputs : a diffusion model checkpoint, filenames of sss, sst and ssh files for guidance with position indexes r1 and r2, and other parameters for the guidance
#Outputs : generated, true and observed images
# -> the design of the function certainly needs improving

def guidance_3var(
    checkpoint: str,
    fname_sss,
    fname_sst,
    fname_ssh,
    index: int,
    size=64,
    r1=0,
    r2=0,
    num_timesteps: int = 50,
    seed: int = 526557,
    grad_enhancing: bool = True,
    sc=1, # parameter for running tests, keep at one 
    device=torch.device("cuda:0")
):
    torch.manual_seed(seed)
    torch.cuda.empty_cache()

    def load_tensor(fname, channel_idx=None):
        arr = np.load(fname)[index]
        if channel_idx is not None:
            arr = arr[channel_idx]
        return torch.tensor(arr[r1:r1+size, r2:r2+size], dtype=torch.float32, device=device)

    true_sss_12 = load_tensor(fname_sss, channel_idx=0)
    true_sst_12 = load_tensor(fname_sst, channel_idx=0)
    true_ssh_12 = load_tensor(fname_ssh)
    obs_sss_3_clean = downsample_to_mean(true_sss_12)
    obs_sss_3 = obs_sss_3_clean # change here if you want to add noise to the observation

    model = DiffusionModel_3var().to(device) #initialize diffusion model
    state = torch.load(checkpoint, map_location=device, weights_only=True) #load model's weights
    model.load_state_dict({k: v for k, v in state["state_dict"].items() if not (k.startswith("noise_buffer") or k.startswith("steps_buffer"))}, strict=False)
    model.eval()

    scheduler = diffusers.DDIMScheduler(
        beta_schedule="squaredcos_cap_v2",
        clip_sample=False
    )
    scheduler.set_timesteps(num_timesteps, device=device)

    alphas_cumprod = scheduler.alphas_cumprod.to(device)
    timesteps = scheduler.timesteps

    current_image = torch.randn(1, 3, size, size, device=device) #initialize the diffusion process with random noise
    current_image.requires_grad = True 

    for t_idx in range(num_timesteps - 1): #denoising timesteps
        t = timesteps[t_idx]
        ts_tensor = torch.full((1,), t, dtype=torch.long, device=device)

        x_in = scheduler.scale_model_input(current_image, t)
        predicted_noise = model.model(x_in, ts_tensor).sample #the model predicts the noise at current timestep

        alpha_t = alphas_cumprod[t]
        x0_pred = (current_image - predicted_noise * torch.sqrt(1 - alpha_t)) / torch.sqrt(alpha_t) #predict denoised image 
        x0_pred = x0_pred.squeeze(0)
        guidance_term = get_grad(obs_sss_3, true_sst_12, true_ssh_12, x0_pred, current_image, scaling_factor=sc) #see obs_operator_3var.py for the get_grad function 
        scaled_guidance = guidance_term * torch.sqrt(alpha_t) * 0.3 # for an unknown reason, the method is unstable (exploding values) if the guidance term isn't diminished by a factor of at least 0.3 (bellow 0.3, we lose guidance precision, and above is unstable)

        t_prev = timesteps[t_idx + 1]
        alpha_s = alphas_cumprod[t_prev]

        c1 = get_c1(alpha_t, alpha_s)
        c2 = get_c2(alpha_t, alpha_s)

        noise = torch.randn_like(predicted_noise)

        if grad_enhancing and t < 250: #we perform gradient enhancing only for the last quarter of the denoising step, once a coherent image has already emerged
            x0_pred = enhance_grad(x0_pred)

        current_image = torch.sqrt(alpha_s) * x0_pred + scaled_guidance + c1 * noise + c2 * predicted_noise #see article on pseudo-inverse guidance for details about this expression

    final_pred_noise = model.model(current_image, timesteps[-1]).sample # one last step of denoising without gradient enhancement or guidance, better for stability
    current_image = scheduler.step(final_pred_noise, timesteps[-1], current_image).prev_sample

    final_image = current_image.squeeze().detach().cpu().numpy()
    return final_image, true_sss_12.cpu().numpy(), true_sst_12.cpu().numpy(), true_ssh_12.cpu().numpy(), obs_sss_3_clean.cpu().numpy(), obs_sss_3.cpu().numpy()