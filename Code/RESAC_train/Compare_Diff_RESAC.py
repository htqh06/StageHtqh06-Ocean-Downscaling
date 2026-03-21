import math
import torch
import diffusers
import jsonargparse
from pathlib import Path
import os
import matplotlib.pyplot as plt
import numpy as np
import sys
#----------------------------------------------------------------------
enzo_code = False
if enzo_code :
    sys.path.append(str(Path(__file__).resolve().parent))  # Add the current directory to sys.path
    sys.path.append('/datatmp/home/eforestier/Diffusion_SSS/Fast_sst_64x64')
    
    data_path = "/datatmp/home/eforestier/Copernicus_processed_data"
    checkpt_path = "/datatmp/home/eforestier/Diffusion_SSS"
    model_path = "/datatmp/home/eforestier/Downscaling_SSS"

else:
    #sys.path.append("../RESAC_train")
    sys.path.append(str(Path(__file__).resolve().parent))  # Add the current directory to sys.path
    sys.path.append("../../CodeEnzo/Diffusion_SSS/Fast_sst_64x64")
    
    data_path = "../data/Copernicus_processed_data"
    checkpt_path = "../../CodeEnzo/Diffusion_SSS/Fast_sst_64x64"
    model_path = "../../CodeEnzo/Downscaling_SSS"

from diff_sst_fast64 import DiffusionModel_sst
from obs_operator_sst import *
from guided_sampling_sst import *
from archi_SSS_SST import *
from plot_utils_SSS_SST import *
from Dataloader_SSS_SST import *

##########################################################################

size =64
device = torch.device("cpu")  # Use GPU if available

index = 0

##########################################################################

fname_sss = os.path.join(data_path, 'so_test.npy')
fname_sst = os.path.join(data_path, 'thetao_test.npy')
################################################################################

# Load the data

pool = torch.nn.AvgPool2d(2,stride=(2,2))
pool2 = torch.nn.AvgPool2d(4,stride=(4,4))
r1, r2 = 10,30

test_sss_12 = torch.Tensor(np.load(os.path.join(data_path, 'so_test.npy'))[:, :, r1:size + r1, r2:size + r2])

test_sst_12 = torch.Tensor(np.load(os.path.join(data_path, 'thetao_test.npy'))[:, :, r1:size + r1, r2:size + r2])

test_sss_6 = pool(torch.Tensor(np.load(os.path.join(data_path, 'so_test.npy'))[:, :, r1:size + r1, r2:size + r2]))

test_sst_6 = pool(torch.Tensor(np.load(os.path.join(data_path, 'thetao_test.npy'))[:, :, r1:size + r1, r2:size + r2]))

test_sss_3 = pool2(torch.Tensor(np.load(os.path.join(data_path, 'so_test.npy'))[:, :, r1:size + r1, r2:size + r2]))

test_loader = ConcatData_rsc([test_sss_3,test_sss_6,test_sss_12,test_sst_6,test_sst_12],shuffle=False)

##########################################################################

# Load the models

# ------------------------------------------------------------------------------------
#checkpoint_guidance_diff = 'Fast_sst_64x64/Save/lightning_logs/version_3/checkpoints/epoch=79-step=24320.ckpt'
#checkpoint_guidance_diff = 'Save/lightning_logs/version_3/checkpoints/epoch=79-step=24320.ckpt'
#checkpoint_guidance_diff = 'Save/lightning_logs/version_4/checkpoints/epoch=99-step=29600.ckpt'
#checkpoint_guidance_diff = 'Save/lightning_logs/version_6/checkpoints/epoch=69-step=2590.ckpt'
checkpoint_guidance_diff = 'Save/lightning_logs/version_7/checkpoints/epoch=99-step=3700.ckpt'
# ------------------------------------------------------------------------------------

checkpoint_diff = os.path.join(checkpt_path, checkpoint_guidance_diff)


# ------------------------------------------------------------------------------------
#checkpoint_state_file = "Save/2025-04-25_17h25_resac-n76-SSS_SST/2025-04-25_17h25_resac.pth"   # Orig in Code ... *** No such file or directory ***
#checkpoint_state_file = "Save/2025-05-05_15h10_resac-n37-SSS_SST/2025-05-05_15h10_resac.pth"
#checkpoint_state_file = "Save/2025-07-11_14h15_resac-n37-SSS_SST/2025-07-11_14h15_resac.pth"
#checkpoint_state_file = "Save/2025-07-18_15h07_resac-n37-SSS_SST/2025-07-18_15h07_resac.pth"
#checkpoint_state_file = "Save/2025-07-18_16h02_resac-n37-SSS_SST/2025-07-18_16h02_resac.pth"
#checkpoint_state_file = "Save/2025-07-22_13h33_resac-n37-SSS_SST/2025-07-22_13h33_resac.pth"
#
checkpoint_state_file = "Save/2026-02-03_17h12_resac-n37-SSS_SST/2026-02-03_17h12_resac.pth"; model_path = '.'
checkpoint_state_file = "Save/2026-02-04_09h16_resac-n37-SSS_SST/2026-02-04_09h16_resac.pth"; model_path = '.'
# ------------------------------------------------------------------------------------

criterion1 = RMSELoss()
model_RESAC = resac_v2()
model_RESAC.to(device)
model_RESAC.load_state_dict(torch.load(os.path.join(model_path, checkpoint_state_file), map_location=torch.device(device)))
model_RESAC.eval()

mean,l_im = model_RESAC.test(criterion1, test_loader, device, data_path, get_im=[index])
for n,line in enumerate(l_im):
    [sss6_rsc,sss12_rsc,target] = line
sss12_rsc = torch.squeeze(sss12_rsc).cpu().numpy()
target = torch.squeeze(target).cpu().numpy()
output_12_diff, true_sss, true_sst = guidance_sst(checkpoint_diff, fname_sss, fname_sst, index=index, r1=r1,r2=r2, num_timesteps=50, seed=544156154)
sss12_diff = output_12_diff[0]
#true_sss = torch.squeeze(true_sss).numpy()

##########################################################################

# Calculate the global min and max values for the color scale
global_min = min(sss12_rsc.min(), sss12_diff.min(), true_sss.min())
global_max = max(sss12_rsc.max(), sss12_diff.max(), true_sss.max())

# Calculate RMSE for sss12_rsc and sss12_diff
rmse_rsc = np.sqrt(np.mean((sss12_rsc - true_sss) ** 2))
rmse_diff = np.sqrt(np.mean((sss12_diff - true_sss) ** 2))

# Calculate mean and std for each image
mean_rsc, std_rsc = np.mean(sss12_rsc), np.std(sss12_rsc)
mean_diff, std_diff = np.mean(sss12_diff), np.std(sss12_diff)
mean_true, std_true = np.mean(true_sss), np.std(true_sss)
mean_true_pt, std_true_pt = np.mean(target), np.std(target)

# Create a figure with 1 row and 3 columns
fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # Adjust to 3 columns

# Plot l_im (RESAC Output)
im_rsc = axes[0].imshow(sss12_rsc, cmap="BuGn", vmin=global_min, vmax=global_max)
axes[0].set_title(f'RESAC Output\nMean: {mean_rsc:.4f}, Std: {std_rsc:.4f}, RMSE: {rmse_rsc:.4f}')
axes[0].axis('off')

# Plot final_image_diff (Diffusion Output)
im_diff = axes[1].imshow(sss12_diff, cmap="BuGn", vmin=global_min, vmax=global_max)
axes[1].set_title(f'Diffusion Output\nMean: {mean_diff:.4f}, Std: {std_diff:.4f}, RMSE: {rmse_diff:.4f}')
axes[1].axis('off')

# Plot true_sss (True SSS)
im_true = axes[2].imshow(target, cmap="BuGn", vmin=global_min, vmax=global_max)
axes[2].set_title(f'True SSS\nMean: {mean_true:.4f}, Std: {std_true:.4f}')
axes[2].axis('off')

# Add a colorbar to the figure
cbar = fig.colorbar(im_true, ax=axes, orientation='horizontal', fraction=0.02, pad=0.1)
cbar.set_label('SSS Value')

# Adjust layout and show the figure
plt.tight_layout()
plt.show()
