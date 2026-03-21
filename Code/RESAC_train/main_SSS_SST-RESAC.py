"""
Programme d'entrainement RESAC_mercator (V.2.0):

    Super-resolution ou downscaling de la hauteur de la surface de l'ocean (sss - Sea
    Surface Height). Utilise une deuxième variable (la SST, Sea Surface Temperature) 
    plus haute resolution pour guider l'augmentation en resolution de la sss.
                      
    Version originale:
        Auteur: Maximilian Wemaere
        https://github.com/MxWmr/RESAC_mercator

V.2.0 ... Version modifiée par C.Mejia (Locean / IPSL) (22 fev. 2024)
    Paramétrisation pour permettre différentes tailles de données
    Modification du code dans data_process.py et main.py pour selection simple
    d'une sous-region et du sous-échantillonnage temporel.
    
--------

File originally in '.../Downscaling_SSS/main_SSS_SST.py'

A executer sur Acratopotes

"""

import os, re, platform
import torch
from datetime import datetime
from Dataloader_SSS_SST import *
import numpy as np 
from archi_SSS_SST import *
from plot_utils_SSS_SST import *
import time
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import wandb

#%%

train_flag = True
# train_flag = False

#n_epochs = 80
n_epochs = 20

bsize = 64

# test_flag = True
test_flag = False

l_files,n_files = 256,37

#----------------------------------------------------------------------
if test_flag :
    verbose = True
    #verbose = False
    
    save_figs = True
    #save_figs = False
    #--------------------
    figs_dir = 'Figs'
    fig_ext = '.png'
    #--------------------
    figs_defaults = { 'dpi' : 300, 'facecolor':'w', 'edgecolor' : 'w'} # ajuter format='png' ou autre a l'appel de savefig()
    #--------------------
    if save_figs and not os.path.exists(figs_dir):
        os.makedirs(figs_dir)
#----------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------
info_type = "SSS_SST"
case_label = f"resac-n{n_files}-{info_type}"
#time.sleep(60*60*10) 
#date = datetime.now().strftime("%m_%d_%H:%M_")
date = datetime.now().strftime("%Y-%m-%d_%Hh%M")
node = platform.node()
print(node,date.replace('-','/').replace('_',' '),end='')

start_time = time.time()

if torch.cuda.is_available():
    # Choix de la carte NVIDIA (voir, sur un terminal, la commande nvidia-smi   - Attention: numerotation inversee)
    device = "cuda:0"      # Nvidia RTX A5000 

#----------------------------------------------------------------------
enzo_code = False
if enzo_code :
    base_save_path="/datatmp/home/eforestier/Downscaling_SSS/Save"
    data_path = "/datatmp/home/eforestier/Copernicus_processed_data"

    if re.match(r"acratopotes*", node) :
        alreadySaved_path = "/datatmp/home/eforestier/Downscaling_SSS/Save"

else:
    base_save_path = "./Save"
    former_base_save_path = "../../CodeEnzo/Downscaling_SSS/Save"
    data_path = r"../data/Copernicus_processed_data"
    
    if re.match(r"acratopotes*", node) :
        alreadySaved_path = base_save_path


save_path = os.path.join(base_save_path,f"{date}_{case_label}")

file_for_test = os.path.join(data_path, "so_00.npy")
print(f"longueur du fichier test : {len(np.load(file_for_test))}")
#%%
print('Load and prepare datasets ...')
pool = torch.nn.AvgPool2d(2,stride=(2,2))
pool2 = torch.nn.AvgPool2d(4,stride=(4,4))
#train_loader = Dataset(l_files,n_files,data_path,'sss_mod_','sst_','u_','v_',batch_size=bsize,first_file=0) # first file 67 to start with 2011
train_dataset = Dataset_rsc(l_files=l_files, n_files=n_files, data_path=data_path, file_name_sss='so_', file_name_sst='thetao_') # first file 67 to start with 2011
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bsize, shuffle=True, num_workers=0, pin_memory=True)

#%%

test_sss_12 = np.load(os.path.join(data_path, 'so_test.npy'))[:,:,20:64+20,20:64+20]
test_sss_12 = torch.Tensor(test_sss_12)
test_sst_12 = np.load(os.path.join(data_path, 'thetao_test.npy'))[:,:,20:64+20,20:64+20]
test_sst_12 = torch.Tensor(test_sst_12)

test_sss_6 = pool(torch.Tensor(np.load(os.path.join(data_path, 'so_test.npy'))[:,:,20:64+20,20:64+20]))
test_sst_6 = pool(torch.Tensor(np.load(os.path.join(data_path, 'thetao_test.npy'))[:,:,20:64+20,20:64+20]))

test_sss_3 = pool2(torch.Tensor(np.load(os.path.join(data_path, 'so_test.npy'))[:,:,20:64+20,20:64+20]))

test_loader = ConcatData_rsc([test_sss_3,test_sss_6,test_sss_12,test_sst_6,test_sst_12],shuffle=False)

#%%

valid_sss_12 = np.load(os.path.join(data_path, 'so_val.npy'))[:,:,20:64+20,20:64+20]
valid_sss_12 = torch.Tensor(valid_sss_12)
valid_sst_12 = np.load(os.path.join(data_path, 'thetao_val.npy'))[:,:,20:64+20,20:64+20]
valid_sst_12 = torch.Tensor(valid_sst_12)

valid_sss_6 = pool(torch.Tensor(np.load(os.path.join(data_path, 'so_val.npy'))[:,:,20:64+20,20:64+20]))
valid_sst_6 = pool(torch.Tensor(np.load(os.path.join(data_path, 'thetao_val.npy'))[:,:,20:64+20,20:64+20]))

valid_sss_3 = pool2(torch.Tensor(np.load(os.path.join(data_path, 'so_val.npy'))[:,:,20:64+20,20:64+20]))

valid_loader = ConcatData_rsc([valid_sss_3,valid_sss_6,valid_sss_12,valid_sst_6,valid_sst_12],shuffle=False)

#%%

print('\nPreparing RESAC model ...')

criterion1 = RMSELoss()
model = resac_v2()

#%%

if train_flag:    #train
    os.makedirs(save_path, exist_ok=True)
    wandb.init(
        project="resac_sss",
        name= datetime.now().strftime("%Y-%m-%d_%Hh%M"),
        #entity="enzo-caperan")
        # entity="carlos-mejia-locean-su"
    )

    print(f'\nTraining RESAC model {case_label} BS-{bsize} [{date}] ...')

    optim = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim,factor=0.3,patience=8)
    #track training with wandb
    wandb.watch(model, criterion1, log="all", log_freq=1)
    training_start_time = time.time()
    model.fit(train_loader,valid_loader,n_epochs,device,criterion1,optim,data_path,scheduler)
    training_end_time = time.time()

    fit_elapsed_time = training_end_time - training_start_time
    time_epoch = fit_elapsed_time / n_epochs
    if time_epoch > 1 :
        time_epoch_label = 'sec/epoch'
    else:
        time_epoch = 1 / time_epoch
        time_epoch_label = 'epochs/sec'
  
    print(f" --- elapsed training time during {n_epochs} epochs: {training_end_time-training_start_time} seconds --- {time_epoch:.1f} {time_epoch_label} ---")
    torch.save(model.state_dict(), os.path.join(save_path,f'{date}_resac.pth'))
    alreadySaved_date = date
    
    #wandb.close()    # NON --> AttributeError: module 'wandb' has no attribute 'close'
    wandb.finish()
else:
    print('\n ** Training not enabled **')
    
#%%
alreadySaved_net_path = None

#alreadySaved_date = "2025-05-05_15h10"; current_base_save_path = former_base_save_path    # *** size mismatch ***
#alreadySaved_date = "2025-07-22_13h33"; current_base_save_path = former_base_save_path
#alreadySaved_date = "2026-02-03_17h12"; current_base_save_path = base_save_path
#alreadySaved_date = "2026-02-04_09h16"; current_base_save_path = base_save_path

if test_flag:   #test

    for alreadySaved_date, current_base_save_path in [["2025-07-22_13h33", former_base_save_path],
                                                      ["2026-02-03_17h12", base_save_path],
                                                      ["2026-02-04_09h16", base_save_path],
                                                      ] :
        
        alreadySaved_path = os.path.join(current_base_save_path,f"{alreadySaved_date}_{case_label}")

        n_test = test_loader.datasets[0].shape[0]
        
        print(f'\nTesting RESAC model {case_label} BS-{bsize} [{alreadySaved_date}] ...')
        print(f"{n_test} patterns in Test loader.")
        
        if alreadySaved_net_path is None :
            alreadySaved_net_path = os.path.join(alreadySaved_path,alreadySaved_date)
    
        device= 'cpu'
        model.load_state_dict(torch.load(os.path.join(alreadySaved_path,f'{alreadySaved_date}_resac.pth'), map_location=torch.device(device)))
        model = model.to(device)
    
        #k = 120
        #k = 64
        for k in [0, 120]:
            print(f" -> Testing with Test pattern {k} ...")
            mean,l_im = model.test(criterion1, test_loader, device, data_path, get_im=[k])
            #plot_test_sss(l_im,alreadySaved_path,date)
            plot_test_sss(l_im, figs_dir, alreadySaved_date, fig_lbl=f'(day {k}) [mean rmse={mean:.4f}]', save=save_figs)
            
            print('test RMSE sss 1/12: {}'.format(mean))

else:
    print('\n ** Test not enabled **')

end_time = time.time()
print(f"\n --- elapsed {case_label} BS-{bsize} [{alreadySaved_date}] process time {end_time-start_time} seconds ---")
