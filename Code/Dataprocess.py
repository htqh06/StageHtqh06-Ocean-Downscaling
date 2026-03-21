"""
Programme de extraction des données pour generation des ensembles Train, Valid,
et Test pour l'entrainement. Voir main.y.

Autheur: Maximilian Wemaere

V.2 version modifiee par C.Mejia (Locean / IPSL) (22 fev. 2024)

"""
import xarray as xr 
import sys, os, platform, glob
from datetime import datetime
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

print('torch version:', torch.__version__)

#%%

#date = datetime.now().strftime("%m_%d_%H:%M_")
date = datetime.now().strftime("%Y-%m-%d_%Hh%M")
node = platform.node()
print(node,date.replace('-','/').replace('_',' '),end='')

#verbose = True
verbose = False

save_flag = True
#save_flag = False


# -----------------------------------------------------------------------------
# Data are normally from 1993 to 2020
# --> data for Test set .... 2020
#     data for Valid ....... 2019
#     data for Train set ... 1993-2018
#
test_year = 2020
valid_year = 2019
# -----------------------------------------------------------------------------

# to clear NaN pixels by linear interpolation
interp_nan = True
#interp_nan = False

# -----------------------------------------------------------------------------
# Initialisations:
select_label = None
select_time_step = None
# -----------------------------------------------------------------------------
# Zone selection (in pixels):
# -----------------------------------------------------------------------------
# Current data covering NATL60 zone (Sargass sea):
#in Mercator GLORYS model data, grid resolution is 1/12 degree and has a size
# in pixels of 270 x 216 (latitude: 216, longitude: 270) and 10227 time steps.
# For selecting a sub-rectangle or a sub-sampling of data declare one or both
# SELECT_LABEL and SELECT_TIME_STEP variables below.
# -----------------------------------------------------------------------------
#select_label = '-128x128'   # see if/else case with select_label below
#select_label = '-96x96'   # dont forget to add a new case in the if/else with select_label below
# -----------------------------------------------------------------------------
#select_time_step = 4   # take une data on 4
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Data dir name:
data_dir_name = 'DataV2'
nb_train_data = 256     # generates 10 files if select_time_step = 4, 38 if None
#nb_train_data = 98     # generates 97 (avec select_time_step = None )
#
data_dir_name += f'-l{nb_train_data}'
#
if select_label is not None or select_time_step is not None :
    data_dir_name += '-'
    if select_label is not None :
        data_dir_name += f'sel{select_label}'
    if select_time_step is not None :
        data_dir_name += f's{select_time_step}'
# -----------------------------------------------------------------------------
print(f"data_dir_name: '{data_dir_name}'")

# -----------------------------------------------------------------------------
# Save base path:
if node == 'acratopotes' :
    #save_base_path = "/data/labo/data/RESAC_Mercator/Data_MWemaere"
   # save_base_path = "/datatmp/data/resac/RESAC_Mercator/Data_Grison_Rossello"
    #save_base_path = "/datatmp/home/projetslong/vgrison/data/"
    save_base_path = "/datatmp/home/projetslong/donnees/resac_mercator/data_ssh"

    #save_base_path = "../resac_data_link"
#
save_path = os.path.join(save_base_path, data_dir_name)
# -----------------------------------------------------------------------------


if save_flag :
    print(" ** save_flag enabled **\n Save path:", save_path)
    os.makedirs(save_path, exist_ok=True)
# `nb_train_data` is `l_files` in main:
#nb_train_data = 98    # as original (as Maximilien code) (generates 97 files by variable for train)
#nb_train_data = 366   # (generates 26 files by variable for train) 
#nb_train_data = 1187  # (generates 8 files by variable for train) too big, out of memory at 6-th epoch: OutOfMemoryError: CUDA out of memory. Tried to allocate 2.08 GiB. GPU 0 has a total capacty of 23.47 GiB of which 885.94 MiB is free. Including non-PyTorch memory, this process has 22.58 GiB memory in use. Of the allocated memory 22.06 GiB is allocated by PyTorch, and 269.53 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF
#nb_train_data = 256; select_time_step=4 # (generates 10 files by variable for train) too big, out of memory at 6-th epoch: OutOfMemoryError: CUDA out of memory. Tried to allocate 2.08 GiB. GPU 0 has a total capacty of 23.47 GiB of which 885.94 MiB is free. Including non-PyTorch memory, this process has 22.58 GiB memory in use. Of the allocated memory 22.06 GiB is allocated by PyTorch, and 269.53 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF

#%%

def divisors(n):
    from itertools import chain
    from math import sqrt
    
    return set(chain.from_iterable((i,n//i) for i in range(1,int(sqrt(n))+1) if n%i == 0))

def remove_seasonality(arr_train,arr_test,leap=False):  
    from sklearn.linear_model import LinearRegression
    #from sklearn.preprocessing import PolynomialFeatures
    #from sklearn.pipeline import make_pipeline
    #from scipy.ndimage import uniform_filter1d
    #from statsmodels.tsa.seasonal import seasonal_decompose

    for i in tqdm(range(arr_test.shape[2])):
        for j in range(arr_test.shape[3]):
            y = torch.squeeze(arr_train[:,:,i,j]).numpy().reshape(-1, 1)
            X = np.arange(len(y)).reshape(-1, 1)
            X2 = np.arange(len(y),len(y)+len(arr_test)).reshape(-1, 1)
            #degree=1
            #model=make_pipeline(PolynomialFeatures(degree),LinearRegression())
            model = LinearRegression()
            model.fit(X, y)
            trend = model.predict(X)
            trend_train = model.predict(X2)
            arr_train[:,:,i,j]-=torch.tensor(trend)
            arr_test[:,:,i,j]-=torch.tensor(trend_train)

            #DecomposeResult = seasonal_decompose(arr_train[:,i,j],model='additive',period=365)
            #arr_train[:,i,j]-=DecomposeResult.seasonal
            #if leap:
            #    arr_test[:,i,j]-=DecomposeResult.seasonal[-366:]       
            #else:
            #    arr_test[:,i,j]-=DecomposeResult.seasonal[-365:]            

    return arr_train,arr_test


def histo_matching(arr_train,arr_test,ref):     # ref = torch.squeeze(arr_train_model.mean(dim=0)).numpy()  
    from skimage.exposure import match_histograms

    for i in range(len(arr_train)):
        arr_train[i] = torch.tensor(match_histograms(torch.squeeze(arr_train[i]).numpy(), ref)) # from skimage.exposure import match_histograms

    for i in range(len(arr_test)):
        arr_test[i] = torch.tensor(match_histograms(torch.squeeze(arr_test[i]).numpy(), ref))

    return arr_train,arr_test


def pool_images(sarray,pool):
    return pool(sarray)

def save_dims(fname,dims,verbose=False):
    """

    Parameters
    ----------
    fname : TYPE
        DESCRIPTION.
    dims : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    Example of reading a pickel:

    import pickle
    
    # Read dictionary pkl file
    with open('ssh_train_dims.pkl', 'rb') as fp:
        ssh_train_dimsB = pickle.load(fp)
        print('ssh_train_dims dictionary')
        print(ssh_train_dimsB)

    """
    filename = os.path.join(save_path, fname+"_dims"+".pt")
    with open(filename, 'wb') as fp:
        pickle.dump(dims, fp)
        if verbose:
            print(f'{fname} dictionary saved successfully to file')

    
def save_test(arr_test,mean,std,mod=False,sst=False,sst_mod=False,u=False,v=False,dims=None):

    arr_test -= mean
    arr_test /= std
    arr_test = torch.flip(arr_test,[2])
    
    print("Saving a TEST set of size: ",arr_test.shape)
    if mod:
        fname = "test_ssh_mod"
    elif sst:
        fname = "test_sst"
    elif sst_mod:
        fname = "test_sst_mod"
    elif u:
        fname = "test_u"
    elif v:
        fname = "test_v"
    else:
        fname = "test_ssh_sat"
    
    torch.save(arr_test,os.path.join(save_path, fname+".pt"))
    if dims is not None:
        save_dims(fname, dims)


def save_valid(arr_valid,mean,std,mod=False,sst=False,sst_mod=False,u=False,v=False,dims=None):

    print("Saving a VALD set of size: ",arr_valid.shape)
    arr_valid = torch.flip(arr_valid,[2])
    if mod:
        fname = "valid_ssh_mod"
    elif sst:
        fname = "valid_sst"
    elif sst_mod:
        fname = "valid_sst_mod"
    elif u:
        fname = "valid_u"
    elif v:
        fname = "valid_v"
    else:
        fname = "valid_ssh_sat"
    
    torch.save(arr_valid,os.path.join(save_path, fname+".pt"))
    if dims is not None:
        save_dims(fname, dims)


def save_train(arr_train,mean,std,mod=False,sst=False,sst_mod=False,u=False,v=False,nbyfile=98,dims=None, timevar='time'):

    print("Saving in a TRAIN set of size: ",arr_train.shape)
    print(f"  saving it in {np.ceil(arr_train.shape[0]/nbyfile):.0f} files of {nbyfile} patterns in each file (max.): ")

    n_patt = arr_train.shape[0]
    arr_train = torch.flip(arr_train,[2])
    time_dims = None
    if dims is not None:
        time_dims = dims['time']
    for n,i in enumerate(tqdm(range(0,len(arr_train),nbyfile))):
        #print(i, end=', ')
        i_end = i+nbyfile if i+nbyfile < n_patt else n_patt
        
        #print('i to i_end:',i,i_end, end='')
        arr = arr_train[i:i_end,:,:,:].clone()
        if dims is not None:
            dims['time'] = time_dims[i:i_end]

        #print(', size:',arr.shape)
        arr = arr.to(torch.float32)
        if mod:
            fname = "ssh_mod"
        elif sst:
            fname = "sst"
        elif sst_mod:
            fname = "sst_mod"
        elif u:
            fname = "u"
        elif v:
            fname = "v"
        else:
            fname = "ssh_sat"
        
        torch.save(arr,os.path.join(save_path, f"{fname}_{n:02d}.pt"))
        if dims is not None:
            save_dims(f"{fname}_{n:02d}", dims)
    print()


def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]


def get_dims_and_coords (ds, var) :
    dims = ds[var].dims
    dic = {}
    
    for d in dims :
        dic[d] = ds[var].coords[d]
    
    return dic


def interp_nan_from_ds(ds, var, timevar='time', before_plot=False, after_plot=False, before_after_plot=False, show_nan_dates=False):
    
    # Interpolating isolated NaN pixels ... 

    if type(var) is not list :
        var = [ var ]
    
    for v in var:
        da = ds[v]
        
        print(f"\n Interpolating isolated NaN pixels for '{v}' ... ", end='')
        boolisnan = np.isnan(da.values)
        if np.sum(boolisnan.flatten()) == 0 :
            print(" ** no NaN pixels found **   Continue ...")
        else:
            booldayisnan = np.sum(boolisnan,axis=(1,2))
            print(f"\n  - Nb. of NaN pixels found: {np.sum(boolisnan.flatten())}")
            print(f"  - no. of time-steps having NaNs: {np.sum(booldayisnan)}")
            
            before_da = da.mean(dim=[timevar],skipna=False).copy()
        
            if before_plot:
                # showing the NaN pixel in the mean througth time
                plt.imshow(da.mean(dim=[timevar],skipna=False).to_numpy())
                plt.title(f"  '{v}' before NaN interpolation  :-\\")
                plt.show()
                
            # gets 3D array having NaNs flatten to a 1D vector
            y_tmp = da.to_numpy().flatten()
            
            nans_tmp, x_tmp = nan_helper(y_tmp)
            
            y_tmp[nans_tmp] = np.interp(x_tmp(nans_tmp), x_tmp(~nans_tmp), y_tmp[~nans_tmp])
            
            arr_tmp = y_tmp.reshape(da.shape)
            
            # put back 3D array where NaN where interpollated
            ds[v] = ((timevar,'latitude','longitude'),arr_tmp)
            
            if after_plot:
                plt.imshow(da.mean(dim=[timevar],skipna=False).to_numpy())
                plt.title(f"'{v}' After NaN interpolation  ;-)")
                plt.show()
        
            if show_nan_dates:
                # to show dates of pixels
                tmp_time = np.array(ds[timevar].values).astype(np.datetime64)
                x_arr  = x_tmp(nans_tmp)
                print(f"\n  '{v}' Nan pixels in {len(x_arr)} dates:\n    [",end='')
                [print(f"{pd.to_datetime(str(tmp_time[i//np.prod(before_da.shape)])).strftime('%Y.%m.%d')}, ",end='') for i in x_arr];
                print(']')
            
            if before_after_plot:
                
                after_da = ds[v].mean(dim=[timevar],skipna=False).copy()
        
                # plot before and after interpolation, showing in a average of all time steps in a singe plane;
                # a NaN here means that at list one pixel is NaN in this position, considering all time steps.
                nan_before = np.array([ 1 if np.isnan(p) else 0 for p in before_da.values.flatten()]).reshape(before_da.shape)
                nan_before_da = xr.DataArray(nan_before,
                                             [('latitude', before_da['latitude'].values),
                                              ('longitude', before_da['longitude'].values),])
                fig,(ax1,ax2,ax3) = plt.subplots(nrows=1,ncols=3,sharex=True, sharey=True, figsize=(18, 4.5))
                
                before_da.plot(ax=ax1)
                ax1.set_title(f"  '{v}' before NaN interpolation  :-\\\n(mean over all time-steps)")
                
                nan_before_da.plot(ax=ax2)
                ax2.set_title('  NaN pixels in yellow')
                
                # showing that NaN pixel has desapeared ...
                after_da.plot(ax=ax3)
                ax3.set_title('  After ... No NaNs ;-)\n(mean over all time-steps)')
                
                fig.show()
        
    return ds

#%%

### ###############################################################################################
### Lecture des donnees Modele
### ###############################################################################################

### ###############################################################################################
### Repertoire de base des fichiers des donnees source:
### 
### Fichiers NetCDF telecharges du site Copernicus.
### Produits:
###     - GLORYS12V1_PRODUCT_001_030 (donnees modele ssh, u, v a une resolution de 1/12 de degre) et
###
### Et plus bas, les donnéees satellite:
###     - SST_PRODUCT_010_024 (donnees sst satellite, resolition de 1/20 de degre)
### ###############################################################################################

##### formating mod datas
#data_path = "/data/labo/data/Copernicus/OutputData/data_LON-64-42-LAT+26+44/GLORYS12V1_PRODUCT_001_030/"
if node == 'acratopotes' :
    base_data_path = "/data/labo/data/Copernicus/OutputData/data_LON-64-42-LAT+26+44"
    
elif os.path.exists('/net/acratopotes/data/labo/data/Copernicus'):
    base_data_path = "/net/acratopotes/data/labo/data/Copernicus/OutputData/data_LON-64-42-LAT+26+44"
    
elif os.path.exists('/net/ether/data/varclim/carlos/data/downscaling/Copernicus'):
    base_data_path = "/net/ether/data/varclim/carlos/data/downscaling/Copernicus/OutputData/data_LON-64-42-LAT+26+44"

elif os.path.exists('/Users/carlos/Downloads/Copernicus'):
    base_data_path = "/Users/carlos/Downloads/Copernicus/OutputData/data_LON-64-42-LAT+26+44"

else:
    print("\n *** data_process error: base_data_path not found. Stopping !!! ***\n")
    sys.exit(1)

print(f"\n base_data_path found: '{base_data_path}'")

data_path = os.path.join(base_data_path,"GLORYS12V1_PRODUCT_001_030")
files = "glorys12v1_mod_product_001_030_*.nc"

filenames = glob.glob(os.path.join(data_path,files))

if False :
    filenames = np.sort(filenames).tolist()
    print("Reading files ... ",filenames)

print(f"\nReading {len(filenames)} NetCDF files containing SSH, U & V data... ")
# lecture toutes des donnees modele en un Dataset
mf_mod_ds = xr.open_mfdataset(filenames, engine='netcdf4')

size_mod_grid = mf_mod_ds.latitude.shape[0], mf_mod_ds.longitude.shape[0]
nb_time_steps = mf_mod_ds.time.shape[0]

if verbose:
    print(mf_mod_ds)

print(f"\n Model data grid size: {size_mod_grid}")
print(f" and {nb_time_steps} time steps.")
#%%

### ###############################################################################################
### Pre traitements des données:
###   - Selection de sous-region ,
###   - sous-echantillonnage temporel,
###   - effacement des NaN par interpolation
### ###############################################################################################

# ----------------------------------------------------
# duplique le Dataset pour traitement
# ----------------------------------------------------
mf_ds = mf_mod_ds.copy()
    
# ----------------------------------------------------
# Selection d'une sous-region
# ----------------------------------------------------
if select_label is not None:
    # lighten data arrays:
    #
    # In case of 128x128, from 216 we keep 128 latitudes top latitudes (the northest ones)
    # and from 270 longitudes we keep only first 128 (the westernst ones)
    #
    # Data dimensions goes from (10227, 216, 270) to (10227, 128, 128)
    #
    if select_label == '-128x128':
        sel_n_lat = -128   # top (North) latitudes
        sel_n_lon = 128    # West longitudes
    
    elif select_label == '-96x96' :
        sel_n_lat = -96   # top (North) latitudes
        sel_n_lon = 96    # West longitudes
    
    else:
        print(f"\n *** Unknown select_label case '{select_label}'. Add it to the if/else case ...")
        raise
    
    # we keep N longitudes top latitudes (the last onas are the northest ones)
    lat_min = mf_ds.latitude.values[sel_n_lat] - 0.01
    lat_max = None

    # we keep first 40 longitudes (the first are the western ones)
    lon_min = None
    lon_max = mf_ds.longitude.values[sel_n_lon] - 0.01

    # new region selected Dataset
    size_region_factor = (
        mf_ds.latitude.shape[0] * mf_ds.longitude.shape[0]) / (np.abs(sel_n_lat) * np.abs(sel_n_lon))
    print(
        f"\n - selecting a smaller pixels sub-region. Size varies from [{mf_ds.latitude.shape[0]} x {mf_ds.longitude.shape[0]}] to [{np.abs(sel_n_lat)} x {np.abs(sel_n_lon)}] ... [Reduce pixel number by a factor of {size_region_factor:.1f}]")
    mf_ds = mf_ds.sel(longitude=slice(lon_min, lon_max)).sel(
        latitude=slice(lat_min, lat_max))

# ----------------------------------------------------
# Sous-echantillonnage dans l'axe de temps 
# ----------------------------------------------------
if select_time_step is not None :
    all_times = mf_ds.time
    new_times = all_times[np.arange(0,all_times.shape[0],select_time_step)]
    size_timestep_factor = all_times.shape[0] / new_times.shape[0]
    print(f"\n - selecting 1 time step from {select_time_step}. Reduce nb.of time steps from {all_times.shape[0]} to {new_times.shape[0]} ... [Reduce patterns number by a factor of {size_timestep_factor:.1f}]")

    # new time reduced Dataset
    mf_ds = mf_ds.sel(time=new_times)


if False:   # if True then some Plots !!
    # pour comparer en cas de sous-echantillionage de 'time' (si select_time_step n'est pas None)
    
    fig,(ax1,ax2,ax3) = plt.subplots(nrows=1,ncols=3,sharex=True, sharey=True, figsize=(18, 4.5))
    
    ax1.axis('image')
    ax2.axis('image')
    ax3.axis('image')

    k = 5   # time step to show
    
    k1 = k*select_time_step
    k2 = k
    da1 = mf_mod_ds['sla'].isel(time=k1)
    da2 = mf_ds['sla'].isel(time=k2)
    da3 = da1 - da2
    
    da1.plot(x='longitude',y='latitude', ax=ax1)
    da2.plot(x='longitude',y='latitude', ax=ax2)
    da3.plot(x='longitude',y='latitude', ax=ax3)
    ax1.set_title(f"T={k1} ({ax1.get_title()})")
    ax2.set_title(f"T={k2} ({ax2.get_title()})")
    ax3.set_title("difference")
    #ax3.set_title(f"difference ({ax3.get_title()})")

if False:   # more Plots !!
    
    fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2,sharex=True, sharey=True, figsize=(12, 4.5))
    
    ax1.axis('image')
    ax2.axis('image')

    k = 1000  # time step to show
    
    mf_ds['sla'].isel(time=k).plot(x='longitude',y='latitude', ax=ax1)
    mf_ds['sst'].isel(time=k).plot(x='longitude',y='latitude', ax=ax2)


# ----------------------------------------------------
# Effacemet des NaN par interpolation des pixels a NaN
# ----------------------------------------------------
# Attention a n'appliquer que en cas des pixels isolés, la pocedure n'est conseillee
# non plus pour resoudre des pixels en frontiere droite ou gauche de la grille.
# L'interpollation se fait de maniere lineaire a l'aide des pixels voisins, a priori,
# dans meme ligne (les pixels a gauche et a droite). Comme l'interpollation se fait
# de maniere lineaire sue l'array 3D applati (flattened), si le pixel NaN est au bout
# de la ligne dans la carte 2D ou 3D alors, apres applatissement un de ces voisins sera
# une valeur venant de l'autre bout d'une autre ligne, celle d'avant ou celle d'apres.
if interp_nan:
    
    mf_ds = interp_nan_from_ds(mf_ds,['sla', 'uo', 'vo'], before_after_plot=True, show_nan_dates=True)


#%%

# split Datasets in Test and the rest (that will be splitted in Val and Train):
#
# Attention, on considere la derniere annee des donnees, 2020, pour Test
# l'avant-derniere, 2019, pour Valid et pour Train le reste (1993 a 2018).
# ** Re-voir le processus de separation si vous changez ces conditions **
#
# Uses xarray Datasets possibilities to split data sets instead of split_sets()
# function defined here.
#
# Test en premier (derniere année des donnees, annee 2020)
mf_test_ds = mf_ds.loc[dict(time=slice(str(test_year), str(test_year)))]
mf_trainval_ds = mf_ds.loc[dict(time=slice(None, str(test_year-1)))]


#%%

# Spliting the rest of data:
# Data without Test, splitted in Val and Train:
#
# Attention, on considere la derniere annee des donnees pour Test
# l'avant-derniere, 2019, pour Valid et pour Train le reste (1993 - 2018).
# Re-voyer le processus de separation si vous changez ces conditions

# Valid en second (avant-derniere année, 2019)
mf_valid_ds = mf_trainval_ds.loc[dict(time=slice(str(valid_year), str(valid_year)))]

# pour Train le reste en dernier (2018 et avant)
mf_train_ds = mf_trainval_ds.loc[dict(time=slice(None, str(valid_year-1)))]

#%%

### ###############################################################################################
###
### Traitements de la SSH (ou 'sla' - Sea Level Anomalie  [m]):
###
### ###############################################################################################

ds_varname = 'sla'   # nom de la SSH dans le dataset

if False:
    # differentes manieres de calculer les MEAN et STD de l'array 3D
    mean_by_year_da = mf_trainval_ds[ds_varname].mean(dim=['latitude','longitude'],skipna=True)
    std_by_year_da = mf_trainval_ds[ds_varname].std(dim=['latitude','longitude'],skipna=True)
    
    mean_by_pixel_da = mf_trainval_ds[ds_varname].mean(dim=['time'],skipna=True)
    std_by_pixel_da = mf_trainval_ds[ds_varname].std(dim=['time'],skipna=True)
    
    meanx = np.mean(mean_by_year_da.to_numpy())
    stdx = np.mean(std_by_year_da.to_numpy())
    
    print(f"By year:\n mean: {meanx}\n std : {stdx}")

    meanx = np.mean(mean_by_pixel_da.to_numpy())
    stdx = np.mean(std_by_pixel_da.to_numpy())
    
    print(f"By pixel:\n mean: {meanx}\n std : {stdx}")
        
    meanx = np.nanmean(mf_trainval_ds[ds_varname].to_numpy())
    stdx = np.nanstd(mf_trainval_ds[ds_varname].to_numpy())

    print(f"meanx: {meanx}\nstdx : {stdx}")

    meanx = np.nanmean(mf_trainval_ds[ds_varname].to_numpy().flatten())
    stdx = np.nanstd(mf_trainval_ds[ds_varname].to_numpy().flatten())

    print(f"meanx: {meanx}\nstdx : {stdx}")

# calcule la MEAN et la STD sur les donnees Train+Valid (calcules de considerant tous les pixels et tous les time-steps):
## get and mean and std from general train set ...
ssh_mean = np.nanmean(mf_trainval_ds[ds_varname].to_numpy())
ssh_std = np.nanstd(mf_trainval_ds[ds_varname].to_numpy())

print(f"SSH:\n mean: {ssh_mean}\n std : {ssh_std}")

# arrays
ssh_train = torch.tensor(mf_train_ds[ds_varname].values).to(torch.float32)
ssh_train = torch.unsqueeze(ssh_train,1)

ssh_valid = torch.tensor(mf_valid_ds[ds_varname].values).to(torch.float32)
ssh_valid = torch.unsqueeze(ssh_valid,1)

ssh_test = torch.tensor(mf_test_ds[ds_varname].values).to(torch.float32)
ssh_test = torch.unsqueeze(ssh_test,1)

# dimensions des arrays
ssh_train_dims = get_dims_and_coords(mf_train_ds, var=ds_varname)
ssh_valid_dims = get_dims_and_coords(mf_valid_ds, var=ds_varname)
ssh_test_dims = get_dims_and_coords(mf_test_ds, var=ds_varname)

print(f"\nSplited sets (year {test_year} for Test, {valid_year} for Valid, the rest for Train):\nssh_train, ssh_valid and ssh_test sizes:\n    ",
      ssh_train.shape, "\n    ",ssh_valid.shape, "\n    ",ssh_test.shape)

# ... and normalize Train and Valid (Test will be done later)
ssh_train -= ssh_mean
ssh_train /= ssh_std

ssh_valid -= ssh_mean
ssh_valid /= ssh_std

print("ssh train nb patterns",ssh_train.shape[0],"\nand divisors:",divisors(ssh_train.shape[0]))

#%%

# ----------------------------------------------------------------------------------------------------
# sauvegarde des differents sets de donnees pour la SSH mod ...
# ----------------------------------------------------------------------------------------------------

if save_flag :
    print(f"\nSaving [mean,std] tensors and normalized data for SSH mod ...\npath= '{save_path}'")

    filemane = os.path.join(save_path,"mean_std_ssh_mod"+".pt")
    print("  Saving SSH mod [mean,std] tensors in file:",filemane)
    torch.save(torch.tensor([ssh_mean,ssh_std]),filemane)

    # normalize and save test set
    #save test set, normalized with general train mean and std
    print("  Normaizing and saving SSH mod test set ...")
    save_test(ssh_test,ssh_mean,ssh_std,mod=True,dims=ssh_test_dims)

    # save normalized valid set
    print("  Saving normalized SSH mod valid set ...")
    save_valid(ssh_valid,ssh_mean,ssh_std,mod=True,dims=ssh_valid_dims)
    
    # save normalized train set
    print("  Saving normalized SSH mod train set in multiple files ...")
    save_train(ssh_train,ssh_mean,ssh_std,mod=True,nbyfile=nb_train_data,dims=ssh_train_dims)
    
else:
    print("\n ** Saving SSH mod data tensors not enabled **\n")

  
#%%

"""
if False :
    # donnees Sat SSH - NON pas pour l'instant
    
    # save train set for histogram matching
    #ref = torch.squeeze(ssh_train.mean(dim=0)).numpy()
    #realize histogram matching
    #ssh_train,ssh_test = histo_matching(ssh_train,ssh_test,ref)

    if save_flag :
        print("\nSaving [mean,std] tensors and normalized data for SSH sat ...")
    
        #normalize and save test set
        print("Saving test set ...")
        save_test(ssh_test,ssh_mean,std)
    
        ## split train and valid set
        ssh_train,ssh_valid= split_sets(ssh_train,year=valid_year)   
    
        # normalize and save valid set
        print("Saving valid set ...")
        save_valid(ssh_valid,ssh_mean,std)
    
    
        # normalize and save train set
        print("Saving train set ...")
        save_train(ssh_train,ssh_mean,std,nbyfile=nb_train_data)
        
    else:
        print("\n ** Saving SSH sat data tensors not enabled **\n")
"""


#%%

if True:

    #%%
    ### ###############################################################################################
    ###
    ### Traitements de la U (ou 'uo' - Eastward velocity [m/s]):
    ###
    ### ###############################################################################################

    ds_varname = 'uo'    # nom de la composante U dans le dataset

    # calcule la MEAN et la STD sur les donnees Train+Valid (calcules de considerant tous les pixels et tous les time-steps):
    ## get and mean and std from general train set ...
    u_mean = np.nanmean(mf_trainval_ds[ds_varname].to_numpy())
    u_std = np.nanstd(mf_trainval_ds[ds_varname].to_numpy())
    
    print(f"U:\n mean: {u_mean}\n std : {u_std}")
    
    # arrays
    u_train = torch.tensor(mf_train_ds[ds_varname].values).to(torch.float32)
    u_train = torch.unsqueeze(u_train,1)
    
    u_valid = torch.tensor(mf_valid_ds[ds_varname].values).to(torch.float32)
    u_valid = torch.unsqueeze(u_valid,1)
    
    u_test = torch.tensor(mf_test_ds[ds_varname].values).to(torch.float32)
    u_test = torch.unsqueeze(u_test,1)
        
    print(f"\nSplited sets (year {test_year} for Test, {valid_year} for Valid, the rest for Train):\nu_train, u_valid and u_test sizes:\n    ",
          u_train.shape, "\n    ",u_valid.shape, "\n    ",u_test.shape)
    
    # ... and normalize Train and Valid (Test will be done later)
    u_train -= u_mean
    u_train /= u_std
    
    u_valid -= u_mean
    u_valid /= u_std


#%%

    # ----------------------------------------------------------------------------------------------------
    # sauvegarde des differents sets de donnees pour U mod ...
    # ----------------------------------------------------------------------------------------------------
    
    if save_flag :
        print(f"\nSaving [mean,std] tensors and normalized data for U mod ...\npath= '{save_path}'")
    
        filemane = os.path.join(save_path,"mean_std_u"+".pt")
        print("  Saving U mod [mean,std] tensors in file:",filemane)
        torch.save(torch.tensor([u_mean,u_std]),filemane)

        # normalize and save test set
        print("  Normaizing and saving U mod test set ...")
        save_test(u_test,u_mean,u_std,u=True)
        
        # save normalized valid set
        print("  Saving normalized U mod valid set ...")
        save_valid(u_valid,u_mean,u_std,u=True)
    
        # save normalized train set
        print("  Saving normalized U mod train set in multiple files ...")
        save_train(u_train,u_mean,u_std,u=True,nbyfile=nb_train_data)    # use same rand_prem than mod !
    
    else:
        print("\n ** Saving U mod data tensors not enabled **\n")

#%%

if True:

    #%%
    ### ###############################################################################################
    ###
    ### Traitements de la V (ou 'vo' - Northward velocity [m/s]):
    ###
    ### ###############################################################################################

    ds_varname = 'vo'    # nom de la composante V dans le dataset

    # calcule la MEAN et la STD sur les donnees Train+Valid (calcules de considerant tous les pixels et tous les time-steps):
    ## get and mean and std from general train set ...
    v_mean = np.nanmean(mf_trainval_ds[ds_varname].to_numpy())
    v_std = np.nanstd(mf_trainval_ds[ds_varname].to_numpy())
    
    print(f"V:\n mean: {v_mean}\n std : {v_std}")
    
    # arrays
    v_train = torch.tensor(mf_train_ds[ds_varname].values).to(torch.float32)
    v_train = torch.unsqueeze(v_train,1)
    
    v_valid = torch.tensor(mf_valid_ds[ds_varname].values).to(torch.float32)
    v_valid = torch.unsqueeze(v_valid,1)
    
    v_test = torch.tensor(mf_test_ds[ds_varname].values).to(torch.float32)
    v_test = torch.unsqueeze(v_test,1)
        
    print(f"\nSplited sets (year {test_year} for Test, {valid_year} for Valid, the rest for Train):\nv_train, v_valid and v_test sizes:\n    ",
          v_train.shape, "\n    ",v_valid.shape, "\n    ",v_test.shape)
    
    # ... and normalize Train and Valid (Test will be done later)
    v_train -= v_mean
    v_train /= v_std
    
    v_valid -= v_mean
    v_valid /= v_std
    
    
    if save_flag :
        print(f"\nSaving [mean,std] tensors and normalized data for V mod ...\npath= '{save_path}'")

        filemane = os.path.join(save_path,"mean_std_v"+".pt")
        print("  Saving V mod [mean,std] tensors in file:",filemane)
        torch.save(torch.tensor([v_mean,v_std]),filemane)
    
        # normalize and save test set
        print("  Normaizing and saving V mod test set ...")
        save_test(v_test,v_mean,v_std,v=True)
        
        # save normalized valid set
        print("  Saving normalized V mod valid set ...")
        save_valid(v_valid,v_mean,v_std,v=True)
    
        # save normalized train set
        print("  Saving normalized V mod train set in multiple files ...")
        save_train(v_train,v_mean,v_std,v=True,nbyfile=nb_train_data)    # use same rand_prem than before !
        
    else:
        print("\n ** Saving V mod data tensors not enabled **\n")

    
#%%

if True:
    
    #%%
    ### ###############################################################################################
    ### Lecture des donnees Satellite
    ### ###############################################################################################
    
    ### ###############################################################################################
    ### 
    ### Produits:
    ###     - SST_PRODUCT_010_024 (donnees sst satellite, resolition de 1/20 de degre)
    ### ###############################################################################################

    ##### formating sat sst data
    #data_path = "/data/labo/data/Copernicus/OutputData/data_LON-64-42-LAT+26+44/SST_PRODUCT_010_024/"

    data_path = os.path.join(base_data_path,"SST_PRODUCT_010_024")
    files = "sst_sat_product_010_024_*.nc"

    filenames = glob.glob(os.path.join(data_path,files))

    if False :
        filenames = np.sort(filenames).tolist()
        print("Reading files ... ",filenames)
    
    print(f"\nReading {len(filenames)} NetCDF files containing SST data ... ")
    mf_sat_ds = xr.open_mfdataset(filenames, engine='netcdf4')

    if verbose:
        print(mf_sat_ds)
    
    size_sat_grid = mf_sat_ds.latitude.shape[0], mf_sat_ds.longitude.shape[0]
    
    print(f"\n Satellite data grid size: {size_sat_grid}")

#%%

    ### ###############################################################################################
    ###
    ### Traitements de la SST sat (ou 'analysed_sst' - analysed sea surface temperature  [°K]):
    ###
    ### ###############################################################################################


    ds_varname = 'analysed_sst'   # nom de la SST dans le dataset

    print(f"\n Sat data grid size will be re-grided from size {size_sat_grid} to same size pixel points as selected Model data.")
    
    mf_sat_da = mf_sat_ds[ds_varname]
        
    # pour eviter les NaN lors de l'interpolation ajouter kwargs={"fill_value": "extrapolate"}
    new_mf_sat_da = mf_sat_da.interp(longitude=mf_ds.longitude, latitude=mf_ds.latitude,
                                     kwargs={"fill_value": "extrapolate"})
    
    if True: # some plots
        fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2,sharex=True, sharey=True, figsize=(12, 4.5))
        
        ax1.axis('image')
        ax2.axis('image')

        k = 100  # time step to show
        vmin,vmax = np.min(mf_sat_da.isel(time=k).values),np.max(mf_sat_da.isel(time=k).values)
                                                           
        mf_sat_da.isel(time=k).plot(x='longitude',y='latitude', ax=ax1, vmin=vmin, vmax=vmax)
        ax1.set_title(ax1.get_title()+f" ({mf_sat_da.latitude.shape[0]}x{mf_sat_da.longitude.shape[0]}) pix.\n(original grid)")
        
        new_mf_sat_da.isel(time=k).plot(x='longitude',y='latitude', ax=ax2, vmin=vmin, vmax=vmax)
        ax2.set_title(ax2.get_title()+f" ({new_mf_sat_da.latitude.shape[0]}x{new_mf_sat_da.longitude.shape[0]}) pix.\n(re-interpolated grid, ssh-mod like)")
        fig.show()
        
    # split Datasets in Test, Valid and Train usin times from model Datasets:    
    mf_sat_test_da = new_mf_sat_da.sel(time=mf_test_ds.time)
    mf_sat_valid_da = new_mf_sat_da.sel(time=mf_valid_ds.time)
    mf_sat_train_da = new_mf_sat_da.sel(time=mf_train_ds.time)

    mf_sat_trainval_da = new_mf_sat_da.sel(time=mf_trainval_ds.time)
    
    # calcule la MEAN et la STD sur les donnees Train+Valid (calcules de considerant tous les pixels et tous les time-steps):
    ## get and mean and std from general train set ...
    sst_mean = np.mean(mf_sat_trainval_da.to_numpy())
    sst_std = np.std(mf_sat_trainval_da.to_numpy())

    print(f"SST:\n mean: {sst_mean}\n std : {sst_std}")
    
    # arrays
    sst_train = torch.tensor(mf_sat_train_da.values).to(torch.float32)
    sst_train = torch.unsqueeze(sst_train,1)
    
    sst_valid = torch.tensor(mf_sat_valid_da.values).to(torch.float32)
    sst_valid = torch.unsqueeze(sst_valid,1)
    
    sst_test = torch.tensor(mf_sat_test_da.values).to(torch.float32)
    sst_test = torch.unsqueeze(sst_test,1)

    print(f"\nSplited sets (year {test_year} for Test, {valid_year} for Valid, the rest for Train):\nsst_train, sst_valid and sst_test sizes:\n    ",
          sst_train.shape, "\n    ",sst_valid.shape, "\n    ",sst_test.shape)
    
    # ... and normalize Train and Valid (Test will be done later)
    sst_train -= sst_mean
    sst_train /= sst_std
    
    sst_valid -= sst_mean
    sst_valid /= sst_std
    
    #print("sst train nb patterns",sst_train.shape[0],"\nand divisors:",divisors(sst_train.shape[0]))
    
    #%%
    
    # ----------------------------------------------------------------------------------------------------
    # sauvegarde des differents sets de donnees pour la SST sat ...
    # ----------------------------------------------------------------------------------------------------
        
    if save_flag :
        print("\nSaving [mean,std] tensors and normalized data for SST sat ...\npath= '{save_path}'")
    
        filemane = os.path.join(save_path,"mean_std_sst"+".pt")
        print("  Saving SST sat [mean,std] tensors in file:",filemane)
        torch.save(torch.tensor([sst_mean,sst_std]),filemane)

        #realize histogram matching
        #sst_train,sst_test = histo_matching(sst_train,sst_test,ref)
    
        #normalize and save test set
        print("  Normaizing and saving SST sat test set ...")
        save_test(sst_test,sst_mean,sst_std,sst=True)
        
        # normalize and save valid set
        print("  Saving normalized SST sat valid set ...")
        save_valid(sst_valid,sst_mean,sst_std,sst=True)
        
        # normalize and save train set
        print("  Saving normalized SST sat train set in multiple files ...")
        save_train(sst_train,sst_mean,sst_std,sst=True,nbyfile=nb_train_data)
    
    else:
        print("\n ** Saving SST sat data tensors not enabled **\n")

n_gen_train_files = len(glob.glob(os.path.join(save_base_path,data_dir_name,'ssh_mod_*[0-9].pt')))

print(f"\n {'*'*80}\n * end of process"+\
      f"\n *   generated data label ... '{data_dir_name}'"+\
      f"\n *   in path ................ '{save_base_path}/'"+\
      f"\n *"+\
      f"\n * SSH Train where splitted in {n_gen_train_files} files. Thus, put next line in main:"+\
      f"\n *  data_dirname = '{data_dir_name}'; l_files,n_files = {nb_train_data},{n_gen_train_files}"+\
      f"\n {'*'*80}\n")

