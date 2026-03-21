import xarray as xr 
import os
import torch
from tqdm import tqdm
import sys
from scipy.ndimage import uniform_filter1d
from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from skimage.exposure import match_histograms

# pip install pytorch-histogram-matching
save_path = "/datatmp/home/eforestier/test_RESAC/Data_test/"


def remove_seasonality(ssh_train,ssh_test,leap=False):  

    for i in tqdm(range(ssh_test.shape[2])):
        for j in range(ssh_test.shape[3]):
            y = torch.squeeze(ssh_train[:,:,i,j]).numpy().reshape(-1, 1)
            X = np.arange(len(y)).reshape(-1, 1)
            X2 = np.arange(len(y),len(y)+len(ssh_test)).reshape(-1, 1)
            degree=1
            #model=make_pipeline(PolynomialFeatures(degree),LinearRegression())
            model = LinearRegression()
            model.fit(X, y)
            trend = model.predict(X)
            trend_train = model.predict(X2)
            ssh_train[:,:,i,j]-=torch.tensor(trend)
            ssh_test[:,:,i,j]-=torch.tensor(trend_train)

            #DecomposeResult = seasonal_decompose(ssh_train[:,i,j],model='additive',period=365)
            #ssh_train[:,i,j]-=DecomposeResult.seasonal
            #if leap:
            #    ssh_test[:,i,j]-=DecomposeResult.seasonal[-366:]       
            #else:
            #    ssh_test[:,i,j]-=DecomposeResult.seasonal[-365:]            

    return ssh_train,ssh_test


def histo_matching(ssh_train,ssh_test,ref):     # ref = torch.squeeze(ssh_train_model.mean(dim=0)).numpy()  
    for i in range(len(ssh_train)):
        ssh_train[i] = torch.tensor(match_histograms(torch.squeeze(ssh_train[i]).numpy(), ref)) # from skimage.exposure import match_histograms

    for i in range(len(ssh_test)):
        ssh_test[i] = torch.tensor(match_histograms(torch.squeeze(ssh_test[i]).numpy(), ref))

    return ssh_train,ssh_test

def split_sets(ssh_array,year=2017,leap=False):
    n_leap = (year-1993)//4
    # if leap:
    #     ssh_test = ssh_array[365*(year-1993)+n_leap:365*(year+1-1993)+n_leap+1,:,:].clone()
    # else:
    #     ssh_test = ssh_array[365*(year-1993)+n_leap:365*(year+1-1993)+n_leap,:,:].clone()
    if leap:
        ssh_test = ssh_array[365*(year-1993)+n_leap+1:,:,:,:].clone()
        ssh_train = ssh_array[:365*(year-1993)+n_leap+1,:,:,:].clone()
    else:
        ssh_test = ssh_array[365*(year-1993)+n_leap:,:,:,:].clone()
        ssh_train = ssh_array[:365*(year-1993)+n_leap,:,:,:].clone()

    # if len(ssh_array[365*(year+1-1993)+n_leap:])==0 or len(ssh_array[365*(year+1-1993)+n_leap+1:])==0:
    #     ssh_train = ssh_array[:365*(year-1993)+n_leap,:,:].clone()
    # else:
    #     if leap:
    #         ssh_train = torch.concat((ssh_array[0:365*(year-1993)+n_leap,:,:],ssh_array [365*(year+1-1993)+n_leap+1:,:,:]),axis=0)
    #     else:
    #         ssh_train = torch.concat((ssh_array[0:365*(year-1993)+n_leap,:,:],ssh_array [365*(year+1-1993)+n_leap:,:,:]),axis=0)
    return ssh_train,ssh_test



def pool_images(sarray,pool):
    return pool(sarray)



def save_test(ssh_test,mean,std,mod=False,sst=False,sst_mod=False,u=False,v=False):

    ssh_test -= mean
    ssh_test /= std
    ssh_test = torch.flip(ssh_test,[2])
    if mod:
        torch.save(ssh_test,save_path+"test_ssh_mod"+".pt")
    elif sst:
        torch.save(ssh_test,save_path+"test_sst"+".pt")
    elif sst_mod:
        torch.save(ssh_test,save_path+"test_sst_mod"+".pt")
    elif u:
        torch.save(ssh_test,save_path+"test_u"+".pt")
    elif v:
        torch.save(ssh_test,save_path+"test_v"+".pt")
    else:
        torch.save(ssh_test,save_path+"test_ssh_sat"+".pt")

def save_valid(ssh_valid,mean,std,mod=False,sst=False,sst_mod=False,u=False,v=False):

    ssh_valid = torch.flip(ssh_valid,[2])
    if mod:
        torch.save(ssh_valid,save_path+"valid_ssh_mod"+".pt")
    elif sst:
        torch.save(ssh_valid,save_path+"valid_sst"+".pt")
    elif sst_mod:
        torch.save(ssh_valid,save_path+"valid_sst_mod"+".pt")
    elif u:
        torch.save(ssh_valid,save_path+"valid_u"+".pt")
    elif v:
        torch.save(ssh_valid,save_path+"valid_v"+".pt")
    else:
        torch.save(ssh_valid,save_path+"valid_ssh_sat"+".pt")



def save_train(ssh_array,mean,std,mod=False,sst=False,sst_mod=False,u=False,v=False):

    ssh_array = torch.flip(ssh_array,[2])
    n=0
    for i in tqdm(range(0,len(ssh_array),98)):
        ssh = ssh_array[i:i+98,:,:,:].clone()
        ssh = ssh.to(torch.float32)
        if mod:
            torch.save(ssh,save_path+"ssh_mod_"+str(n)+".pt")
        elif sst:
            torch.save(ssh,save_path+"sst_"+str(n)+".pt")
        elif sst_mod:
            torch.save(ssh,save_path+"sst_mod_"+str(n)+".pt")
        elif u:
            torch.save(ssh,save_path+"u_"+str(n)+".pt")
        elif v:
            torch.save(ssh,save_path+"v_"+str(n)+".pt")
        else:
            torch.save(ssh,save_path+"ssh_sat_"+str(n)+".pt")
        n+=1




if True:  


    ##### formating mod data
    data_path = "/data/labo/data/Copernicus/OutputData/data_LON-64-42-LAT+26+44/GLORYS12V1_PRODUCT_001_030/"

    files = "glorys12v1_mod_product_001_030_*.nc"


    filenames = os.path.join(data_path,files)

    mf_ds = xr.open_mfdataset(filenames)


    ssh_array = torch.tensor(mf_ds['sla'].values).to(torch.float32)
    ssh_array = torch.unsqueeze(ssh_array,1)

    time = np.array(mf_ds['time'].values).astype(np.datetime64)

    #define pool layer to size the mod data
    #pool = torch.nn.AvgPool2d(3,stride=(3,3))

    ## pool images
    #ssh_array = pool_images(ssh_array,pool)


    #mean_dic = compute_anual_mean(ssh_array,time)
    #ssh_array = substract_annual_mean(ssh_array,time,mean_dic)

    ## split train and test set
    ssh_train,ssh_test = split_sets(ssh_array,year=2020,leap=True)


    ## remove trend and seasonality
    #ssh_train,ssh_test = remove_seasonality(ssh_train,ssh_test,leap=False)
    


    ## get and save mean and std
    mean = torch.mean(ssh_train)
    ssh_train-=mean
    std = torch.std(ssh_train) 
    ssh_train/=std

    torch.save(torch.tensor([mean,std]),save_path+"mean_std_mod"+".pt")

    ##save test set
    save_test(ssh_test,mean,std,mod=True)

    ## split train and valid set
    ssh_train,ssh_valid= split_sets(ssh_train,year=2019)   

    # normalize and save valid set
    save_valid(ssh_valid,mean,std,mod=True)

    # normalize and save train set
    save_train(ssh_train,mean,std,mod=True) 
    
    # save train set for histogram matching
    ref = torch.squeeze(ssh_train.mean(dim=0)).numpy()

    
  

    ##### formating mod sat data
    data_path = "/data/labo/data/Copernicus/OutputData/data_LON-64-42-LAT+26+44/GLORYS12V1_PRODUCT_001_030/"

    files = "glorys12v1_mod_product_001_030_*.nc"

    filenames = os.path.join(data_path,files)

    mf_ds = xr.open_mfdataset(filenames)

    time = np.array(mf_ds['time'].values).astype(np.datetime64)

    ssh_array = torch.tensor(mf_ds['sla'].values).to(torch.float32)
    ssh_array = torch.unsqueeze(ssh_array,1)

    #mean_dic = compute_anual_mean(ssh_array,time)
    #ssh_array = substract_annual_mean(ssh_array,time,mean_dic)

    
    ## split train and test set
    ssh_train,ssh_test = split_sets(ssh_array,year=2020,leap=True)

    ## remove trend and seasonality
    #ssh_train,ssh_test = remove_seasonality(ssh_train,ssh_test,leap=False)

    ## get and save mean and std
    mean = torch.mean(ssh_train)
    ssh_train-=mean
    std = torch.std(ssh_train) 
    ssh_train/=std
    

    #realize histogram matching
    #ssh_train,ssh_test = histo_matching(ssh_train,ssh_test,ref)



    #normalize and save test set
    save_test(ssh_test,mean,std)

    ## split train and valid set
    ssh_train,ssh_valid= split_sets(ssh_train,year=2019)   

    # normalize and save valid set
    save_valid(ssh_valid,mean,std)


    # normalize and save train set
    save_train(ssh_train,mean,std)








if True:




     ##### formating u data 
    data_path = "/data/labo/data/Copernicus/OutputData/data_LON-64-42-LAT+26+44/GLORYS12V1_PRODUCT_001_030/"

    files = "glorys12v1_mod_product_001_030_*.nc"


    filenames = os.path.join(data_path,files)

    mf_ds = xr.open_mfdataset(filenames)


    ssh_array = torch.tensor(mf_ds['uo'].values).to(torch.float32)
    ssh_array = torch.unsqueeze(ssh_array,1)

    ssh_in = ssh_array.clone()

    ssh_train,ssh_test = split_sets(ssh_in,year=2020,leap=True)

    ## get and save mean and std
    mean = torch.mean(ssh_train)
    ssh_train-=mean
    std = torch.std(ssh_train) 
    ssh_train/=std
    torch.save(torch.tensor([mean,std]),save_path+"mean_std_u"+".pt")

    ##save test set
    save_test(ssh_test,mean,std,u=True)

    ## split train and valid set
    ssh_train,ssh_valid= split_sets(ssh_train,year=2019)   

    # normalize and save valid set
    save_valid(ssh_valid,mean,std,u=True)


    # normalize and save train set
    save_train(ssh_train,mean,std,u=True)    # use same rand_prem than sat !

    



    ##### formating v data 
    data_path = "/data/labo/data/Copernicus/OutputData/data_LON-64-42-LAT+26+44/GLORYS12V1_PRODUCT_001_030/"

    files = "glorys12v1_mod_product_001_030_*.nc"


    filenames = os.path.join(data_path,files)

    mf_ds = xr.open_mfdataset(filenames)


    ssh_array = torch.tensor(mf_ds['vo'].values).to(torch.float32)
    ssh_array = torch.unsqueeze(ssh_array,1)

    ssh_in = ssh_array.clone()

    ssh_train,ssh_test = split_sets(ssh_in,year=2020,leap=True)

    ## get and save mean and std
    mean = torch.mean(ssh_train)
    ssh_train-=mean
    std = torch.std(ssh_train) 
    ssh_train/=std
    torch.save(torch.tensor([mean,std]),save_path+"mean_std_v"+".pt")

    ##save test set
    save_test(ssh_test,mean,std,v=True)

    ## split train and valid set
    ssh_train,ssh_valid= split_sets(ssh_train,year=2019)   

    # normalize and save valid set
    save_valid(ssh_valid,mean,std,v=True)


    # normalize and save train set
    save_train(ssh_train,mean,std,v=True)    # use same rand_prem than sat !

    







if True:
    ##### formating sat sst data
    data_path = "/data/labo/data/Copernicus/OutputData/data_LON-64-42-LAT+26+44/SST_PRODUCT_010_024/"

    files = "sst_sat_product_010_024_*.nc"

    filenames = os.path.join(data_path,files)

    mf_ds = xr.open_mfdataset(filenames)

    ssh_array = torch.tensor(mf_ds['analysed_sst'].values).to(torch.float32)
    ssh_array = torch.unsqueeze(ssh_array,1)

    time = np.array(mf_ds['time'].values).astype(np.datetime64)

    #interpolate to good dimension
    ssh_array = torch.nn.functional.interpolate(ssh_array,size=(216,270),mode='bicubic')

    #mean_dic = compute_anual_mean(ssh_array,time)
    #ssh_array = substract_annual_mean(ssh_array,time,mean_dic)


    ## split train and test set
    ssh_train,ssh_test = split_sets(ssh_array,year=2020,leap=True)

    ## remove trend and seasonality
    #ssh_train,ssh_test = remove_seasonality(ssh_train,ssh_test,leap=False)

    ## get and save mean and std
    mean = torch.mean(ssh_train)
    ssh_train-=mean
    std = torch.std(ssh_train) 
    ssh_train/=std


    #realize histogram matching
    #ssh_train,ssh_test = histo_matching(ssh_train,ssh_test,ref)


    #normalize and save test set
    save_test(ssh_test,mean,std,sst=True)

    ## split train and valid set
    ssh_train,ssh_valid= split_sets(ssh_train,year=2019)   

    # normalize and save valid set
    save_valid(ssh_valid,mean,std,sst=True)
    
    # normalize and save train set
    save_train(ssh_train,mean,std,sst=True)






