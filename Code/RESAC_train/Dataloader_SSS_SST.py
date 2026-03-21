import os
import torch
import numpy as np

class Dataset_rsc(torch.utils.data.Dataset):
    def __init__(self, l_files, n_files, data_path, file_name_sss, file_name_sst, first_file=0):
        super().__init__()
        self.l_files = l_files
        self.n_files = n_files - first_file
        self.data_path = data_path
        self.first_file = first_file

        self.file_name_sss = file_name_sss
        self.file_name_sst = file_name_sst

        self.pool = torch.nn.AvgPool2d(2, stride=2)
        self.pool2 = torch.nn.AvgPool2d(4, stride=4)

        self.samples_per_file = l_files
        self.total_samples = self.samples_per_file * self.n_files

        self.current_file_index = None

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        file_idx = idx // self.samples_per_file + self.first_file
        sample_idx = idx % self.samples_per_file

        if self.current_file_index != file_idx:
            self.current_file_index = file_idx

            # Load current file into memory
            path_sss = os.path.join(self.data_path, f'{self.file_name_sss}{file_idx:02d}.npy')
            path_sst = os.path.join(self.data_path, f'{self.file_name_sst}{file_idx:02d}.npy')

            self.sss_data = torch.from_numpy(np.load(path_sss)).float()
            self.sst_data = torch.from_numpy(np.load(path_sst)).float()

        # Random spatial crop (64x64)

        r1 = np.random.randint(0, 60)
        r2 = np.random.randint(0, 60)

        sss_12 = self.sss_data[sample_idx, :, r1:r1+64, r2:r2+64]
        sst_12 = self.sst_data[sample_idx, :, r1:r1+64, r2:r2+64]

        sss_6 = self.pool(sss_12.unsqueeze(0)).squeeze(0)
        sst_6 = self.pool(sst_12.unsqueeze(0)).squeeze(0)
        sss_3 = self.pool2(sss_12.unsqueeze(0)).squeeze(0)

        return sss_3, sss_6, sss_12, sst_6, sst_12  
            
        
class ConcatData_rsc(torch.utils.data.Dataset):
    def __init__(self,datasets,shuffle=False,batch_size=1):
        super().__init__()
        self.datasets = datasets
        self.batch_size = batch_size

        if shuffle:
            n = len(datasets[0])
            id_rd = torch.randperm(n)
            for d in self.datasets:
                d = d[list(id_rd)]

    def __getitem__(self,i):
        self.datasets[0][(i+1)*self.batch_size]
        return tuple(d[i*self.batch_size:(i+1)*self.batch_size] for d in self.datasets)


    def __len__(self):
        return min(int(len(d)/self.batch_size) for d in self.datasets)

