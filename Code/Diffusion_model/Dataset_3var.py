import os
import torch
import numpy as np

#the train and valid datasets load numpy files of 1/12° sss, sst and ssh, select random patches of 64x64, and concatenates them in a tensor

class Dataset_3var_train(torch.utils.data.Dataset):
    def __init__(self, l_files, n_files, data_path, file_name_sss, file_name_sst, file_name_ssh, first_file=0):
        super().__init__()
        self.l_files = l_files
        self.data_path = data_path
        self.file_name_sss = file_name_sss
        self.file_name_sst = file_name_sst
        self.file_name_ssh = file_name_ssh
        self.first_file = first_file
        self.n_files = n_files - first_file
        self.total_samples = self.n_files * self.l_files
        self.current_file_idx = None

        self.d_sss = None
        self.d_sst = None
        self.d_ssh = None

    def __len__(self):
        return self.total_samples

    def load_current_file(self, file_idx):
        fname_sss = os.path.join(self.data_path, f'{self.file_name_sss}{file_idx:02d}.npy')
        fname_sst = os.path.join(self.data_path, f'{self.file_name_sst}{file_idx:02d}.npy')
        fname_ssh = os.path.join(self.data_path, f'{self.file_name_ssh}{file_idx:02d}.npy')
        print

        # Random crop (on the CPU for speed gain)
        r1 = np.random.randint(0, 60)
        r2 = np.random.randint(0, 60)
        d_sss_np = np.load(fname_sss)[:, :, r1:r1+64, r2:r2+64]
        d_sst_np = np.load(fname_sst)[:, :, r1:r1+64, r2:r2+64]
        d_ssh_np = np.load(fname_ssh)[:, r1:r1+64, r2:r2+64]

        self.d_sss = torch.from_numpy(d_sss_np).float().to(memory_format=torch.channels_last)
        self.d_sst = torch.from_numpy(d_sst_np).float().to(memory_format=torch.channels_last)
        self.d_ssh = torch.from_numpy(d_ssh_np).float().unsqueeze(1).to(memory_format=torch.channels_last)
        self.current_file_idx = file_idx

    def __getitem__(self, idx):
        file_idx = idx // self.l_files + self.first_file
        sample_idx = idx % self.l_files

        if self.current_file_idx != file_idx:
            self.load_current_file(file_idx)

        sss = self.d_sss[sample_idx]
        sst = self.d_sst[sample_idx]
        ssh = self.d_ssh[sample_idx]  

        return torch.cat((sss, sst, ssh), dim=0)
    
class Dataset_3var_valid(torch.utils.data.Dataset):
    def __init__(self, data_path, file_name_sss, file_name_sst, file_name_ssh, crop_size=64, crop_positions=None):
        super().__init__()
        self.data_path = data_path
        self.file_name_sss = file_name_sss
        self.file_name_sst = file_name_sst
        self.file_name_ssh = file_name_ssh
        self.crop_size = crop_size

        fname_sss = os.path.join(self.data_path, self.file_name_sss)
        fname_sst = os.path.join(self.data_path, self.file_name_sst)
        fname_ssh = os.path.join(self.data_path, self.file_name_ssh)

        self.d_sss_full = torch.Tensor(np.load(fname_sss))
        self.d_sst_full = torch.Tensor(np.load(fname_sst))
        self.d_ssh_full = torch.Tensor(np.load(fname_ssh))

        height = self.d_sss_full.shape[2]
        width = self.d_sss_full.shape[3]
        max_r1 = height - self.crop_size
        max_r2 = width - self.crop_size

        default_positions = [
            (0, 0),
            (0, max_r2),
            (max_r1, 0),
            (max_r1, max_r2),
            (max_r1 // 2, max_r2 // 2),
        ]
        self.crop_positions = crop_positions if crop_positions is not None else default_positions

        self.l_files = self.d_sss_full.shape[0]
        self.total_samples = self.l_files * len(self.crop_positions)

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        sample_idx = idx % self.l_files
        crop_idx = idx // self.l_files
        r1, r2 = self.crop_positions[crop_idx]

        sss = self.d_sss_full[sample_idx, :, r1:r1 + self.crop_size, r2:r2 + self.crop_size]
        sst = self.d_sst_full[sample_idx, :, r1:r1 + self.crop_size, r2:r2 + self.crop_size]
        ssh = self.d_ssh_full[sample_idx, r1:r1 + self.crop_size, r2:r2 + self.crop_size].unsqueeze(0)

        return torch.cat((sss, sst, ssh), dim=0)
