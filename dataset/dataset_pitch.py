import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


class PitchDataset(Dataset):
    
    def __init__(self, config):
        self.config = config
        self.data_path = config.data_params.data_dir
        names = os.listdir(os.path.join(self.data_path, 'f0'))
        file_names = [name.replace('_f0.npy', '') for name in names]
        self.file_names = file_names
        self.data = self.get_item()
    
       
    def get_item(self):
        data = []
        for i, file_name in enumerate(tqdm(self.file_names)):
            # if i >= 10: break
            spec = np.load(os.path.join(self.data_path, 'spec', f'{file_name}_spec.npy'))
            spec = torch.tensor(spec, dtype=torch.float32)
            pitch = np.load(os.path.join(self.data_path, 'f0', f'{file_name}_f0.npy'))
            pitch = torch.tensor(pitch, dtype=torch.float32)
            t = len(pitch)
            N = 2000
            for i in range(t//N):
                start = N * i
                end = N * (i + 1)
                batch = {'spec':spec[start:end], 'pitch':pitch[start:end], 'file_name': file_name}
                data.append(batch) 
            # batch = {'spec':spec, 'pitch':pitch, 'file_name': file_name}
            # data.append(batch)

        return data
    
    
    def __getitem__(self, index):
        batch = self.data[index]        
        return list(batch.values())
    
    
    def __len__(self):
        return len(self.data)