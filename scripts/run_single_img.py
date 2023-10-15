import math
import os
import json
import sys
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from dotmap import DotMap
from dataset.dataset_pitch import PitchDataset
from torch.utils.data import DataLoader
from model.model_pitch import PitchModel_LSTM, PitchModel_Transformer, PitchModel_Linear, PitchModel_CNN, PitchModel_CREPE, PitchModel_CNN10, PitchModel_CNNAE
from utils.trainer_tester import trainer, tester


# python scripts/run_single_img.py configs/config_lstm.json --gpuid 0


def load_json(f_path):
    with open(f_path, 'r') as f:
        return json.load(f)

def save_json(obj, f_path):
    with open(f_path, 'w') as f:
        json.dump(obj, f, ensure_ascii=False)
        
def load_config(config_path):
    config_json = load_json(config_path)
    config = DotMap(config_json)   
    return config

def seed_everything(seed):
    random.seed(seed)                     
    torch.manual_seed(seed)               
    torch.cuda.manual_seed_all(seed)      
    np.random.seed(seed)                  
    os.environ['PYTHONHASHSEED'] = str(seed)
    

def main(args):
    
    config = load_config(args.config)
    seed_everything(config.seed)
    
    if args.gpuid >= 0:
        config.gpu_device = args.gpuid
    if config.cuda:
        device = torch.device('cuda:{}'.format(config.gpu_device))
    else:
        device = torch.device('cpu')
    
    full_dataset = PitchDataset(config)
    train_size = int(0.8 * len(full_dataset))
    val_size = int(0.1 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=config.optim_params.batch_size)
    val_loader = DataLoader(val_dataset, batch_size=config.optim_params.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=config.optim_params.batch_size)
    loader_dict = {'train': train_loader, 'val': val_loader, 'test': test_loader}
    del train_dataset
    del val_dataset
    del test_dataset

    if config.model == 0:
        model = PitchModel_LSTM(config, device)
    elif config.model == 1:
        model = PitchModel_Transformer(config, device)
    elif config.model == 2:
        model = PitchModel_Linear(config, device)
    elif config.model == 3:
        model = PitchModel_CNN(config, device)
    elif config.model == 4:
        model = PitchModel_CREPE(config, device)
    elif config.model == 10:
        model = PitchModel_CNN10(config, device)
    elif config.model == 11:
        model = PitchModel_CNNAE(config, device)
    else:
        print('model error')
        exit()  
    
    model.load_state_dict(torch.load(os.path.join(config.outdir, 'best_val_loss_model.pth'))) 
    model.to(device)
    
    os.makedirs(config.img_params.data_dir, exist_ok=True)
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            
            if i >= 10: break
            
            pred, target, pred2, _ = model.inference(batch)
            pred = pred.reshape(-1)
            pred2 = pred2.reshape(-1)
            target = target.reshape(-1)
            pred = pred.detach().cpu().numpy()
            pred2 = pred2.detach().cpu().numpy()
            target = target.detach().cpu().numpy()
            pred_all = pred * pred2
            
            t = len(pred)
            start = config.img_params.start // config.img_params.data_shift
            end = config.img_params.end // config.img_params.data_shift
            t_datas = [t_data * config.img_params.data_shift for t_data in range(start, end)]
            plt.plot(t_datas, target[start:end], color='red', label='target')
            plt.plot(t_datas, pred_all[start:end], color='blue', label='pred')
            plt.legend()
            plt.savefig(os.path.join(config.img_params.data_dir, f'{batch[2][0]}_{config.img_params.start}ms_{config.img_params.end}ms.png'))
            plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, default='configs/config.path', help='path to config file')
    parser.add_argument('--gpuid', type=int, default=-1, help='gpu device id')
    args = parser.parse_args()
    main(args)