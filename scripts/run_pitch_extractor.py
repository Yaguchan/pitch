import math
import os
import json
import sys
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
from dotmap import DotMap
from dataset.dataset_pitch import PitchDataset
from torch.utils.data import DataLoader
from model.model_pitch import PitchModel_LSTM, PitchModel_Transformer, PitchModel_Linear, PitchModel_CNN, PitchModel_CREPE, PitchModel_MB_CNNLSTM, PitchModel_CNN10, PitchModel_CNNAE
from utils.trainer_tester import trainer, tester


# python scripts/run_pitch_extractor.py configs/config_lstm.json --gpuid 0 


def load_json(f_path):
    with open(f_path, 'r') as f:
        return json.load(f)
        
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
    elif config.model == 5:
        model = PitchModel_MB_CNNLSTM(config, device)
    elif config.model == 10:
        model = PitchModel_CNN10(config, device)
    elif config.model == 11:
        model = PitchModel_CNNAE(config, device)
    else:
        print('model error')
        exit()  
    
    model.to(device)
    parameters = model.parameters()
    optimizer = torch.optim.AdamW(
        parameters,
        lr=config.optim_params.learning_rate
    )
    
    os.makedirs(os.path.join(config.outdir), exist_ok=True)
    trainer(
        num_epochs=config.num_epochs,
        model=model,
        loader_dict=loader_dict,
        optimizer=optimizer,
        device=device,
        outdir=os.path.join(config.outdir)
    )
    tester(
        model=model,
        loader_dict=loader_dict,
        modeldir=os.path.join(config.outdir),
        device=device
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, default='configs/config.path', help='path to config file')
    parser.add_argument('--gpuid', type=int, default=-1, help='gpu device id')
    args = parser.parse_args()
    main(args)