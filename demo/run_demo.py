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
from model.model_pitch import PitchModel_LSTM, PitchModel_Transformer, PitchModel_Linear, PitchModel_CNN, PitchModel_CREPE, PitchModel_MB_CNNLSTM, PitchModel_CNN10, PitchModel_CNNAE
from utils.trainer_tester import trainer, tester
import wave
import librosa
import sflib.sound.sigproc.spec_image as spec_image


# python demo/run_demo.py configs/config_lstm.json --gpuid 0
# WAV_PATH = './demo/wav/BASIC5000_0001.wav'
WAV_PATH = './demo/wav/yaguchi_yorosiku.wav'
IMAGE_WIDTH=1
IMAGE_SHIFT=1


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
    
    model.load_state_dict(torch.load(os.path.join(config.outdir, 'best_val_loss_model.pth'))) 
    model.to(device)
    
    os.makedirs(config.demo.data_dir, exist_ok=True)
    
    with torch.no_grad():
        
        wf = wave.open(WAV_PATH)
        x = np.frombuffer(wf.readframes(wf.getnframes()), np.int16)
        generator = spec_image.SpectrogramImageGenerator(image_width=IMAGE_WIDTH,image_shift=IMAGE_SHIFT)
        spec = generator.input_wave(x)
        spec = torch.tensor(np.array(spec), dtype=torch.float32)
        spec = torch.unsqueeze(spec, dim=0)
        
        pred, pred2 = model.demo_inference(spec)
        pred = pred.reshape(-1)
        pred2 = pred2.reshape(-1)
        pred = pred.detach().cpu().numpy()
        pred2 = pred2.detach().cpu().numpy()
        pred_all = pred * pred2
        start = 0
        end = 300
        t_datas = [t_data * IMAGE_SHIFT * 10 for t_data in range(start, end)]
        plt.plot(t_datas, pred_all[start:end], color='blue', label='pred')
        plt.legend()
        save_path = WAV_PATH.split('/')[-1].replace('.wav', '')
        plt.savefig(os.path.join(config.demo.data_dir, f'{save_path}.png'))
        plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, default='configs/config.path', help='path to config file')
    parser.add_argument('--gpuid', type=int, default=-1, help='gpu device id')
    args = parser.parse_args()
    main(args)