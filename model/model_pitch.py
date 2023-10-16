import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import matplotlib.pyplot as plt
import numpy as np
from torch.nn.parameter import Parameter



class PitchModel_Linear(nn.Module):
    
    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.device = device
        self.fc0 = nn.Linear(config.model_params.input_dim, config.model_params.hidden_dim)
        self.fc1 = nn.Linear(config.model_params.hidden_dim, config.model_params.hidden_dim)
        self.fc2 = nn.Linear(config.model_params.hidden_dim, 1)
        self.fc3 = nn.Linear(config.model_params.hidden_dim, 1)
        self.relu = nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.critation = nn.MSELoss(reduction='sum')
        self.critation2 = nn.BCELoss()

    
    def forward(self, batch, split='train'):
        
        x = batch[0].to(self.device) 
        target = batch[1].to(self.device) 
        target2 = target.float()  
        target2 = torch.where(target2 == 0, 0.0, 1.0)
        b, t, _, _ = x.shape
        x = x.reshape(b, t, -1)
        x = self.relu(self.fc0(x))
        x = self.relu(self.fc1(x))
        
        x1 = self.fc2(x)
        pred = x1.reshape(b, -1)
        loss_mask = (target > 0).int()
        non_zero = torch.sum(loss_mask, axis=1)
        loss1 = self.critation(pred*loss_mask, target) / torch.where(non_zero == 0, 1, non_zero)
        loss1 = sum(loss1)
  
        x2 = self.fc3(x)
        pred2 = x2.reshape(b, -1)
        pred2 = self.sigmoid(pred2)
        pred2 = torch.reshape(pred2, (b, t))
        loss2 = self.critation2(pred2, target2)

        return loss1, loss2
    
    
    def inference(self, batch):
        x = batch[0].to(self.device)
        target = batch[1].to(self.device)
        target2 = target.float()  
        target2 = torch.where(target2 == 0, 0.0, 1.0)
        b, t, _, _ = x.shape
        x = x.reshape(b, t, -1)
        x = self.relu(self.fc0(x))
        x = self.relu(self.fc1(x))
        x1 = self.fc2(x)
        pred = x1.reshape(b, -1)
        x2 = self.fc3(x)
        pred2 = x2.reshape(b, -1)
        pred2 = (self.sigmoid(pred2) >= 0.5).int()
        
        return pred, target, pred2, target2
    
    
    def demo_inference(self, spec):
        x = spec.to(self.device)
        b, t, _, _ = x.shape
        x = x.reshape(b, t, -1)
        x = self.relu(self.fc0(x))
        x = self.relu(self.fc1(x))
        x1 = self.fc2(x)
        pred = x1.reshape(b, -1)
        x2 = self.fc3(x)
        pred2 = x2.reshape(b, -1)
        pred2 = (self.sigmoid(pred2) >= 0.5).int()
        
        return pred, pred2



class PitchModel_LSTM(nn.Module):
    
    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.device = device   
        self.lstm = nn.LSTM(input_size=config.model_params.input_dim, hidden_size=config.model_params.hidden_dim, num_layers=config.model_params.num_layers, batch_first=True, bidirectional=config.model_params.bidirectional)
        self.fc2 = nn.Linear(config.model_params.hidden_dim, 1)
        self.fc3 = nn.Linear(config.model_params.hidden_dim, 1)
        self.relu = nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.critation = nn.MSELoss(reduction='sum')
        self.critation2 = nn.BCELoss()

    
    def forward(self, batch, split='train'):
        
        x = batch[0].to(self.device) 
        target = batch[1].to(self.device) 
        target2 = target.float()  
        target2 = torch.where(target2 == 0, 0.0, 1.0)
        b, t, _, _ = x.shape
        x = x.reshape(b, t, -1)
        x, _ = self.lstm(x)
        
        x1 = self.fc2(x)
        pred = x1.reshape(b, -1)
        loss_mask = (target > 0).int()
        non_zero = torch.sum(loss_mask, axis=1)
        loss1 = self.critation(pred*loss_mask, target) / torch.where(non_zero == 0, 1, non_zero)
        loss1 = sum(loss1)
  
        x2 = self.fc3(x)
        pred2 = x2.reshape(b, -1)
        pred2 = self.sigmoid(pred2)
        pred2 = torch.reshape(pred2, (b, t))
        loss2 = self.critation2(pred2, target2)

        return loss1, loss2

    
    def inference(self, batch):
        x = batch[0].to(self.device)
        target = batch[1].to(self.device)
        target2 = target.float()  
        target2 = torch.where(target2 == 0, 0.0, 1.0)
        b, t, _, _ = x.shape
        x = x.reshape(b, t, -1)
        x, _ = self.lstm(x)
        x1 = self.fc2(x)
        pred = x1.reshape(b, -1)
        x2 = self.fc3(x)
        pred2 = x2.reshape(b, -1)
        pred2 = (self.sigmoid(pred2) >= 0.5).int()

        return pred, target, pred2, target2
    
    
    def demo_inference(self, spec):
        x = spec.to(self.device)
        b, t, _, _ = x.shape
        x = x.reshape(b, t, -1)
        x, _ = self.lstm(x)
        x1 = self.fc2(x)
        pred = x1.reshape(b, -1)
        x2 = self.fc3(x)
        pred2 = x2.reshape(b, -1)
        pred2 = (self.sigmoid(pred2) >= 0.5).int()

        return pred, pred2


class PitchModel_Transformer(nn.Module):
    
    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.device = device   
        encoder_layers = nn.TransformerEncoderLayer(d_model=config.model_params.input_dim, nhead=config.model_params.nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers=config.model_params.num_layers)
        self.fc2 = nn.Linear(config.model_params.input_dim, 1)
        self.fc3 = nn.Linear(config.model_params.input_dim, 1)
        self.relu = nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.critation = nn.MSELoss(reduction='sum')
        self.critation2 = nn.BCELoss()
    
    
    def forward(self, batch, split='train'):
        
        x = batch[0].to(self.device)
        target = batch[1].to(self.device)
        target2 = target.float()  
        target2 = torch.where(target2 == 0, 0.0, 1.0)
        b, t, _, _ = x.shape
        x = x.reshape(b, t, -1)
        x = self.transformer(x)
        
        x1 = self.fc2(x)
        pred = x1.reshape(b, -1)
        loss_mask = (target > 0).int()
        non_zero = torch.sum(loss_mask, axis=1)
        loss1 = self.critation(pred*loss_mask, target) / torch.where(non_zero == 0, 1, non_zero)
        loss1 = sum(loss1)
  
        x2 = self.fc3(x)
        pred2 = x2.reshape(b, -1)
        pred2 = self.sigmoid(pred2)
        pred2 = torch.reshape(pred2, (b, t))
        loss2 = self.critation2(pred2, target2)

        return loss1, loss2

    
    def inference(self, batch):
        x = batch[0].to(self.device)
        target = batch[1].to(self.device)
        target2 = target.float()  
        target2 = torch.where(target2 == 0, 0.0, 1.0)
        b, t, _, _ = x.shape
        x = x.reshape(b, t, -1)
        x = self.transformer(x)
        x1 = self.fc2(x)
        pred = x1.reshape(b, -1)
        x2 = self.fc3(x)
        pred2 = x2.reshape(b, -1)
        pred2 = (self.sigmoid(pred2) >= 0.5).int()

        return pred, target, pred2, target2



class Floor(nn.Module):
    def __init__(self, shape):
        super(Floor, self).__init__()
        self.shape = shape
        self.weight = Parameter(torch.zeros(shape))
        self.bias = Parameter(torch.zeros(shape))
        self.reset_parameters()

    def reset_parameters(self):
        init.zeros_(self.weight)
        init.zeros_(self.bias)

    def forward(self, x):
        return self.weight.exp() * x + self.bias 


class PitchModel_CNN(nn.Module):
    
    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.device = device
        self.input_dim = config.model_params.input_dim
        self.hidden_dim = config.model_params.hidden_dim

        self.floor = Floor((1, self.input_dim, 1,))
        self.c1 = nn.Conv2d(1, 32, (5, 1), stride=(2,1), padding=(1,0))
        self.bn1 = nn.BatchNorm2d(32)
        self.c2 = nn.Conv2d(32, 32, (7, 1), stride=(3,1), padding=(1,0))
        self.bn2 = nn.BatchNorm2d(32)
        self.c3 = nn.Conv2d(32, 32, (7, 1), stride=(3,1), padding=(1,0))
        self.bn3 = nn.BatchNorm2d(32)
        self.c4 = nn.Conv2d(32, 32, (7, 1), stride=(3,1), padding=(1,0))
        self.bn4 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(256, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, 1)
        self.fc3 = nn.Linear(self.hidden_dim, 1)
        self.relu = nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.critation = nn.MSELoss(reduction='sum')
        self.critation2 = nn.BCELoss()

    
    def forward(self, batch, split='train'):
        
        x = batch[0].to(self.device)
        target = batch[1].to(self.device)
        target2 = target.float()
        target2 = torch.where(target2 == 0, 0.0, 1.0)

        b, t, _, _ = x.shape
        x = x.reshape(b*t, self.input_dim, -1)
        x = torch.unsqueeze(x, dim=1)
        x = F.relu(self.floor(x))
        x = self.bn1(F.relu(self.c1(x)))
        x = self.bn2(F.relu(self.c2(x)))
        x = self.bn3(F.relu(self.c3(x)))
        x = self.bn4(F.relu(self.c4(x)))
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = x.reshape(b, t, -1)
        
        x1 = self.fc2(x)
        pred = x1.reshape(b, -1)
        loss_mask = (target > 0).int()
        non_zero = torch.sum(loss_mask, axis=1)
        loss1 = self.critation(pred*loss_mask, target) / torch.where(non_zero == 0, 1, non_zero)
        loss1 = sum(loss1)

        x2 = self.fc3(x)
        pred2 = x2.reshape(b, -1)
        pred2 = self.sigmoid(pred2)
        pred2 = torch.reshape(pred2, (b, t))
        loss2 = self.critation2(pred2, target2)

        return loss1, loss2

    
    def inference(self, batch):
        x = batch[0].to(self.device)
        target = batch[1].to(self.device)
        b, t, _, _ = x.shape
        target2 = target.float()
        target2 = torch.where(target2 == 0, 0.0, 1.0)

        x = x.reshape(b*t, self.input_dim, -1) 
        x = torch.unsqueeze(x, dim=1)
        x = F.relu(self.floor(x))
        x = self.bn1(F.relu(self.c1(x)))
        x = self.bn2(F.relu(self.c2(x)))
        x = self.bn3(F.relu(self.c3(x)))
        x = self.bn4(F.relu(self.c4(x)))
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = x.reshape(b, t, -1)

        x1 = self.fc2(x)
        pred = x1.reshape(b, -1)
        x2 = self.fc3(x)
        pred2 = x2.reshape(b, -1)
        pred2 = (self.sigmoid(pred2) >= 0.5).int()

        return pred, target, pred2, target2
    
    
    def demo_inference(self, spec):
        x = spec.to(self.device)
        b, t, _, _ = x.shape
        x = x.reshape(b*t, self.input_dim, -1) 
        x = torch.unsqueeze(x, dim=1)
        x = F.relu(self.floor(x))
        x = self.bn1(F.relu(self.c1(x)))
        x = self.bn2(F.relu(self.c2(x)))
        x = self.bn3(F.relu(self.c3(x)))
        x = self.bn4(F.relu(self.c4(x)))
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = x.reshape(b, t, -1)
        x1 = self.fc2(x)
        pred = x1.reshape(b, -1)
        x2 = self.fc3(x)
        pred2 = x2.reshape(b, -1)
        pred2 = (self.sigmoid(pred2) >= 0.5).int()

        return pred, pred2



class PitchModel_CNNLSTM(nn.Module):
    
    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.device = device
        self.input_dim = config.model_params.input_dim
        self.lstm_input_dim = config.model_params.lstm_input_dim
        self.lstm_hidden_dim = config.model_params.lstm_hidden_dim

        self.c1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=2, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.c2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=7, stride=3, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.c3 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=7, stride=3, padding=1)
        self.bn3 = nn.BatchNorm1d(32)
        self.c4 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=7, stride=3, padding=1)
        self.bn4 = nn.BatchNorm1d(32)
        self.lstm = nn.LSTM(input_size=self.lstm_input_dim, hidden_size=self.lstm_hidden_dim, num_layers=1, batch_first=True)
        self.fc2 = nn.Linear(self.lstm_hidden_dim, 1)
        self.fc3 = nn.Linear(self.lstm_hidden_dim, 1)
        self.relu = nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.critation = nn.MSELoss(reduction='sum')
        self.critation2 = nn.BCELoss()

    
    def forward(self, batch, split='train'):
        
        x = batch[0].to(self.device)
        target = batch[1].to(self.device)
        target2 = target.float()
        target2 = torch.where(target2 == 0, 0.0, 1.0)

        b, t, _, _ = x.shape
        x = x.reshape(b*t, 1, self.input_dim)
        x = self.bn1(F.relu(self.c1(x)))
        x = self.bn2(F.relu(self.c2(x)))
        x = self.bn3(F.relu(self.c3(x)))
        x = self.bn4(F.relu(self.c4(x))) 
        x = x.reshape(b, t, -1)
        x, _ = self.lstm(x)
        
        x1 = self.fc2(x)
        pred = x1.reshape(b, -1)
        loss_mask = (target > 0).int()
        non_zero = torch.sum(loss_mask, axis=1)
        loss1 = self.critation(pred*loss_mask, target) / torch.where(non_zero == 0, 1, non_zero)
        loss1 = sum(loss1)

        x2 = self.fc3(x)
        pred2 = x2.reshape(b, -1)
        pred2 = self.sigmoid(pred2)
        pred2 = torch.reshape(pred2, (b, t))
        loss2 = self.critation2(pred2, target2)

        return loss1, loss2

    
    def inference(self, batch):
        x = batch[0].to(self.device)
        target = batch[1].to(self.device)
        b, t, _, _ = x.shape
        target2 = target.float()
        target2 = torch.where(target2 == 0, 0.0, 1.0)

        x = x.reshape(b*t, 1, self.input_dim)
        x = self.bn1(F.relu(self.c1(x)))
        x = self.bn2(F.relu(self.c2(x)))
        x = self.bn3(F.relu(self.c3(x)))
        x = self.bn4(F.relu(self.c4(x)))
        x = x.reshape(b, t, -1)
        x, _ = self.lstm(x)

        x1 = self.fc2(x)
        pred = x1.reshape(b, -1)
        x2 = self.fc3(x)
        pred2 = x2.reshape(b, -1)
        pred2 = (self.sigmoid(pred2) >= 0.5).int()

        return pred, target, pred2, target2
    
    
    def demo_inference(self, spec):
        x = spec.to(self.device)
        b, t, _, _ = x.shape
        x = x.reshape(b*t, 1, self.input_dim)
        x = self.bn1(F.relu(self.c1(x)))
        x = self.bn2(F.relu(self.c2(x)))
        x = self.bn3(F.relu(self.c3(x)))
        x = self.bn4(F.relu(self.c4(x)))
        x = x.reshape(b, t, -1)
        x, _ = self.lstm(x)
        x1 = self.fc2(x)
        pred = x1.reshape(b, -1)
        x2 = self.fc3(x)
        pred2 = x2.reshape(b, -1)
        pred2 = (self.sigmoid(pred2) >= 0.5).int()

        return pred, pred2



class PitchModel_CNNTransformer(nn.Module):
    
    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.device = device
        self.input_dim = config.model_params.input_dim
        self.transformer_input_dim = config.model_params.transformer_input_dim

        self.c1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=2, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.c2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=7, stride=3, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.c3 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=7, stride=3, padding=1)
        self.bn3 = nn.BatchNorm1d(32)
        self.c4 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=7, stride=3, padding=1)
        self.bn4 = nn.BatchNorm1d(32)
        encoder_layers = nn.TransformerEncoderLayer(d_model=self.transformer_input_dim, nhead=8, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers=1)
        self.fc2 = nn.Linear(self.transformer_input_dim, 1)
        self.fc3 = nn.Linear(self.transformer_input_dim, 1)
        self.relu = nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.critation = nn.MSELoss(reduction='sum')
        self.critation2 = nn.BCELoss()

    
    def forward(self, batch, split='train'):
        
        x = batch[0].to(self.device)
        target = batch[1].to(self.device)
        target2 = target.float()
        target2 = torch.where(target2 == 0, 0.0, 1.0)

        b, t, _, _ = x.shape
        x = x.reshape(b*t, 1, self.input_dim)
        x = self.bn1(F.relu(self.c1(x)))
        x = self.bn2(F.relu(self.c2(x)))
        x = self.bn3(F.relu(self.c3(x)))
        x = self.bn4(F.relu(self.c4(x))) 
        x = x.reshape(b, t, -1)
        x = self.transformer(x)
        
        x1 = self.fc2(x)
        pred = x1.reshape(b, -1)
        loss_mask = (target > 0).int()
        non_zero = torch.sum(loss_mask, axis=1)
        loss1 = self.critation(pred*loss_mask, target) / torch.where(non_zero == 0, 1, non_zero)
        loss1 = sum(loss1)

        x2 = self.fc3(x)
        pred2 = x2.reshape(b, -1)
        pred2 = self.sigmoid(pred2)
        pred2 = torch.reshape(pred2, (b, t))
        loss2 = self.critation2(pred2, target2)

        return loss1, loss2

    
    def inference(self, batch):
        x = batch[0].to(self.device)
        target = batch[1].to(self.device)
        b, t, _, _ = x.shape
        target2 = target.float()
        target2 = torch.where(target2 == 0, 0.0, 1.0)

        x = x.reshape(b*t, 1, self.input_dim)
        x = self.bn1(F.relu(self.c1(x)))
        x = self.bn2(F.relu(self.c2(x)))
        x = self.bn3(F.relu(self.c3(x)))
        x = self.bn4(F.relu(self.c4(x)))
        x = x.reshape(b, t, -1)
        x = self.transformer(x)

        x1 = self.fc2(x)
        pred = x1.reshape(b, -1)
        x2 = self.fc3(x)
        pred2 = x2.reshape(b, -1)
        pred2 = (self.sigmoid(pred2) >= 0.5).int()

        return pred, target, pred2, target2
    
    
    def demo_inference(self, spec):
        x = spec.to(self.device)
        b, t, _, _ = x.shape
        x = x.reshape(b*t, 1, self.input_dim)
        x = self.bn1(F.relu(self.c1(x)))
        x = self.bn2(F.relu(self.c2(x)))
        x = self.bn3(F.relu(self.c3(x)))
        x = self.bn4(F.relu(self.c4(x)))
        x = x.reshape(b, t, -1)
        x = self.transformer(x)
        x1 = self.fc2(x)
        pred = x1.reshape(b, -1)
        x2 = self.fc3(x)
        pred2 = x2.reshape(b, -1)
        pred2 = (self.sigmoid(pred2) >= 0.5).int()

        return pred, pred2



class PitchModel_CREPE(nn.Module):
    
    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.device = device
        self.input_dim = config.model_params.input_dim
        self.hidden_dim = config.model_params.hidden_dim

        if config.model_params.size == 'tiny':
            out_channels = [128, 16, 16, 16, 16, 32, 64]
        elif config.model_params.size == 'small':
            out_channels = [256, 32, 32, 32, 32, 64, 128]
        elif config.model_params.size == 'midium':
            out_channels = [512, 64, 64, 64, 64, 128, 256]
        elif config.model_params.size == 'large':
            out_channels = [768, 96, 96, 96, 96, 192, 384]
        elif config.model_params.size == 'full':
            out_channels = [1024, 128, 128, 128, 128, 256, 512]
        else:
            print('check config.model_params.size')
            exit()
        
        self.c1 = nn.Conv1d(in_channels=1, out_channels=out_channels[0], kernel_size=512, stride=4, padding=256)
        self.bn1 = nn.BatchNorm1d(out_channels[0])
        self.c2 = nn.Conv1d(in_channels=out_channels[0], out_channels=out_channels[1], kernel_size=64, stride=1, padding=32)
        self.bn2 = nn.BatchNorm1d(out_channels[1])
        self.c3 = nn.Conv1d(in_channels=out_channels[1], out_channels=out_channels[2], kernel_size=64, stride=1, padding=32)
        self.bn3 = nn.BatchNorm1d(out_channels[2])
        self.c4 = nn.Conv1d(in_channels=out_channels[2], out_channels=out_channels[3], kernel_size=64, stride=1, padding=32)
        self.bn4 = nn.BatchNorm1d(out_channels[3])
        self.c5 = nn.Conv1d(in_channels=out_channels[3], out_channels=out_channels[4], kernel_size=64, stride=1, padding=32)
        self.bn5 = nn.BatchNorm1d(out_channels[4])
        self.c6 = nn.Conv1d(in_channels=out_channels[4], out_channels=out_channels[5], kernel_size=64, stride=1, padding=32)
        self.bn6 = nn.BatchNorm1d(out_channels[5])
        self.maxpool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(out_channels[5]*2, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, 1)
        self.fc3 = nn.Linear(self.hidden_dim, 1)
        self.relu = nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.critation = nn.MSELoss(reduction='sum')
        self.critation2 = nn.BCELoss()

    
    def forward(self, batch, split='train'):
        
        x = batch[0].to(self.device)
        target = batch[1].to(self.device)
        target2 = target.float()
        target2 = torch.where(target2 == 0, 0.0, 1.0)

        b, t, _, _ = x.shape
        x = x.reshape(b*t, self.input_dim)
        x = torch.unsqueeze(x, 1)
        x = self.dropout(self.maxpool(self.bn1(F.relu(self.c1(x)))))
        x = self.dropout(self.maxpool(self.bn2(F.relu(self.c2(x)))))
        x = self.dropout(self.maxpool(self.bn3(F.relu(self.c3(x)))))
        x = self.dropout(self.maxpool(self.bn4(F.relu(self.c4(x)))))
        x = self.dropout(self.maxpool(self.bn5(F.relu(self.c5(x)))))
        x = self.dropout(self.maxpool(self.bn6(F.relu(self.c6(x)))))
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = x.reshape(b, t, -1)
        
        x1 = self.fc2(x)
        pred = x1.reshape(b, -1)
        loss_mask = (target > 0).int()
        non_zero = torch.sum(loss_mask, axis=1)
        loss1 = self.critation(pred*loss_mask, target) / torch.where(non_zero == 0, 1, non_zero)
        loss1 = sum(loss1)

        x2 = self.fc3(x)
        pred2 = x2.reshape(b, -1)
        pred2 = self.sigmoid(pred2)
        pred2 = torch.reshape(pred2, (b, t))
        loss2 = self.critation2(pred2, target2)

        return loss1, loss2

    
    def inference(self, batch):
        x = batch[0].to(self.device)
        target = batch[1].to(self.device)
        b, t, _, _ = x.shape
        target2 = target.float()
        target2 = torch.where(target2 == 0, 0.0, 1.0)

        x = x.reshape(b*t, self.input_dim)
        x = torch.unsqueeze(x, 1)
        x = self.dropout(self.maxpool(self.bn1(F.relu(self.c1(x)))))
        x = self.dropout(self.maxpool(self.bn2(F.relu(self.c2(x)))))
        x = self.dropout(self.maxpool(self.bn3(F.relu(self.c3(x)))))
        x = self.dropout(self.maxpool(self.bn4(F.relu(self.c4(x)))))
        x = self.dropout(self.maxpool(self.bn5(F.relu(self.c5(x)))))
        x = self.dropout(self.maxpool(self.bn6(F.relu(self.c6(x)))))
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = x.reshape(b, t, -1)

        x1 = self.fc2(x)
        pred = x1.reshape(b, -1)
        x2 = self.fc3(x)
        pred2 = x2.reshape(b, -1)
        pred2 = (self.sigmoid(pred2) >= 0.5).int()

        return pred, target, pred2, target2
    
    
    def demo_inference(self, spec):
        x = spec.to(self.device)
        b, t, _, _ = x.shape
        x = x.reshape(b*t, self.input_dim)
        x = torch.unsqueeze(x, 1)
        x = self.dropout(self.maxpool(self.bn1(F.relu(self.c1(x)))))
        x = self.dropout(self.maxpool(self.bn2(F.relu(self.c2(x)))))
        x = self.dropout(self.maxpool(self.bn3(F.relu(self.c3(x)))))
        x = self.dropout(self.maxpool(self.bn4(F.relu(self.c4(x)))))
        x = self.dropout(self.maxpool(self.bn5(F.relu(self.c5(x)))))
        x = self.dropout(self.maxpool(self.bn6(F.relu(self.c6(x)))))
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = x.reshape(b, t, -1)
        x1 = self.fc2(x)
        pred = x1.reshape(b, -1)
        x2 = self.fc3(x)
        pred2 = x2.reshape(b, -1)
        pred2 = (self.sigmoid(pred2) >= 0.5).int()

        return pred, pred2



class PitchModel_MB_CNNLSTM(nn.Module):
    
    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.device = device
        self.input_dim = config.model_params.input_dim
        self.lstm_input_dim = config.model_params.lstm_input_dim
        self.lstm_hidden_dim = config.model_params.lstm_hidden_dim
        
        self.convA1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=5, stride=2)
        self.convA2 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=9, stride=2)
        self.convA3 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=15, stride=2)
        self.convB1 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=2)
        self.convB2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5, stride=2)
        self.convB3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=7, stride=2)
        self.maxpool = nn.MaxPool1d(16)
        self.dropout = nn.Dropout(0.25)
        self.lstm = nn.LSTM(input_size=self.lstm_input_dim, hidden_size=self.lstm_hidden_dim, num_layers=1, batch_first=True, bidirectional=False)
        self.fc2 = nn.Linear(self.lstm_hidden_dim, 1)
        self.fc3 = nn.Linear(self.lstm_hidden_dim, 1)
        self.relu = nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.critation = nn.MSELoss(reduction='sum')
        self.critation2 = nn.BCELoss()

    
    def forward(self, batch, split='train'):
        
        x = batch[0].to(self.device)
        target = batch[1].to(self.device)
        target2 = target.float()
        target2 = torch.where(target2 == 0, 0.0, 1.0)

        b, t, _, _ = x.shape
        x = x.reshape(b*t, self.input_dim)
        x = torch.unsqueeze(x, 1)
        x1 = self.dropout(F.relu(self.convA1(x)))
        x2 = self.dropout(F.relu(self.convA2(x)))
        x3 = self.dropout(F.relu(self.convA3(x)))
        x = torch.cat([x1, x2, x3], dim=-1)
        x1 = self.dropout(F.relu(self.convB1(x)))
        x2 = self.dropout(F.relu(self.convB2(x)))
        x3 = self.dropout(F.relu(self.convB3(x)))
        x = torch.cat([x1, x2, x3], dim=-1)
        x = self.maxpool(self.maxpool(x))
        x = x.reshape(b*t, -1)
        x, _ = self.lstm(x)
        
        x1 = self.fc2(x)
        pred = x1.reshape(b, -1)
        loss_mask = (target > 0).int()
        non_zero = torch.sum(loss_mask, axis=1)
        loss1 = self.critation(pred*loss_mask, target) / torch.where(non_zero == 0, 1, non_zero)
        loss1 = sum(loss1)

        x2 = self.fc3(x)
        pred2 = x2.reshape(b, -1)
        pred2 = self.sigmoid(pred2)
        pred2 = torch.reshape(pred2, (b, t))
        loss2 = self.critation2(pred2, target2)

        return loss1, loss2

    
    def inference(self, batch):
        x = batch[0].to(self.device)
        target = batch[1].to(self.device)
        b, t, _, _ = x.shape
        target2 = target.float()
        target2 = torch.where(target2 == 0, 0.0, 1.0)

        b, t, _, _ = x.shape
        x = x.reshape(b*t, self.input_dim)
        x = torch.unsqueeze(x, 1)
        x1 = self.dropout(F.relu(self.convA1(x)))
        x2 = self.dropout(F.relu(self.convA2(x)))
        x3 = self.dropout(F.relu(self.convA3(x)))
        x = torch.cat([x1, x2, x3], dim=-1)
        x1 = self.dropout(F.relu(self.convB1(x)))
        x2 = self.dropout(F.relu(self.convB2(x)))
        x3 = self.dropout(F.relu(self.convB3(x)))
        x = torch.cat([x1, x2, x3], dim=-1)
        x = self.maxpool(self.maxpool(x))
        x = x.reshape(b*t, -1)
        x, _ = self.lstm(x)

        x1 = self.fc2(x)
        pred = x1.reshape(b, -1)
        x2 = self.fc3(x)
        pred2 = x2.reshape(b, -1)
        pred2 = (self.sigmoid(pred2) >= 0.5).int()

        return pred, target, pred2, target2
    
    
    def demo_inference(self, spec):
        x = spec.to(self.device)
        b, t, _, _ = x.shape
        b, t, _, _ = x.shape
        x = x.reshape(b*t, self.input_dim)
        x = torch.unsqueeze(x, 1)
        x1 = self.dropout(F.relu(self.convA1(x)))
        x2 = self.dropout(F.relu(self.convA2(x)))
        x3 = self.dropout(F.relu(self.convA3(x)))
        x = torch.cat([x1, x2, x3], dim=-1)
        x1 = self.dropout(F.relu(self.convB1(x)))
        x2 = self.dropout(F.relu(self.convB2(x)))
        x3 = self.dropout(F.relu(self.convB3(x)))
        x = torch.cat([x1, x2, x3], dim=-1)
        x = self.maxpool(self.maxpool(x))
        x = x.reshape(b*t, -1)
        x, _ = self.lstm(x)
        x1 = self.fc2(x)
        pred = x1.reshape(b, -1)
        x2 = self.fc3(x)
        pred2 = x2.reshape(b, -1)
        pred2 = (self.sigmoid(pred2) >= 0.5).int()

        return pred, pred2



class PitchModel_CNN10(nn.Module):
    
    def __init__(self, config, device):
        super().__init__() 
        self.config = config
        self.device = device
        self.input_dim = config.model_params.input_dim
        self.hidden_dim = config.model_params.hidden_dim

        self.floor = Floor((1, self.input_dim, 10,))
        self.c1 = nn.Conv2d(1, 32, (5, 5), padding=2, stride=(2, 2))
        self.bn1 = nn.BatchNorm2d(32)
        self.c2 = nn.Conv2d(32, 32, (7, 5), padding=(3, 2), stride=(3, 2))
        self.bn2 = nn.BatchNorm2d(32)
        self.c3 = nn.Conv2d(32, 32, (7, 5), padding=(3, 2), stride=(3, 2))
        self.bn3 = nn.BatchNorm2d(32)
        self.c4 = nn.Conv2d(32, 32, (7, 5), padding=(3, 2), stride=(3, 2))
        self.bn4 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(320, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, 1)
        self.fc3 = nn.Linear(self.hidden_dim, 1)
        self.relu = nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid() 
        self.critation = nn.MSELoss(reduction='sum')
        self.critation2 = nn.BCELoss()
    
    
    def forward(self, batch, split='train'):
        
        x = batch[0].to(self.device)
        target = batch[1].to(self.device)
        target2 = target.float()
        target2 = torch.where(target2 == 0, 0.0, 1.0)

        b, t, _, _ = x.shape
        x = x.reshape(b*t, self.input_dim, 10)
        x = torch.unsqueeze(x, dim=1)
        x = F.relu(self.floor(x))
        x = self.bn1(F.relu(self.c1(x)))
        x = self.bn2(F.relu(self.c2(x)))
        x = self.bn3(F.relu(self.c3(x)))
        x = self.bn4(F.relu(self.c4(x)))
        x = x.reshape(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        x = x.reshape(b, t, -1)
        
        x1 = self.fc2(x)
        pred = x1.reshape(b, -1)
        loss_mask = (target > 0).int()
        loss1 = self.critation(pred*loss_mask, target) / torch.sum(loss_mask, axis=1)
        loss1 = sum(loss1)
        
        x2 = self.fc3(x)
        pred2 = x2.reshape(b, -1)
        pred2 = self.sigmoid(pred2)
        pred2 = torch.reshape(pred2, (b, t))
        loss2 = self.critation2(pred2, target2)
        
        return loss1, loss2

    
    def inference(self, batch):
        
        x = batch[0].to(self.device)
        target = batch[1].to(self.device)
        target2 = target.float()
        target2 = torch.where(target2 == 0, 0.0, 1.0)

        b, t, _, _ = x.shape
        x = x.reshape(b*t, self.input_dim, 10)
        x = torch.unsqueeze(x, dim=1)
        x = F.relu(self.floor(x))
        x = self.bn1(F.relu(self.c1(x)))
        x = self.bn2(F.relu(self.c2(x)))
        x = self.bn3(F.relu(self.c3(x)))
        x = self.bn4(F.relu(self.c4(x)))
        x = x.reshape(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        x = x.reshape(b, t, -1)
        
        x1 = self.fc2(x)
        pred = x1.reshape(b, -1)
        x2 = self.fc3(x)
        pred2 = x2.reshape(b, -1)
        pred2 = (self.sigmoid(pred2) >= 0.5).int()
        
        return pred, target, pred2, target2
    
    
    def demo_inference(self, spec):   
        x = spec.to(self.device)
        b, t, _, _ = x.shape
        x = x.reshape(b*t, self.input_dim, 10)
        x = torch.unsqueeze(x, dim=1)
        x = F.relu(self.floor(x))
        x = self.bn1(F.relu(self.c1(x)))
        x = self.bn2(F.relu(self.c2(x)))
        x = self.bn3(F.relu(self.c3(x)))
        x = self.bn4(F.relu(self.c4(x)))
        x = x.reshape(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        x = x.reshape(b, t, -1)
        x1 = self.fc2(x)
        pred = x1.reshape(b, -1)
        x2 = self.fc3(x)
        pred2 = x2.reshape(b, -1)
        pred2 = (self.sigmoid(pred2) >= 0.5).int()
        
        return pred, pred2



class PitchModel_CNN10LSTM(nn.Module):
    
    def __init__(self, config, device):
        super().__init__() 
        self.config = config
        self.device = device
        self.input_dim = config.model_params.input_dim
        self.lstm_input_dim = config.model_params.lstm_input_dim
        self.lstm_hidden_dim = config.model_params.lstm_hidden_dim

        self.c1 = nn.Conv2d(1, 32, (5, 5), padding=2, stride=(2, 2))
        self.bn1 = nn.BatchNorm2d(32)
        self.c2 = nn.Conv2d(32, 32, (7, 5), padding=(3, 2), stride=(3, 2))
        self.bn2 = nn.BatchNorm2d(32)
        self.c3 = nn.Conv2d(32, 32, (7, 5), padding=(3, 2), stride=(3, 2))
        self.bn3 = nn.BatchNorm2d(32)
        self.c4 = nn.Conv2d(32, 32, (7, 5), padding=(3, 2), stride=(3, 2))
        self.bn4 = nn.BatchNorm2d(32)
        self.lstm = nn.LSTM(input_size=self.lstm_input_dim, hidden_size=self.lstm_hidden_dim, num_layers=1, batch_first=True, bidirectional=False)
        self.fc2 = nn.Linear(self.lstm_hidden_dim, 1)
        self.fc3 = nn.Linear(self.lstm_hidden_dim, 1)
        self.relu = nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid() 
        self.critation = nn.MSELoss(reduction='sum')
        self.critation2 = nn.BCELoss()
    
    
    def forward(self, batch, split='train'):
        
        x = batch[0].to(self.device)
        target = batch[1].to(self.device)
        target2 = target.float()
        target2 = torch.where(target2 == 0, 0.0, 1.0)

        b, t, _, _ = x.shape
        x = x.reshape(b*t, self.input_dim, 10)
        x = torch.unsqueeze(x, dim=1)
        x = self.bn1(F.relu(self.c1(x)))
        x = self.bn2(F.relu(self.c2(x)))
        x = self.bn3(F.relu(self.c3(x)))
        x = self.bn4(F.relu(self.c4(x)))
        x = x.reshape(b, t, -1)
        x, _ = self.lstm(x)
        
        x1 = self.fc2(x)
        pred = x1.reshape(b, -1)
        loss_mask = (target > 0).int()
        loss1 = self.critation(pred*loss_mask, target) / torch.sum(loss_mask, axis=1)
        loss1 = sum(loss1)
        
        x2 = self.fc3(x)
        pred2 = x2.reshape(b, -1)
        pred2 = self.sigmoid(pred2)
        pred2 = torch.reshape(pred2, (b, t))
        loss2 = self.critation2(pred2, target2)
        
        return loss1, loss2

    
    def inference(self, batch):
        
        x = batch[0].to(self.device)
        target = batch[1].to(self.device)
        target2 = target.float()
        target2 = torch.where(target2 == 0, 0.0, 1.0)

        b, t, _, _ = x.shape
        x = x.reshape(b*t, self.input_dim, 10)
        x = torch.unsqueeze(x, dim=1)
        x = self.bn1(F.relu(self.c1(x)))
        x = self.bn2(F.relu(self.c2(x)))
        x = self.bn3(F.relu(self.c3(x)))
        x = self.bn4(F.relu(self.c4(x)))
        x = x.reshape(b, t, -1)
        x, _ = self.lstm(x)
        
        x1 = self.fc2(x)
        pred = x1.reshape(b, -1)
        x2 = self.fc3(x)
        pred2 = x2.reshape(b, -1)
        pred2 = (self.sigmoid(pred2) >= 0.5).int()
        
        return pred, target, pred2, target2
    
    
    def demo_inference(self, spec):   
        x = spec.to(self.device)
        b, t, _, _ = x.shape
        x = x.reshape(b*t, self.input_dim, 10)
        x = torch.unsqueeze(x, dim=1)
        x = self.bn1(F.relu(self.c1(x)))
        x = self.bn2(F.relu(self.c2(x)))
        x = self.bn3(F.relu(self.c3(x)))
        x = self.bn4(F.relu(self.c4(x)))
        x = x.reshape(b, t, -1)
        x, _ = self.lstm(x)
        x1 = self.fc2(x)
        pred = x1.reshape(b, -1)
        x2 = self.fc3(x)
        pred2 = x2.reshape(b, -1)
        pred2 = (self.sigmoid(pred2) >= 0.5).int()
        
        return pred, pred2



class PitchModel_CNNAE(nn.Module):
    
    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.device = device
        self.lstm = nn.LSTM(input_size=config.model_params.input_dim, hidden_size=config.model_params.hidden_dim, num_layers=config.model_params.num_layers, batch_first=True, bidirectional=False)
        self.fc2 = nn.Linear(config.model_params.hidden_dim, 1)
        self.fc3 = nn.Linear(config.model_params.hidden_dim, 1)
        self.sigmoid = torch.nn.Sigmoid()
        self.critation = nn.MSELoss(reduction='sum')
        self.critation2 = nn.BCELoss()

    
    def forward(self, batch, split='train'):
        
        x = batch[0].to(self.device) 
        target = batch[1].to(self.device) 
        target2 = target.float()  
        target2 = torch.where(target2 == 0, 0.0, 1.0)
        b, t, _ = x.shape
        x = x.reshape(b, t, -1)
        x, _ = self.lstm(x)
        
        x1 = self.fc2(x)
        pred = x1.reshape(b, -1)
        loss_mask = (target > 0).int()
        non_zero = torch.sum(loss_mask, axis=1)
        loss1 = self.critation(pred*loss_mask, target) / torch.where(non_zero == 0, 1, non_zero)
        loss1 = sum(loss1)
  
        x2 = self.fc3(x)
        pred2 = x2.reshape(b, -1)
        pred2 = self.sigmoid(pred2)
        pred2 = torch.reshape(pred2, (b, t))
        loss2 = self.critation2(pred2, target2)

        return loss1, loss2
    
    
    def inference(self, batch):
        x = batch[0].to(self.device)
        target = batch[1].to(self.device)
        target2 = target.float()  
        target2 = torch.where(target2 == 0, 0.0, 1.0)
        b, t, _ = x.shape
        x = x.reshape(b, t, -1)
        x, _ = self.lstm(x)
        x1 = self.fc2(x)
        pred = x1.reshape(b, -1)
        x2 = self.fc3(x)
        pred2 = x2.reshape(b, -1)
        pred2 = (self.sigmoid(pred2) >= 0.5).int()
        
        return pred, target, pred2, target2
    
    
    def demo_inference(self, spec):
        x = spec.to(self.device)
        b, t, _ = x.shape
        x = x.reshape(b, t, -1)
        x, _ = self.lstm(x)
        x1 = self.fc2(x)
        pred = x1.reshape(b, -1)
        x2 = self.fc3(x)
        pred2 = x2.reshape(b, -1)
        pred2 = (self.sigmoid(pred2) >= 0.5).int()
        
        return pred, pred2