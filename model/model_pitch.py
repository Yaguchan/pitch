import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import matplotlib.pyplot as plt
import numpy as np
from torch.nn.parameter import Parameter
from torchvision.models import resnet50, ResNet50_Weights, vgg16, VGG16_Weights
from efficientnet_pytorch import EfficientNet



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
        self.bnc1 = nn.BatchNorm2d(32)
        self.c2 = nn.Conv2d(32, 32, (7, 1), stride=(3,1), padding=(1,0))
        self.bnc2 = nn.BatchNorm2d(32)
        self.c3 = nn.Conv2d(32, 32, (7, 1), stride=(3,1), padding=(1,0))
        self.bnc3 = nn.BatchNorm2d(32)
        self.c4 = nn.Conv2d(32, 32, (7, 1), stride=(3,1), padding=(1,0))
        self.bnc4 = nn.BatchNorm2d(32)
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
        x = F.relu(self.c1(x))
        x = self.bnc1(x)
        x = F.relu(self.c2(x))
        x = self.bnc2(x)
        x = F.relu(self.c3(x))
        x = self.bnc3(x)
        x = F.relu(self.c4(x))
        x = self.bnc4(x)
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
        x = F.relu(self.c1(x))
        x = self.bnc1(x)
        x = F.relu(self.c2(x))
        x = self.bnc2(x)
        x = F.relu(self.c3(x))
        x = self.bnc3(x)
        x = F.relu(self.c4(x))
        x = self.bnc4(x)
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
        x = F.relu(self.c1(x))
        x = self.bnc1(x)
        x = F.relu(self.c2(x))
        x = self.bnc2(x)
        x = F.relu(self.c3(x))
        x = self.bnc3(x)
        x = F.relu(self.c4(x))
        x = self.bnc4(x)
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = x.reshape(b, t, -1)
        x1 = self.fc2(x)
        pred = x1.reshape(b, -1)
        x2 = self.fc3(x)
        pred2 = x2.reshape(b, -1)
        pred2 = (self.sigmoid(pred2) >= 0.5).int()

        return pred, pred2, 



class PitchModel_CNN10(nn.Module):
    
    def __init__(self, config, device):
        super().__init__() 
        self.config = config
        self.device = device
        self.input_dim = config.model_params.input_dim
        self.hidden_dim = config.model_params.hidden_dim

        self.floor = Floor((1, self.input_dim, 10,))
        self.c1 = nn.Conv2d(1, 32, (5, 5), padding=2, stride=(2, 2))
        self.bnc1 = nn.BatchNorm2d(32)
        self.c2 = nn.Conv2d(32, 32, (7, 5), padding=(3, 2), stride=(3, 2))
        self.bnc2 = nn.BatchNorm2d(32)
        self.c3 = nn.Conv2d(32, 32, (7, 5), padding=(3, 2), stride=(3, 2))
        self.bnc3 = nn.BatchNorm2d(32)
        self.c4 = nn.Conv2d(32, 32, (7, 5), padding=(3, 2), stride=(3, 2))
        self.bnc4 = nn.BatchNorm2d(32)
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
        x = F.relu(self.c1(x))
        x = self.bnc1(x)
        x = F.relu(self.c2(x))
        x = self.bnc2(x)
        x = F.relu(self.c3(x))
        x = self.bnc3(x)
        x = F.relu(self.c4(x))
        x = self.bnc4(x)
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
        x = F.relu(self.c1(x))
        x = self.bnc1(x)
        x = F.relu(self.c2(x))
        x = self.bnc2(x)
        x = F.relu(self.c3(x))
        x = self.bnc3(x)
        x = F.relu(self.c4(x))
        x = self.bnc4(x)
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
        x = F.relu(self.c1(x))
        x = self.bnc1(x)
        x = F.relu(self.c2(x))
        x = self.bnc2(x)
        x = F.relu(self.c3(x))
        x = self.bnc3(x)
        x = F.relu(self.c4(x))
        x = self.bnc4(x)
        x = x.reshape(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        x = x.reshape(b, t, -1)
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