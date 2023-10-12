import os
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


def train(model, optimizer, data_loader, device):
    model.train()
    total_loss = 0.0
    for batch in tqdm(data_loader):
        optimizer.zero_grad() 
        loss1, loss2 = model(batch, 'train')
        loss = loss1 + loss2
        if np.isnan(loss.detach().cpu().numpy()):
            continue
        loss.backward()
        optimizer.step()
        total_loss += loss.detach().cpu().numpy()
    total_loss = total_loss / len(data_loader)
    return total_loss


def val(model, data_loader, deivce):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(data_loader):
            loss1, loss2 = model(batch, 'val')
            loss = loss1 + loss2
            total_loss += loss.detach().cpu().numpy()

    total_loss = total_loss / len(data_loader)

    return total_loss


def test(model, data_loader, save_path, deivce):
    model.eval()
    total_loss = 0.0
    pred_all = np.array([])
    target_all = np.array([])
    with torch.no_grad():
        for batch in tqdm(data_loader):
            loss1, loss2 = model(batch, 'val')
            loss = loss1 + loss2
            total_loss += loss.detach().cpu().numpy()
            _, _, pred, target = model.inference(batch)
            pred = pred.reshape(-1)
            target = target.reshape(-1)
            pred = pred.detach().cpu().numpy()
            target = target.detach().cpu().numpy()
            pred_all = np.concatenate([pred_all, pred])
            target_all = np.concatenate([target_all, target])

    total_loss = total_loss / len(data_loader)
    cm = confusion_matrix(target_all, pred_all)
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')
    plt.savefig(os.path.join(save_path, 'cm.png'))
    print('------------------------------------')
    print('Test loss : {}'.format(total_loss))
    print('------------------------------------')
    print(f'accuracy : {accuracy_score(target_all, pred_all)}')
    print(f'precision: {precision_score(target_all, pred_all)}')
    print(f'recall   : {recall_score(target_all, pred_all)}')
    print(f'f1       : {f1_score(target_all, pred_all)}')
    print('------------------------------------')
    with open(os.path.join(save_path, 'vad.txt'), 'w') as f:
        f.write('------------------------------------\n')
        f.write('Test loss : {}\n'.format(total_loss))
        f.write('------------------------------------\n')
        f.write(f'accuracy : {accuracy_score(target_all, pred_all)}\n')
        f.write(f'precision: {precision_score(target_all, pred_all)}\n')
        f.write(f'recall   : {recall_score(target_all, pred_all)}\n')
        f.write(f'f1       : {f1_score(target_all, pred_all)}\n')
        f.write('------------------------------------\n')
    
    
def trainer(num_epochs, model, loader_dict, optimizer, device, outdir):
    best_val_loss = 1000000000
    t_loss=[]
    v_loss=[]
    for epoch in range(num_epochs):
        print('Epoch:{}'.format(epoch+1))
        train_loss = train(model, optimizer, loader_dict['train'], device)
        val_loss = val(model, loader_dict['val'], device)
        print('Train loss: {}'.format(train_loss))
        print('Val loss: {}'.format(val_loss))
        if best_val_loss > val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(outdir, 'best_val_loss_model.pth'))
        t_loss.append(train_loss) 
        v_loss.append(val_loss)
                
            
def tester(model, loader_dict, modeldir, device):
    model.load_state_dict(torch.load(os.path.join(modeldir, 'best_val_loss_model.pth')))
    model.to(device)
    test(model, loader_dict['test'], modeldir, device)