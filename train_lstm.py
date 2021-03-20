import pandas as pd
import numpy as np
import os
import gc
import argparse

parser = argparse.ArgumentParser()
arg = parser.add_argument
arg('--l', type=int)
arg('--f', type=int)
args = parser.parse_args()

l = args.l
valid_idx = int(args.f)
print('The validation fold is {}'.format(valid_idx))

model_path = './models/Fold{}/{}/'.format(valid_idx, l)
if not os.path.exists(model_path):
    os.makedirs(model_path)
filename = './pp_data/data-1d-{}-pp.csv'.format(l)
data = pd.read_csv(filename, sep=';')

from sklearn.model_selection import KFold

data['fold'] = 0
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for idx, (train_idx, valid_idx) in enumerate(kf.split(data)):
    data['fold'].iloc[valid_idx] = idx

from torch import nn, optim
from torch.nn import functional as F
from torch.nn import LSTM
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import *
from tqdm import tqdm

valid_idx = int(args.f)
train_df = data[data['fold']!=valid_idx].reset_index(drop=True)
valid_df = data[data['fold']==valid_idx].reset_index(drop=True)
print('There are {} samples in the training set.'.format(len(train_df)))
print('There are {} samples in the validation set.'.format(len(valid_df)))

class AnDiDataset(Dataset):
    def __init__(self, df, label=True):
        self.df = df.copy()
        self.label = label
        
    def __getitem__(self, index): 
        data_seq = torch.Tensor([float(i) for i in self.df['new_pos'].iloc[index].split(',')])
        if self.label:
            target = self.df['label'].iloc[index]
        else:
            target = 0.
        return data_seq.unsqueeze(-1), target
    
    def __len__(self):
        return len(self.df)

train_loader = DataLoader(AnDiDataset(train_df), batch_size=512, shuffle=True, num_workers=2)
valid_loader = DataLoader(AnDiDataset(valid_df), batch_size=512, shuffle=True, num_workers=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class AnDiModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(hidden_dim, output_dim))
    
    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        out, state = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:,-1,:])
        return out

criterion = nn.MSELoss()
metric = nn.L1Loss()
model = AnDiModel(1, 64, 3, 1).to(device)

lr = 0.001
optimizer = optim.Adam(model.parameters(), lr=lr)

def train_model(epoch, history=None):
    model.train() 
    t = tqdm(train_loader)
    
    for batch_idx, (seq_batch, label_batch) in enumerate(t):
        
        seq_batch = seq_batch.to(device)
        label_batch = label_batch.to(device)
        
        optimizer.zero_grad()
        output = model(seq_batch)
        loss = criterion(output.squeeze(), label_batch.float())
        t.set_description(f'train_loss (l={loss:.4f})')
        
        if history is not None:
            history.loc[epoch + batch_idx / len(train_loader), 'train_loss'] = loss.data.cpu().numpy()
        
        loss.backward()    
        optimizer.step()
    
    torch.save(model.state_dict(), model_path+'epoch{}.pth'.format(epoch))

def evaluate(epoch, history=None): 
    model.eval() 
    valid_loss = 0.
    mae_metric = 0.
    all_predictions, all_targets = [], []
    
    with torch.no_grad():
        for batch_idx, (seq_batch, label_batch) in enumerate(valid_loader):
            all_targets.append(label_batch.numpy().copy())
            seq_batch = seq_batch.to(device)
            label_batch = label_batch.to(device)
            
            output = model(seq_batch)
            loss = criterion(output.squeeze(), label_batch.float())
            mae = metric(output.squeeze(), label_batch.float())
            valid_loss += loss.data
            mae_metric += mae.data
            all_predictions.append(output.cpu().numpy())
    
    all_predictions = np.concatenate(all_predictions)
    all_targets = np.concatenate(all_targets)
    
    valid_loss /= (batch_idx+1)
    mae_metric /= (batch_idx+1)
    
    if history is not None:
        history.loc[epoch, 'valid_loss'] = valid_loss.cpu().numpy()
    
    valid_status = 'Epoch: {}\tLR: {:.6f}\tValid Loss: {:.4f}\tValid MAE: {:.4f}'.format(
        epoch, optimizer.state_dict()['param_groups'][0]['lr'], valid_loss, mae_metric)
    print(valid_status)
    with open(model_path+'log.txt', 'a+') as f:
        f.write(valid_status+'\n')
        f.close()
    
    return valid_loss, mae_metric

history_train = pd.DataFrame()
history_valid = pd.DataFrame()

n_epochs = 100
init_epoch = 0
max_lr_changes = 1
valid_losses = []
mae_metrics = []
lr_reset_epoch = init_epoch
patience = 2
lr_changes = 0
best_valid_loss = 1000.

for epoch in range(init_epoch, n_epochs):
    torch.cuda.empty_cache()
    gc.collect()
    train_model(epoch, history_train)
    valid_loss, mae_metric = evaluate(epoch, history_valid)
    valid_losses.append(valid_loss)
    mae_metrics.append(mae_metric)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
    elif (patience and epoch - lr_reset_epoch > patience and
          min(valid_losses[-patience:]) > best_valid_loss):
        # "patience" epochs without improvement
        lr_changes +=1
        if lr_changes > max_lr_changes: # 早期停止
            break
        lr /= 5 # 学习率衰减
        print(f'lr updated to {lr}')
        lr_reset_epoch = epoch
        optimizer.param_groups[0]['lr'] = lr
