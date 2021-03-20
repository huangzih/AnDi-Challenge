import numpy as np
import pandas as pd
import sys
import gc
import os
from os.path import isfile
from copy import deepcopy

from torch import nn, optim
from torch.nn import functional as F
from torch.nn import LSTM
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import *
from tqdm import tqdm

# Data Preprocess
with open('./data/task1.txt', 'r') as f:
    words = f.readlines()
    f.close()

d1_data = []
for word in words:
    idx = int(float(word.split(';')[0]))
    if idx == 1: d1_data.append(','.join(word.split(';')[1:])[:-2])

with open('./data/task1-1d.csv', 'w') as f:
    f.write('pos;length\n')
    for word in d1_data:
        f.write(word+';')
        length = len(word.split(','))
        f.write(str(length)+'\n')
    f.close()

del words, d1_data
gc.collect()

data = pd.read_csv('./data/task1-1d.csv', sep=';')

def normalize(x):
    data = np.array([float(i) for i in x.split(',')])
    mean = np.mean(data)
    std = np.std(data)
    data2 = (data - mean)/std
    return ','.join([str(i) for i in data2])

data['new_pos'] = data['pos'].apply(lambda x: normalize(x))

# Check Model File
MarkLength = [10,15,20,25,30,40,45,50,55,60,70,80,90,100,
              105,110,115,120,125,150,175,200,225,250,
              275,300,325,350,375,
              400,425,450,475,500,550,600,650,700,750,800,850,900,950]

flag = False
for fold in range(3):
    for mark in MarkLength:
        if not isfile('./models/Fold{}/{}/bestmodel.pth'.format(fold, mark)):
            print('Model file is missing for length {} at fold {}'.format(mark, fold))
            flag = True

if flag: sys.exit(0)

# PyTorch Dataset
def fixlength(x):
    assert (x>=10)
    if x in MarkLength:
        return x
    MarkLengthTemp = deepcopy(MarkLength)
    MarkLengthTemp.append(x)
    MarkLengthTemp.sort()
    Mark = MarkLengthTemp.index(x)
    return MarkLengthTemp[Mark-1]

data['fix_length'] = data['length'].apply(lambda x: fixlength(x))

class AnDiDataset(Dataset):
    def __init__(self, df):        
        self.df = df.copy()
        
    def __getitem__(self, index): 
        data_seq = torch.Tensor([float(i) for i in self.df['new_pos'].iloc[index].split(',')])       
        ori_length = self.df['length'].iloc[index]
        fix_length = self.df['fix_length'].iloc[index]
        
        if fix_length == ori_length:
            return data_seq.unsqueeze(-1), fix_length, 1
        else:
            data_seq_list = []
            for i in [0, ori_length-fix_length]: 
                seq = data_seq[i:i+fix_length].unsqueeze(-1)
                data_seq_list.append(seq)
            return data_seq_list, fix_length, 2
    
    def __len__(self):
        return len(self.df)

test_loader = DataLoader(AnDiDataset(data), batch_size=1, shuffle=False, num_workers=2)

# PyTorch Model
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

model = AnDiModel(1, 64, 3, 1).to(device)

# Check PyTorch Version
try:
    model.load_state_dict(torch.load('./models/Fold0/10/bestmodel.pth'))
except:
    print('fail to load model file, please check the PyTorch version (1.6.0 is required).')

# Inference
output_list_folds = []

for fold in range(3):
    output_list = []

    for seq_batch, seq_length, seq_mark in tqdm(test_loader):

        model.load_state_dict(torch.load('./models/Fold{}/{}/bestmodel.pth'.format(fold, int(seq_length))));
        model.eval()

        with torch.no_grad():
            if int(seq_mark) == 1:
                seq_batch = seq_batch.to(device)
                output = model(seq_batch)
            elif int(seq_mark) == 2:
                output_sum = 0.
                for seq in seq_batch:
                    output = model(seq.to(device))
                    output_sum += output
                output = output_sum/len(seq_batch)
            output_list.append(output.detach().cpu())

    output_list = np.array(torch.tensor(output_list).detach().numpy())
    output_list_folds.append(deepcopy(output_list))

output_list = sum(output_list_folds)/3.
output_list_final = (output_list*1.011).clip(0.05,2.0)

with open('./output/task1-1d.txt', 'w') as f:
    for i in output_list_final:
        f.write('1.0;'+str(i)+'\n')
    f.close()
