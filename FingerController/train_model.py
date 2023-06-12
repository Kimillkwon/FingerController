import numpy as np
import os

actions = ['play', 'pause', 'max_size', 'min_size', 'vol_up', 'vol_down', 'close_screen','video_forward_10s','video_backward_10s']

data = np.concatenate([
    np.load('dataset/seq_close_screen.npy'),
    np.load('dataset/seq_max_size.npy'),
    np.load('dataset/seq_min_size.npy'),
    np.load('dataset/seq_pause.npy'),
    np.load('dataset/seq_play.npy'),
    np.load('dataset/seq_vol_down.npy'),
    np.load('dataset/seq_vol_up.npy'),
    np.load('dataset/seq_video_forward_10s.npy'),
    np.load('dataset/seq_video_backward_10s.npy'),
], axis=0)
x_data = data[:, :, :-1]
labels = data[:, 0, -1]

print(x_data.shape)
print(labels.shape)

data_len=len(data)

x_data = np.concatenate([x_data, np.zeros((data_len, 30, 2))], axis=2)

print(x_data.shape)
print(labels.shape)
x_data = x_data.reshape((data_len, 30, 11, 11))

import torch.nn.functional as F
import torch
labels = torch.Tensor(labels)

num_class = 9
y_data = F.one_hot(labels.to(torch.int64))
y_data.shape


from sklearn.model_selection import train_test_split

x_data = x_data.astype(np.float32)
y_data = y_data.numpy().astype(np.float32)

x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.2, random_state=2023)

print(x_train.shape, y_train.shape)
print(x_val.shape, y_val.shape)


import torch.nn as nn

class SimpleConvLSTM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(30, 64, 3, 1, 1)
        self.relu = torch.nn.ReLU(inplace=False)
        self.lstm = torch.nn.LSTM(121, 64, 1, batch_first = True)
        self.relu1 = torch.nn.ReLU(inplace=False)
        self.lin1 = torch.nn.Linear(64, 32, bias=False)
        self.relu2 = torch.nn.ReLU(inplace=False)
        self.lin2 = torch.nn.Linear(32, 9, bias=True)
        self.pad = torch.zeros(1, 1, 64).to('cpu')
        
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        b, c, _, _ = x.shape
        x = x.view(b, c, -1)
        x, _ = self.lstm(x, (self.pad, self.pad))
        x = x[-1, -1]
        x = self.relu1(x)
        x = self.lin1(x)
        x = self.relu2(x)
        out = self.lin2(x)
        out = out.unsqueeze(0)
        return out
     
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
        
model = SimpleConvLSTM()
epochs = 200
device = 'cpu'
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50)
criterion = torch.nn.CrossEntropyLoss().to(device)

x_train = torch.Tensor(x_train)
y_train = torch.Tensor(y_train)
x_val = torch.Tensor(x_val)
y_val = torch.Tensor(y_val)

model(x_train[0].unsqueeze(0))

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, datas, labels, transform=None):
        self.datas = datas
        self.labels = labels
        self.transform=transform
    
    def __len__(self):
        return len(self.datas)
    
    def __getitem__(self, index):
        return self.datas[index], self.labels[index]

train_set = CustomDataset(x_train, y_train)
val_set = CustomDataset(x_val, y_val)
train_loader = DataLoader(dataset=train_set, batch_size=1, shuffle=True)
validation_loader = DataLoader(dataset=val_set, batch_size=1, shuffle=False)

import torchinfo

torchinfo.summary(model, (1, 30, 11, 11))

from trainer import Trainer
params_train = {
    'num_epochs': 15,
    'optimizer': optimizer,
    'loss_func': criterion,
    'train_loader': train_loader,
    'validation_loader': validation_loader,
    'lr_scheduler': scheduler,
    'device': 'cpu',
    'save_path': './test9_6_new.pth'
}

trainer = Trainer(model, params_train)
model, loss, metric = trainer.train_val()


from plot import plot_show
plot_show(15, loss, metric)
