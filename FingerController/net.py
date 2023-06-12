import torch
import torch.nn as nn

class SimpleLSTM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = torch.nn.LSTM(99, 64, 1, batch_first = True)
        self.relu1 = torch.nn.ReLU(inplace=False)
        self.lin1 = torch.nn.Linear(64, 32, bias=False)
        self.relu2 = torch.nn.ReLU(inplace=False)
        self.lin2 = torch.nn.Linear(32, 3, bias=True)
        self.pad = torch.zeros(1, 1, 64).to('cuda:0')
        
    def forward(self, x):
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

class SimpleConvLSTM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(30, 64, 3, 1, 1)
        self.relu = torch.nn.ReLU(inplace=False)
        self.lstm = torch.nn.LSTM(100, 64, 1, batch_first = True)
        self.relu1 = torch.nn.ReLU(inplace=False)
        self.lin1 = torch.nn.Linear(64, 32, bias=False)
        self.relu2 = torch.nn.ReLU(inplace=False)
        self.lin2 = torch.nn.Linear(32, 3, bias=True)
        self.pad = torch.zeros(1, 1, 64).to('cuda:0')
        
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

class SimpleConvLSTM2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(30, 64, 3, 1, 1)
        self.relu = torch.nn.ReLU(inplace=False)
        self.lstm = torch.nn.LSTM(121, 64, 1, batch_first=True)
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
