import torch
import csv
import numpy as np
import pandas as pd
import torchaudio.transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch import nn
from torch.nn import functional as F
from torchinfo import summary
import queue


def try_gpu(i=0):  # @save
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.device = try_gpu()
        self.conv1 = nn.Sequential(
            # nn.Conv2d(1, 64, kernel_size=3, stride=3),
            nn.Conv1d(39, 64, kernel_size=3, stride=3),
            nn.BatchNorm1d(64, eps=1e-5, momentum=0.1,
                           affine=True, track_running_stats=True),
            # nn.GELU(),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=(2, 2))
        )
        # self.linear1 = nn.Sequential(
        #     nn.Linear(16128, 128),
        #     nn.Dropout(0.3)
        # )
        # self.linear2 = nn.Linear(128, 256)
        self.BiLSTM = nn.LSTM(input_size=1344, hidden_size=128, batch_first=True, bidirectional=True)
        self.encoder_layer1 = nn.TransformerEncoderLayer(d_model=256, nhead=8, batch_first=True)
        self.encoder_layer2 = nn.TransformerEncoderLayer(d_model=256, nhead=8, batch_first=True)

    def forward(self, x):
        x = self.conv1(x)  ##输入为[b,1,128,42]时输出为([b,64,21,7])      输入为[b,1,128,73]时输出为([b,64,21,12])
        # x = Reshape(x.shape[0], 1, x.shape[1] * x.shape[2] * x.shape[3])(x)
        x = Reshape(x.shape[0], 1, x.shape[1] * x.shape[2] )(x)


        # x = Reshape(x.shape[0],  x.shape[1] * x.shape[2] * x.shape[3])(x)
        # ##消融CNN加的MLP
        # x = self.linear1(x)
        # x = self.linear2(x)


        out, (h_n, c_n) = self.BiLSTM(x)  ##输入为[b,1,128,42]时h_n输出为([2,b,128])
        x = torch.permute(h_n, (1, 0, 2)).contiguous()  ##[b,2,128]
        x = Reshape(x.shape[0], x.shape[1] * x.shape[2])(x)  ##[b,256]
        # #x = out.squeeze(dim=1)
        x = self.encoder_layer1(x)  ##([b,256])只是在全局范围内进行了多头自注意力机制
        x = self.encoder_layer2(x)
        mean = torch.mean(x, dim=0)
        std = torch.sqrt(x.clamp(min=1e-9))
        x2 = x + mean + std  ##[b,256]


        return x2 ####输出尺寸是(b,256),后续的操作不影响尺寸，这个128由BiLSTM决定，后续的全局注意力跟特殊池化操作均不改变

class projection(nn.Module):
    def __init__(self):
        super(projection, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, stride=3),
            nn.BatchNorm1d(64, eps=1e-5, momentum=0.1, affine=True,
                           track_running_stats=True),
            # nn.GELU())
            nn.ReLU())
        self.fc = nn.Sequential(nn.Linear(85,256),
                                nn.Dropout(0.2),
                                nn.ReLU(),
                                nn.Linear(256,256*3))
    def forward(self,x):
        x = x.unsqueeze(1)
        x1 = self.conv1(x)
        x2 = self.fc(x1)
        return x2
if __name__ == '__main__':
    device = try_gpu()
    model  = Encoder().to(device)
    pro = projection().to(device)
    #summary(model,input_size=(5,1,256,39))
    # x=torch.rand(5,1,256,39).to(device=device)
    x = torch.rand(5,  39, 256).to(device=device)
    summary(pro, input_size=(5, 256))
    out1 = model(x)
    print(out1.shape)
    # out2 = out1.unsqueeze(1)
    out3 = pro(out1)
    print(out3.shape)
