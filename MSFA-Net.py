import  torch
import csv
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch import nn
from torch.utils.data import Dataset,random_split
from tqdm import tqdm
from torchinfo import summary
class ConvModule(nn.Module):
    def __init__(self,c_in,c_out,k,s,p):
        super(ConvModule, self).__init__()
        self.conv = nn.Conv2d(in_channels=c_in,out_channels=c_out,kernel_size=k,stride=s,padding=p)
        self.bn = nn.BatchNorm2d(c_out)
        self.activation = nn.ReLU()
        # self.activation = nn.GELU()
    def forward(self,x):
        return self.activation(self.bn(self.conv(x)))

class SPPFBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pool_size=5):
        super(SPPFBlock, self).__init__()
        self.initial_conv = ConvModule(in_channels,out_channels,1,1,0)

        self.pool1 = nn.MaxPool2d(kernel_size=pool_size, stride=1, padding=pool_size // 2)
        self.pool2 = nn.MaxPool2d(kernel_size=pool_size, stride=1, padding=pool_size // 2)
        self.pool3 = nn.MaxPool2d(kernel_size=pool_size, stride=1, padding=pool_size // 2)

        self.final_conv = ConvModule(out_channels*4,out_channels,1,1,0)

    def forward(self, x):
        x_initial = self.initial_conv(x)
        x1 = self.pool1(x_initial)
        x2 = self.pool2(x1)
        x3 = self.pool3(x2)
        x_concat = torch.cat((x_initial, x1, x2, x3), dim=1)
        x_final = self.final_conv(x_concat)
        return x_final

class Bottleneck(nn.Module):
    def __init__(self,c_in):
        super(Bottleneck, self).__init__()
        # self.conv1 = nn.Conv1d(in_channels=c,out_channels=c//2,kernel_size=3,stride=1,padding=1)
        self.conv1 = ConvModule(c_in,c_in//2,3,1,1)
        # self.conv2 = nn.Conv1d(in_channels=c//2, out_channels=c, kernel_size=3, stride=1, padding=1)
        self.conv2 = ConvModule(c_in//2,c_in,3,1,1)
    def forward(self,x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        return x2+x

class C2f(nn.Module):
    def __init__(self,c_in,c_out,n):
        super(C2f, self).__init__()
        self.n = n
        self.conv1 = ConvModule(c_in,c_out,1,1,0)
        self.bottle1 = Bottleneck(c_out//2)
        self.finalconv = ConvModule(((self.n+2)//2)*c_out,c_out,1,1,0)
    def forward(self,x):
        x1 =self.conv1(x)
        x2,x3 = torch.split(x1,x1.shape[1]//2,dim=1)##x2直接接到最后
        bottle_out = []
        for i in range(self.n):
            x4 = self.bottle1(x3)
            bottle_out.append(x4)
        x_con1 = torch.cat((x2,x3),dim=1)
        x_con2 = torch.cat(bottle_out,dim=1)
        x_con = torch.cat((x_con1,x_con2),dim=1)##c=0.5*(n+2)*c_out
        x5 = self.finalconv(x_con)
        return x5

class backbonemodel(nn.Module):
    def __init__(self,c_in):
        super(backbonemodel, self).__init__()
        self.stemlayer = ConvModule(c_in,64,3,2,1)
        self.layer1 = nn.Sequential(
            ConvModule(64,128,3,2,1),
            C2f(128,128,8)
        )
        self.layer2 = nn.Sequential(
            ConvModule(128, 256, 3, 2, 1),
            C2f(256, 256, 18)
        )
        self.layer3 = nn.Sequential(
            ConvModule(256, 512, 3, 2, 1),
            C2f(512, 512, 18)
        )
        # self.layer4 = nn.Sequential(
        #     ConvModule(512, 512, 3, 2, 1),
        #     C2f(512, 512, 8),
        #     SPPFBlock(512,512)
        self.layer4 = nn.Sequential(
            ConvModule(512, 256, 3, 2, 1),
            C2f(256, 128, 8),
            SPPFBlock(128, 32)#####输出[batch,32,2,24],拉直为1536
        )
        self.layer5 = nn.Sequential(
            nn.Linear(1536, 512),
            nn.ReLU(),
            # nn.GELU(),
            # nn.Dropout(0.2),
            # nn.Linear(512, 256),
            # nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(),
            # nn.GELU(),
            nn.Linear(128, 60)
        )
    def forward(self,x):
        x1 = self.stemlayer(x)
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        x6 = x5.reshape(x5.shape[0],-1)
        x7 = self.layer5(x6)
        return x7


if __name__ == '__main__':
    device = 'cuda:0'
    mymodel = backbonemodel(2).to(device)
    x = torch.rand(50,2,64,768).to(device)
    summary(mymodel,input_size=(50,2,64,768))
    out1 = mymodel(x)
    print(out1.shape)
