import  torch
import csv
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch import nn
from torch.utils.data import Dataset,random_split
from tqdm import tqdm
from sincnet_layers_5 import SincConv
from CBEncoder import Encoder,projection
from backbone import backbonemodel
from torchinfo import summary
def try_gpu(i=0):  #@save
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')
torch.manual_seed(42)
class CustomDataset(Dataset):
    def __init__(self, file_path):
        # 读取数据文件
        self.data = np.loadtxt(file_path, delimiter=',')  # 假设数据是以逗号分隔的
        self.num_samples = self.data.shape[0]  # 样本数量
        self.num_frames = 256  # 每个样本包含256帧
        self.mfcc_dim = 39  # MFCC的维度
        #self.wavelet_dim = 3  # 小波系数的维度
        self.wavelet_dim = 5  # 小波系数的维度
        self.num_features = self.num_frames * (self.mfcc_dim + self.wavelet_dim)  # 特征总维度

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 获取单个样本
        sample = self.data[idx]

        # 特征数据：前256*(39+4)个数据
        features = sample[:self.num_features].reshape(self.num_frames, self.mfcc_dim + self.wavelet_dim)

        # 标签：最后一个值
        label = sample[-1]  # 标签是每行数据的最后一个值

        # 分割成 MFCC 和小波系数
        mfcc = features[:, :self.mfcc_dim]  # 取前39列作为MFCC
        wavelet = features[:, self.mfcc_dim:]  # 取后5列作为小波系数

        # 转换为 tensor
        mfcc_tensor = torch.tensor(mfcc, dtype=torch.float32)
        wavelet_tensor = torch.tensor(wavelet, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)  # 标签一般为整型

        return mfcc_tensor, wavelet_tensor, label_tensor

class Mymodel(nn.Module):
    def __init__(self):
        super(Mymodel, self).__init__()
        self.mfcc_encoder = Encoder()
        self.mfcc_neck = projection()
        self.wavlet_encoder = SincConv(64,101,32000)
        self.backbone = backbonemodel(2)
        # 定义两个可学习的权重 alpha 和 beta
        self.alpha = nn.Parameter(torch.ones(1))  # 对 a2 的权重
        self.beta = nn.Parameter(torch.ones(1))  # 对 b1 的权重
        self.linear = nn.Linear(1280, 768)
    def forward(self,mfcc_input,wavlet_input):
        a1 = self.mfcc_encoder(mfcc_input)
        b1 = self.wavlet_encoder(wavlet_input)
        a2 = self.mfcc_neck(a1)
        # 对 a2 和 b1 应用加权
        a2_weighted = a2 * self.alpha #(50, 64, 768)
        b1_weighted = b1 * self.beta #(50, 64, 1280)
        b1_weighted = self.linear(b1_weighted.view(-1, 1280))
        b1_weighted = b1_weighted.view(-1, 64, 768)
        c = torch.stack((a2_weighted,b1_weighted),dim=1)
        out = self.backbone(c)
        return out

device = try_gpu()
model = Mymodel().to(device)
# summary(model, input_size=[(5, 39,256),(5,1,256*3)])
file="D:\FYD\多波段特征提取（MFCC+DWT_5+F1+F2）\CCNUPLUS_train_300_256_MFCC+5cA+F1+F2_label.csv"
train_dataset= CustomDataset(file)
# print(dataset[0])
# train_lodar=DataLoader(train_dataset,batch_size=50,shuffle=True,num_workers=0)

test_file="D:\FYD\多波段特征提取（MFCC+DWT_5+F1+F2）\CCNUPLUS_test_216_256_MFCC+5cA+F1+F2_label.csv"
test_dataset= CustomDataset(test_file)
test_size = len(test_dataset)
train_size=int(len(train_dataset)*0.8)
val_size = len(train_dataset)-train_size
train_data,val_data = random_split(train_dataset,[train_size,val_size])
train_lodar=DataLoader(train_data,batch_size=50 ,shuffle=True,num_workers=0)
val_lodar = DataLoader(val_data,batch_size=50,shuffle=True)

test_lodar=DataLoader(test_dataset,batch_size=50,shuffle=False,num_workers=0)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=1e-4,weight_decay=1e-3)


for epoch in range(300):
    model.train()
    total=0
    num_correct=0
    for step,(x,y,z) in enumerate(tqdm(train_lodar)):
        mfcc_inputs , wavelet_inputs,label = (x,y,z)
        mfcc_inputs=torch.reshape(mfcc_inputs,(-1,39,256))
        #wavelet_inputs = torch.reshape(wavelet_inputs, (-1, 1, 256*3))
        wavelet_inputs = torch.reshape(wavelet_inputs, (-1, 1, 256 * 5))
        mfcc_inputs=mfcc_inputs.to(device=device)
        wavelet_inputs = wavelet_inputs.to(device=device)
        label=label.to(device=device)

        output = model(mfcc_inputs,wavelet_inputs)
        label=label.squeeze(-1)
        loss = criterion(torch.squeeze(output),label.long())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        res=torch.softmax(output.data,dim=1)
        w, out = torch.max(res, dim=1)  # 取出预测的最大值
        total += y.size(0)
        num_correct += (out == label).sum().item()
        train_acc = num_correct / total
    # print('[epoch : %d] train-loss: %.3f  train-acc:%.6f' % (epoch+1 , loss,train_acc))
    tqdm.write(f"[epoch : {epoch+1}] train-loss: {loss.item():.3f}  train-acc:{train_acc:.6f}")
    ##验证集步骤开始
    model.eval()
    total_val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for step, (x, y, z) in enumerate(val_lodar):
            mfcc_inputs, wavelet_inputs, label = (x, y, z)
            mfcc_inputs = torch.reshape(mfcc_inputs, (-1, 39, 256))
            #wavelet_inputs = torch.reshape(wavelet_inputs, (-1, 1, 256 * 3))
            wavelet_inputs = torch.reshape(wavelet_inputs, (-1, 1, 256 * 5))
            mfcc_inputs = mfcc_inputs.to(device=device)
            wavelet_inputs = wavelet_inputs.to(device=device)
            label = label.to(device=device)
            label = label.squeeze(-1)

            output = model(mfcc_inputs, wavelet_inputs)

            loss = criterion(torch.squeeze(output), label.long())
            total_val_loss += loss
            res = torch.softmax(output.data, dim=1)
            w, out = torch.max(res, dim=1)  # 取出预测的最大值
            total += y.size(0)
            correct += (out == label).sum().item()
            val_acc = correct / total
    # print("验证集上的Loss:%.3f,  验证集上的准确率:%.6f"  %  (total_val_loss,val_acc))
    tqdm.write(f"[验证集上的Loss : {total_val_loss.item():.3f}]   验证集上的准确率:{val_acc:.6f}")



print("--------------TEST--------------")


model.eval()
with torch.no_grad():
    correct=0
    total=0
    for step , (x,y,z) in enumerate(test_lodar):
        mfcc_inputs, wavelet_inputs, label = (x, y, z)
        mfcc_inputs = torch.reshape(mfcc_inputs, (-1, 39, 256))
        # = torch.reshape(wavelet_inputs, (-1, 1, 256 * 3))
        wavelet_inputs = torch.reshape(wavelet_inputs, (-1, 1, 256 * 5))
        mfcc_inputs = mfcc_inputs.to(device=device)
        wavelet_inputs = wavelet_inputs.to(device=device)
        label = label.to(device=device)
        output = model(mfcc_inputs,wavelet_inputs)
        label = label.squeeze(-1)
        res = torch.softmax(output.data, dim=1)
        _ , pred_y = torch.max(res,dim=1)

        total += y.size(0)
        correct += (pred_y == label).sum().item()
    print('Accuracy on test set : %.6f ' % (correct/total))

# torch.save(model.state_dict(),'300epoch_lr1e-4_batch50_双权重_feng.pt')
torch.save(model.state_dict(),'')

print("模型保存成功")