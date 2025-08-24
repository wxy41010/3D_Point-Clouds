# -*- coding: utf-8 -*-
"""
乐乐感知学堂公众号
@author: https://blog.csdn.net/suiyingy
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 20:30:41 2020
@author: yehx
"""
 
import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
 
 
# 构建输入数据集
class Data_set(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.x_data = np.random.randint(0, 10, (10,4,5))
        self.y_data = np.random.randint(0, 10, (10,1))
    
    def __getitem__(self, index):
        x, y = self.pull_item(index)
        return x, y
        
    def __len__(self):
        return self.x_data.shape[0]   
    
    def pull_item(self, index):
        return self.x_data[index, :, :], self.y_data[index, :]
   
    
    
class MyModel(nn.Module):
    def __init__(self, num_classes=10):
        super(MyModel, self).__init__()
        self.model_name = "test"
        self.conv = nn.Conv1d(4, 1, 3, 1, 1)
        self.bn = nn.BatchNorm1d(1)
        self.relu = nn.ReLU(inplace=True)
        self.fc   = nn.Linear(5, 1)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = x.view(-1, 5)
        return self.fc(x)
    
class MyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x, y):
        return torch.mean(torch.pow((x - y), 2))
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='基础模型参数配置')
    train_set = parser.add_mutually_exclusive_group()
    parser.add_argument('--batch_size', default=2, type=int,
                    help='Batch size for training')
    args = parser.parse_args()
    
    
    dataset = Data_set()
    data_loader = DataLoader(dataset, args.batch_size, shuffle=True)
    Net = MyModel()
    criterion = MyLoss()
    #criterion = nn.MSELoss() #也可使用pytorch自带的损失函数
    optimzer = torch.optim.SGD(Net.parameters(), lr=0.001)
    Net.train()
    loss_list = []
    num_epoches = 200
    for epoch in range(num_epoches):
        for i, data in enumerate(data_loader):
            inputs, labels = data
            inputs, labels = Variable(inputs).float(), Variable(labels).float()
            out = Net(inputs)
            loss = criterion(out, labels)  # 计算误差
            optimzer.zero_grad()  # 清除梯度
            loss.backward()
            optimzer.step()
        loss_list.append(loss.item())
        if (epoch+1) % 10 == 0:
            print('[INFO] {}/{}: Loss: {:.4f}'.format(epoch+1, num_epoches, loss.item()))
  
    #作图：误差loss在迭代过程中的变化情况
    plt.plot(loss_list, label='loss for every epoch')
    plt.legend()
    plt.show()  
  
    #训练的模型参数   
    print('[INFO] 训练后模型的参数：')
    for name,parameters in Net.named_parameters():
        print(name,':',parameters)
 
    
    
    #测试模型结果
    print('[INFO] 计算某个样本模型运算结果：')
    Net.eval()
    x_data = np.random.randint(0, 10, (4,5))
    x_data = torch.tensor(x_data).float()
    x_data = x_data.unsqueeze(0)
    y_data = Net(x_data)
    print(y_data.item())
    
    #模型保存
    torch.save(Net, 'model0.pth')
    
    #模型加载
    print('[INFO] 验证模型加载运算结果：')
    model0 =torch.load('model0.pth')
    y_data = model0 (x_data)
    print(y_data.item())
 
 
 
 

 