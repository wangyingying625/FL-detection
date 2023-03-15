#计算单个节点独自训练的准确率
import numpy as np
import torch
import wandb
import csv
import time
import copy

import matplotlib.pyplot as plt

import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from model import CNN
from data import KddData
import numpy as np
import ssl
from bak.clients import groups, client

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
ssl._create_default_https_context = ssl._create_unverified_context
USE_GPU = torch.cuda.is_available()


if USE_GPU:
    model = model.cuda()
class client():
    def __init__(self, id, dataset,epoch,lr):
        self.id=id
        self.dataset=dataset
        self.epoch=epoch
        self.lr=lr
    def getModel(params):
        net = CNN(1, 4)
        dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        net = net.to(dev)
        net.load_state_dict(params,strict=True)
        return net

    def single_train(self,myModel):
        loss_lists=[]
        params={}
        acc_lists=[]
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(myModel.parameters(), lr=self.lr)
        for epoch in range(self.epoch):
            print("*******training************")
            print(f'第{self.id+1} client的 {epoch + 1}(共有{self.epoch}轮)')
            running_loss = 0.0
            running_acc = 0.0
            for i, data in enumerate(self.dataset.train_dataloader, 1):
                img, label = data
                if USE_GPU:
                    img = img.cuda()
                    label = label.cuda()
                img = Variable(img)
                label = Variable(label)
                # 向前传播
                out = myModel(img)
                loss = criterion(out, label)
                running_loss += loss.item() * label.size(0)
                _, pred = torch.max(out, 1)
                num_correct = (pred == label).sum()
                running_acc += num_correct.item()
                # 向后传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print('Finish {} epoch, Loss: {:.6f}, Acc: {:.6f}'.format(
                epoch + 1, running_loss / (len(self.dataset.train_dataset)), running_acc / (len(
                    self.dataset.train_dataset))))
            # 数据展示

            myModel.eval()
            eval_loss = 0
            eval_acc = 0
            for data in self.dataset.test_dataloader:
                img, label = data
                if USE_GPU:
                    img = Variable(img, volatile=True).cuda()
                    label = Variable(label, volatile=True).cuda()
                else:
                    img = Variable(img, volatile=True)
                    label = Variable(label, volatile=True)
                out1 = myModel(img)
                loss = criterion(out1, label)
                eval_loss += loss.item() * label.size(0)
                _, pred = torch.max(out1, 1)
                num_correct = (pred == label).sum()
                eval_acc += num_correct.item()
            print("*******test************")
            print(f"第{self.id + 1}个客户端单独训练的测试结果")
            print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
                self.dataset.test_dataset)), eval_acc / (len(self.dataset.test_dataset))))
            print()
            acc_list=round(eval_acc / (len(self.dataset.test_dataset)),3)
            acc_lists.append(acc_list)
            loss_list=round(eval_loss / (len(self.dataset.test_dataset)),3)
            loss_lists.append(loss_list)
            if epoch==self.epoch-1:
                for key, var in myModel.state_dict().items():
                    params[key] = var.clone()

        print('\n')
        print('\n')
        print('\n')
        print('\n')
        print('\n')
        return params,acc_lists,loss_lists