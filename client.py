import numpy as np
import torch
import csv
import time

from matplotlib import pyplot
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from model import CNN
from data import KddData
import numpy as np
import ssl

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
ssl._create_default_https_context = ssl._create_unverified_context
USE_GPU = torch.cuda.is_available()

global_parameters = {}

if USE_GPU:
    model = model.cuda()
class dataSet():
    def __init__(self, data, target):
        self.data=data
        self.target=target
def showChart(xs,name,data):
    year = xs
    people = data
    # 生成图表
    pyplot.plot(year, people)
    # 设置横坐标为year，纵坐标为population，标题为Population year correspondence
    pyplot.xlabel('epoch')
    pyplot.title(name)
    pyplot.xticks([1, 2, 3, 4, 5,6,7])

    # 显示图表
    pyplot.show()

def readData(path):
    data=[]
    target=[]
    with open(path,'r') as data_source:
        csv_reader=csv.reader(data_source)
        for row in csv_reader:
            temp_line=row

            target_line=temp_line.pop()
            data_line=temp_line
            data.append(data_line)
            target.append(target_line)
    return data,target

# def readData1(path):
#     with open(path, 'r') as f:
#         ff = f.read()
#     return ff


class client():
    def __init__(self, id, dataset,epoch,lr):
        self.id=id
        self.dataset=dataset
        self.epoch=epoch
        self.lr=lr
    def getModel(params):
        myModel = CNN(1, 4)

        loss_lists = []
        param1 = {}
        params = []
        acc_lists = []
        acc_temp = 0
        p = 0
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(myModel.parameters(), lr=self.lr)
        for epoch in range(self.epoch):
            print("*******training************")
            print(f'第{self.id + 1} client的 {epoch + 1}(共有{self.epoch}轮)')
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
            acc_list = running_acc / (len(self.dataset.train_dataset))
            acc_lists.append(acc_list)
            loss_list = running_loss / (len(self.dataset.train_dataset))
            loss_lists.append(loss_list)
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
            if epoch != 0:
                if (epoch % 4 == 0 or epoch == self.epoch - 1):
                    params.append({})
                    for key, var in myModel.state_dict().items():
                        params[p][key] = var.clone()
                    p = p + 1

        print(f"----------------第{self.id + 1}个客户端训练结束---------------")
        print('\n')
        print('\n')
        print('\n')
        print('\n')
        print('\n')
        return params, acc_lists, loss_lists

    def single_train(self,myModel):
        loss_lists=[]
        params=[]
        acc_lists=[]
        acc_temp=0
        p=0
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
            acc_list=running_acc / (len(self.dataset.train_dataset))
            acc_lists.append(acc_list)
            loss_list=running_loss / (len(self.dataset.train_dataset))
            loss_lists.append(loss_list)
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
            # if (eval_acc / (len(self.dataset.test_dataset)) > acc_temp ):
            #     # 得到网络每一层上
            #     acc_temp = eval_acc / (len(self.dataset.test_dataset))
            #     print("*******test************")
            #     print(f"第{self.id + 1}个客户端单独训练的测试结果")
            #     print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
            #         self.dataset.test_dataset)), eval_acc / (len(self.dataset.test_dataset))))
            #     print()
            if epoch!=0:
                if (epoch%4==0 or epoch==self.epoch-1):
                    params.append({})
                    for key, var in myModel.state_dict().items():
                        params[p][key] = var.clone()
                    p = p + 1

        print(f"----------------第{self.id+1}个客户端训练结束---------------")
        print('\n')
        print('\n')
        print('\n')
        print('\n')
        print('\n')
        return params,acc_lists,loss_lists


class groups():
    def __init__(self, num, all_data,epoches,lr):
        self.num=num
        self.all_data=all_data
        self.epoches=epoches
        self.lr=lr
    def train(self):
        sum_parameters = []
        for i in range(self.num):

            dataset = self.all_data[i]
            clienti=client(i, dataset,self.epoches[i],self.lr[i])
            model=CNN(1,4)
            param,acc_lists,loss_lists=clienti.single_train(model)
            list=[k+1 for k in range(self.epoches[i])]
            showChart(list,f'{i+1}_train_acc',acc_lists)
            showChart(list,f'{i+1}_train_loss',loss_lists)
            for k in range(len(param)):
                if len(sum_parameters)==k:
                    sum_parameters.append( {})
                    for key, var in param[k].items():
                        sum_parameters[k][key] = var.clone()
                else:
                    for var in sum_parameters[k]:
                        sum_parameters[k][var] = sum_parameters[k][var] + param[k][var]

        # return data_test,data_test_len
        return sum_parameters
    # def train(self):
    #     sum_parameters = []
    #     # for i in range(self.num):
    #     i=0
    #     dataset = self.all_data[i]
    #     clienti=client(i, dataset,self.epoches[i],self.lr[i])
    #     param,acc_lists,loss_lists=clienti.single_train()
    #     # param=[[],[],[]]
    #     list=[k+1 for k in range(self.epoches[i])]
    #     showChart(list,f'{i+1}_train_acc',acc_lists)
    #     showChart(list,f'{i+1}_train_loss',loss_lists)
    #     # return data_test,data_test_len
    #     # return sum_parameters





