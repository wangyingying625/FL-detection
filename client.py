import numpy as np
import torch
import csv
import time
import copy

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

if USE_GPU:
    model = model.cuda()


class dataSet():
    def __init__(self, data, target):
        self.data = data
        self.target = target


def readData(path):
    data = []
    target = []
    with open(path, 'r') as data_source:
        csv_reader = csv.reader(data_source)
        for row in csv_reader:
            temp_line = row

            target_line = temp_line.pop()
            data_line = temp_line
            data.append(data_line)
            target.append(target_line)
    return data, target


# def readData1(path):
#     with open(path, 'r') as f:
#         ff = f.read()
#     return ff


class client():
    def __init__(self, id, dataset, epoch, lr):
        self.id = id
        self.dataset = dataset
        self.epoch = epoch
        self.lr = lr

    def getModel(self, params1):
        net = CNN(1, 4)
        dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        net = net.to(dev)
        net.load_state_dict(params1, strict=True)
        return net

    def single_train_test(self, myModel):
        loss_lists = []
        params = {}
        acc_lists = []
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
            acc_list = round(eval_acc / (len(self.dataset.test_dataset)), 3)
            acc_lists.append(acc_list)
            loss_list = round(eval_loss / (len(self.dataset.test_dataset)), 3)
            loss_lists.append(loss_list)
            if epoch == self.epoch - 1:
                for key, var in myModel.state_dict().items():
                    params[key] = var.clone()

        print('\n')
        print('\n')
        print('\n')
        print('\n')
        print('\n')
        return params, acc_lists, loss_lists
    # 模型攻击时用来仅测试的
    def test(self,myModel):
        params = {}
        loss_lists = []
        acc_lists = []
        criterion = nn.CrossEntropyLoss()
        # 优化器
        # 一轮一轮开始训练
        for epoch in range(self.epoch):
            print("*******training************")
            print(f'第{self.id + 1} client的 {epoch + 1}(共有{self.epoch}轮)')
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
            acc_list = round(eval_acc / (len(self.dataset.test_dataset)), 3)
            acc_lists.append(acc_list)
            loss_list = round(eval_loss / (len(self.dataset.test_dataset)), 3)
            loss_lists.append(loss_list)
        print('\n')
        print('\n')
        print('\n')
        print('\n')
        print('\n')
        if epoch == self.epoch - 1:
            for key, var in myModel.state_dict().items():
                params[key] = var.clone()
        return params, acc_lists, loss_lists

    def single_test_train(self, myModel):
        loss_lists = []
        params = {}
        acc_lists = []
        criterion = nn.CrossEntropyLoss()
        # 优化器
        optimizer = optim.SGD(myModel.parameters(), lr=self.lr)
        # 一轮一轮开始训练
        for epoch in range(self.epoch):
            print("*******training************")
            print(f'第{self.id + 1} client的 {epoch + 1}(共有{self.epoch}轮)')
            running_loss = 0.0
            running_acc = 0.0
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
            acc_list = round(eval_acc / (len(self.dataset.test_dataset)), 3)
            acc_lists.append(acc_list)
            loss_list = round(eval_loss / (len(self.dataset.test_dataset)), 3)
            loss_lists.append(loss_list)
            for i, data in enumerate(self.dataset.train_dataloader, 1):
                img, label = data
                if USE_GPU:
                    img = img.cuda()
                    label = label.cuda()
                else:
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
            print("*******train************")
            print(f"第{self.id + 1}个客户端单独训练的训练结果")
            print('Finish {} epoch, Loss: {:.6f}, Acc: {:.6f}'.format(
                epoch + 1, running_loss / (len(self.dataset.train_dataset)), running_acc / (len(
                    self.dataset.train_dataset))))
            # 数据展示

            myModel.eval()

            if epoch == self.epoch - 1:
                for key, var in myModel.state_dict().items():
                    params[key] = var.clone()

        print('\n')
        print('\n')
        print('\n')
        print('\n')
        print('\n')
        return params, acc_lists, loss_lists


class groups():
    def __init__(self, num, all_data, epoches, lr):
        self.num = num
        self.all_data = all_data
        self.epoches = epoches
        self.lr = lr
        self.parameters = {}

    def train(self, parameters,model_attacker):
        accs = []
        # params_list用来存储本轮次训练最后的各个client模型参数params_list=[[参与方1的本轮参数],[],[]]
        params_list = []
        for i in range(self.num):
            dataset = self.all_data[i]
            clienti = client(i, dataset, self.epoches[i], self.lr[i])
            # 常数模型攻击
            if (model_attacker[i]==1):
                model = CNN(1, 4)
                param = model.state_dict()
                for key in param.keys():
                    param[key] = torch.ones(param[key].size())
                model.load_state_dict(param)
                param, acc_lists, loss_lists = clienti.test(model)
                params_list.append(copy.deepcopy(param))
                accs.append(acc_lists)
            elif (model_attacker[i]==2):
                if (parameters != {}):
                    model = clienti.getModel(parameters)
                    param, acc_lists, loss_lists = clienti.single_test_train(model)
                    for key in param.keys():
                        param[key] = torch.mul(param[key], -1)
                        print(f"{i}进行了符号翻转，paramters=空")
                    params_list.append(copy.deepcopy(param))
                    accs.append(acc_lists)
                else:
                    model = CNN(1, 4)
                    param, acc_lists, loss_lists = clienti.single_train_test(model)
                    for key in param.keys():
                        param[key] = torch.mul(param[key], -1)
                        print(f"{i}进行了符号翻转，paramters！=空")
                    params_list.append(copy.deepcopy(param))
                    accs.append(acc_lists)
            # parameters是这个模型当前的参数，用来判断模型是不是初次训练
            elif(model_attacker[i]==0):
                if (parameters != {}):
                    model = clienti.getModel(parameters)
                    param, acc_lists, loss_lists = clienti.single_test_train(model)
                    params_list.append(copy.deepcopy(param))
                    accs.append(acc_lists)
                else:
                    model = CNN(1, 4)
                    param, acc_lists, loss_lists = clienti.single_train_test(model)
                    params_list.append(copy.deepcopy(param))
                    accs.append(acc_lists)
        return accs, params_list





