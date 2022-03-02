import numpy as np
import torch
import wandb
import csv
import time

import matplotlib.pyplot as plt

import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from model import CNN
from data import KddData
import numpy as np
import ssl
from clients import groups, client

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
ssl._create_default_https_context = ssl._create_unverified_context
USE_GPU = torch.cuda.is_available()
class dataSet():
    def __init__(self, data, target):
        self.data=data
        self.target=target
def showChart1(xs,name,data):
    year = xs
    people = data
    # 生成图表
    plt.plot(year, people)
    # 设置横坐标为year，纵坐标为population，标题为Population year correspondence
    plt.xlabel('epoch')
    plt.ylabel('acc')  # y轴标题
    plt.title(name)
    plt.xticks([1, 2, 3, 4, 5,6,7])

    # 显示图表
    plt.show()
def showChart(xs,data,legend):

    plt.title('Avg_Acc')  # 折线图标题
    plt.xlabel('step')  # x轴标题
    plt.ylabel('acc')  # y轴标题
    for i in range(len(data)):
        plt.plot(xs, data[i], marker='o', markersize=3)  # 绘制折线图，添加数据点，设置点的大小

    # for item in data:
    #     for a, b in zip(xs, item):
    #         plt.text(a, b, b, ha='center', va='bottom', fontsize=10)  # 设置数据标签位置及大小

    plt.legend(legend)  # 设置折线名称

    plt.show()  # 显示折线图


def predict(data, multiple=False):
    _data = dataset.encode(data)
    _data = torch.from_numpy(
        np.pad(_data, (0, 64 - len(_data)), 'constant').astype(np.float32)
    ).reshape(-1, 1, 8, 8).cuda()
    _out = int(torch.max(model(_data).data, 1)[1].cpu().numpy())
    return dataset.decode(_out, label=True)
def _aggregate(w_locals):
    # for key in w_locals[0].keys():
    #     paranew[key] = (param0[key] + param1[key]) / 2

    length=len(w_locals)
    param_new = {}
    for key in w_locals[0].keys():
        for idx in range(len(w_locals)):
            if(idx==0):
                param_new[key] = w_locals[idx][key]
            else:
                param_new[key] = (w_locals[idx][key] + param_new[key])

        param_new[key]=param_new[key]/length
    return param_new
def update_model(w_locals,model):
    model1 = model
    params_up=_aggregate(w_locals)
    model1.load_state_dict(params_up)
    return model1
def single_test(model,dataset,test_data_len,idx):

    eval_acc = 0
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    for data in dataset:
        img, label = data
        img, label = img.to(dev), label.to(dev)
        out = model(img)
        _, pred = torch.max(out, 1)
        num_correct = (pred == label).sum()
        eval_acc += num_correct.item()
    print(f"第{idx + 1}个客户端测试结果")
    print('Test Acc: {:.6f}'.format( eval_acc / test_data_len))
    return  round((eval_acc / test_data_len),3)

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
def getData(client_num):
    data, target = readData('./data/kddcup.txt')
    kddcup99 = dataSet(data, target)
    np.random.seed(166)
    rng_state = np.random.get_state()
    np.random.shuffle(kddcup99.data)
    np.random.set_state(rng_state)
    np.random.shuffle(kddcup99.target)
    kddcup99.data = np.array_split(kddcup99.data, client_num, axis=0)
    kddcup99.target = np.array_split(kddcup99.target, client_num, axis=0)
    return kddcup99


if __name__ == '__main__':
    # 设置客户端参数
    ls=[0.003,0.002,0.003]
    # ls=[1e-3,1e-3]
    num_epoches = [5,5,5]
    # epoches=[]
    test_acc=[]
    global_parameters={}
    client_num = 3
    batch_size = 64
    # print(f"共有{client_num}个客户端,每个客户端的epoch为{num_epoches[0]}")
    print(f"共有{client_num}个客户端,每个客户端的epoch为{num_epoches[0]},{num_epoches[1]},{num_epoches[2]}")

    net=CNN(1,4)
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    net = net.to(dev)
    xs=[]
    rounds=5
    legend=[]
    acc_list=[]
    for key, var in net.state_dict().items():
        global_parameters[key] = var.clone()
    # getData
    print("------loadData-------")
    data_total = getData(client_num)
    all_data=[]
    for i in range(client_num):
        test_acc.append([])
        legend.append(f"client_{i+1}")
        dataset = KddData(batch_size, data_total.data[i],data_total.target[i])
        all_data.append(dataset)
    print("------loadData结束-------")
    #  获取数据结束
    myGroup=groups(client_num,all_data,num_epoches,ls)
    # rounds次更新
    for round in range(rounds):
        if round==0:
            print("---------聚合前训练-----------")
        else:
            print(f"-----------第{round}次聚合后结果-----------")
        sum_parameters,accs = myGroup.train()
        for var in global_parameters:
            global_parameters[var] = (sum_parameters[var] / client_num)
        if len(acc_list)<len(accs):
            acc_list=accs.copy()
        else:
            for q in range(len(acc_list)):
                acc_list[q] = acc_list[q] + accs[q]
                # acc_list[q]+(accs[q])
    # acc_list=[[],[],[],[]]
    # 显示每个client的曲线
    # for client in range(client_num):
    #
    #     showChart1(list1, f'{client + 1}_train_acc', acc_list[client])
    list1 = [f'{k + 1}_acc' for k in range(client_num)]
    xs = [k+1 for k in range(rounds*num_epoches[0])]
    showChart(xs,acc_list,list1)


