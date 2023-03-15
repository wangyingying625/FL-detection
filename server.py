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
from client import groups, client

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
class dataSet():
    def __init__(self, data, target):
        self.data=data
        self.target=target
# 函数功能：加载数据集，把每一条数据划分为特征和标签
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
# 函数功能：加载数据集，随机打乱，按照参与方数据平均划分
def getData(client_num):
    # 数据集的最终形式--41个特征（特征中原本是字符串的仍然保留字符串形式），1个label，删除了label中U2R的数据，并且把label处理为数字(0,1,2,3)
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
# 显示单个图表
# def showChart1(xs,name,data):
#     year = xs
#     people = data
#     # 生成图表
#     plt.plot(year, people)
#     # 设置横坐标为year，纵坐标为population，标题为Population year correspondence
#     plt.xlabel('epoch')
#     plt.ylabel('acc')  # y轴标题
#     plt.title(name)
#     plt.xticks([1, 2, 3, 4, 5,6,7])
#
#     # 显示图表
#     plt.show()
# 显示多条折线的图
def showChart(xs,data,legend):

    plt.title('Avg_Acc')  # 折线图标题
    plt.xlabel('step')  # x轴标题
    plt.ylabel('acc')  # y轴标题
    for i in range(len(data)):
        plt.plot(xs, data[i], marker='o', markersize=3)  # 绘制折线图，添加数据点，设置点的大小
    x_major_locator = MultipleLocator(2)
    # 把x轴的刻度间隔设置为1，并存在变量里
    y_major_locator = MultipleLocator(0.1)
    # 把y轴的刻度间隔设置为10，并存在变量里
    ax = plt.gca()
    # ax为两条坐标轴的实例
    ax.xaxis.set_major_locator(x_major_locator)
    # 把x轴的主刻度设置为1的倍数
    ax.yaxis.set_major_locator(y_major_locator)
    # 把y轴的主刻度设置为10的倍数
    plt.xlim(0, 30)
    # 把x轴的刻度范围设置为-0.5到11，因为0.5不满一个刻度间隔，所以数字不会显示出来，但是能看到一点空白
    plt.ylim(0.0, 1.0)

    # for item in data:
    #     for a, b in zip(xs, item):
    #         plt.text(a, b, b, ha='center', va='bottom', fontsize=10)  # 设置数据标签位置及大小
    # plt.xticks([2, 4, 6, 8, 10, 12, 14,16,18,20,22,24,26,28,30])
    plt.legend(legend)  # 设置折线名称
    plt.show()  # 显示折线图
    plt.savefig("./test.jpg")


def predict(data, multiple=False):
    _data = dataset.encode(data)
    _data = torch.from_numpy(
        np.pad(_data, (0, 64 - len(_data)), 'constant').astype(np.float32)
    ).reshape(-1, 1, 8, 8).cuda()
    _out = int(torch.max(model(_data).data, 1)[1].cpu().numpy())
    return dataset.decode(_out, label=True)

def cosine_similarity(x,y):
    num = x.dot(y.T)
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return num / denom
def EuclideanDistances(a, b):
    # dist = numpy.sqrt(numpy.sum(numpy.square(a - b)))
    # return dist
    sq_a = a ** 2
    sum_sq_a = torch.sum(sq_a, dim=1).unsqueeze(1)  # m->[m, 1]
    sq_b = b ** 2
    sum_sq_b = torch.sum(sq_b, dim=1).unsqueeze(0)  # n->[1, n]
    bt = b.t()
    torch.sqrt(a.to(torch.double))
    # temp=sum_sq_a + sum_sq_b - 2 * a.mm(bt)
    return torch.sqrt((sum_sq_a + sum_sq_b - 2 * a.mm(bt)).to(torch.float))
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
def detect(status,params,len):
    sum_parameters = None
    credit=0
    # credit是可信节点数量
    dis={}
    parms_num=0
    # dis用来存放检测器中心与可信点之间的最大距离，parms_num是dis中的数据个数，超过一半大于就认为是恶意
    for j in range(len):
        if status[j]==1: credit=credit+1
    #     开始聚合可信节点
    for i in range(len):
        if(status[i]==1):
            print("----------")
            print(f'开始聚合--可信节点{i + 1}参与聚合')
            if sum_parameters == None:
                sum_parameters = {}
                for key, var in params[i].items():
                    sum_parameters[key] = var.clone()
            else:
                for var in sum_parameters:
                    sum_parameters[var] = sum_parameters[var] + params[i][var]
    #             sum_paramsters是检测器的中心点，可信节点聚合结束
    # 开始生成检测器
    for var in sum_parameters:
        sum_parameters[var] = (sum_parameters[var] / credit)
        dis[var]=0
        parms_num=parms_num+1
    for i in range(len):
        if (status[i] == 1):
            # 求可信点的范围，第一次尝试，对两个tensor进行reshape，然后求欧式距离，求出来是一个数值，存在dis中
            for item in sum_parameters:
                # temp_sum=sum_parameters[item].reshape(-1).numpy()
                # temp_sum=noramlization(temp_sum)
                temp_sum=torch.reshape(sum_parameters[item],(1,-1))
                temp_parm=torch.reshape(params[i][item],(1,-1))
                # temp_parm=params[i][item].reshape(-1).numpy()
                # temp_parm=noramlization(temp_parm)
                # t[i]=np.linalg.norm(temp_sum - temp_parm)
                t=EuclideanDistances(temp_sum,temp_parm)
                if(dis[item]<t):
                    dis[item]=t
    #                 检测器生成结束
    # 开始检测
    for j in range(len):
        largej=0
        if (status[j]==0):
            for tem in sum_parameters:
                temp_sum1 = torch.reshape(sum_parameters[tem],(1,-1))
                temp_parm1 = torch.reshape(params[j][tem],(1,-1))
                # temp_parm1 = torch.where(torch.isnan(temp_parm1), torch.full_like(temp_parm1, 100), temp_parm1)
                t1 = EuclideanDistances(temp_sum1, temp_parm1)
                if(t1>dis[tem]):
                    largej=largej+1
            if(largej>(parms_num*3/5)):
                print(f"节点{j+1}异常，不参与聚合")
            else:
                print(f"节点{j + 1}正常，参与聚合")
                for var in sum_parameters:
                    sum_parameters[var] = (sum_parameters[var]*(len-1)/len + params[j][var]/len)
    return sum_parameters


if __name__ == '__main__':
    # 设置客户端参数ls是各个参与方的lr,num_epoches是每个参与方在本地训练empoch次之后参与聚合
    # 6个client
    ls=[0.005,0.005,0.005,0.005,0.005,0.005]
    # 四个client
    # ls=[0.004,0.006,0.007,0.007]

    num_epoches = [2,2,2,2,2,2]
    test_acc=[]
    global_parameters={}
    # 总的客户端数目
    client_num = 6
    batch_size = 64
    print(f"共有{client_num}个客户端,每个客户端的epoch为{num_epoches[0]},{num_epoches[1]},{num_epoches[2]}")
    # 使用的是CNN(1,4)
    net=CNN(1,4)
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    net = net.to(dev)

    # 共聚合的轮次, params_list用来存储每次训练模型参数
    rounds=3
    params_lists=[]
    # legend,acc_list,xs都是用作画图的参数
    xs = []
    legend=[]
    acc_list=[]
    # 初始化联邦模型参数
    for key, var in net.state_dict().items():
        global_parameters[key] = var.clone()
    # getData
    print("------loadData-------")
    data_total = getData(client_num)
    print(f"------Data分割结束-------")
    # data_total.target是一个list，可以修改其中的[1][2][3]可以直接修改label
    all_data=[]
    for i in range(client_num):
        test_acc.append([])
        legend.append(f"client_{i+1}")
        if((i+1)==3):
            # KddData的导数第二个参数传1的时候代表当前节点遭受标签攻击，导数第一个表示遭受常数攻击
            dataset = KddData(batch_size, data_total.data[i], data_total.target[i], 0,0)
        else:
            dataset = KddData(batch_size, data_total.data[i],data_total.target[i],0,0)
        all_data.append(dataset)
    print("------loadData结束-------")
    #  获取数据结束。开始初始化参与方，client_num是参与方数量，all_data是所有参与方数据集，
    myGroup=groups(client_num,all_data,num_epoches,ls)
    status = [1,0,0,1,0,1]
    parameters={}
    # 模型攻击标志，为1表示遭受常数攻击
    modelAttack=[2,0,0,0,0,0]
    for round in range(rounds):
        print(f"-----------第{round}次聚合后结果-----------")
        # 所有参与方在此进行round轮训练
        accs,params_list = myGroup.train(parameters,modelAttack)
        # 随后params_list收集参与方生成的模型参数
        # number=0
        # for j in range(client_num):
        #     if status[j] ==1:
        #         number=number+1
        # for var in global_parameters:
        #     global_parameters[var] = (sum_parameters[var] / number)
        parameters=detect(status,params_list,client_num)
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


