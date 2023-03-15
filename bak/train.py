import numpy as np
import torch
import wandb
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
def loadData(client_num):

    # kddcup99 = datasets.fetch_kddcup99()
    data,target=readData('../data/kddcup.txt')
    kddcup99 = dataSet(data,target)
    np.random.seed(66)
    rng_state = np.random.get_state()
    np.random.shuffle(kddcup99.data)
    np.random.set_state(rng_state)
    np.random.shuffle(kddcup99.target)
    kddcup99.data = np.array_split(kddcup99.data, client_num, axis=0)
    kddcup99.target = np.array_split(kddcup99.target, client_num, axis=0)
    # return kddcup99.data,kddcup99.target
    return kddcup99


def single_train(dataset,idx,num_epoches,lr):
    myModel = CNN(1,4)
    loss_lists=[]
    acc_lists=[]
    acc_temp=0
    param1={}
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(myModel.parameters(), lr=lr)
    for epoch in range(num_epoches):
        print("*******training************")
        print(f'第{idx+1} client的 {epoch + 1}(共有{num_epoches}轮)')
        running_loss = 0.0
        running_acc = 0.0
        for i, data in enumerate(dataset.train_dataloader, 1):
            img, label = data
            if USE_GPU:
                img = img.cuda()
                label = label.cuda()
            img = Variable(img)
            label = Variable(label)
            # 向前传播
            # print(myModel.state_dict())
            out =  myModel(img)
            loss = criterion(out, label)
            running_loss += loss.item() * label.size(0)
            _, pred = torch.max(out, 1)
            num_correct = (pred == label).sum()
            # accuracy = (pred == label).float().mean()
            running_acc += num_correct.item()
            # 向后传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('Finish {} epoch, Loss: {:.6f}, Acc: {:.6f}'.format(
            epoch + 1, running_loss / (len(dataset.train_dataset)), running_acc / (len(
                dataset.train_dataset))))
        # 数据展示
        acc_list=running_acc / (len(dataset.train_dataset))
        acc_lists.append(acc_list)
        loss_list=running_loss / (len(dataset.train_dataset))
        loss_lists.append(loss_list)
        myModel.eval()
        eval_loss = 0
        eval_acc = 0
        for data in dataset.test_dataloader:
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
        # param = myModel.state_dict()

        if (eval_acc / (len(dataset.test_dataset)) > acc_temp and eval_acc / (len(dataset.test_dataset))>0.5 ):
            # 得到网络每一层上
            for key, var in myModel.state_dict().items():
                param1[key] = var.clone()
            acc_temp = eval_acc / (len(dataset.test_dataset))
            print("*******test************")
            print(f"第{idx + 1}个客户端单独训练的测试结果")
            print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
                dataset.test_dataset)), eval_acc / (len(dataset.test_dataset))))
            print()

    print(f"----------------第{idx+1}个客户端训练结束---------------")
    print('\n')
    print('\n')
    print('\n')
    print('\n')
    print('\n')
    return param1,acc_lists,loss_lists,acc_temp,myModel



def predict(data, multiple=False):
    _data = dataset.encode(data)
    _data = torch.from_numpy(
        np.pad(_data, (0, 64 - len(_data)), 'constant').astype(np.float32)
    ).reshape(-1, 1, 8, 8).cuda()
    _out = int(torch.max(model(_data).data, 1)[1].cpu().numpy())
    return dataset.decode(_out, label=True)


def train(client_num,batch_size,epoch,ls):
    # 返回测试集数据。测试集数据条数，模型参数集
    data_test=[]
    data_test_len=[]
    data_total=loadData(client_num)
    sum_parameters = None
    acc_total=0
    accs=[]
    params=[]
    # data_total.data 是client_num大的数组
    for i in range(client_num):
        dataset = KddData(batch_size, data_total.data[i],data_total.target[i])
        param,acc_lists,loss_lists,acc,model=single_train(dataset,i,epoch[i],ls[i])
        acc_total=acc_total+acc
        accs.append(acc)
        params.append(param)
        list=[k+1 for k in range(epoch[i])]
        showChart(list,f'{i}_train_acc',acc_lists)
        showChart(list,f'{i}_train_loss',loss_lists)
        if sum_parameters is None:
            sum_parameters = {}
            for key, var in param.items():
                sum_parameters[key] = var.clone()
                # sum_parameters[key]=trmp*acc
        else:
            for var in sum_parameters:
                sum_parameters[var] = sum_parameters[var] + param[var]
        data_test.append(dataset.test_dataloader)
        data_test_len.append(len(dataset.test_dataset))
    for var in global_parameters:
        global_parameters[var] = (sum_parameters[var] / client_num)
    return data_test,data_test_len
def single_test(model,dataset,test_data_len,idx):
    eval_loss = 0
    eval_acc = 0
    criterion = nn.CrossEntropyLoss()
    for data in dataset:
        img, label = data
        if USE_GPU:
            img = Variable(img, volatile=True).cuda()
            label = Variable(label, volatile=True).cuda()
        else:
            img = Variable(img, volatile=True)
            label = Variable(label, volatile=True)
        out = model(img)
        loss = criterion(out, label)
        eval_loss += loss.item() * label.size(0)
        _, pred = torch.max(out, 1)
        num_correct = (pred == label).sum()
        eval_acc += num_correct.item()
    print(f"第{idx + 1}个客户端测试结果")
    print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (test_data_len), eval_acc / test_data_len))





if __name__ == '__main__':
    # 参数输入
    ls=[0.0004,0.0005,1e-3]
    num_epoches = [10,10,10]
    epoches=[]

    client_num = 3
    batch_size = 128

    print(f"共有{client_num}个客户端,每个客户端的epoch为{num_epoches[0]},{num_epoches[1]},{num_epoches[2]}")
    model1 = CNN(1,4)
    for key, var in model1.state_dict().items():
        global_parameters[key] = var.clone()

    test_data,test_data_len=train(client_num,batch_size,num_epoches,ls)

    model1.load_state_dict(global_parameters,strict=True)
    print("----------------模型聚合后测试结果----------------------")
    for i in range(client_num):
        single_test(model1,test_data[i],test_data_len[i],i)
    # model_new=update_model(model,test_data,test_data_len)

