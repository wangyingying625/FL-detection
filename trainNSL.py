import numpy as np
import torch
import csv
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from model import CNN
from dataNSL import KddData
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
        self.data=data
        self.target=target
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
def loadData(client_num):

    data_temp,target_temp=readData('./data/notUse/train1.txt')
    data_test,target_test=readData('./data/notUse/test1.txt')
    train=dataSet(data_temp,target_temp)
    test=dataSet(data_test,target_test)
    # # 打乱train
    np.random.seed(96)
    rng_state = np.random.get_state()
    np.random.shuffle(train.data)
    np.random.set_state(rng_state)
    np.random.shuffle(train.target)
    # # 打乱test
    np.random.seed(97)
    rng_state = np.random.get_state()
    np.random.shuffle(test.data)
    np.random.set_state(rng_state)
    np.random.shuffle(test.target)
    # # 切分数据集为n份
    train.data = np.array_split(train.data, client_num, axis=0)
    test.data = np.array_split(test.data, client_num, axis=0)
    train.target = np.array_split(train.target, client_num, axis=0)
    test.target = np.array_split(test.target, client_num, axis=0)
    # # kddcup99是一个对象数组，每个数组元素中存的是一个client需要训练的数据
    return train,test


def single_train(dataset,idx,num_epoches):
    myModel = CNN(1, 24)
    acc_temp=0
    param={}
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(myModel.parameters(), lr=learning_rate)
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

        # if epoch==num_epoches-1:
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


        if (eval_acc / (len(dataset.test_dataset)) > acc_temp and eval_acc / (len(dataset.test_dataset))>0.5 ):
            acc_temp = eval_acc / (len(dataset.test_dataset))
            param = myModel.state_dict()
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
    if param=={}:
        return
    else:
        return param



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

def train(client_num,batch_size,epoch):
    # 返回测试集数据。测试集数据条数，模型参数集
    data_test=[]
    data_test_len=[]
    params=[]
    train_total,test_total=loadData(client_num)
    # data_total.data 是client_num大的数组
    for i in range(client_num):
        # dataset = KddData(batch_size, data_total.data[i],data_total.target[i])
        dataset = KddData(batch_size, train_total.data[i],train_total.target[i],test_total.data[i],test_total.target[i])
        param=single_train(dataset,i,epoch)
        if (param!={}):
            # param=single_train(dataset,i,epoch)
            params.append(param)
        data_test.append(dataset.test_dataloader)
        data_test_len.append(len(dataset.test_dataset))
    return data_test,data_test_len,params

def update_model(w_locals):
    model = CNN(1, 24)
    model1 = model
    params_up=_aggregate(w_locals)
    model1.load_state_dict(params_up)
    # param1 = model1.net.state_dict()
    return model1

def single_test(model,dataset,test_data_len,idx):
    model.eval()
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
    learning_rate = 1e-4
    num_epoches = 10

    client_num = 3
    batch_size = 64
    print(f"共有{client_num}个客户端,每个客户端的epoch都为{num_epoches}")

    test_data,test_data_len,params=train(client_num,batch_size,num_epoches)
    model_new=update_model(params)
    print("----------------模型聚合后测试结果----------------------")
    single_test(model_new,test_data[0],test_data_len[0],0)
    print('\n')
    single_test(model_new,test_data[1],test_data_len[1],1)
    print('\n')
    single_test(model_new,test_data[2],test_data_len[2],2)
# 数据集定制