import numpy as np
import torch
import csv
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
U2R = ['buffer_overflow.', 'loadmodule.', 'perl.', 'ps.', 'rootkit.', 'sqlattack.', 'xterm.']
def four_data(target):
    data_file = open(target, 'w')
    csv_writer = csv.writer(data_file)
    with open(source, 'r') as f:
        ff = f.read()
        item=ff.split(',')
        for i in range(len(item)):
            if item[i] in R2L:
                item[i] = 1
            elif item[i] in DOS:
                item[i] = 2
            elif item[i] in Probe:
                item[i] = 3
            elif item[i] in U2R:
                print(item[i])
                item[i] = 4
                print(i)
            elif item[i] == 'normal':
                item[i] = 0
        csv_writer.writerow(item)
            #输出每行数据中所修改后的状态
    data_file.close()
if __name__ == '__main__':
    kddcup99 = datasets.fetch_kddcup99()
    data_file = open('./data/kddcup991.txt', 'w')
    csv_writer = csv.writer(data_file)
    for i in range(len(kddcup99.data)):
        item=list(kddcup99.data[i])
        item[1]=item[1].decode("utf-8")
        item[2]=item[2].decode("utf-8")
        item[3]=item[3].decode("utf-8")
        target =kddcup99.target[i].decode("utf-8")
        if target in U2R:
            continue
        item.append(target)
        csv_writer.writerow(item)
    data_file.close()

