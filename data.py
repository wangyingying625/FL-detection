import numpy as np
import ssl
import tensorflow as tf
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
ssl._create_default_https_context = ssl._create_unverified_context
def getBinaryTensor(imgTensor):
    # one = tf.ones_like(imgTensor)
    one=np.ones(imgTensor.shape)
    return one

def convert(label):
    list1=label.tolist()
    list_new=list1
    for (i) in (range(len(list1))):
        if list1[i]==0:
            list_new[i]=0
        elif list1[i]==1:
            list_new[i]=0
        elif list1[i] == 2:
            list_new[i] = 0
        elif list1[i] == 3:
            list_new[i] = 3
    return  np.array(list_new)
# 数据集分成三份（每一份中有test+train），装进tensor的数组里返回
class KddData(object):

    def __init__(self, batch_size,data_x,data_y,convert_client,constant_data):
    # def __init__(self, batch_size, train_data,train_target, test_data,test_target):
        self._encoder = {
            'protocal': LabelEncoder(),
            'service':  LabelEncoder(),
            'flag':     LabelEncoder(),
            'label':    LabelEncoder()
        }
        self.batch_size = batch_size
        self.convert_client=convert_client
        self.constant_data=constant_data
        data_X, data_y = self.__encode_data(data_x, data_y)
        self.train_dataset, self.test_dataset = self.__split_data_to_tensor(data_X, data_y)
        self.train_dataloader = DataLoader(self.train_dataset, self.batch_size, shuffle=False)
        self.test_dataloader = DataLoader(self.test_dataset, self.batch_size, shuffle=False)


    """将数据中字符串部分转换为数字，并将输入的41维特征转换为8*8的矩阵"""
    def __encode_data(self, data_X, data_y):
        self._encoder['protocal'].fit(list(set(data_X[:, 1])))
        self._encoder['service'].fit(list(set(data_X[:, 2])))
        self._encoder['flag'].fit((list(set(data_X[:, 3]))))
        self._encoder['label'].fit(list(set(data_y)))
        data_X[:, 1] = self._encoder['protocal'].transform(data_X[:, 1])
        data_X[:, 2] = self._encoder['service'].transform(data_X[:, 2])
        data_X[:, 3] = self._encoder['flag'].transform(data_X[:, 3])
        data_X = np.pad(data_X, ((0, 0), (0, 64 - len(data_X[0]))), 'constant').reshape(-1, 1, 8, 8)
        data_y = self._encoder['label'].transform(data_y)
        # print(min(data_y))
        # print(max(data_y))
        return data_X, data_y


    """将数据拆分为训练集和测试集，并转换为TensorDataset对象"""
    def __split_data_to_tensor(self, data_X, data_y):
        X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.3,random_state=100)
        if(self.convert_client!=0):
            y_train=convert(y_train)
        if (self.constant_data != 0):
            X_train = getBinaryTensor(X_train.astype(np.float32))
        train_dataset = TensorDataset(
            torch.from_numpy(X_train.astype(np.float32)),
            torch.from_numpy(y_train.astype(np.int64))
        )
        test_dataset = TensorDataset(
            torch.from_numpy(X_test.astype(np.float32)),
            torch.from_numpy(y_test.astype(np.int64))
        )
        return train_dataset, test_dataset

    def batch_data(data, batch_size):
        '''
        data is a dict := {'x': [numpy array], 'y': [numpy array]} (on one client)
        returns x, y, which are both numpy array of length: batch_size
        切分数据集
        '''
        # 一个data是一个用户的数据，x是特征，y是标签
        data_x = data['x']
        data_y = data['y']

        # randomly shuffle data 打乱数据集
        np.random.seed(90)
        rng_state = np.random.get_state()
        np.random.shuffle(data_x)
        np.random.set_state(rng_state)
        np.random.shuffle(data_y)

        # loop through mini-batches
        # 把data分成batch大小的小数组
        batch_data = list()
        for i in range(0, len(data_x), batch_size):
            batched_x = data_x[i:i + batch_size]
            batched_y = data_y[i:i + batch_size]
            batched_x = torch.from_numpy(np.asarray(batched_x)).float()
            batched_y = torch.from_numpy(np.asarray(batched_y)).long()
            batch_data.append((batched_x, batched_y))
        return batch_data
    """接受一个数组进行解码"""
    def decode(self, data, label=False):
        if not label:
            _data = list(data)
            _data[1] = self._encoder['protocal'].inverse_transform(_data[1])
            _data[2] = self._encoder['service'].inverse_transform(_data[2])
            _data[2] = self._encoder['flag'].inverse_transform(_data[3])
            return _data
        return self._encoder['label'].inverse_transform(data)


    def encode(self, data, label=False):
        if not label:
            _data = list(data)
            _data[1] = self._encoder['protocal'].transform([_data[1]])[0]
            _data[2] = self._encoder['service'].transform([_data[2]])[0]
            _data[3] = self._encoder['flag'].transform([_data[3]])[0]
            return _data
        return self._encoder['label'].transform([data])[0]
# 数据集调整
