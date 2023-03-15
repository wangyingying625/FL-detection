import os
import csv
import numpy as np

R2L = ['ftp_write', 'guess_passwd', 'imap', 'multihop', 'named', 'phf', 'sendmail', 'snmpgetattack', 'snmpguess', 'spy',
       'warezclient', 'warezmaster', 'xlock', 'xsnoop', 'httptunnel']
DOS = ['back', 'land', 'mailbomb', 'neptune', 'pod', 'processtable', 'smurf', 'teardrop', 'udpstorm', 'apache2']
Probe = ['ipsweep', 'mscan', 'nmap', 'portsweep', 'saint', 'satan']
U2R = ['buffer_overflow', 'loadmodule', 'perl', 'ps', 'rootkit', 'sqlattack', 'xterm']
def get_data(source,target):
    data_file = open(target, 'w', newline='')
    with open(source,'r') as data_source:
        csv_reader=csv.reader(data_source)
        csv_writer=csv.writer(data_file)
        for row in csv_reader:
            temp_line=row  #将每行数据存入temp_line数组里
            temp_line[41]=temp_line[41].strip('.')
            if temp_line[41] in R2L:
                temp_line[41]=1
            elif temp_line[41]  in DOS:
                temp_line[41]=2
            elif temp_line[41] in Probe:
                temp_line[41]=3
            elif temp_line[41] in U2R:
                temp_line[41]=4
            elif temp_line[41] =='normal':
                temp_line[41] =0
            csv_writer.writerow(temp_line)
            #输出每行数据中所修改后的状态
        data_file.close()
def readData(path):
    data=[]
    with open(path,'r') as data_source:
        csv_reader=csv.reader(data_source)
        for row in csv_reader:
            temp_line=row
            data.append(temp_line)
    return data
def four_data(source,target):
    data_file = open(target, 'w')
    csv_writer = csv.writer(data_file)
    # data_file = open(target, 'w')
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
    # path='./data/KDDTrain.txt'
    # path1='./data/train1.txt'
    # get_data(path,path1)
    get_data('./data/kddcup991.txt','./data/kddcup.txt')