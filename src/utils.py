import numpy as np
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 读取文件夹下所有后缀为csv的文件
def read_csv_files(path):
    files = os.listdir(path)
    files.sort()
    csv_files = []
    for file in files:
        if file.endswith('.csv'):
            csv_files.append(path + '/' + file)
    return csv_files


# 判断是否存在文件夹，不存在则创建
def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)


def sliding_windows(args, data):
    '''
    对单变量时间序列进行滑窗处理
    '''
    train = []
    for i in range(data.shape[0]-args.cond_size-args.horizon+1):
        train.append(data[i:i+args.cond_size])
    train = np.array(train)
    train.reshape(-1, args.cond_size)
    target = data[args.cond_size+args.horizon-1:]
    try:
        target = np.array(target).reshape(-1, 1, data.shape[1])
    except:
        target = np.array(target).reshape(-1, 1)
    return train, target

def save_result_pic(y_true,y_pred,result_pic_path):
    plt.plot(y_true,label="true")
    plt.plot(y_pred,label="pred")
    plt.legend()
    plt.savefig(result_pic_path)
    plt.close()


def normalize(data):
    '''
    归一化
    '''
    max = np.max(data, axis=0)
    min = np.min(data, axis=0)
    return (data - min) / (max - min), max, min

def denormalize(data, max, min):
    '''
    反归一化
    ''' 
    return data * (max - min) + min

def standardize(data):
    '''
    标准化
    '''
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return (data - mean) / std, mean, std

def destandardize(data, mean, std):
    '''
    反标准化
    '''
    return data * std + mean