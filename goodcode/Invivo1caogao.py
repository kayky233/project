#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from pandas import read_csv
from matplotlib import pyplot as plt
import math
from keras.models import Sequential
from keras.layers import Dense,LSTM
from keras.layers.core import Dropout
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from matplotlib import gridspec
import scipy 
from scipy import io 

seed = 7
batch_size = 30
epochs = 10
filename = '/home/kayky/project/POI motion data tracked with  TPS model/Invivo-I.mat'
footer = 3
look_back=10
 



def create_dataset(dataset):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 3):
        x = dataset[i: i + look_back, 0]
        dataX.append(x)
        y = dataset[i + look_back, 0]
        dataY.append(y)
        print('X: %s, Y: %s' % (x, y))
    return np.array(dataX), np.array(dataY)

def build_model():
    model = Sequential()
    model.add(LSTM(units=10, input_shape=(1, look_back)))
    
    model.add(Dense(units=1,activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


if __name__ == '__main__':

    # 设置随机种子
    np.random.seed(seed)

    # 导入数据
    features_struct = scipy.io.loadmat(filename)
    Poi = features_struct['Poi']
    #以曲线形式展示数据
    fig = plt.figure(1)
    plt.subplot(331) 
    plot_Poi, = plt.plot(Poi[0,:],color='blue' ,label='datax') 
    plt.subplot(332) 
    plot_Poi, = plt.plot(Poi[1,:],color='red' ,label='datay')
    plt.subplot(333) 
    plot_Poi, = plt.plot(Poi[2,:],color='green' ,label='dataz')
  

    #预测数据预处理
    features = np.reshape(features_struct['Poi'], (3,750), order='c')
    data = pd.DataFrame(features) 
    datas = data.values.astype('float32')
    #标准化预测数据
    ss = StandardScaler()
    cps = ss.fit_transform(datas)
    fig = plt.figure(1)
    plt.subplot(334) 
    plot_cps, = plt.plot(cps[0,:],color='blue' ,label='stdx')
    plt.subplot(335) 
    plot_cps, = plt.plot(cps[1,:],color='red' ,label='stdy')
    plt.subplot(336) 
    plot_cps, = plt.plot(cps[2,:],color='green' ,label='stdz')
    
    datase = pd.DataFrame(cps.reshape(1, 2250, order='F'))
    dataset = datase.values.astype('float32')
    dataset = dataset.T
    
    #划分训练数据集和预测数据集尺寸
    train_size = int(len(dataset) * 0.67)
    validation_size = len(dataset) - train_size
    train, validation = dataset[0: train_size], dataset[train_size: len(dataset)]

    

    # 创建dataset，让数据产生相关性
    X_train, y_train = create_dataset(train)
 

    X_validation, y_validation = create_dataset(validation)
    
    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    
    X_validation = np.reshape(X_validation, (X_validation.shape[0], 1, X_validation.shape[1]))
    
  


    # 训练模型
    model = build_model()
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2)
    print(model.summary())
    # 模型预测数据
    #train 预测数据处理
    predict_train = model.predict(X_train)
    trainshow = predict_train[2:1493] 
    print(trainshow)
    print(len(trainshow))
    print(trainshow.shape)
    
    trainpre = trainshow.reshape(3, 497, order='F')
    print(trainpre)
    #展示train预测曲线    
    fig = plt.figure(1)
    plt.subplot(337)
    plot_trainpre, = plt.plot(trainpre[0,:],color='blue' ,label='trainprex')
    plt.subplot(338) 
    plot_trainpre, = plt.plot(trainpre[1,:],color='red' ,label='trainprey')
    plt.subplot(339)
    plot_trainpre, = plt.plot(trainpre[2,:],color='green' ,label='trainprez')
    plt.show()
    
    #validation 预测数据处理
    predict_validation = model.predict(X_validation)
    validationshow = predict_validation[1:730]
    validationpre = validationshow.reshape(3, 243, order='F')
    print(validationpre)
    
    print(validationshow)
    print(len(validationshow))
    print(validationshow.shape)


    # 评估模型
    train_score = model.evaluate(X_train, y_train, verbose=0)
    print('Train Score: %.2f MSE (%.2f RMSE)' % (train_score, math.sqrt(train_score)))
    validation_score = model.evaluate(X_validation, y_validation, verbose=0)
    print('Validatin Score: %.2f MSE (%.2f RMSE)' % (validation_score, math.sqrt(validation_score)))
