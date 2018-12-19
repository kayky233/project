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
look_back=3
 



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
    plt.subplot(431) 
    plot_Poix, = plt.plot(Poi[0,:],color='blue' ,label='datax') 
    plt.subplot(432) 
    plot_Poiy, = plt.plot(Poi[1,:],color='red' ,label='datay')
    plt.subplot(433) 
    plot_Poiz, = plt.plot(Poi[2,:],color='green' ,label='dataz')
    plt.legend([plot_Poix, plot_Poiy,plot_Poiz], ['datax', 'datay','dataz']) 
  

    #预测数据预处理
    features = np.reshape(features_struct['Poi'], (3,750), order='c')
    data = pd.DataFrame(features) 
    datas = data.values.astype('float32')
    #标准化预测数据
    ss = StandardScaler()
    cps = ss.fit_transform(datas)
    fig = plt.figure(1)
    plt.subplot(434) 
    plot_cpsx, = plt.plot(cps[0,:],color='blue' ,label='stdx')
    plt.subplot(435) 
    plot_cpsy, = plt.plot(cps[1,:],color='red' ,label='stdy')
    plt.subplot(436) 
    plot_cpsz, = plt.plot(cps[2,:],color='green' ,label='stdz')
    plt.legend([plot_cpsx, plot_cpsy,plot_cpsz], ['stdx', 'stdy','stdz']) 
    
    datase = pd.DataFrame(cps.reshape(1, 2250, order='F'))
    dataset = datase.values.astype('float32')
    dataset = dataset.T
    
    #划分训练数据集和预测数据集尺寸
    train_size = int(len(dataset) * 0.67)
    validation_size = len(dataset) - train_size
    train, validation = dataset[0: train_size], dataset[train_size: len(dataset)]

    

    # 创建dataset，让数据产生相关性
    X_train, y_train = create_dataset(train)
    print(y_train.shape)
 

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
    
    trainpre = trainshow.reshape(3, 497, order='F')
    
    #展示train预测曲线    
    fig = plt.figure(1)
    plt.subplot(437)
    plot_trainprex, = plt.plot(trainpre[0,:],color='blue' ,label='trainprex')
    plt.subplot(438) 
    plot_trainprey, = plt.plot(trainpre[1,:],color='red' ,label='trainprey')
    plt.subplot(439)
    plot_trainprez, = plt.plot(trainpre[2,:],color='green' ,label='trainprez')
    plt.legend([plot_trainprex, plot_trainprey,plot_trainprez], ['trainprex', 'trainprey','trainprez'])
    
    #validation 预测数据处理
    predict_validation = model.predict(X_validation)
    validationshow = predict_validation[1:730]
    validationpre = validationshow.reshape(3, 243, order='F')
    
    #展示validation预测曲线
    fig = plt.figure(1)
    plt.subplot(4,3,10)
    plot_validationprex, = plt.plot(validationpre[0,:],color='blue' ,label='validationprex')
    plt.subplot(4,3,11) 
    plot_validationprey, = plt.plot(validationpre[1,:],color='red' ,label='validationprey')
    plt.subplot(4,3,12)
    plot_validationprez, = plt.plot(validationpre[2,:],color='green' ,label='validationprez')
    plt.legend([plot_validationprex, plot_validationprey,plot_validationprez], ['validationprex', 'validationprey','validationprez'])
    plt.show()


    # 评估模型
    train_score = model.evaluate(X_train, y_train, verbose=0)
    print('Train Score: %.2f MSE (%.2f RMSE)' % (train_score, math.sqrt(train_score)))
    validation_score = model.evaluate(X_validation, y_validation, verbose=0)
    print('Validatin Score: %.2f MSE (%.2f RMSE)' % (validation_score, math.sqrt(validation_score)))
   


    #展示train和trainpre（standard）
    fig = plt.figure(2)
    #展示train
    data = pd.DataFrame(y_train) 
    datas = data.values.astype('float32')
    ytrainshow = datas[2:1493] 
    
    ytrainpre = ytrainshow.reshape(3, 497, order='F')
    print('ytrain',ytrainpre)
    print('validationpre',trainpre)
    plt.subplot(231)
    plot_y_trainx, = plt.plot(ytrainpre[1,:],color='blue' ,label='y_trainx')
    plt.subplot(232) 
    plot_y_trainy, = plt.plot(ytrainpre[2,:],color='red' ,label='y_trainy')
    plt.subplot(233)
    plot_y_trainz, = plt.plot(ytrainpre[0,:],color='green' ,label='y_trainz')
    plt.legend([plot_y_trainx, plot_y_trainy,plot_y_trainz], ['y_trainx', 'y_trainy','y_trainz'])
    #展示trainpre（standard）

    plt.subplot(234)
    plot_trainprex, = plt.plot(trainpre[1,:],color='blue' ,label='trainprex')
    plt.subplot(235) 
    plot_trainprey, = plt.plot(trainpre[2,:],color='red' ,label='trainprey')
    plt.subplot(236)
    plot_trainprez, = plt.plot(trainpre[0,:],color='green' ,label='trainprez')
    plt.legend([plot_trainprex, plot_trainprey,plot_trainprez], ['trainprex', 'trainprey','trainprez'])
    plt.show()
    
    #展示validation及其预测曲线
   
    fig = plt.figure(3)
    #展示y_validation
    data = pd.DataFrame(y_validation) 
    datas = data.values.astype('float32')
    y_validationshow = datas[1:730] 
    
    y_validationpre = y_validationshow.reshape(3, 243, order='F')
    print('y_validation',y_validationpre)
    print('validationpre',validationpre)
    plt.subplot(231)
    plot_y_validationx, = plt.plot(y_validationpre[1,:],color='blue' ,label='y_validationx')
    plt.subplot(232) 
    plot_y_validationy, = plt.plot(y_validationpre[2,:],color='red' ,label='y_validationy')
    plt.subplot(233)
    plot_y_validationz, = plt.plot(y_validationpre[0,:],color='green' ,label='y_validationz')
    plt.legend([plot_y_validationx, plot_y_validationy,plot_y_validationz], ['y_validationx', 'y_validationy','y_validationz'])
    #展示y_validationpre（standard）

    fig = plt.figure(3)
    plt.subplot(234)
    plot_validationprex, = plt.plot(validationpre[1,:],color='blue' ,label='validationprex')
    plt.subplot(235) 
    plot_validationprey, = plt.plot(validationpre[2,:],color='red' ,label='validationprey')
    plt.subplot(236)
    plot_validationprez, = plt.plot(validationpre[0,:],color='green' ,label='validationprez')
    plt.legend([plot_validationprex, plot_validationprey,plot_validationprez], ['validationprex', 'validationprey','validationprez'])
    plt.show()

     
    
    
