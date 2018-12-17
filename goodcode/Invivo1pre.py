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
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy 
from scipy import io 
import numpy as np

seed = 7
batch_size = 30
epochs = 20
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

    features = np.reshape(features_struct['Poi'], (750,3), order='F')
    data = pd.DataFrame(features) 
   
    datas = data.values.astype('float32')
    
    ss = StandardScaler()
    cps = ss.fit_transform(datas)
    datase = pd.DataFrame(cps.reshape(1, 2250, order='F'))
    datases = datase.values.astype('float32')
    dataset = datases.T
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
    predict_train = model.predict(X_train)
    predict_validation = model.predict(X_validation)
    print(predict_train)
    print(predict_validation)
    # 评估模型
    train_score = model.evaluate(X_train, y_train, verbose=0)
    print('Train Score: %.2f MSE (%.2f RMSE)' % (train_score, math.sqrt(train_score)))
    validation_score = model.evaluate(X_validation, y_validation, verbose=0)
    print('Validatin Score: %.2f MSE (%.2f RMSE)' % (validation_score, math.sqrt(validation_score)))
# 对预测曲线绘图，并存储到sin.jpg 
fig = plt.figure() 
plot_predict_validation, = plt.plot(predict_validation[:200],color='red' ,label='predict_validation') 
plot_y_validation, = plt.plot(y_validation[:200], label='y_validation') 
plt.legend([plot_predict_validation, plot_y_validation], ['predict_validation', 'y_validation']) 
plt.show()
fig = plt.figure() 
plot_predict_train, = plt.plot(predict_train[:200],color='red' ,label='predict_train') 
plot_y_train, = plt.plot(y_train[:200], label='y_train') 
plt.legend([plot_predict_train, plot_y_train], ['predict_train', 'y_train']) 
plt.show()
