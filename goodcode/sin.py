#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np 
import tensorflow as tf 
import matplotlib as mpl 
from matplotlib import pyplot as plt 
from tensorflow.contrib.learn.python.learn.estimators.estimator import SKCompat 
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

# TensorFlow的高层封装TFLearn 
learn = tf.contrib.learn 
seed = 7
batch_size = 30
epochs = 20
filename = '/home/kayky/project/POI motion data tracked with  TPS model/Phantom-II.csv'
footer = 3
look_back=10
# 神经网络参数 
HIDDEN_SIZE = 30     # LSTM隐藏节点个数 
NUM_LAYERS = 2       # LSTM层数 
TIMESTEPS = 10       # 循环神经网络截断长度 
BATCH_SIZE = 32 
# batch大小 
# 数据参数 
TRAINING_STEPS = 3000 
# 训练轮数 
TRAINING_EXAMPLES = 10000
 # 训练数据个数 
TESTING_EXAMPLES = 1000 
# 测试数据个数 
SAMPLE_GAP = 0.01 # 采样间隔 
def generate_data(seq): # 序列的第i项和后面的TIMESTEPS-1项合在一起作为输入，第i+TIMESTEPS项作为输出 
     X = []    
     y = [] 
     for i in range(len(seq) - TIMESTEPS - 1): 
         X.append([seq[i:i + TIMESTEPS]]) 
         y.append([seq[i + TIMESTEPS]]) 
     return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32) 
   
# 用sin生成训练和测试数据集 
test_start = TRAINING_EXAMPLES * SAMPLE_GAP 
test_end = (TRAINING_EXAMPLES + TESTING_EXAMPLES) * SAMPLE_GAP 
train_X, train_y = generate_data( 
     np.sin(np.linspace(0, test_start, TRAINING_EXAMPLES, dtype=np.float32))) 
print(train_X)
print(train_y)
test_X, test_y = generate_data( 
     np.sin( np.linspace(test_start, test_end, TESTING_EXAMPLES, dtype=np.float32))) 
print(test_X)
print(test_y)
def build_model():
    model = Sequential()
    model.add(LSTM(units=10, input_shape=(1, look_back)))
    model.add(Dense(units=1,activation='sigmoid'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


if __name__ == '__main__':

    # 设置随机种子
    np.random.seed(seed)

    # 导入数据
   

    



    # 训练模型
    model = build_model()
    model.fit(train_X, train_y, epochs=epochs, batch_size=batch_size, verbose=2)
    print(model.summary())
    # 模型预测数据
    predict_train = model.predict(train_X)
    predict_validation = model.predict(test_X)
    print(predict_train)
    print(predict_validation)
    # 评估模型
    train_score = model.evaluate(train_X, train_y, verbose=0)
    print('Train Score: %.2f MSE (%.2f RMSE)' % (train_score, math.sqrt(train_score)))
    validation_score = model.evaluate(test_X, test_y, verbose=0)
    print('Validatin Score: %.2f MSE (%.2f RMSE)' % (validation_score, math.sqrt(validation_score)))


   
