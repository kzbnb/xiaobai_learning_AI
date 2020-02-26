from __future__ import division
from pandas import read_csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout
import sklearn.model_selection
from sklearn.preprocessing import MinMaxScaler
from collections import Counter
import matplotlib.pyplot as plt
import pandas
import numpy
from keras.layers.core import Dense,Activation,Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import matplotlib.pyplot as plt
from itertools import chain
import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

from keras import backend as K
K.set_image_dim_ordering('tf')

import matplotlib.pyplot as plt


#  数据预处理
f = open('1234.csv', encoding='gbk')
names = ['证券代码', '证券名称', '开盘价', '转股价格', '上市日期', '网上中签率', '第一持有人持有比例', '上市转股溢价率','转股价值溢价','转股价值折价']
data = pd.read_csv(f, names=names)
plt.rcParams['font.sans-serif']=['SimHei']


over_price_precentage=[]
for n in data['上市转股溢价率']:
  over_price_precentage.append(float(n))

start_price=[]
for n in  data['开盘价']:
  start_price.append(float(n))

hit_rate=[]
for n in data['网上中签率']:
  hit_rate.append(float(n))

owner_rate=[]
for n in  data['第一持有人持有比例']:
  owner_rate.append(float(n))
'''
plt.scatter(over_price_precentage, start_price)
plt.show()
'''

dataframe = pd.DataFrame(data)



#  线性相关性度量
print(dataframe.corr())
#dataframe.corr().to_csv('corr1.csv', index=False, sep=',')

#  读取新的数据集文件，进行后续处理
#file = open('result1.csv', encoding='UTF-8')



print(data.shape)


#  获取属性
x = dataframe.loc[:, [ '证券代码', '转股价格','上市日期', '网上中签率', '第一持有人持有比例','上市转股溢价率']]
print(x)
print(x.shape)  # (256, 5)

#  以下标形式获取数据（此处为开盘价）
y = dataframe.loc[:, ['转股价值溢价','转股价值折价']]
print(y.shape)  # (256, 1)
print(y)

#  切分数据集（90%训练集、10%测试集) 不过数据量有点小
x_train, x_valid, y_train, y_valid = sklearn.model_selection.train_test_split(x, y, test_size=0.1)


print(x_train.shape)
print(y_train.shape)
print(x_valid.shape)
print(y_valid.shape)

# 转成DataFrame格式方便数据处理
x_train_pd = pd.DataFrame(x_train)
y_train_pd = pd.DataFrame(y_train)
x_valid_pd = pd.DataFrame(x_valid)
y_valid_pd = pd.DataFrame(y_valid)

# 训练集归一化
min_max_scaler = MinMaxScaler()
min_max_scaler.fit(x_train_pd)
x_train = min_max_scaler.transform(x_train_pd)


# 验证集归一化
min_max_scaler.fit(x_valid_pd)
x_valid = min_max_scaler.transform(x_valid_pd)




x_test=x_valid
y_test=y_valid


x_train = x_train.reshape(x_train.shape[0], 3, 2, 1)
x_test = x_test.reshape(x_test.shape[0], 3, 2, 1)



model = Sequential()
model.add(Conv2D(64,(2,2),strides=(1,1),input_shape=x_train.shape[1:],padding='same',activation='relu',kernel_initializer='uniform'))

model.add(Conv2D(64,(2,2),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(256,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])
model.summary()

#230/230 [==============================] - 1s 3ms/step - loss: 0.5004 - acc: 0.8000 - val_loss: 0.4369 - val_acc: 0.8462
#230/230 [==============================] - 1s 3ms/step - loss: 0.6237 - acc: 0.8130 - val_loss: 0.6413 - val_acc: 0.7308
# train the model 训练模型
history=model.fit(x_train, y_train,
          batch_size=32,
          epochs=200,
          verbose=1,
          validation_data=(x_test, y_test))
# verbose = 0 为不在标准输出流输出日志信息，verbose = 1 为输出进度条记录，verbose = 2 为每个epoch输出一行记录

# 创建一个绘图窗口
epochs = len(history.history['loss'])
plt.plot(range(epochs), history.history['loss'], label='loss')
plt.plot(range(epochs), history.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

epochs = len(history.history['loss'])
plt.plot(range(epochs), history.history['acc'], label='acc')
plt.plot(range(epochs), history.history['val_acc'], label='val_acc')
plt.legend()
plt.show()




