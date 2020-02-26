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
from keras.layers.recurrent import LSTM,GRU
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
f = open('12345.csv', encoding='gbk')
names = ['证券代码', '证券名称', '开盘价', '转股价格', '上市日期', '网上中签率', '第一持有人持有比例', '上市转股溢价率']
data = pd.read_csv(f, names=names)
plt.rcParams['font.sans-serif']=['SimHei']
'''
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

plt.scatter(over_price_precentage, start_price)
plt.show()


dataframe = pd.DataFrame(data)



#  线性相关性度量
print(dataframe.corr())
#dataframe.corr().to_csv('corr1.csv', index=False, sep=',')

#  读取新的数据集文件，进行后续处理
#file = open('result1.csv', encoding='UTF-8')



print(data.shape)
'''
dataframe = pd.DataFrame(data)
#  获取属性
x = dataframe.loc[:, [ '上市日期', '网上中签率', '第一持有人持有比例', '上市转股溢价率','转股价格']]
print(x)
print(x.shape)  # (256, 5)

#  以下标形式获取数据（此处为开盘价）
y = dataframe.loc[:, ['开盘价']]
print(y.shape)  # (256, 1)
print(y)
cut=30
x_train=x[cut:]
x_valid=x[:cut]
y_train=y[cut:]
y_valid=y[:cut]
#  切分数据集（90%训练集、10%测试集) 不过数据量有点小 BP用
#x_train, x_valid, y_train, y_valid = sklearn.model_selection.train_test_split(x, y, test_size=0.1)


print(x_train.shape)
print(y_train.shape)
print(x_valid.shape)
print(y_valid.shape)
'''
(230, 5)
(230, 1)
(26, 5)
(26, 1)
'''

# 转成DataFrame格式方便数据处理
x_train_pd = pd.DataFrame(x_train)
y_train_pd = pd.DataFrame(y_train)
x_valid_pd = pd.DataFrame(x_valid)
y_valid_pd = pd.DataFrame(y_valid)

# 训练集归一化
min_max_scaler = MinMaxScaler()
min_max_scaler.fit(x_train_pd)
x_train = min_max_scaler.transform(x_train_pd)
print(x_train)

min_max_scaler.fit(y_train_pd)
y_train = min_max_scaler.transform(y_train_pd)

# 验证集归一化
min_max_scaler.fit(x_valid_pd)
x_valid = min_max_scaler.transform(x_valid_pd)

min_max_scaler.fit(y_valid_pd)
y_valid = min_max_scaler.transform(y_valid_pd)
print(y_valid)




'''
#普通BP

model = Sequential()  # 初始化
model.add(Dense(units = 4,   # 输入大小
                activation='relu',  # 激励函数
                input_shape=(x_train_pd.shape[1],)  # 输入大小, 也就是列的大小
            )
        )

model.add(Dropout(0.2))

model.add(Dense(units=8,
                activation='relu'  # 激励函数
            )
        )

model.add(Dense(units=1,
                activation='linear'  # 线性激励函数 回归一般在输出层用这个激励函数
            )
        )

print(model.summary())  # 打印网络层次结构

model.compile(loss='mse',  # 损失均方误差
            optimizer='adam',  # 优化器
            )

history=model.fit(x_train, y_train,
        epochs=1000,  # 迭代次数
        batch_size=128,  # 每次用来梯度下降的批处理数据大小
        shuffle=True,
        verbose=2,
        validation_data = (x_valid, y_valid)  # 验证集
    )

'''
#LSTM 效果差

x_train = np.array(x_train)
x_valid = np.array(x_valid)
timesteps=5
x_train = np.reshape(x_train,(x_train.shape[0],timesteps,1))
x_valid = np.reshape(x_valid,(x_valid.shape[0],timesteps,1))

model = Sequential()


model.add(LSTM(128,input_shape=(timesteps,1),return_sequences= True))

# 防止过拟合 丢弃
model.add(
  Dropout(0.2)
)

model.add(
  LSTM(
    200,
    return_sequences=False
  )
)

model.add(
  Dropout(0.2)
)

model.add(Dense(output_dim=1))

model.add(Activation('linear'))

model.compile(loss='mse', optimizer='rmsprop')

# 3训练模型

history=model.fit(x_train, y_train,
          batch_size=120,
          epochs=300,

            validation_data = (x_valid, y_valid)  # 验证集
          )




# 预测
y_guess = model.predict(x_valid)

# 反归一化
min_max_scaler.fit(y_valid_pd)
y_guess = min_max_scaler.inverse_transform(y_guess)
y_valid = min_max_scaler.inverse_transform(y_valid)

plt.plot(y_guess)
plt.plot(y_valid)


plt.show()

epochs = len(history.history['loss'])
plt.plot(range(epochs), history.history['loss'], label='loss')
plt.plot(range(epochs), history.history['val_loss'], label='val_loss')
plt.legend()
plt.show()
