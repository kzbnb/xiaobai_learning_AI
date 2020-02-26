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




#  数据预处理
f = open('南北资金流向.csv', encoding='gbk')
names = ['日期', '日成交净买额', '上证指数', '上日涨跌幅','昨日收盘价',"涨","跌",'开盘价','成交金额']
data = pd.read_csv(f, names=names)
plt.rcParams['font.sans-serif']=['SimHei']

dataframe = pd.DataFrame(data)



print(data.shape)


#  获取属性
x = dataframe.loc[:, ['开盘价','成交金额','日期','上日涨跌幅','日成交净买额','昨日收盘价']]
print(x)
print(x.shape)  # (256, 5)

#  以下标形式获取数据（此处为开盘价）
y = dataframe.loc[:, ['涨','跌']]
print(y.shape)  # (256, 1)
print(y)

#  切分数据集（90%训练集、10%测试集) 不过数据量有点小
#x_train, x_valid, y_train, y_valid = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
#(1223, 3)
cut=300
x_train=x[cut:]
x_valid=x[:cut]
y_train=y[cut:]
y_valid=y[:cut]

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
print(x_train)

min_max_scaler.fit(y_train_pd)
y_train = min_max_scaler.transform(y_train_pd)

# 验证集归一化
min_max_scaler.fit(x_valid_pd)
x_valid = min_max_scaler.transform(x_valid_pd)

min_max_scaler.fit(y_valid_pd)
y_valid = min_max_scaler.transform(y_valid_pd)
print(y_valid)



x_train = np.array(x_train)
x_valid = np.array(x_valid)
timesteps=6
x_train = np.reshape(x_train,(x_train.shape[0],timesteps,1))
x_valid = np.reshape(x_valid,(x_valid.shape[0],timesteps,1))

model = Sequential()

model.add(GRU(128,input_shape=(timesteps,1),return_sequences= True))

# 防止过拟合 丢弃
model.add(
  Dropout(0.2)
)

model.add(
  GRU(
    200,
    return_sequences=False
  )
)

model.add(
  Dropout(0.2)
)

model.add(Dense(2,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])
# 3训练模型

history=model.fit(x_train, y_train,
          batch_size=256,
          epochs=1500,
          validation_data = (x_valid, y_valid)  # 验证集

          )




# 预测
y_guess = model.predict(x_valid)

# 反归一化
min_max_scaler.fit(y_valid_pd)
y_guess = min_max_scaler.inverse_transform(y_guess)
y_valid = min_max_scaler.inverse_transform(y_valid)



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
