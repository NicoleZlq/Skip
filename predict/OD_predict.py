from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import GRU, LSTM, Bidirectional, Dense, Flatten, Conv1D, BatchNormalization, LeakyReLU, Dropout
from tensorflow.keras import Sequential
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import metrics
from utils import *
from model import *
import tensorflow as tf
import os
import time



cut = 500

instance = 0
direction = 'start'
df = pd.read_csv("data_instance5.csv")

# 步骤2: 筛选出start为0的数据
df_start_zero = df[df[direction] == instance]

# 步骤3: 按time分组并计数
time_grouped = df_start_zero.groupby('time').size()

# 步骤4: 创建新DataFrame
new_df = pd.DataFrame({
    'num': time_grouped.values,
    'time': time_grouped.index
})

# 确保列的顺序
data1 = new_df[['time', 'num']]
data1.to_csv("predict/{}_station_{}.csv".format(direction,instance),index=False,sep=',')


#data1 = data1.loc[:, ['num']]
data_yuan = data1
data = data1.iloc[1:cut, :] #data 训练  data1全部 data2测试
data2 = data1.iloc[cut:, :] 

print(len(data2))

step = 10

data = np.array(data)


pollution_data = data[:, 1].reshape(len(data), 1)  #get the close price


train_X, _ = create_dataset(data, step)
_, train_Y = create_dataset(pollution_data, step)


print(train_X.shape, train_Y.shape)

dim = train_X.shape[-1]


m = attention_model(dim, step,5)


adam = tf.keras.optimizers.Adam(learning_rate=0.001)
m.compile(optimizer=adam, loss='mse') 

history = m.fit([train_X], train_Y, epochs=200, batch_size=8, validation_split=0.1)
# 保存模型为 SavedModel 格式

save_path = "predict/OD_save_weights.h5"


m.summary()  #输出模型信息
m.save_weights(save_path)
#tf.keras.models.save_model(m, "predict/stock_model")
#np.save("predict/OD_normalize.npy", normalize)





plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show(block=True)
#########################################################



# normalize = np.load("normalize.npy")
# loadmodelname = "model.h5"

class Config:
    def __init__(self):
        self.dimname = 'num'

config = Config()
name = config.dimname
# normalize = np.load("normalize.npy")
y_hat, y_test = PredictWithData(data2, data_yuan, name, save_path, dim, step,cut)
y_hat = np.array(y_hat, dtype='float64')
y_test = np.array(y_test, dtype='float64')
evaluation_metric(y_test,y_hat)
time = data1['time'][cut:]
plt.plot(time, y_test, label='True')
plt.plot(time, y_hat, label='Prediction')
plt.title('Hybrid model prediction')
plt.xlabel('Time', fontsize=12, verticalalignment='top')
plt.ylabel('Price', fontsize=14, horizontalalignment='center')
plt.legend()
plt.show(block=True)