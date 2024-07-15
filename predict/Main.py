from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import metrics
from utils import *
from model import *
import tensorflow as tf

data1 = pd.read_csv("predict/601988.SH.csv")
data1.index = pd.to_datetime(data1['trade_date'], format='%Y%m%d')
#data1 = data1.drop(['ts_code', 'trade_date', 'turnover_rate', 'volume_ratio', 'pb', 'total_share', 'float_share', 'free_share'], axis=1)
data1 = data1.loc[:, ['open', 'high', 'low', 'close', 'vol', 'amount']]
data_yuan = data1
residuals = pd.read_csv('predict/ARIMA_residuals1.csv')
residuals.index = pd.to_datetime(residuals['trade_date'])
residuals.pop('trade_date')
data1 = pd.merge(data1, residuals, on='trade_date')
#数据切割 训练+测试
data = data1.iloc[1:3500, :] #data 训练  data1全部 data2测试
data2 = data1.iloc[3500:, :] 

TIME_STEPS = 20

data, normalize = NormalizeMult(data)
print('#', normalize) 

pollution_data = data[:, 3].reshape(len(data), 1)  #get the close price
print(pollution_data)

train_X, _ = create_dataset(data, TIME_STEPS)
_, train_Y = create_dataset(pollution_data, TIME_STEPS)

print(train_X.shape, train_Y.shape)

m = attention_model(INPUT_DIMS=7)
m.summary()  #输出模型信息
adam = tf.keras.optimizers.Adam(learning_rate=0.01)
m.compile(optimizer=adam, loss='mse') 
history = m.fit([train_X], train_Y, epochs=5, batch_size=64, validation_split=0.1)
m.save_weights("path_to_save_weights")
#tf.keras.models.save_model(m, "predict/stock_model")
np.save("predict/stock_normalize.npy", normalize)

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show(block=True)

# normalize = np.load("normalize.npy")
# loadmodelname = "model.h5"

class Config:
    def __init__(self):
        self.dimname = 'close'

config = Config()
name = config.dimname
# normalize = np.load("normalize.npy")
y_hat, y_test = PredictWithData(data2, data_yuan, name, "path_to_save_weights",7)
y_hat = np.array(y_hat, dtype='float64')
y_test = np.array(y_test, dtype='float64')
evaluation_metric(y_test,y_hat)
time = pd.Series(data1.index[3500:])
plt.plot(time, y_test, label='True')
plt.plot(time, y_hat, label='Prediction')
plt.title('Hybrid model prediction')
plt.xlabel('Time', fontsize=12, verticalalignment='top')
plt.ylabel('Price', fontsize=14, horizontalalignment='center')
plt.legend()
plt.show(block=True)