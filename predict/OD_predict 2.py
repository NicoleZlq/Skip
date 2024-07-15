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

class GAN:
    def __init__(self, generator, discriminator, opt):
        self.opt = opt
        self.learning_rate = opt["learning_rate"]
        self.generator = generator
        self.discriminator = discriminator
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.batch_size = self.opt['bs']
        self.checkpoint_dir = 'predict/gan/training_checkpoints'
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                              discriminator_optimizer=self.discriminator_optimizer,
                                              generator=self.generator,
                                              discriminator=self.discriminator)

    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(self, fake_output):
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)

    @tf.function
    def train_step(self, real_x, real_y, yc):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_data = self.generator(real_x, training=True)
            generated_data_reshape = tf.reshape(generated_data, [generated_data.shape[0], generated_data.shape[1], 1])
            d_fake_input = tf.concat([tf.cast(generated_data_reshape, tf.int64), yc], axis=1)
            real_y_reshape = tf.reshape(real_y, [real_y.shape[0], real_y.shape[1], 1])
            d_real_input = tf.concat([real_y_reshape, yc], axis=1)

            # Reshape for MLP
            # d_fake_input = tf.reshape(d_fake_input, [d_fake_input.shape[0], d_fake_input.shape[1]])
            # d_real_input = tf.reshape(d_real_input, [d_real_input.shape[0], d_real_input.shape[1]])

            real_output = self.discriminator(d_real_input, training=True)
            fake_output = self.discriminator(d_fake_input, training=True)

            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        return real_y, generated_data, {'d_loss': disc_loss, 'g_loss': gen_loss}

    def train(self, real_x, real_y, yc, opt):
        train_hist = {}
        train_hist['D_losses'] = []
        train_hist['G_losses'] = []
        train_hist['per_epoch_times'] = []
        train_hist['total_ptime'] = []

        epochs = opt["epoch"]
        for epoch in range(epochs):
            start = time.time()

            real_price, fake_price, loss = self.train_step(real_x, real_y, yc)

            G_losses = []
            D_losses = []

            Real_price = []
            Predicted_price = []

            D_losses.append(loss['d_loss'].numpy())
            G_losses.append(loss['g_loss'].numpy())

            Predicted_price.append(fake_price.numpy())
            Real_price.append(real_price.numpy())

            # Save the model every 15 epochs
            if (epoch + 1) % 15 == 0:
                tf.keras.models.save_model(generator, 'gen_model_3_1_%d.h5' % epoch)
                self.checkpoint.save(file_prefix=self.checkpoint_prefix + f'-{epoch}')
                print('epoch', epoch + 1, 'd_loss', loss['d_loss'].numpy(), 'g_loss', loss['g_loss'].numpy())
            # print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))
            # For printing loss
            epoch_end_time = time.time()
            per_epoch_ptime = epoch_end_time - start
            train_hist['D_losses'].append(D_losses)
            train_hist['G_losses'].append(G_losses)
            train_hist['per_epoch_times'].append(per_epoch_ptime)

        # Reshape the predicted result & real
        Predicted_price = np.array(Predicted_price)
        Predicted_price = Predicted_price.reshape(Predicted_price.shape[1], Predicted_price.shape[2])
        Real_price = np.array(Real_price)
        Real_price = Real_price.reshape(Real_price.shape[1], Real_price.shape[2])

        plt.plot(train_hist['D_losses'], label='D_loss')
        plt.plot(train_hist['G_losses'], label='G_loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        return Predicted_price, Real_price, np.sqrt(mean_squared_error(Real_price, Predicted_price)) / np.mean(
            Real_price)




def make_generator_model(input_dim, output_dim, feature_size) -> tf.keras.models.Model:

    model = Sequential()
    model.add(GRU(units=1024, return_sequences = True, input_shape=(input_dim, feature_size),
                  recurrent_dropout=0.2))
    model.add(GRU(units=512, return_sequences = True, recurrent_dropout=0.2)) # 256, return_sequences = True
    model.add(GRU(units=256, recurrent_dropout=0.2)) #, recurrent_dropout=0.1
    # , recurrent_dropout = 0.2
    model.add(Dense(128))
    # model.add(Dense(128))
    model.add(Dense(64))
    #model.add(Dense(16))
    model.add(Dense(units=output_dim))
    return model

def make_discriminator_model():

    cnn_net = tf.keras.Sequential()
    cnn_net.add(Conv1D(32, input_shape=(4, 1), kernel_size=3, strides=2, padding='same', activation=LeakyReLU(alpha=0.01)))
    cnn_net.add(Conv1D(64, kernel_size=5, strides=2, padding='same', activation=LeakyReLU(alpha=0.01)))
    cnn_net.add(Conv1D(128, kernel_size=5, strides=2, padding='same', activation=LeakyReLU(alpha=0.01)))
    cnn_net.add(Flatten())
    cnn_net.add(Dense(220, use_bias=False))
    cnn_net.add(LeakyReLU())
    cnn_net.add(Dense(220, use_bias=False, activation='relu'))
    cnn_net.add(Dense(1, activation='sigmoid'))
    return cnn_net


def get_X_y(y_data,n_steps_in,n_steps_out):

    yc = list()

    length = len(y_data)
    for i in range(0, length, 1):
        
        yc_value = y_data[i: i + n_steps_in][:, :]
        if len(yc_value) == n_steps_in:

            yc.append(yc_value)

    return np.array(yc)


df = pd.read_csv("data_instance3.csv")

# 步骤2: 筛选出start为0的数据
df_start_zero = df[df['start'] == 0]

# 步骤3: 按time分组并计数
time_grouped = df_start_zero.groupby('time').size()

# 步骤4: 创建新DataFrame
new_df = pd.DataFrame({
    'num': time_grouped.values,
    'time': time_grouped.index
})

# 确保列的顺序
data1 = new_df[['time', 'num']]

# 步骤5: 保存到CSV
data1.to_csv('predict/gan/new_file.csv', index=False)

#data1 = data1.loc[:, ['num']]
data_yuan = data1
data = data1.iloc[1:680, :] #data 训练  data1全部 data2测试
data2 = data1.iloc[680:, :] 

step = 3

data = np.array(data)


# data, normalize = NormalizeMult(data)
# print('#', normalize) 


pollution_data = data[:, 1].reshape(len(data), 1)  #get the close price
print(pollution_data)

train_X, _ = create_dataset(data, step)
_, train_Y = create_dataset(pollution_data, step)

yc = get_X_y(train_Y, step,1)

train_X = train_X[0:len(yc)]
train_Y = train_Y[0:len(yc)]


print(train_X.shape, train_Y.shape)

dim = train_X.shape[-1]

#gan

model = make_discriminator_model()
print(model.summary())
opt = {"learning_rate": 0.00006, "epoch": 10, 'bs': 4}

generator = make_generator_model(train_X.shape[1], train_Y.shape[1], train_X.shape[2])
discriminator = make_discriminator_model()
gan = GAN(generator, discriminator, opt)
Predicted_price, Real_price, RMSPE = gan.train(train_X, train_Y, yc, opt)



#####################

m = attention_model(INPUT_DIMS=dim, TIME_STEPS=step)
m.summary()  #输出模型信息

adam = tf.keras.optimizers.Adam(learning_rate=0.01)
m.compile(optimizer=adam, loss='mse') 
history = m.fit([train_X], train_Y, epochs=200, batch_size=32, validation_split=0.1)
m.save_weights("OD_save_weights")
#tf.keras.models.save_model(m, "predict/stock_model")
np.save("predict/OD_normalize.npy", normalize)


data1 = pd.read_csv("data_instance3.csv")


data1.sort_values(by='time', inplace=True)
data1['passenger'] = range(1, len(data1) + 1)
data1.set_index('passenger', inplace=True)
data1.to_csv('predictOD.csv')

data1 = data1.loc[:, ['start', 'end', 'time']]
#数据切割 训练+测试
total_samples=len(data1)
data = data1.iloc[1:3500, :] #data 训练  data1全部 data2测试
data2 = data1.iloc[3500:, :] 
data1 = data1.loc[:, ['start', 'end', 'time']]
data_yuan = data1


step = 20


data, normalize = NormalizeMult(data)
print('#', normalize) 


pollution_data =np.vstack((data[:, 0], data[:, 1], data[:, 2])).T #get the close price
print(pollution_data)

train_X, _ = create_dataset(data, step)
_, train_Y = create_dataset(pollution_data, step)

print(train_X.shape, train_Y.shape)

dim = train_X.shape[-1]

m = attention_model(INPUT_DIMS=dim, TIME_STEPS=step)
m.summary()  #输出模型信息

adam = tf.keras.optimizers.Adam(learning_rate=0.05)
m.compile(optimizer=adam, loss='mse') 
history = m.fit([train_X], train_Y, epochs=200, batch_size=64, validation_split=0.1)
m.save_weights("OD_save_weights")
#tf.keras.models.save_model(m, "predict/stock_model")
np.save("predict/OD_normalize.npy", normalize)

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show(block=True)
#########################################################


data11 = pd.read_csv("predict/601988.SH.csv")
data11.index = pd.to_datetime(data11['trade_date'], format='%Y%m%d')
#data1 = data1.drop(['ts_code', 'trade_date', 'turnover_rate', 'volume_ratio', 'pb', 'total_share', 'float_share', 'free_share'], axis=1)
data11 = data11.loc[:, ['open', 'high', 'low', 'close', 'vol', 'amount']]
data_yuan = data11
residuals = pd.read_csv('predict/ARIMA_residuals1.csv')
residuals.index = pd.to_datetime(residuals['trade_date'])
residuals.pop('trade_date')
data1 = pd.merge(data1, residuals, on='trade_date')
#数据切割 训练+测试
data = data11.iloc[1:3500, :] #data 训练  data1全部 data2测试
data2 = data11.iloc[3500:, :] 

TIME_STEPS = 20

data, normalize = NormalizeMult(data)

pollution_data = data[:, 3].reshape(len(data), 1)  #get the close price
print(pollution_data)

train_X, _ = create_dataset(data, TIME_STEPS)
_, train_Y = create_dataset(pollution_data, TIME_STEPS)

print(train_X.shape, train_Y.shape)

m = attention_model(INPUT_DIMS=7)
m.summary()  #输出模型信息
adam = tf.keras.optimizers.Adam(learning_rate=0.01)
m.compile(optimizer=adam, loss='mse') 
history = m.fit([train_X], train_Y, epochs=5, batch_size=32, validation_split=0.1)
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