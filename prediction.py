from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers.core import Dense, Activation
from tensorflow.python.keras.layers.recurrent import LSTM
from tensorflow.python.keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import math
import random

io = 1
hidden = 300
sequence_size = 100

def make_dataset(data):

    x, y = [], []
    db = pd.DataFrame(data)
    for i in range(len(data) - sequence_size):
        x.append(db.iloc[i: i + sequence_size].value())
        y.append(db.iloc[i + sequence_size].value())

    return np.array(x), np.array(y)


def main():

    wave = hoge()
    x_train, y_train = make_dataset(wave)

    model = Sequential([
        LSTM(hidden, batch_input_shape=(None, sequence_size, io), return_sequences=False),
        Dense(io),
        Activation('linear')
    ])

    model.compile(loss='mean_squared_error', optimizer='adam')
    stopping = EarlyStopping(monitor='val_loss', mode='auto', patience=10)
    model.fit(x_train, y_train, batch_size=256, nb_epoch=100, validation_split=0.1, callbacks=[stopping])

    result = []
    future_steps = 400
    future_data = x_train[-sequence_size:]
    for i in range(future_steps):
      print(i)
      pred = model.predict(future_data)
      future_data = np.delete(future_data, 0)
      future_data = np.append(future_data, pred[-1:])

      result = np.append(result, pred[-1:])


    plt.figure()
    #plt.plot(y_test.flatten())
    plt.plot(range(0, len(x_train)), x_train, label='input')
    plt.plot(range(len(x_train), len(x_train) + future_steps), result, label='future')
    plt.title('Wave prediction')
    plt.legend(loc='upper right')
    plt.savefig('result.png')


if __name__ == '__main__':
    main()
