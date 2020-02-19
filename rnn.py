from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers.core import Dense, Activation
from tensorflow.python.keras.layers.recurrent import LSTM
from tensorflow.python.keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt

import make_data as md

import pandas as pd
import numpy as np
import math
import random

io = 1
hidden = 50

def load_data(data, n_prev):

    x, y = [], []
    for i in range(len(data) - n_prev):
        x.append(data.iloc[i: i + n_prev].as_matrix())
        y.append(data.iloc[i + n_prev].as_matrix())
    alsx = np.array(x)
    alsy = np.array(y)

    return alsx, alsy


def train_test_split(df, test_size=0.1, n_prev=100):

    ntrn = round(len(df) * (1 - test_size))
    ntrn = int(ntrn)
    x_train, y_train = load_data(df.iloc[0:ntrn], n_prev)
    x_test, y_test = load_data(df.iloc[ntrn:], n_prev)

    return (x_train, y_train), (x_test, y_test)


def main():

    sin = md.make_noised_sin()
    size = 60
    (x_train, y_train), (x_test, y_test) = train_test_split(sin, n_prev = size)

    model = Sequential([
        LSTM(hidden, batch_input_shape=(None, size, io), return_sequences=False),
        Dense(io),
        Activation('linear')
    ])

    model.compile(loss='mean_squared_error', optimizer='adam')
    stopping = EarlyStopping(monitor='val_loss', mode='auto', patience=10)
    model.fit(x_train, y_train, batch_size=256, nb_epoch=1000, validation_split=0.1, callbacks=[stopping])

    result = []
    future_steps = 1000
    future_data = [x_train[-1:][-1:][-size:]]
    for i in range(future_steps):
      print(i)
      pred = model.predict(future_data)
      future_data = np.delete(future_data, 0)
      future_data = np.append(future_data, pred[-1:]).reshape(1, size, 1)
      result = np.append(result, pred[-1:])

    plt.figure()
    #plt.plot(y_test.flatten())
    output = y_train[-100:]
    plt.plot(range(0, len(output)), output, label='input')
    plt.plot(range(len(output), len(output) + future_steps), result, label='future')
    plt.title('Sin Curve prediction')
    plt.legend(loc='upper right')
    plt.savefig('result.png')


if __name__ == '__main__':
    main()
