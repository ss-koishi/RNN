from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers.core import Dense, Activation
from tensorflow.python.keras.layers.recurrent import LSTM
from tensorflow.python.keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt

from make_noised_sin import make_noised_sin

import pandas as pd
import numpy as np
import math
import random

io = 1
hidden = 300

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

    sin = make_noised_sin()
    size = 100
    (x_train, y_train), (x_test, y_test) = train_test_split(sin, n_prev = size)

    model = Sequential([
        LSTM(hidden, batch_input_shape=(None, size, io), return_sequences=False),
        Dense(io),
        Activation('linear')
    ])

    model.compile(loss='mean_squared_error', optimizer='adam')
    stopping = EarlyStopping(monitor='val_loss', mode='auto', patience=10)
    model.fit(x_train, y_train, batch_size=256, nb_epoch=100, validation_split=0.1, callbacks=[stopping])

    pred_out = model.predict(x_test)
    predict = pred_out
    output = []
    future_data = x_test
    future_steps = 400
    for i in range(future_steps):
      print(i)
      future_data = np.delete(future_data, 0, 0)
      future_data = np.append(future_data, predict[-size:]).reshape(300, 100, 1)

      predict = model.predict(future_data)
      output = np.append(output, predict[len(predict) - 1])


    plt.figure()
    #plt.plot(y_test.flatten())
    plt.plot(range(0, 300), y_test.flatten(), label='input')
    plt.plot(range(0, 300), pred_out.flatten(), label='predict')
    plt.plot(range(300, 300 + future_steps), output.flatten(), label='future')
    plt.title('Sin Curve prediction')
    plt.legend(loc='upper right')
    plt.savefig('result.png')


if __name__ == '__main__':
    main()
