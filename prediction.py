from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers.core import Dense, Activation
from tensorflow.python.keras.layers.recurrent import LSTM
from tensorflow.python.keras.callbacks import EarlyStopping

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import math
import random

io = 1
hidden = 50
sequence_size = 80


def load_data(data):

    x, y = [], []
    db = pd.DataFrame(data)
    for i in range(len(data) - sequence_size):
        x.append(db.iloc[i: i + sequence_size].as_matrix())
        y.append(db.iloc[i + sequence_size].as_matrix())

    return np.array(x), np.array(y)


def make_dataset(test_size=0.1):

    scaler = StandardScaler()
    db = pd.read_csv('./wave.csv')
    db['ok'] = scaler.fit_transform(db['ok'])

    pos = int(len(db) * (1 - test_size))
    x_train, y_train = load_data(db['ok'].iloc[0:pos])
    x_test, y_test = load_data(db['ok'].iloc[pos:])

    return (x_train, y_train), (x_test, y_test)


def main():

    (x_train, y_train), (x_test, y_test) = make_dataset()

    model = Sequential([
        LSTM(hidden, batch_input_shape=(None, sequence_size, io), return_sequences=False),
        Dense(io),
        Activation('linear')
    ])

    model.compile(loss='mean_squared_error', optimizer='adam')
    stopping = EarlyStopping(monitor='val_loss', mode='auto', patience=10)
    model.fit(x_train, y_train, batch_size=256, epochs=100, validation_split=0.2, callbacks=[stopping])

    result = []
    future_steps = 400
    future_data = x_test[-1:][-sequence_size:]
    for i in range(future_steps):
        print(i)
        pred = model.predict(future_data)
        future_data = np.delete(future_data, 0)
        future_data = np.append(future_data, pred[-1:])

        result = np.append(result, pred[-1:])

    input_pred = model.predict(x_test).flatten()
    plt.figure()
    plt.plot(range(0, len(y_train)), y_train, label='train')
    plt.plot(range(len(y_train), len(y_train) + len(y_test)), y_test, label='input')
    plt.plot(range(len(y_train), len(y_train) + len(input_pred)), input_pred, label='input')
    plt.plot(range(len(y_train) + len(input_pred), len(y_train) + len(input_pred) + future_steps), result, label='future')
    plt.title('Wave prediction')
    plt.legend(loc='upper right')
    plt.savefig('result.png')

    model.save('./model/model.h5')

if __name__ == '__main__':
    main()
