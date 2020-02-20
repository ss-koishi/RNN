from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers.core import Dense, Activation
from tensorflow.python.keras.layers.recurrent import LSTM
from tensorflow.python.keras.callbacks import EarlyStopping

from sklearn import preprocessing

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import math
import random

io = 1
hidden = 25
sequence_size = 50

def load_data(df):
  
    x, y = [], []
    for i in range(len(df) - sequence_size):
        x.append(df.iloc[i:i + sequence_size].as_matrix())
        y.append(df.iloc[i + sequence_size].as_matrix())

    return np.array(x), np.array(y)


def make_dataset(test_size=0.01):

    db = pd.read_csv('./nikkei-225-index-historical-chart-data.csv')
    db['v'] = preprocessing.scale(db['v'])
    db = db.sort_values(by='date')
    db = db.reset_index(drop=True)
    df = db[['v']]

    pos = int(round(len(df) * (1 - test_size)))
    x_train, y_train = load_data(df.iloc[0:pos])
    x_test, y_test = load_data(df.iloc[pos:])

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
    model.fit(x_train, y_train, batch_size=512, epochs=200, validation_split=0.24, callbacks=[stopping])

    result = []
    future_steps = 50
    future_data = x_test[-1:][-1:][-sequence_size:]
    print(future_data)
    for i in range(future_steps):
      print(i)
      pred = model.predict(future_data)
      future_data = np.delete(future_data, 0)
      #print(future_data)
      future_data = np.append(future_data, pred[-1:]).reshape(1, sequence_size, 1)
      #print(future_data)
      result = np.append(result, pred[-1:][-1:])


    input_pred = model.predict(x_test).flatten()
    plt.figure()
    #plt.plot(y_test.flatten())
    plt.plot(range(0, len(y_test)), y_test, label='input')
    plt.plot(range(0, len(input_pred)), input_pred, label='predict') 
    plt.plot(range(len(input_pred), len(input_pred) + future_steps), result, label='future')
    plt.title('Wave prediction')
    plt.legend(loc='upper right')
    plt.savefig('result.png')


if __name__ == '__main__':
    main()
