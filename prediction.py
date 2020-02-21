from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Activation, Input
from tensorflow.python.keras.layers.recurrent import LSTM
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.regularizers import l1, l2

from sklearn.preprocessing import minmax_scale

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import math
import random

import sys

io = 1
hidden = 150
sequence_size = 20


def load_data(data):

    x = []
    db = pd.DataFrame(data)
    for i in range(len(data) - sequence_size):
        #print(db.iloc[i : i + sequence_size])
        #print(db.iloc[i : i + sequence_size].as_matrix().flatten())
        x.append(db.iloc[i:i + sequence_size].as_matrix().flatten())

    return np.array(x)


def make_dataset(test_size=0.15):

    #scaler = StandardScaler()
    db = pd.read_csv('./wave.csv')
    db['ok'] = minmax_scale(db[['ok']])
    db['ok2'] = minmax_scale(db[['ok2']])
  
    #with open('parameters.txt', 'w') as f:
    #  f.write(str(scaler.mean_[0]) + '\n')
    #  f.write(str(scaler.var_[0]))

    pos = int(len(db) * (1 - test_size))
    x2_train = load_data(db['ok2'].iloc[:])
    x_train = load_data(db['ok'].iloc[0:pos])
    x_test = load_data(db['ok'].iloc[pos:])
    
    return (np.append(x_train, x2_train).reshape([-1, sequence_size]), x_test)


def main():

    (x_train, x_test) = make_dataset()
    
    print(x_train)
    print(x_test)
    #sys.exit(0)
      
    #model = Sequential([
    #    LSTM(hidden, batch_input_shape=(None, sequence_size, io), return_sequences=False),
    #    Dense(io),
    #    Activation('linear')
    #])

    #model.compile(loss='mean_squared_error', optimizer='adam')
  
    model = Sequential([
        Dense(64, input_dim=sequence_size, activation='relu'),
        Dense(32, activation='relu'),
        Dense(8, activation='relu', activity_regularizer=l2(0.001)),
        Dense(32, activation='relu'),
        Dense(64, activation='relu'),
        Dense(sequence_size, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='mse')

    stopping = EarlyStopping(monitor='val_loss', mode='auto', patience=10000)

    model.fit(x_train, x_train, batch_size=128, shuffle=True, epochs=10000, validation_split=0.2, callbacks=[stopping])

    #result = []
    #future_steps = 400
    future_data = x_test[-1:][-sequence_size:]
    #print(future_data)
    #for i in range(future_steps):
     #   print(i)
     #   pred = model.predict(future_data).flatten()
     #   future_data = np.delete(future_data, 0)
     #   future_data = np.append(future_data, pred[-1:]).reshape(-1, sequence_size)
#
#        result = np.append(result, pred[-1:])
    
    print(len(x_train[:, -1:]))
    tr_w = x_train[:, -1:].flatten()
    ts_w = x_test[:, -1:].flatten()
    input_pred = model.predict(x_test)[:,-1:].flatten()
    print(input_pred)

    plt.figure()
#    plt.plot(range(0, len(tr_w)), tr_w, label='train')
    plt.plot(range(len(tr_w), len(tr_w) + len(ts_w)), ts_w, label='input')
    plt.plot(range(len(tr_w), len(tr_w) + len(input_pred)), input_pred, label='predict')
   # plt.plot(range(len(y_train) + len(input_pred), len(y_train) + len(input_pred) + future_steps), result, label='future')
    plt.title('Wave prediction')
    plt.legend(loc='upper right')
    plt.savefig('result.png')

    model.save('./model/model.h5')

if __name__ == '__main__':
    main()
