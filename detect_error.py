from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers.core import Dense, Activation
from tensorflow.python.keras.layers.recurrent import LSTM
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.models import load_model

from sklearn import preprocessing

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import math
import random

io = 1
hidden = 25
sequence_size = 50


def make_dataset(df):

    ret = []
    for i in range(0, len(df) - sequence_size):
        ret.append(df.iloc[i:i + sequence_size])

    return ret

def main():

    model = load_model('./model/model.h5')
    model.compile(loss='mean_squared_error', optimizer='adam')

    db = pd.read_csv('./ok.csv')

    data = make_dataset(db[['ng']])
    pred = model.predict(data)

    mse = np.sqrt(np.sum((pred - data)**2, axis=1))

    plt.figure()
    #plt.plot(y_test.flatten())
    plt.plot(range(0, len(db['ng'])), db['ng'], label='input')
    plt.plot(range(sequence_size, sequence_size + len(pred)), pred, label='predict') 
    plt.plot(range(sequence_size, sequence_size + len(pred)), mse, label='mse')
    plt.title('Wave prediction')
    plt.legend(loc='upper right')
    plt.savefig('result.png')


if __name__ == '__main__':
    main()
