from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers.core import Dense, Activation
from tensorflow.python.keras.layers.recurrent import LSTM
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.models import load_model

from sklearn.preprocessing import minmax_scale

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import math
import random 

sequence_size = 20

def make_dataset(df):

    ret = []
    for i in range(len(df) - sequence_size):
        ret.append(df.iloc[i:i + sequence_size].as_matrix().flatten())

    return np.array(ret)


def main():

    model = load_model('./model/model.h5')
    model.compile(loss='mse', optimizer='adam')

    #scaler = StandardScaler()
    db = pd.read_csv('./wave.csv')
    mean, var = None, None
    #with open('parameters.txt', 'r') as f:
    #  mean = float(f.readline())
    #  var = float(f.readline())

    #scaler.fit(db[['ok']]);
    #scaler.mean_ = np.array(mean)
    #scaler.var_ = np.array(var)
    db['ok'] = minmax_scale(db[['ok']])
    db['ng'] = minmax_scale(db[['ng']])
    db['ok2'] = minmax_scale(db[['ok2']])

    ok_data = make_dataset(db[['ok']].iloc[0:-1])
    ng_data = make_dataset(db[['ng']].iloc[0:-1])
   
    ok2_data = make_dataset(db[['ok2']].iloc[0:-1])
    
    ok_pred = model.predict(ok_data)
    ng_pred = model.predict(ng_data)
    ok2_pred = model.predict(ok2_data)
    ok_mse = np.sqrt(np.sum((ok_pred - ok_data)**2, axis=1))
    ng_mse = np.sqrt(np.sum((ng_pred - ng_data)**2, axis=1))
    ok2_mse = np.sqrt(np.sum((ok2_pred - ok2_data)**2, axis=1))

    plt.figure()
    ok_pred = ok_pred[:, -1:]
    ng_pred = ng_pred[:, -1:]
    #plt.plot(range(0, len(db['ng'])), db['ng'], label='ng_input')
    #plt.plot(range(0, len(db['ok'])), db['ok'], label='ok_input')
    #plt.plot(range(sequence_size, sequence_size + len(ng_pred)), ng_pred, label='NG_predict')
    #plt.plot(range(sequence_size, sequence_size + len(ok_pred)), ok_pred, label='OK_predict')
    plt.plot(range(sequence_size, sequence_size + len(ng_mse)), ng_mse, label='NG mse')
    plt.plot(range(sequence_size, sequence_size + len(ok_mse)), ok_mse, label='OK mse')
    plt.plot(range(sequence_size, sequence_size + len(ok2_mse)), ok2_mse, label='OK2 mse')
    plt.title('Wave prediction')
    plt.legend(loc='upper right')
    plt.savefig('error.png')


if __name__ == '__main__':
    main()
