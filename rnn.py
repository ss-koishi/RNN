from tensorflow.keras.models import Sequential
from tensorflow.keras.layers.core import Dense, Activation
from tensorflow.layers.recurrent import LSTM

import matplotli.pyplot as plt

from make_noised_sin import make_noised_sin

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
        LSTM(hidden, batch_input_shape=(None, size, io), return_sequence=False),
        Dense(io),
        Activation('linear')
    ])

    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(x_train, y_train, batch_size=16, nb_epoch=15, validation_split=0.05)

    plt.figure()
    plt.plot(x_test, y_test, label='input')
    plt.plot(predict[:200], y_test, label='predict')
    plt.saveimg('result.png')

if __name__ == '__main__':
    main()
