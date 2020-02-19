import pandas as pd
import numpy as np
import math
import random

def make_noised_sin():
    random.seed(0)
    # 乱数の係数
    random_factor = 0.05

    # サイクルあたりのステップ数
    steps_per_cycle = 80
    # 生成するサイクル数
    number_of_cycles = 50

    df = pd.DataFrame(np.arange(steps_per_cycle * number_of_cycles + 1), columns=["t"])
    df["sin_t"] = df.t.apply(lambda x: math.sin(x * (2 * math.pi / steps_per_cycle)+ random.uniform(-1.0, +1.0) * random_factor))

    return df[["sin_t"]]

def make_sincos():
    df = pd.DataFrame(np.arange(0, 4000), columns=['t'])
    df["sincos"] = df.t.apply(lambda x: math.sin(np.pi * x / 20) + math.cos(np.pi * x * 3 / 20))
    df["sincos"] = df.sincos.apply(lambda x: x * (0.75 + 0.5 * random.random()))

    return df[["sincos"]]
