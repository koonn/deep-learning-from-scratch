"""ネットワークの関数
"""

import numpy as np
from nn.functions import sigmoid, identity


def init_network():
    """
    ネットワークの初期値を定義する関数

    Returns:
        Dict: ネットワークのDict

    """
    network = {
        'W1': np.random.rand(2, 3),
        'B1': np.random.rand(1, 3),
        'W2': np.random.rand(3, 2),
        'B2': np.random.rand(1, 2),
        'W3': np.random.rand(2, 2),
        'B3': np.random.rand(1, 2),
        }

    return network


def forward(network, x):
    """
    入力信号を出力に変換する関数

    Args:
        network: ネットワークのDict
        x: Inputの配列

    Returns:
        出力信号

    """
    w1, w2, w3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['B1'], network['B2'], network['B3']

    # 1層目
    a1 = np.dot(x, w1) + b1
    z1 = sigmoid(a1)

    # 2層目
    a2 = np.dot(z1, w2) + b2
    z2 = sigmoid(a2)

    # 3層目
    a3 = np.dot(z2, w3) + b3
    y = identity(a3)

    return y
