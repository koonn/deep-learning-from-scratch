"""活性化関数
"""

import numpy as np


def step(x):
    """
    ステップ関数

    Args:
        x(np.array): 入力値

    Returns: 活性化関数によって変換した出力値

    """
    # xの要素が0より大ならTrue/そうでなければFalseに変換
    y = x > 0

    # Trueを1, Falseを0のフラグに変換
    return y.astype(np.int)


def sigmoid(x):
    """
    シグモイド関数

    Args:
        x(np.array): 入力値

    Returns: 活性化関数によって変換した出力値

    """
    return 1 / (1 + np.exp(-x))


def relu(x):
    """
    ReLU関数

    Args:
        x(np.array): 入力値

    Returns: 活性化関数によって変換した出力値

    """
    return np.maximum(0, x)  # xと0の大きい方を返す


def identity(x):
    """
    恒等関数

    Args:
        x(np.array): 入力値

    Returns: 入力値と同じ配列

    """
    return x


def softmax(x):
    """

    Args:
        x(np.array): 入力値

    Returns: 入力信号をSoftmaxで変換した配列

    """
    exp_x = np.exp(x)
    sum_exp_x = np.sum(exp_x)

    y = exp_x / sum_exp_x

    return y

