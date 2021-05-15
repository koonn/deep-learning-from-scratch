"""活性化関数
"""

import numpy as np


def step_function(x):
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


def sigmoid_function(x):
    """
    シグモイド関数

    Args:
        x(np.array): 入力値

    Returns: 活性化関数によって変換した出力値

    """
    return 1 / (1 + np.exp(-x))


def relu_function(x):
    """
    ReLU関数

    Args:
        x(np.array): 入力値

    Returns: 活性化関数によって変換した出力値

    """
    return np.maximum(0, x)  # xと0の大きい方を返す