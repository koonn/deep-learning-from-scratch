"""損失関数
"""

import numpy as np


def mean_squared_error(y, t):
    """最小二乗誤差を計算する関数

    Args:
        y (np.array): 予測値の配列
        t (np.array): 実データ値の配列

    Returns:
        float: 最小二乗誤差の値
    """
    return 0.5 * np.sum((y - t)**2)


def cross_entropy_error(y, t):
    """交差エントロピー誤差を計算する関数

    Args:
        y (np.array): 予測値の配列
        t (np.array): 実データ値の配列

    Returns:
        float: 交差エントロピー誤差の値
    """
    delta = 1e-7

    return -np.sum(t * np.log(y + delta))