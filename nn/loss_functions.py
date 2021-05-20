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
    # 一次元の場合は、形式を変更する(ndimは次元数(shapeの要素数))
    # 多次元配列と同じ形式に直して誤差計算を行えるようにするため
    if y.ndim == 1:
        t = t.reshape(1, y.size)  # shapeが(t.size,)の配列を、(1, t.size)に変換
        y = y.reshape(1, y.size)  # shapeが(y.size,)の配列を、(1, y.size)に変換

    batch_size = y.shape[0]

    delta = 1e-7  # yが0のときに-∞に発散しないように加算する定数
    return -np.sum(t * np.log(y + delta)) / batch_size
