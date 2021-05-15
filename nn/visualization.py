"""可視化用の関数
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_curve(f):
    """
    活性化関数の可視化関数

    Args:
        f(Callable[np.array]): 活性化関数

    """
    x = np.arange(-5.0, 5.0, 0.1)
    y = f(x)

    plt.plot(x, y)
    plt.ylim(-0.1, 1.1)  # y軸の描画範囲の指定
    plt.show()
