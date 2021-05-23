"""NNの学習用の関数群
"""


def numerical_diff(f, x):
    """数値微分のための関数
    """
    h = 10e-50

    return (f(x+h) - f(x)) / h
