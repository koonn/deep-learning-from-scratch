"""NNの学習用の関数群
"""
import numpy as np


def numerical_diff(f, x):
    """数値微分のための関数

    Args:
        f (function): 微分を求めたい関数(1変数)
        x (numeric): 微分を求めたい点(x座標)

    Returns:

    """
    h = 1e-4

    return (f(x+h) - f(x-h)) / (2*h)


def numerical_gradient(f, x):
    """

    Args:
        f (function): 偏微分を求めたい関数
        x (numeric): 偏微分を求めたい点(x座標)

    Returns:

    Examples:
        functionの形式は以下のように、numpy配列を引数にとるものを想定

        def function_2(x):
            return x[0]**2 + x[1]**2
    """
    h = 1e-4

    grad = np.zeros_like(x)  # xと同じ形状の配列を作成

    for idx in range(x.size):
        tmp_val = x[idx]

        # f(x+h)の計算
        x[idx] = tmp_val + h
        fxh1 = f(x)

        # f(x-h)の計算
        x[idx] = tmp_val - h
        fxh2 = f(x)

        # 勾配の計算
        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val #値を元に戻す

    return grad


def gradient_descent(
        f,
        init_x,
        lr=0.01,
        step_num=100):
    """

    Args:
        f: 関数
        init_x (np.array): xの初期値
        lr: 学習率
        step_num: ステップ数

    Returns:
        関数が最小値をとるときのパラメータ(十分学習できていれば)
    """
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)

        x -= lr * grad

    return x
