import numpy as np
from numpy._typing import NDArray
import math


def rls(x: list[float], y: list[float]) -> float:
    """
    Regresión lineal para un modelo de la forma y = kx

    Parametros
    ----------
    independientes : list[float]
        Variables explicativas
    dependientes : list[float]
        Variables de respuesta

    Regresa
    _______
    k : float
        Pendiente
    """
    xy = [xi * yi for xi, yi in zip(x, y)]
    x2 = [xi**2 for xi in x]

    suma_xy = sum(xy)
    suma_x2 = sum(x2)

    k = suma_xy / suma_x2

    return k


def rlm(x1: list[float], x2: list[float], y: list[float]) -> NDArray:
    """
    Regresión lineal para un modelo de la forma y = ax1 + bx2

    Parametros
    __________
    x1 : list[float]
        Primer lista de variables explicativas
    x2 : list[float]
        Segunda lista de variables explicativas
    y : list[float]
        Variables de respuesta

    Regresa
    _______
    params : ndarray
        Parámetros del modelo

    """
    x1sq = [xi**2 for xi in x1]
    x2sq = [xi**2 for xi in x2]
    x1x2 = [xi1 * xi2 for xi1, xi2 in zip(x1, x2)]

    suma_x1x2 = sum(x1x2)
    suma_x1sq = sum(x1sq)
    suma_x2sq = sum(x2sq)

    yx1 = [y * xi for y, xi in zip(y, x1)]
    yx2 = [y * xi for y, xi in zip(y, x2)]

    suma_yx1 = sum(yx1)
    suma_yx2 = sum(yx2)

    A = np.array([[suma_x1sq, suma_x1x2], [suma_x1x2, suma_x2sq]])
    B = np.array([[suma_yx1], [suma_yx2]])

    params = np.linalg.solve(A, B)

    return params


def pearson(x: list[float], y: list[float]) -> float:
    """
    Calcula el coeficiente de Pearson

    Parámetros
    __________
    x : list[float]
        Variables explicativas
    y : list[float]
        Variables de respuesta

    Regresa
    _______
    coef : float
        coeficiente de Pearson
    """
    n = len(x)
    suma_x = sum(x)
    suma_y = sum(y)
    suma_x2 = sum([xi**2 for xi in x])
    suma_y2 = sum([yi**2 for yi in y])
    suma_xy = sum([xi * yi for xi, yi in zip(x, y)])

    num = (n * suma_xy) - (suma_x * suma_y)
    den = math.sqrt(n * suma_x2 - suma_x**2) * math.sqrt(n * suma_y2 - suma_y**2)

    coef = num / den

    return coef


def calc_error(pred: list[float], truth: list[float]) -> float:
    """
    Calcula el error entre una predicción y la verdad de los datos

    Parámetros
    __________
    pred : list[float]
        Valores de predicción del modelo
    truth : list[float]
        Valores reales de los datos

    Regresa
    _______
    error : float
        Error promedio del modelo

    """
    n = len(pred)

    error = sum([(p - t) ** 2 for p, t in zip(pred, truth)]) / n

    return error
