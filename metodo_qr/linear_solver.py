#!/usr/bin/env python

from gram_schmidt import matvec, transpose
from qr import qr


def solve(A: list[list[float]], b: list[float]) -> list[float]:

    m = len(A)  # número de filas
    n = len(A[0])  # número de columnas

    if len(b) != m:
        return ValueError(
            "No existe la solución pues hay dimensiones diferentes entre A y b"
        )

    Q, R = qr(A)
    Qt = transpose(Q)
    Qt_b = matvec(Qt, b)

    Qt_b = Qt_b[:n]

    x = [0.0] * n

    for i in range(n - 1, -1, -1):
        suma = 0.0
        for j in range(i + 1, n):
            suma += R[i][j] * x[j]
        x[i] = (Qt_b[i] - suma) / R[i][i]

    return x


# ─── Prueba rápida ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    A = [[1.0, 1.0], [1.0, -1.0]]

    b = [2.0, 0.0]

    x = solve(A, b)

    print("La solución es:", x)
