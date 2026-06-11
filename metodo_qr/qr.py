#!/usr/bin/env python

"""
Realiza la factorización QR de una matriz
"""

from gram_schmidt import gm, transpose, dot


def qr(M: list[list[float]]) -> tuple[list[list[float]], list[list[float]]]:

    # 1. Extraemos las columnas de M  (M está guardada como lista de filas)
    columnas = transpose(M)  # cada elemento es una columna de M

    # 2. Aplicamos Gram-Schmidt a las columnas → obtenemos columnas ortonormales
    columnas_Q = gm(columnas)  # lista de vectores ortonormales

    # 3. Q se arma poniendo esas columnas ortonormales de vuelta como matriz
    #    columnas_Q tiene forma [[col0], [col1], ...]  así que trasponemos
    Q = transpose(columnas_Q)

    # 4. R = Q^T * M
    #    Como Q es unitaria, Q^T = Q^{-1}, entonces Q^T * M = R (triangular sup.)
    Qt = transpose(Q)
    n = len(M)
    R = [[dot(Qt[i], [M[k][j] for k in range(n)]) for j in range(n)] for i in range(n)]

    # Limpieza numérica: valores muy pequeños bajo la diagonal los ponemos en 0
    # (errores de punto flotante que deberían ser exactamente 0)
    for i in range(n):
        for j in range(i):  # j < i  →  debajo de la diagonal
            if abs(R[i][j]) < 1e-10:
                R[i][j] = 0.0

    return Q, R


def main(): ...


if __name__ == "__main__":
    main()
