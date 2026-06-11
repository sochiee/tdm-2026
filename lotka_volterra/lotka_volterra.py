#!/usr/bin/env python

from matplotlib import pyplot as plt
from models import lotka_volterra


def plot_lotvol(valores, name, punto=False) -> None:
    """
    Este apartado grafica los valores del sistema de ecucaciones Lotka-Volterra

    valores: Las dos listas resultantes de la funcion
    """

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 3))
    fig.tight_layout()

    ax1.plot(valores[0], label="Presa", color="darkcyan")
    ax1.set_title("n vs C(n)")
    ax1.set_xlabel("n")
    ax1.set_ylabel("C(n)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(valores[1], label="Depredadores", color="coral")
    ax2.set_title("n vs Z(n)")
    ax2.set_xlabel("n")
    ax2.set_ylabel("Z(n)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    ax3.plot(valores[0], valores[1], color="mediumvioletred")
    ax3.set_title("C(n) vs Z(n)")
    ax3.set_xlabel("C(n)")
    ax3.set_ylabel("Z(n)")
    ax3.grid(True, alpha=0.3)

    if punto:
        r = valores[2][0]
        c = valores[2][1]
        b = valores[2][2]
        d = valores[2][3]

        ax3.plot(d / c, ((r * (c - d)) / (b * c)), "o")

    plt.savefig("media/" + name + ".png")

    return None


# En esta sección definimos los parámetros del ejercicio 1
if __name__ == "__main__":
    params: list[list] = [
        [0.08, 0.1, 0.01, 0.15, 100, 0.2, 0.9],
        [0.25, 1.8, 0.91, 0.6, 1000, 0.2, 0.5],
        [0.25, 0.95, 1.1, 0.55, 300, 0.2, 0.5],
    ]

    for i, param in enumerate(params):
        iniciales = [(param[5], param[6]), (0.5, 0.5), (0.7, 0.2), (0.1, 0.1)]
        for j, inicial in enumerate(iniciales):
            vals = lotka_volterra(
                param[0], param[1], param[2], param[3], param[4], *inicial
            )
            if i == 0:
                plot_lotvol(vals, f"{i}{j}")
            else:
                plot_lotvol(vals, f"{i}{j}", True)
