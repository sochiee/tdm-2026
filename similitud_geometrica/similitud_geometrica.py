from matplotlib import pyplot as plt
from numpy.typing import NDArray
import seaborn as sns
import numpy as np
import pandas as pd

from linear_regression import rls, rlm, pearson, calc_error


def plot_aprox(
    x: NDArray, aprox: NDArray, truth: NDArray, name: str, title: str
) -> None:
    plt.figure(figsize=(8, 5))

    sns.scatterplot(x=x, y=truth, linewidth=0, s=75, color="cornflowerblue")
    sns.lineplot(x=x, y=aprox, color="orange")

    plt.xlabel("Volumen (cm$^3$)", fontsize=10)
    plt.ylabel("Peso (kg)", fontsize=10)
    plt.title(title, fontsize=13, pad=15)
    plt.savefig("media/" + name + ".png", dpi=500)
    plt.clf()


def plot_data(x: NDArray, y: NDArray) -> None:
    plt.figure(figsize=(8, 5))

    sns.scatterplot(x=x, y=y, linewidth=0, s=75, color="cornflowerblue")
    plt.xlabel("Volumen (cm$^3$)", fontsize=10)
    plt.ylabel("Peso (kg)", fontsize=10)
    plt.title("Relación entre el peso y volumen de los Róbalos", fontsize=13, pad=15)

    plt.savefig("media/pesovolumen.png", dpi=500)
    plt.clf()


def main():
    df = pd.read_csv("data/pescados.csv")
    long = df["longitud"].to_numpy()
    peso = df["peso"].to_numpy()

    r = pearson(long, peso)
    print(r)

    sns.set_theme(style="darkgrid", palette="pastel")

    plot_data(long**3, peso)

    k = rls(long**3, peso)
    print(k)
    lineal_aprox = k * (long**3)
    plot_aprox(
        long**3,
        lineal_aprox,
        peso,
        "aproximacion_lineal",
        "Aproximación lineal del modelo",
    )

    circ = df["circunferencia"].to_numpy()
    a, b = rlm(long**3, long * (circ**2), peso)
    print(a, b)
    multilineal_aprox = a * (long**3) + b * long * (circ**2)
    plot_aprox(
        long**3,
        multilineal_aprox,
        peso,
        "aproximacion_multilineal",
        "Aproximación polinomial del modelo",
    )

    linear_error = calc_error(lineal_aprox, peso)
    multilineal_error = calc_error(multilineal_aprox, peso)

    print(f"error lineal: {linear_error}")
    print(f"error polinomial: {multilineal_error}")


if __name__ == "__main__":
    main()
