#!/usr/bin/env python
import collections
import numbers
from math import sqrt
from linear_solver import solve
from collections.abc import Sequence
from qr import qr


from eigenvalues import eigen_epsilon, eigenvals


class linspace(collections.abc.Sequence):
    """linspace(start, stop, num) -> linspace object

    Return a virtual sequence of num numbers from start to stop (inclusive).

    If you need a half-open range, use linspace(start, stop, num+1)[:-1].
    """

    def __init__(self, start, stop, num):
        if not isinstance(num, numbers.Integral) or num <= 1:
            raise ValueError("num must be an integer > 1")
        self.start, self.stop, self.num = start, stop, num
        self.step = (stop - start) / (num - 1)

    def __len__(self):
        return self.num

    def __getitem__(self, i):
        if isinstance(i, slice):
            return [self[x] for x in range(*i.indices(len(self)))]
        if i < 0:
            i = self.num + i
        if i >= self.num:
            raise IndexError("linspace object index out of range")
        if i == self.num - 1:
            return self.stop
        return self.start + i * self.step

    def __repr__(self):
        return "{}({}, {}, {})".format(
            type(self).__name__, self.start, self.stop, self.num
        )

    def __eq__(self, other):
        if not isinstance(other, linspace):
            return False
        return (self.start, self.stop, self.num) == (other.start, other.stop, other.num)

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash((type(self), self.start, self.stop, self.num))


# ─────────────────────────────────────────────────────────────────────────────
# Utilidades para el Ejercicio 1
# ─────────────────────────────────────────────────────────────────────────────


def polinomio_caracteristico_2x2(A: list[list[float]]) -> tuple[float, float, float]:
    """Calcula los coeficientes (a, b, c) del polinomio característico
    de una matriz 2x2, donde aλ² + bλ + c = 0.

    Para A = [[a11, a12],
              [a21, a22]]:

      det(A - λI) = (a11 - λ)(a22 - λ) - a12*a21
                  = λ² - (a11 + a22)λ + (a11*a22 - a12*a21)

    Devuelve (a, b, c) con a=1 (coeficiente de λ²).
    """
    a11, a12 = A[0]
    a21, a22 = A[1]

    a = 1.0
    b = -(a11 + a22)  # negativo de la traza
    c = a11 * a22 - a12 * a21  # determinante
    return a, b, c


def formula_cuadratica(a: float, b: float, c: float) -> tuple[float, float]:
    """Resuelve aλ² + bλ + c = 0 con la fórmula general.

    Devuelve las dos raíces (λ1, λ2) con λ1 ≤ λ2.
    Lanza ValueError si el discriminante es negativo (raíces complejas).
    """
    discriminante = b**2 - 4 * a * c

    if discriminante < 0:
        raise ValueError(
            f"Discriminante negativo ({discriminante:.4f}): "
            "los eigenvalores son complejos."
        )

    raiz_disc = sqrt(discriminante)
    lambda1 = (-b - raiz_disc) / (2 * a)
    lambda2 = (-b + raiz_disc) / (2 * a)
    return lambda1, lambda2


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main():
    # ── Ejercicio 1: Eigenvalores por el polinomio característico ─────────────

    print("=" * 55)
    print("  Ejercicio 1: Polinomio Característico")
    print("=" * 55)

    A = [
        [5.0, -2.0],
        [-2.0, 8.0],
    ]

    print("\nMatriz A:")
    for fila in A:
        print(f"  {fila}")

    # Paso 1 — coeficientes del polinomio característico
    a, b, c = polinomio_caracteristico_2x2(A)

    print("\nPolinomio característico det(A - λI) = 0:")
    print(f"  λ² + ({b})λ + ({c}) = 0")
    print(f"  (coeficientes: a={a}, b={b}, c={c})")

    # Paso 2 — discriminante
    discriminante = b**2 - 4 * a * c
    print(f"\nDiscriminante = b² - 4ac = {b}² - 4·{a}·{c} = {discriminante}")

    # Paso 3 — raíces (eigenvalores)
    lambda1, lambda2 = formula_cuadratica(a, b, c)

    print("\nEigenvalores:")
    print(f"  λ₁ = {lambda1}")
    print(f"  λ₂ = {lambda2}\n")

    # Ejercicio 2:

    print("=" * 55)
    print("  Ejercicio 2: Método QR")
    print("=" * 55)

    eigs_qr_10 = eigenvals(A, 10)

    print("Eigenvalores aproximados (QR):")
    for val in eigs_qr_10:
        print(f"  {val}")

    lambda_exactos = sorted([lambda1, lambda2])
    lambda_qr = sorted(eigs_qr_10)

    # Error relativo:
    for i in range(len(lambda_exactos)):
        error_real = abs(lambda_exactos[i] - lambda_qr[i])
        print(f"Error relativo λ{i + 1}: {error_real}")

    # Ejercicio 3:

    print("=" * 55)
    print(" Ejercicio 3: Método QR")
    print("=" * 55)

    print("\nMatriz A:")
    for row in A:
        print(f"  {row}")

    Ak = eigen_epsilon(A, 1000, 1e-10)

    print("\nMatriz con eigenvalores:")
    for row in Ak:
        print(f"  {row}")


if __name__ == "__main__":
    main()
