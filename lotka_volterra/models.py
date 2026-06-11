def lotka_volterra(
    r: float, c: float, b: float, d: float, n: int, c_0: float, z_0: float
):
    """
    La funcion regresa dos listas con los valores de Lotka-Volterra para n iteraciones
    r: Tasa de crecimiento de los conejos.
    b: Tasa de exito de caza de los zorros.
    d: Tasa de crecimiento de los zorros.
    c: Tasa de exito en la caza para los zorros.
    n: Numero de iteraciones
    z_0, c_0: Generaciones iniciales de zorros y conejos respectivamente.
    """

    c_vals = [c_0]
    z_vals = [z_0]

    for k in range(n - 1):
        c_k = c_vals[k]
        z_k = z_vals[k]

        c_vals.append(c_k + r * c_k * (1 - c_k) - b * c_k * z_k)
        z_vals.append(z_k - d * z_k + c * c_k * z_k)

    return c_vals, z_vals, [r, c, b, d]
