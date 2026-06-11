def dot(x: list[float], y: list[float]) -> float:
    """Producto punto entre dos vectores.

    Multiplica elemento por elemento y suma todo.
    Ej: dot([1,2,3], [4,5,6]) = 1*4 + 2*5 + 3*6 = 32
    """
    return sum(xi * yi for xi, yi in zip(x, y))


def transpose(M: list[list[float]]) -> list[tuple[float]]:
    """Devuelve la traspuesta de una matriz.

    Convierte filas en columnas y columnas en filas.
    Ej: [[1,2],[3,4]] -> [[1,3],[2,4]]
    """
    return [list(row) for row in zip(*M)]


def matmul(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    """Multiplicación de dos matrices A y B.

    El elemento (i,j) del resultado es el producto punto
    de la fila i de A con la columna j de B.
    """
    B_T = transpose(B)  # Trasponemos B para acceder a sus columnas como filas
    return [[dot(fila, col) for col in B_T] for fila in A]


def matvec(A: list[list[float]], v: list[float]) -> list[float]:
    """Multiplicación de una matriz A por un vector v.

    Cada elemento del resultado es el producto punto
    de una fila de A con el vector v.
    """
    return [dot(fila, v) for fila in A]


def norm(x: list[float]) -> float:
    """Obtiene la norma 2 (longitud) de un vector.

    Es la raíz cuadrada de la suma de los cuadrados de sus elementos.
    Ej: norm([3, 4]) = sqrt(9 + 16) = 5.0
    """
    return dot(x, x) ** 0.5


def proj(u: list[float], v: list[float]) -> list[float]:
    """Calcula la proyección del vector u sobre el vector v.

    Fórmula: proj_v(u) = (<u, v> / <v, v>) * v
    Es la "sombra" de u en la dirección de v.
    """
    escalar = dot(u, v) / dot(v, v)
    return [escalar * vi for vi in v]


def normalize(u: list[float]) -> list[float]:
    """Normaliza un vector para que tenga norma 1.

    Divide cada componente entre la norma del vector.
    """
    n = norm(u)
    return [ui / n for ui in u]


def gram_schmidt(V: list[list[float]]) -> list[list[float]]:
    """Aplica el proceso de Gram-Schmidt a una lista de vectores.

    Recibe V como lista de vectores columna (cada vector es una lista).
    Devuelve una lista de vectores ortonormales.
    """
    U = []  # Aquí guardaremos los vectores ortogonales (sin normalizar aún)

    for v in V:
        # Empezamos con el vector original
        u = list(v)

        # Le restamos las proyecciones sobre todos los vectores ya calculados
        for u_prev in U:
            p = proj(v, u_prev)
            u = [ui - pi for ui, pi in zip(u, p)]

        # Normalizamos para que tenga norma 1
        U.append(normalize(u))

    return U


gm = gram_schmidt  # Alias para facilitar la importación desde qr.py


def matrix_to_str(matrix: list[list[float]]) -> str:
    """Convierte una matriz a texto formateado con columnas alineadas."""
    s = [[str(e) for e in row] for row in matrix]
    lens = [max(map(len, col)) for col in zip(*s)]
    fmt = "\t".join("{{:{}}}".format(x) for x in lens)
    table = [fmt.format(*row) for row in s]
    return "\n".join(table)


def main(): ...


if __name__ == "__main__":
    main()
