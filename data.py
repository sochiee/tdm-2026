import csv


def read_csv(path: str) -> list[list[float]] | list[tuple]:
    """
    Lee los datos de un csv y los regresa en una lista de listas

    Parámetros
    __________
    path : str
        Recorrido al archivo .csv

    Regresa
    _______
    data : list[tuple]
        Lista con las columnas del csv

    """
    with open(path, newline="") as csvfile:
        spamreader = csv.reader(csvfile, delimiter=",")
        next(spamreader)
        rows = list(spamreader)
        data = list(zip(*rows))

    return data


if __name__ == "__main__":
    ...
