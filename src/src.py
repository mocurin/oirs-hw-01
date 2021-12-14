import numpy as np


# Среднеквадратичное отклонение - для оценки работы алгоритма
def st_der(A: np.ndarray):
    return np.mean((A - np.mean(A)) ** 2) ** (1/2)


# Оценка степени зашумления
def nrc(A: np.ndarray, B: np.ndarray):
    return np.sum(A ** 2) / np.sum(B ** 2)


# Для подсчета MSE надо бы иметь последовательности одинаковой длины
def expand(A: np.ndarray, times: int = 2):
    return np.array(
        list(
            e
            for elem in A
            for e in [elem] * times
        )
    )


# Позволит групировать последовтельность по N последовательных элементов
def grouper(n, iterable):
    args = [iter(iterable)] * n
    return zip(*args)


# Матрица хаара для наименьшего шага преобразования
H2 = np.array(
    [[1, 1], [1, -1]]
) * 1/2


# Шаг фильтрации хаара для одномерной последовательности
def h_filter(sequence: np.ndarray):
    res = list(np.dot(chunk, H2) for chunk in grouper(2, sequence))

    return np.transpose(res)


def multi_h_filter(n, xi, yi):
    x, y = list(), list()
    for _ in range(n):
        xi, _ = h_filter(xi)
        yi, _ = h_filter(yi)

        x.append(xi)
        y.append(yi)

    return x, y
