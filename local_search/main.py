from math import cos, sqrt
import sys
import random
import numpy as np
import time


def happycat_value(xx):
    return pow(pow((pow(np.linalg.norm(xx), 2) - 4), 2), 1 / 8) + (1 / 4) * (
                1 / 2 * pow(np.linalg.norm(xx), 2) + np.sum(xx)) + 1 / 2


def griewank_value(xx):
    g_x = 1 + np.sum(np.array([xx[i] ** 2 / 4000 for i in range(0, len(xx) - 1)])) - np.prod(
        np.array([cos(xx[i] / sqrt(i + 1)) for i in range(0, len(xx) - 1)]))
    return g_x


def local_search(t, f):
    tmp_array = np.array([random.random() for i in range(4)])
    best_array = np.copy(tmp_array)
    best_value = f(best_array)
    t_end = time.time() + t
    sigma = 0.005
    n = len(tmp_array)

    while time.time() < t_end:
        mask = np.random.normal(0, sigma, n)
        tmp_array = np.copy(best_array) + mask
        tmp_value = f(tmp_array)

        # sprawdz czy lepszy
        if tmp_value < best_value:
            best_array = tmp_array
            best_value = tmp_value

    return best_value, best_array


def read_data_from_file(fname):
    with open(fname, 'r') as f:
        lines = f.readlines()
        # czas w sekundach
        t = lines[0].split(' ')[0]
        # b
        b = lines[0].split(' ')[1]
        return int(t), int(b)


if __name__ == '__main__':
    max_time, func_num = list(map(int, input().split()))
    func = happycat_value if func_num == 0 else griewank_value
    # print(max_time, func_num)
    result = local_search(max_time, func)

    best_value, best_array = result
    for i in best_array:
        print(i, end=' ')
    print(best_value)
