from sys import stderr
from Image import Image
import decimal
import numpy as np
import time
import copy
import random


def value(M, M2):
    return ((M - M2) ** 2).mean()


def acceptance_probability(v_neigh, v_curr, temp, c=1):
    return 1 / (1 + decimal.Decimal(c * (v_neigh - v_curr) / temp).exp())


def new_temperature(temp, temp_multiplier):
    return temp * temp_multiplier


def sa(time_limit, image, n, m, k):
    time_to_end = time.time() + time_limit

    curr = image.blocks
    curr_M = image.get_matrix_from_blocks(curr)
    v_curr = image.get_distance_between(image.original_image, curr_M)

    START_TEMPERATURE = 1000
    TEMP_MULTIPLIER = 0.97
    END_TEMPERATURE = 1

    best = copy.deepcopy(curr)
    v_best = v_curr

    temperature = START_TEMPERATURE

    while time.time() < time_to_end:
        neighbour = image.get_neighbour_solution(curr)
        v_n = image.get_distance_between(image.original_image, image.get_matrix_from_blocks(neighbour))

        if v_n < v_curr or random.random() < acceptance_probability(v_n, v_curr, temperature):
            curr = copy.deepcopy(neighbour)
            v_curr = v_n

        if v_curr < v_best:
            best = copy.deepcopy(curr)
            v_best = v_curr

        temperature = new_temperature(temperature, TEMP_MULTIPLIER)
        if temperature <= END_TEMPERATURE:
            # restart from best point
            curr = copy.deepcopy(best)
            v_curr = v_best
            temperature = START_TEMPERATURE

    return best


if __name__ == '__main__':
    first_line = input()
    t, n, m, k = [int(num) for num in first_line.split()]
    M = np.array([[int(num) for num in input().split()]
                  for _ in range(n)])
    image = Image(M, n, m, k)
    blocks = image.blocks

    best = sa(t, image, n, m, k)
    best_matrix = image.get_matrix_from_blocks(best)
    best_cost = image.get_distance_between(image.original_image, best_matrix)

    if not image.blocks_legal_test:
        print("ERROR: Wynikowa macierz nie spełnia wymagań")
        exit(1)

    print(best_cost)
    for row in best_matrix:
        for val in row:
            print(val, end=' ', file=stderr)
        print(file=stderr)
