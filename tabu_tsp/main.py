import numpy as np
import time
from Tabu import SolutionTabu
import random
import sys


def tsp(cost_matrix, n, t):
    time_to_end = time.time() + t
    candidate_solution, candidate_cost = get_best_greedy_route(n, cost_matrix)
    # print("GREEDY ",candidate_solution, candidate_cost)
    best_solution, best_cost = np.copy(candidate_solution), candidate_cost
    tabu = SolutionTabu(elem_size=n, tabu_size=2*n)
    tabu.add_solution(candidate_solution)

    while time.time() < time_to_end:

        while len(tabu.tabu_array) > tabu.size:
            tabu.remove_oldest()

        new_candidate = get_random_route(n)
        new_candidate_cost = route_cost(new_candidate, cost_matrix)

        # two opt
        for i in range(1, n - 1):
            for k in range(i + 1, n):
                new_route = two_city_swap(np.copy(candidate_solution), i, k)
                if not tabu.contains(new_route):
                    cost = route_cost(new_route, cost_matrix)
                    if cost < new_candidate_cost or tabu.contains(new_candidate):
                        new_candidate = new_route
                        new_candidate_cost = cost
            if time.time() > time_to_end:
                break

        if not tabu.contains(new_candidate):
            tabu.add_solution(new_candidate)
            if new_candidate_cost <= candidate_cost:
                candidate_cost = new_candidate_cost
                candidate_solution = new_candidate

        if candidate_cost < best_cost:
            best_cost = candidate_cost
            best_solution = candidate_solution

        # print(new_candidate_cost, best_cost,tabu.tabu_array)

    return best_solution, best_cost


def two_city_swap(route, kth, nth):
    route[kth], route[nth] = route[nth], route[kth]
    return route


def two_opt_swap(route, kth, nth):
    """ dwukrawedziowa zmiana
        1. take route[0] to route[i-1] and add them in order to new_route
        2. take route[i] to route[k] and add them in reverse order to new_route
        3. take route[k+1] to end and add them in order to new_route """

    rev_part = route[kth: nth + 1][::-1]
    swapped = route[0:kth]
    swapped = np.append(swapped, rev_part)
    swapped = np.append(swapped, route[nth + 1:])
    # print(swapped)
    return swapped


def get_random_route(n):
    """ return random route that starts at city 0 """
    return np.insert(np.random.permutation([i for i in range(1, n)]), 0, 0)


def get_best_greedy_route(n, cost_matrix):
    best_route = get_greedy_route(n,cost_matrix,0)
    best_cost = route_cost(best_route,cost_matrix)

    for i in range(1,n):
        route = get_greedy_route(n,cost_matrix,1)
        cost = route_cost(route,cost_matrix)
        if cost < best_cost:
            best_route = np.copy(route)
            best_cost = cost

    return best_route, best_cost


def get_greedy_route(n, cost_matrix, start=0):
    unvisited_cities = np.array([i for i in range(0, n)])
    unvisited_cities = unvisited_cities[unvisited_cities !=start]
    # every route starts with 0

    route = np.array([start])
    min_city = start

    while unvisited_cities.size != 0:
        min_city = _get_min_cost_city(cost_matrix[min_city], unvisited_cities)
        route = np.append(route, [min_city])
        unvisited_cities = unvisited_cities[unvisited_cities != min_city]

    while route[0] != 0:
        route = np.roll(route, 1)

    return route


# na input miasto z ktorego chcemy wybrac droge do innego o mozliwym koszcie
def _get_min_cost_city(cost_cities, unvisited):
    min_val = max(cost_cities)
    min_city = -1
    for city in unvisited:
        if cost_cities[city] <= min_val:
            min_city = city
            min_val = cost_cities[city]
    return min_city


def route_cost(route, cost_matrix):
    # dołączamy pierwszy element na koniec aby można było łatwiej obliczyć koszt
    route = np.append(route, route[0])
    # enumeracja od 0 do przedostatniego elementu
    dist_sum = sum(np.array([cost_matrix[route[index]][route[index + 1]] for index, a in enumerate(route[:-1])]))
    return dist_sum


def read_data_from_file(fname):
    with open(fname, 'r') as f:
        lines = f.readlines()
        # czas w sekundach
        t = lines[0].split(' ')[0]
        # ilość miast
        n = lines[0].split(' ')[1]

        cost_matrix = [list(map(int, line.split(' '))) for line in lines[1:]]

        return int(t), int(n), np.array(cost_matrix)


def print_route(route):
    data_to_print = list(map(lambda x: x + 1, route))
    data_to_print.append(1)
    print(' '.join(str(city) for city in data_to_print), file=sys.stderr)

    # print(' '.join(str(city) for city in best_path), file=stderr)


if __name__ == '__main__':
    first_line = input()
    max_time, cities_count = [int(num) for num in first_line.split()]
    cities = [[int(num) for num in input().split()]
              for _ in range(cities_count)]

    best_solution, best_cost = tsp(cities, cities_count, max_time)

    print(best_cost)
    print_route(best_solution)
