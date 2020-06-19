import operator
from grid import Grid
import math
import random
import time
import copy
import numpy as np
import sys


def ga_without_crossover(grid, initial_solution, pop_size=20, NO_IMPROVEMENT_LIMIT=50):
    time_to_end = time.time() + t

    initial_solution = grid.compress(grid.trim_route(initial_solution))
    population = get_initial_solutions(grid, 2 * pop_size - 1, initial_solution)
    population.append(initial_solution)
    best_route, best_value = initial_solution, value(grid, initial_solution)

    no_improvement = 0
    # print(pop_size)
    # print(evaluate_population(grid, population))

    while time.time() < time_to_end:
        population = list(set(population))
        evaluated = evaluate_population(grid, population)
        population = select_parents(evaluated, pop_size // 2)

        if evaluated[0][1] > best_value:
            best_route, best_value = evaluated[0]
            no_improvement = 0
            NO_IMPROVEMENT_LIMIT = min(25, len(best_route))

        while len(population) < pop_size:
            rand = random.random()
            sol = np.random.choice(population)

            if rand < 0.6:
                sol = swap_two_letters(sol)
            elif rand < 0.7:
                sol = reverse_midroute(sol)
            elif rand < 0.8:
                if len(best_route) > 20:
                    sol = shuffle_beginning(best_route,
                                            max(np.random.randint(len(best_route) // 5, len(best_route) // 5 * 2), 1))
                    sol = grid.compress(sol)
                else:
                    sol = get_random_solutions(grid, 1)[0]
            else:
                if len(best_route) > 20:
                    index = np.random.randint(len(best_route) // 5 * 2, len(best_route) // 5 * 4)
                    sol = grid.append_safe_random_walk(best_route[:index], max_extra_length=grid.m * grid.n // 4)
                else:
                    sol = get_random_solutions(grid, 1)[0]

            sol = grid.compress(sol)
            population.append(sol)

        no_improvement += 1
        if no_improvement == NO_IMPROVEMENT_LIMIT:
            # print('TIME LEFT ', time.time() - time_to_end)
            break

    return best_route, best_value


def evaluate_population(grid, population):
    ''' evaluate whole population '''
    val_pop = [(route, value(grid, route)) for route in
               population]
    sorted_val_pop = sorted(val_pop, key=operator.itemgetter(1), reverse=True)
    return sorted_val_pop


def select_parents(evaluated, n):
    ''' parents selection '''
    ev = copy.copy(evaluated)
    ev_words = [w[0] for w in ev]
    res1 = ev_words[0:2]
    ev_values = np.array([w[1] for w in ev])
    res2 = list(np.random.choice(ev_words, replace=True, size=n, p=ev_values / sum(ev_values)))
    return res1 + res2


def value(grid, solution):
    ''' evaluate single solution '''
    if solution == '':
        return 0

    val = 2 * grid.m * grid.n

    if not grid.is_safe_route(solution):
        val = val / 2
    if grid.route_exits(solution):
        val *= 1.6

    val -= len(solution)

    if val <= 0:
        return 0.0001

    return val


def get_random_solutions(grid, size):
    ''' return random solutions'''
    return [grid.compress(grid.get_safe_random_walk(max_route_length=grid.n * grid.m, start_pos=grid.initial_agent_pos))
            for i in range(size)]


def get_initial_solutions(grid, size, initial):
    ''' returns initial solutions'''
    population = []
    while len(population) < size:
        # new random route
        if random.random() < 0.85 or len(initial) < 10:
            candidate = grid.compress(
                grid.get_safe_random_walk(max_route_length=grid.n * grid.m, start_pos=grid.initial_agent_pos))
        # modification of given initial route
        else:
            index = np.random.randint(len(initial) // 5, len(initial) // 5 * 4)
            sol = grid.append_safe_random_walk(initial[:index], max_extra_length=grid.m * grid.n // 4)
            candidate = grid.compress(sol)
        population.append(candidate)
    return population


def swap_two_letters(route):
    if len(route) <= 1:
        return route
    i = random.randint(0, len(route) - 1)
    k = random.randint(0, len(route) - 1)
    route_tab = list(route)
    route_tab[i], route_tab[k] = route_tab[k], route_tab[i]
    return ''.join(route_tab)


def reverse_midroute(route):
    ''' reversing mid part of the route'''
    if len(route) <= 1:
        return route
    i = random.randint(0, len(route))
    k = random.randint(i, len(route))
    return route[:i] + route[k:i - 1:-1] + route[k + 1:]


def shuffle_beginning(route, shuffle_size):
    ''' random modifying beggining of [:shuffles_size] of the route'''
    new_route = ''
    beginning = route[:shuffle_size]
    Us, Ds, Rs, Ls = beginning.count('U'), beginning.count('D'), beginning.count('R'), beginning.count('L')
    if Us > Ds:
        new_route = new_route + (Us - Ds) * 'U'
    else:
        new_route = new_route + (Ds - Us) * 'D'
    if Rs > Ls:
        new_route = new_route + (Rs - Ls) * 'R'
    else:
        new_route = new_route + (Ls - Rs) * 'U'
    new_route = (shuffle_size // 4 + 2) * 'URDL' + new_route

    new_route = ''.join(random.sample(new_route, len(new_route))) + route[shuffle_size:]
    return new_route


if __name__ == '__main__':
    first_line = input()

    # czas i wymiary
    t, n, m = [int(word) for word in first_line.split()]
    # grid = [list(input()) for _ in range(n)]
    grid_matrix = [list(map(int, list(input().strip('\r')))) for _ in range(n)]
    initial_solution = input()
    grid = Grid(n, m, grid_matrix)

    NO_IMPROVEMENT_LIMIT = 2 * int(math.sqrt(grid.m * grid.n))
    pop_size = max(22, int(0.75 * int(math.sqrt(grid.m * grid.n))))
    best_route, best_value = ga_without_crossover(grid, initial_solution, pop_size=pop_size,
                                                  NO_IMPROVEMENT_LIMIT=NO_IMPROVEMENT_LIMIT)

    print(len(best_route))
    print(best_route, file=sys.stderr)
