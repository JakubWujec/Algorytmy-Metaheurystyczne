import numpy as np


# tabu ma stałą długość
class SolutionTabu():
    def __init__(self, elem_size, tabu_size):
        self.elem_size = elem_size
        self.size = tabu_size
        self.tabu_array = np.array([np.array([0 for j in range(self.elem_size)]) for i in range(tabu_size)])

    def add_solution(self, solution):
        d = np.vstack((self.tabu_array, np.array([solution])))
        self.tabu_array = d

    def contains(self, solution):
        k = False
        for sol in self.tabu_array:
            if np.array_equal(sol, solution):
                k = True
        return k

    def remove_oldest(self):
        self.tabu_array = self.tabu_array[1:]



