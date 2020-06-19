import copy
import numpy as np
import random
from fieldEnum import Field


class Grid:
    def __init__(self, n, m, grid_matrix):
        self.n = n
        self.m = m
        self.grid_matrix = grid_matrix
        self.initial_grid_matrix = copy.deepcopy(self.grid_matrix)

        self.exits = self._find_exits()
        self.initial_agent_pos = self._find_agent()
        self.agent_pos = np.copy(self.initial_agent_pos)  # [r,c]
        self.steps = {'U': np.array([-1, 0]), 'D': np.array([1, 0]), 'R': np.array([0, 1]), 'L': np.array([0, -1])}

    def _find_agent(self):
        ''' returns location of agent on the map '''

        # musze wyszukac agenta aby moc go przesunac
        for r, row in enumerate(self.grid_matrix):
            for c, col in enumerate(row):
                if col == Field.AGENT.value:
                    return np.array([r, c])

        print("Error 404: agent '5' not found")
        exit(0)

    def _find_exits(self):
        ''' returns location of all exits on the map'''
        exits = []
        # musze wyszukac wyjscia aby móc sprawdzac czy dotarl
        for r, row in enumerate(self.grid_matrix):
            for c, col in enumerate(row):
                if col == Field.EXIT.value:
                    exits.append(np.array([r, c]))

        if not exits:
            print("Error 404: exit '8' not found")
            exit(0)
        return exits

    def reset(self):
        ''' returns to the initial grid '''
        self.grid_matrix = copy.deepcopy(self.initial_grid_matrix)
        self.agent_pos = copy.copy(self.initial_agent_pos)

    def is_safe_step(self, step, actual_position=None):
        ''' checks if given step from actual_position is safe '''
        if actual_position is None:
            actual_position = copy.copy(self.agent_pos)

        actual_field = self.initial_grid_matrix[actual_position[0]][actual_position[1]]
        desired_position = actual_position + self.steps[step]
        desired_field = self.initial_grid_matrix[desired_position[0]][desired_position[1]]

        # out of map check
        if desired_position[0] < 0 or desired_position[0] >= self.n:
            return False
        if desired_position[1] < 0 or desired_position[1] >= self.m:
            return False

        # wall check
        if desired_field == Field.WALL.value:
            return False

        # already inside tunnel check
        if actual_field == Field.VERTICAL_TUNNEL.value and step not in ['U', 'D']:
            return False
        if actual_field == Field.HORIZONTAL_TUNNEL.value and step not in ['R', 'L']:
            return False

        # wrong tunnel enter check
        if desired_field == Field.VERTICAL_TUNNEL.value and step not in ['U', 'D']:
            return False
        if desired_field == Field.HORIZONTAL_TUNNEL.value and step not in ['R', 'L']:
            return False

        return True

    def is_safe_route(self, route, start_position=None):
        ''' checks if whole route from start_position is safe
            route is also safe when it reaches exit field before executing all steps
        '''
        if start_position is None:
            start_position = self.initial_agent_pos
        position = np.copy(start_position)
        for step in route:
            if not self.is_safe_step(step, position):
                return False
            position += self.steps[step]
            # check if exits
            if self.initial_grid_matrix[position[0]][position[1]] == Field.EXIT.value:
                return True
        return True

    def end_position(self, route, start_position=None):
        ''' returns end_position of route starting at start_position '''
        if start_position is None:
            start_position = self.initial_agent_pos
        start_position = np.copy(start_position)
        end_position = start_position + self.steps['U'] * route.count('U') + self.steps['L'] * route.count('L') + self.steps['D'] * route.count('D') + self.steps['R'] * route.count('R')
        return end_position

    def get_safe_random_walk(self, max_route_length, start_pos=None, exit_ends_walk=True):
        ''' returns compressed safe random route with length < max_route_length
            exit_end_walk - determines if walking into EXIT field finish the route '''
        if start_pos is None:
            start_pos = self.initial_agent_pos
        position = copy.copy(start_pos)
        route = ''
        if exit_ends_walk:
            for step_index in range(max_route_length):
                neighbourhood = self.get_neighbourhood_UDRL_fields_of_position(position)
                if Field.EXIT.value in neighbourhood:
                    route += 'UDRL'[neighbourhood.index(Field.EXIT.value)]
                    return route
                else:
                    step = random.choice('UDRL')
                    while not self.is_safe_step(step, actual_position=position):
                        step = 'UDRL'[('UDRL'.index(step)+1) % 4]
                    route += step
                    position += self.steps[step]
        return route

    def append_safe_random_walk(self, route_prefix, max_extra_length):
        ''' add < max_extra_length random steps to given route_prefix '''
        end_prefix = self.end_position(route_prefix)
        extra = self.get_safe_random_walk(max_extra_length, start_pos=end_prefix)
        new_route = route_prefix + extra
        return new_route

    def position_in_bounds(self, position):
        """ check if position is in bounds"""
        if position[0] < 0 or position[0] >= self.n:
            return False
        if position[1] < 0 or position[1] >= self.m:
            return False
        return True

    def get_neighbourhood_UDRL_fields_of_position(self, position):
        ''' return list of neigbourhood (UDRL) of given position  '''
        if not self.position_in_bounds(position):
            'TODO przypadek w ktorym wejdzie w 8 na scianie nawali'
            return []
        return [self.initial_grid_matrix[position[0] - 1][position[1]],
                self.initial_grid_matrix[position[0] + 1][position[1]],
                self.initial_grid_matrix[position[0]][position[1] + 1],
                self.initial_grid_matrix[position[0]][position[1] - 1]]

    def compress(self, route):
        ''' removes redundant pieces of a route '''
        carpe_diem = True
        while carpe_diem:
            route = route.replace('URDL', '').replace('ULDR', '').replace('RULD', '').replace('RDLU', '').replace('DLUR','').replace('DRUL','').replace('LURD','').replace('LDRU','')
            if 'UD' in route:
                route = route.replace('UD', '')
            elif 'DU' in route:
                route = route.replace('DU', '')
            elif 'LR' in route:
                route = route.replace('LR', '')
            elif 'RL' in route:
                route = route.replace('RL', '')
            else:
                carpe_diem = False
        return route

    def distances_from_exit(self, pos):
        ''' return euclidean distances pos from exits  '''
        int_pos = [int(pos[0]), int(pos[1])]
        dists = [np.linalg.norm(np.array(int_pos) - np.array(ex)) for ex in self.exits]
        return dists

    def min_distance_from_exit(self, pos):
        ''' returns min distance to exit'''
        return min(self.distances_from_exit(pos))

    def trim_route(self, route):
        ''' trim route to be feasible '''
        for i in range(len(route)):
            if self.route_exits(route):
                return route
            route = route[:-1]
        return route

    def route_exits(self, route):
        ''' checks if route exits '''
        end_pos = self.end_position(route)
        return self.min_distance_from_exit(end_pos) == 0


def safe_route_tests(grid, safe_routes):
    for route in safe_routes:
        if not grid.is_safe_route(route):
            print("FAIL: ", route)
        else:
            print("GOOD")
        print(grid.end_position(route))


if __name__ == '__main__':
    first_line = input()
    t, n, m = [int(word) for word in first_line.split()]
    grid_matrix = [list(map(int, list(input().strip('\r')))) for _ in range(n)]
    initial_solution = input()
    grid = Grid(n, m, grid_matrix)
    grid.reset()
