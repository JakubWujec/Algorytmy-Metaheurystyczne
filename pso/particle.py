import random
import numpy as np


class Particle:
    def __init__(self, location, velocity, params):
        self.n = len(location)
        self.location = location
        self.velocity = velocity
        self.params = params

        self.value = self.yang()
        self.best_location = self.location
        self.best_value = self.value

    def __repr__(self):
        return f'loc: {self.location}, val: {self.yang()}'

    def update(self, w, c1, c2, best_ever_in_swarm):
        self.update_particle_velocity(w, c1, c2, best_ever_in_swarm)
        self.update_particle_location()

        self.value = self.yang()
        if self.value < self.best_value:
            self.best_location = np.copy(self.location)
            self.best_value = self.value

    def update_particle_velocity(self, w, c1, c2, best_particle_in_swarm):
        self.velocity = w * self.velocity + c1*random.random() * (self.best_location - self.location) \
                        + c2 * random.random() * (best_particle_in_swarm.best_location - self.location)
        # print(self.velocity)

    def update_particle_location(self):
        self.location = self.location + self.velocity

    def yang(self):
        # assert abs(max(self.location, key=abs)) <= 5
        result = np.sum([self.params[i] * pow(abs(self.location[i]), i+1) for i in range(self.n)])
        return result



