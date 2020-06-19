from particle import Particle
import random
import numpy as np
import copy


class Swarm:
    def __init__(self, swarm_size, vector_size, params, w, c1, c2):
        self.swarm_size = swarm_size
        self.vector_size = vector_size
        self.params = params
        self.w = w
        self.c1 = c1
        self.c2 = c2

        self.particles = self.initial_swarm()
        self.best_particle = self.particles[0]
        self.set_best_particle()

    def initial_swarm(self):
        return [Particle(
            np.array([random.uniform(0, 5) for i in range(self.vector_size)]),
            np.array([1 for i in range(self.vector_size)]),
            self.params
        ) for j in range(self.swarm_size)]

    def set_best_particle(self):
        for particle in self.particles:
            if particle.best_value < self.best_particle.value:
                self.best_particle = copy.deepcopy(particle)

    def update_swarm(self):
        for particle in self.particles:
            particle.update(self.w, self.c1, self.c2, self.best_particle)
