from math import cos, sqrt
import numpy as np
import time
from swarm import Swarm


# algorytm rojowy
# input 10 1 1 1 1 1 1 1 1 1 1

class PSO:
    def __init__(self, t, xs, es):
        self.t = t
        self.initial_xs = xs
        self.params = es

        self.SWARM_SIZE = 100
        self.VECTOR_SIZE = len(xs)

    def pso(self, w=0.45, c1=1.70, c2=2.49):
        time_to_end = time.time() + self.t
        # C₁ importance of personal best value
        # C₂ importance of social best value.
        # w meaning of velocity vector
        swarm = Swarm(self.SWARM_SIZE, self.VECTOR_SIZE, self.params, w, c1, c2)

        while time.time() < time_to_end:
            swarm.update_swarm()
            swarm.set_best_particle()
            #print(swarm.best_particle)

        #print("#BEST", swarm.best_particle)
        return swarm.best_particle

    def tester(self, w=0.3, c1=1.5, c2=2.0, n=1000):
        best_pso_params = []
        best_val = 100
        without_change = 0
        sigma = 0.03
        for i in range(n):
            val1 = algorithm.pso(w, c1, c2).value
            val2 = algorithm.pso(w, c1, c2).value
            val3 = algorithm.pso(w, c1, c2).value
            val4 = algorithm.pso(w, c1, c2).value
            val = (val1+val2+val3+val4) / 4
            if val < best_val:
                best_val = val
                best_pso_params = [w, c1, c2]
                without_change = 0
                sigma = 0.03
                print(best_val, best_pso_params)
            if without_change == 15:
                sigma += 0.02
            else:
                w = round(float(np.random.normal(best_pso_params[0], 0.03, 1)),5)
                c1 = round(float(np.random.normal(best_pso_params[1], 0.03, 1)),5)
                c2 = round(float(np.random.normal(best_pso_params[2], 0.03, 1)),5)
            print(w,c1,c2,val)
        return best_val, best_pso_params



if __name__ == '__main__':
    t, x1, x2, x3, x4, x5, e1, e2, e3, e4, e5 = list(map(int, input().split()))
    xs = np.array([x1, x2, x3, x4, x5])
    es = np.array([e1, e2, e3, e4, e5])
    algorithm = PSO(t, xs, es)
    best_particle = (algorithm.pso())
    # print(best_particle)

    for v in best_particle.location:
        print(v, end=" ")
    print(best_particle.value)
    # solution = sa(t, args)
    # print(*solution, value(solution))
