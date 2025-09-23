import numpy as np
from src import OptimizationProblem

class Rosenbrock(OptimizationProblem):
        f = lambda v: 100*(v[1] - v[0]**2)**2 + (1 - v[0])**2
        grad = lambda v: np.array([-400*v[0]*(v[1] - v[0]**2) - 2*(1 - v[0]), 200*(v[1] - v[0]**2)], dtype=float)
        hess = lambda v: np.array([[1200*v[0]**2 - 400*v[1] + 2, -400*v[0]], [-400*v[0], 200]], dtype=float)
        x0 = np.array([-1.5, 1.5], dtype=float)
