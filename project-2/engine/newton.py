from src import OptimizationMethod, OptimizationProblem, State
from objectives import Rosenbrock
import numpy as np

class Newton(OptimizationMethod):
    def get_direction(self, state: State) -> np.ndarray:
        try:
            s = np.linalg.solve(state.hess, -state.grad) # solve Hs = -g, giving us the search direction s
        except np.linalg.LinAlgError:
            s = np.linalg.lstsq(state.hess, -state.grad, rcond=None)[0]  # use least squares if Hessian is singular
        return s
    
    def get_alpha(self, state: State) -> float:
        return 1.0  # fixed step size
    
    def _hess_approx(self, x: np.ndarray, h: float = 1e-5) -> np.ndarray:
        return self._fd_hess(x, h)


if __name__ == "__main__":
    Rosenbrock.grad = None # to test finite difference gradient
    Rosenbrock.hess = None # to test finite difference hessian
    Rosenbrock.x0 = np.array([-1.2, 1.0], dtype=float) # change initial guess if needed

    newton = Newton(Rosenbrock, cauchy_tol=1e-6, grad_tol=1e-6, max_iter=1000)

    result = newton.optimize(Rosenbrock.x0)

    print("Converged:", result.converged)
    print("Iterations:", result.iter)
    print("x* =", result.x)
    print("f(x*) =", result.f)
    print("Convergence mode:", result.converge_mode)


    """
    optional way to do it without using a predefined class:
    def f(x: np.ndarray) -> float:
        return 100*(x[1] - x[0]**2)**2 + (1 - x[0])**2

    def grad(x: np.ndarray) -> np.ndarray:
        return np.array([-400*x[0]*(x[1] - x[0]**2) - 2*(1 - x[0]), 200*(x[1] - x[0]**2)], dtype=float)

    def hess(x: np.ndarray) -> np.ndarray:
        return np.array([[1200*x[0]**2 - 400*x[1] + 2, -400*x[0]], [-400*x[0], 200]], dtype=float)

    problem = OptimizationProblem(f=f, grad=grad, hess=hess)

    newton = Newton(problem, cauchy_tol=1e-6, grad_tol=1e-6, max_iter=1000)
    x0 = np.array([1.0, 3.0], dtype=float)

    result = newton.optimize(x0)
    """


# maybe create some sort of wrapper that turns input into np.ndarray if it is not already?





