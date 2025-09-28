from src import OptimizationMethod, OptimizationProblem, State
from linesearch import ExactLineSearch, InExactLineSearch
import numpy as np
from objectives import Rosenbrock, Chebyquad

# this can be combined with normal Newton method since they are very similar
class QuasiNewton(OptimizationMethod):
   def get_direction(self, state: State) -> np.ndarray:
       if self.hess_update not in (None, "good broyden", "symmetric broyden"):
            # quasi-Newton: state.hess is inverse Hessian (B_k)
            return -state.hess @ state.grad
       else:
            # Newton: state.hess is Hessian (H_k)
            try:
                return np.linalg.solve(state.hess, -state.grad)
            except np.linalg.LinAlgError:
                return np.linalg.lstsq(state.hess, -state.grad, rcond=None)[0]

        
   def get_alpha(self, state: State) -> float:
        return 1.0  # fixed step size
    
   def _hess_approx(self, x: np.ndarray, h: float = 1e-5) -> np.ndarray:
        return self._fd_hess(x, h)
    

if __name__ == "__main__":

    quasinewton = QuasiNewton(Rosenbrock, cauchy_tol=1e-6, grad_tol=1e-6, max_iter=100)

    result = quasinewton.optimize(Rosenbrock.x0)

    print(f"---Before Quasi Newton---\n{np.linalg.inv(result.hess)}\n")

    Rosenbrock.grad = None # to test finite difference gradient
    Rosenbrock.hess = None # to test finite difference hessian
    Rosenbrock.x0 = np.array([-1.2, 1.0], dtype=float) # change initial guess if needed

    """Good Broyden"""
    quasinewton = QuasiNewton(Rosenbrock, line_search=ExactLineSearch(), cauchy_tol=1e-6, grad_tol=1e-6, max_iter=1000, hess_update="good broyden")
    result = quasinewton.optimize(Rosenbrock.x0)
    print(f"---Good Broyden---\n{result.x}\n {result.iter}\n")

    """Bad Broyden"""
    quasinewton = QuasiNewton(Rosenbrock, cauchy_tol=1e-6, grad_tol=1e-6, max_iter=1000, hess_update="bad broyden")
    result = quasinewton.optimize(Rosenbrock.x0)
    print(f"---Bad Broyden---\n{result.hess}\n")

    """Symmetric Broyden"""
    quasinewton = QuasiNewton(Rosenbrock, cauchy_tol=1e-6, grad_tol=1e-6, max_iter=1000, hess_update="symmetric broyden")
    result = quasinewton.optimize(Rosenbrock.x0)
    print(f"---Symmetric Broyden---\n{result.hess}\n")

    """DFP rank-2"""
    quasinewton = QuasiNewton(Rosenbrock, cauchy_tol=1e-6, grad_tol=1e-6, max_iter=1000, hess_update="dfp2")
    result = quasinewton.optimize(Rosenbrock.x0)
    print(f"---DFP rank-2---\n{result.hess}\n")

    """BFGS rank-2"""
    quasinewton = QuasiNewton(Rosenbrock, cauchy_tol=1e-6, grad_tol=1e-6, max_iter=1000, hess_update="bfgs2")
    result = quasinewton.optimize(Rosenbrock.x0)
    print(f"---BFGS rank-2---\n{result.hess}\n")




   
