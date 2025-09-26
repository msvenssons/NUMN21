# TODO: error handling, testing, saving and plotting, docstrings, Hessian update for quasi-newton methods
# TODO: implement a registry pattern for the different methods so they can be called by name

from dataclasses import dataclass
from typing import Callable, Optional, Protocol, Tuple
import numpy as np
from abc import ABC, abstractmethod

class Objective(Protocol):
    def __call__(self, x: np.ndarray) -> float:
        ...

class Gradient(Protocol):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        ...

class Hessian(Protocol):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        ...


@dataclass
class State:
    x: np.ndarray
    f: float
    grad: np.ndarray
    hess: np.ndarray   
    s: np.ndarray  
    alpha: float | None = None
    iter: int = 0
    converged: bool = False
    converge_mode: str | None = None
    

@dataclass(frozen=True)
class OptimizationProblem:
    f: Objective
    grad: Gradient | None = None
    hess: Hessian | None = None
    x0 : np.ndarray | None = None


class LineSearch(ABC):
    @abstractmethod
    def get_line_alpha(self, state: State) -> float:
        ...


class OptimizationMethod(ABC):
    """
    Wrapper class for optimization methods.
    Args: 
        problem: OptimizationProblem - optimization problem to be solved
        line_search: LineSearch | None (Optional) - line search method to be used (Default: None)
        alpha_0: float - the initial step size (Default: 1.0)
        cauchy_tol: float - tolerance for the Cauchy stopping criterion (Default: 1e-5)
        grad_tol: float - tolerance for the gradient stopping criterion (Default: 1e-5)
        max_iter: int - maximum number of iterations (Default: 1000)
        verbose: bool - print iteration details (Default: False)
        hess_name: str | None (Optional) - name of the Hessian approximation method used (Default: None)
    """

    def __init__(self, problem: OptimizationProblem, line_search: LineSearch | None = None, alpha_0: float = 1.0, cauchy_tol: float = 1e-5, grad_tol: float = 1e-5, max_iter: int = 1000, verbose = False):
        self.problem = problem
        self.cauchy_tol = cauchy_tol
        self.grad_tol = grad_tol
        self.max_iter = max_iter
        self.line_search = line_search
        self.alpha_0 = alpha_0
        self.verbose = verbose
        self.hess_name: str | None = "provided"

        if problem.grad is None:
            print("Warning: No gradient provided, using gradient approximation (finite difference).")
        
        if problem.hess is None:
            self.set_hess_name()
            print(f"Warning: No Hessian provided, using Hessian approximation ({self.hess_name}).")

        assert grad_tol >= 0 and cauchy_tol >= 0, "Tolerances must be positive or 0."
        assert max_iter > 0, "Maximum number of iterations must be positive."


    # @abstractmethod means that you have to implement this method in any subclass you want to create; i.e. this is a template method that can change between different optimization methods 
    @abstractmethod
    def get_direction(self, state: State) -> np.ndarray:
        ...

    def _hess_approx(self, x: np.ndarray, h: float = 1e-5) -> np.ndarray:
        # maybe option to add name of approximation so it can be printed in the init warning?
        # overwrite this in subclass if we need a different approximation, otherwise default will be finite difference approx.
        return self._fd_hess(x, h)
    
    # bug prone implementation since it is easy to forget to change this if hess_approx has been overwritten in a subclass
    def set_hess_name(self) -> None:
        self.hess_name = "finite difference" # default name, can be changed in subclass if needed

    def get_alpha(self, state: State) -> float:
        if self.line_search is not None:
            return self.line_search.get_line_alpha(state)
        else:
            return self.alpha_0 # default to fixed step size (can be dynamic in subclass if needed)


    def _step(self, x: np.ndarray, s: np.ndarray, alpha: float) -> np.ndarray:
        return x + s * alpha


    def _grad(self, x: np.ndarray) -> np.ndarray:
        return self.problem.grad(x) if self.problem.grad else self._fd_grad(x)


    def _fd_grad(self, x: np.ndarray, h: float = 1e-8) -> np.ndarray:
        e_i = np.eye(len(x))
        grad = np.array([(self.problem.f(x + h * e) - self.problem.f(x - h * e)) / (2 * h) for e in e_i])
        return grad
    

    def _hess(self, x: np.ndarray) -> np.ndarray:
        return self.problem.hess(x) if self.problem.hess else self._hess_approx(x)
    

    def _fd_hess(self, x: np.ndarray, h: float = 1e-5) -> np.ndarray:
        """
        Approximates the Hessian matrix using finite differences (derivative of the gradient).
        """
        e_i = np.eye(len(x))
        hess = np.array([(self._fd_grad(x + h * e) - self._fd_grad(x - h * e)) / (2 * h) for e in e_i])
        hess = np.column_stack(hess)
        return 0.5*(hess + hess.T)   # symmetrizing step


    def _cauchy_stopping(self, x: np.ndarray, x_new: np.ndarray) -> bool:
        return np.linalg.norm(x_new - x) < self.cauchy_tol


    def _gradient_stopping(self, grad: np.ndarray) -> bool:
        return np.linalg.norm(grad) < self.grad_tol


    def _init_state(self, x0: np.ndarray) -> State:
        f = self.problem.f(x0)
        grad = self._grad(x0)
        hess = self._hess(x0)
        s = np.zeros_like(x0)

        if self.verbose:
            print(f"Iteration {0} \n x: {x0} \n f: {f} \n grad: {grad} \n hess: {hess} \n")
        
        return State(x=x0, f=f, grad=grad, hess=hess, s=s, iter=0)
    

    # might have to add a hessian update step for some later methods, just add a function for that here if you need that and update optimize method
    # but keep those that don't need it invariant to that method if possible


    # for plotting; save the states
    def optimize(self, x0: np.ndarray) -> State:

        state = self._init_state(x0) # needs error handling if x0 is None or not provided

        for it in range(self.max_iter):

            if self._gradient_stopping(state.grad):
                state.converged = True
                state.iter = it
                state.converge_mode = "gradient"
                return state

            s = self.get_direction(state)
            a = self.get_alpha(state)
            x_new = self._step(state.x, s, a)

            f_new = self.problem.f(x_new)
            g_new = self._grad(x_new)
            H_new = self._hess(x_new)

            new_state = State(x=x_new, f=f_new, grad=g_new, hess=H_new, s=s, alpha=a, iter=it+1, converged=False)

            if self.verbose:
                print(f"Iteration {new_state.iter} \n x: {new_state.x} \n f: {new_state.f} \n grad: {new_state.grad} \n hess: {new_state.hess} \n")

            if self._cauchy_stopping(state.x, x_new):
                new_state.converged = True
                new_state.converge_mode = "cauchy"
                return new_state
            
            state = new_state
        
        state.converged = False
        return state
