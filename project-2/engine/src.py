# TODO: error handling, testing, saving and plotting, docstrings, Hessian update for quasi-newton methods


from dataclasses import dataclass
from typing import Callable, Optional, Protocol, Tuple
import numpy as np
from abc import ABC, abstractmethod

# arrays for multidimensional problems
Gradient = Callable[[np.ndarray], np.ndarray]
Hessian = Callable[[np.ndarray], np.ndarray]

class Objective(Protocol):
    def __call__(self, x: np.ndarray) -> float:
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


class OptimizationMethod(ABC):
    def __init__(self, problem: OptimizationProblem, cauchy_tol: float = 1e-5, grad_tol: float = 1e-5, max_iter: int = 1000):
        self.problem = problem
        self.cauchy_tol = cauchy_tol
        self.grad_tol = grad_tol
        self.max_iter = max_iter

    # @abstractmethod means that you have to implement this method in any subclass you want to create; i.e. this is a template method that can change between different optimization methods 
    @abstractmethod
    def get_direction(self, state: State) -> np.ndarray:
        ...


    # maybe like this to get adjustable alpha?
    @abstractmethod
    def get_alpha(self, state: State) -> float:
        ...


    @abstractmethod
    def _fd_hess(self, x: np.ndarray, h: float = 1e-5) -> np.ndarray:
        ...


    def _step(self, x: np.ndarray, s: np.ndarray, alpha: float) -> np.ndarray:
        return x + s * alpha


    def _grad(self, x: np.ndarray) -> np.ndarray:
        return self.problem.grad(x) if self.problem.grad else self._fd_grad(x)


    def _fd_grad(self, x: np.ndarray, h: float = 1e-8) -> np.ndarray:
        e_i = np.eye(len(x))
        grad = np.array([(self.problem.f(x + h * e) - self.problem.f(x - h * e)) / (2 * h) for e in e_i])
        return grad
    

    def _hess(self, x: np.ndarray) -> np.ndarray:
        return self.problem.hess(x) if self.problem.hess else self._fd_hess(x)
    

    def _cauchy_stopping(self, x: np.ndarray, x_new: np.ndarray) -> bool:
        return np.linalg.norm(x_new - x) < self.cauchy_tol


    def _gradient_stopping(self, grad: np.ndarray) -> bool:
        return np.linalg.norm(grad) < self.grad_tol


    def _init_state(self, x0: np.ndarray) -> State:
        f = self.problem.f(x0)
        grad = self._grad(x0)
        hess = self._hess(x0)
        s = np.zeros_like(x0)
        return State(x=x0, f=f, grad=grad, hess=hess, s=s, iter=0)
    

    # might have to add a hessian update step for some later methods, just add a function for that here if you need that and update optimize method
    # but keep those that don't need it invariant to that method if possible


    # for plotting; save the states
    def optimize(self, x0: np.ndarray) -> State:
        state = self.init_state(x0)

        for it in range(self.max_iter):

            if self._gradient_stopping(state.grad):
                state.converged = True
                state.iter = it
                state.converge_mode = "gradient"
                return state

            s     = self.get_direction(state)
            a     = self.get_alpha(state)
            x_new = self._step(state.x, s, a)

            f_new = self.problem.f(x_new)
            g_new = self._grad(x_new)
            H_new = self._hess(x_new)

            if self._cauchy_stopping(state.x, x_new):
                return State(x=x_new, f=f_new, grad=g_new, hess=H_new, s=s, alpha=a, iter=it+1, converged=True, converge_mode="cauchy")
            
            state = State(x=x_new, f=f_new, grad=g_new, hess=H_new, s=s, alpha=a, iter=it+1, converged=False)

        state.converged = False
        return state

