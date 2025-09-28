# TODO: error handling, testing, saving and plotting, docstrings, Hessian update for quasi-newton methods
# TODO: implement a registry pattern for the different methods so they can be called by name

from dataclasses import dataclass
from typing import Callable, Optional, Protocol, Tuple
import warnings
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
    def get_line_alpha(self, state: State, problem : OptimizationProblem) -> float:
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

    def __init__(self, problem: OptimizationProblem, line_search: LineSearch | None = None, alpha_0: float = 1.0, cauchy_tol: float = 1e-5, grad_tol: float = 1e-5, max_iter: int = 1000, hess_update: str | None = None, verbose = False):
        self.problem = problem
        self.cauchy_tol = cauchy_tol
        self.grad_tol = grad_tol
        self.max_iter = max_iter
        self.line_search = line_search
        self.alpha_0 = alpha_0
        self.hess_update = hess_update
        self.verbose = verbose

        if problem.grad is None:
            print("Warning: No gradient provided, using gradient approximation (finite difference).")
        
        if problem.hess is None:
            print(f"Warning: No Hessian provided, using Hessian approximation.")

        if grad_tol < 0 or cauchy_tol < 0:
            raise ValueError("Tolerances must be positive.")

        if max_iter <= 0:
            raise ValueError("Maximum number of iterations must be positive.")


    # @abstractmethod means that you have to implement this method in any subclass you want to create; i.e. this is a template method that can change between different optimization methods 
    @abstractmethod
    def get_direction(self, state: State) -> np.ndarray:
        ...


    def _hess_approx(self, x: np.ndarray, h: float = 1e-5) -> np.ndarray:
        # maybe option to add name of approximation so it can be printed in the init warning?
        # overwrite this in subclass if we need a different approximation, otherwise default will be finite difference approx.
        return self._fd_hess(x, h)
    

    def _good_broyden_update(self, G: np.ndarray, delta: np.ndarray, gamma: np.ndarray, damp=0.3) -> np.ndarray:
        delta = delta.reshape(-1, 1)
        gamma = gamma.reshape(-1, 1)
        denom = delta.T @ G @ gamma
        if abs(denom) < 1e-12:
            print("Too small denominator, skipping update")
            return G
        update = ((delta - G @ gamma) @ (delta.T @ G)) / denom
        G += damp * update 
        return G
     
     
    def _bad_broyden_update(self, H: np.ndarray, delta: np.ndarray, gamma: np.ndarray, damp=0.3):
        delta = delta.reshape(-1, 1)
        gamma = gamma.reshape(-1, 1)
        denom = gamma.T @ gamma
        if abs(denom) < 1e-12:
            print("Too small denominator, skipping update")
            return H
        update = ((delta - H @ gamma) @ gamma.T) / denom
        H += damp * update  # in-place
        return H
    

    def _sym_broyden_update(self, G: np.ndarray, delta: np.ndarray, gamma: np.ndarray, damp=0.3) -> np.ndarray:
        delta = delta.reshape(-1,1)
        gamma = gamma.reshape(-1,1)
        diff = delta - G @ gamma
        denom = float(diff.T @ gamma)

        if abs(denom) < 1e-12:
            print("Too small denominator, skipping update")
            return G
        
        update = (diff @ diff.T) / denom
        G += damp * update
        return G
    

    def _dfp_update(self, H: np.ndarray, delta: np.ndarray, gamma: np.ndarray, damp=0.7) -> np.ndarray:
        delta = delta.reshape(-1,1)
        gamma = gamma.reshape(-1,1)
        denom_1 = delta.T @gamma
        denom_2 = gamma.T @ H @ gamma

        if abs(denom_1) < 1e-12 or abs(denom_2) < 1e-12:
            print("Too small denominator, skipping update")
            return H


        term1 = np.outer(delta, delta) / denom_1
        term2 = H @ np.outer(gamma, gamma) @ H / denom_2
        update = term1 - term2
        H += damp * update
        return H
    

    def _bfgs_update(self, H: np.ndarray, delta: np.ndarray, gamma: np.ndarray, damp = 1.0) -> np.ndarray:
        delta = delta.reshape(-1,1)
        gamma = gamma.reshape(-1,1)
        denom = delta.T @ gamma

        if denom < 1e-12:
            print("Too small denominator, skipping update")
            return H
        
        Hgamma = H @ gamma
        term1 = (1 + (gamma.T @ Hgamma) / denom) * (delta @ delta.T) / denom
        term2 = (delta @ (gamma.T @ H) + Hgamma @ delta.T) / denom
        update = term1 - term2
        H += damp * update
        return H


    def get_alpha(self, state: State) -> float:
        if self.line_search is not None:
            return self.line_search.get_line_alpha(self, state, self.problem)
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
        if self.hess_update:
        # Quasi-Newton: start with identity matrix
            hess = np.eye(len(x0))
        else:
            hess = self._hess(x0)
        
        f = self.problem.f(x0)
        grad = self._grad(x0)
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

            delta = x_new - state.x
            gamma = g_new - state.grad


            if self.hess_update == "good broyden":
                H_new = self._good_broyden_update(state.hess, delta, gamma)
            elif self.hess_update == "bad broyden":
                H_new = self._bad_broyden_update(state.hess, delta, gamma)
            elif self.hess_update == "symmetric broyden":
                H_new = self._sym_broyden_update(state.hess, delta, gamma)
            elif self.hess_update == "dfp2":
                H_new = self._dfp_update(state.hess, delta, gamma)
            elif self.hess_update == "bfgs2":
                H_new = self._bfgs_update(state.hess, delta, gamma)
            else:
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
