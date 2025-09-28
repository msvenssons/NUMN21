from src import LineSearch, State, OptimizationProblem 
import numpy as np


# example of a line search class
class ExactLineSearch(LineSearch):
    def get_line_alpha(self, state: State, problem : OptimizationProblem) -> float:
        a = 0.9
        itr = 0
        for i in range(1,1000):
            if(problem.f(state.x) < problem.f(state.x + a*state.s)):
                itr = i
                break
            a = a**i
        return a**itr
    
class InExactLineSearch(LineSearch):
    def get_line_alpha(self, state: State, problem : OptimizationProblem) -> float:
        alpha_k = 0
        alpha_i = 0.9
        a = 0
        b = 0
        itr = 0
        p = 0.25
        tao = 9
        sigma = 0.5
        zero = np.zeros_like(state.x)
        phi_prime_0 = state.grad.T @ state.s
        my = (0 - problem.f(zero))/(p*phi_prime_0)
        
        
        for i in range(1,1000):
            if(problem.f(state.x) >= problem.f(state.x + alpha_i*state.s)):
                itr = i
                break
            if(problem.f(state.x + alpha_i*state.s) > problem.f(zero) + p*alpha_i*phi_prime_0 or problem.f(state.x + alpha_i*state.s) >=problem.f(state.x + alpha_k*state.s)):
                a = alpha_k
                b = alpha_i
            if (abs(self._grad(state.x + alpha_i*state.s))<= - sigma*phi_prime_0):
                break
            if(self._grad(state.x + alpha_i*state.s) > 0):
                a = alpha_i
                b = alpha_k
            if my <= 2*alpha_i - alpha_k:
                alpha_i = my
            else:
                alpha_i =np.mean((2*alpha_i - alpha_k), min(my, alpha_i + tao*(alpha_i - alpha_k)))
            alpha_k = alpha_i
            alpha_i = alpha_i**i
        return alpha_i
