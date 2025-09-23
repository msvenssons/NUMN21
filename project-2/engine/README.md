* Initialize the new method, e.g. "Newton", by having it inherit from the OptimizationMethod class
* Give the class a "get_direction", "get_alpha", and "_fd_hess" method which differs depending on the optimizer
* Initialize the problem using the OptimizationProblem class (the objective function and optionally its exact gradient and Hessian)
* If exact gradient or Hessian is not provided, the gradient will be computed using a finite difference and the Hessian using your _fd_hess function
* Then use method.optimize and it should run the optimizer for that algorithm
* The results are saved in "states" which are objects containing x position, gradient, Hessian, etc. at a particular iteration
