from src import LineSearch, State 
import numpy as np


# example of a line search class
class ExactLineSearch(LineSearch):
    def get_line_alpha(self, state: State) -> float:
        pass
