"""
Python version: 3.12.7
Author: Zhen Chen, chen.zhen5526@gmail.com
Date: 2025/2/17 13:39
Description:
    Methods that solve the stochastic programming.

"""
from enum import Enum
import sppy.solve as sp


class SolveMethod(Enum):
    GENERAL = object
    PROBLEM_SPECIFIC = object


# either solve a specific problem or solve general problems
solve_method = SolveMethod.PROBLEM_SPECIFIC
if solve_method == SolveMethod.PROBLEM_SPECIFIC:
    problem_name = ''
    sp.problem_specific(problem_name)
else:
    sp.general()
