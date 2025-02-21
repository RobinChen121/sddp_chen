"""
Python version: 3.12.7
Author: Zhen Chen, chen.zhen5526@gmail.com
Date: 2025/2/20 16:06
Description: 
    Solve newsvendor problem by paralleling computing SDDP

"""

import time
from gurobipy import Model, GRB
from sppy.utils.sampling import Sampling, generate_scenario_paths

# problem settings
mean_demands = [10, 20, 10, 20]
distribution = "poisson"
T = len(mean_demands)
ini_I = 0
unit_vari_costs = [1 for _ in range(T)]
unit_back_costs = [10 for _ in range(T)]
unit_hold_costs = [2 for _ in range(T)]

# sddp settings
sample_num = 10
iter_num = 10
scenario_forward_num = 10  # sampled number of scenarios for forward computing

sample_nums = [sample_num for t in range(T)]
# detailed samples in each period
sample_details = [[0.0 for _ in range(sample_nums[t])] for t in range(T)]
for t in range(T):
    sampling = Sampling(dist_name=distribution, mu=mean_demands[t])
    sample_details[t] = sampling.generate_samples(sample_nums[t])

iter_ = 0
theta_iniValue = 0  # initial theta values (profit) in each period
models = [Model() for _ in range(T + 1)]
q = models[0].addVar(vtype=GRB.CONTINUOUS, name="q_1")
theta = models[0].addVar(lb=theta_iniValue * T, vtype=GRB.CONTINUOUS, name="theta_2")
models[0].setObjective(unit_vari_costs[0] * q + theta, GRB.MINIMIZE)
