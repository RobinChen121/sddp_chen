"""
Python version: 3.12.7
Author: Zhen Chen, chen.zhen5526@gmail.com
Date: 2025/3/3 22:41
Description: 
    SDDP for single product with lead time.

"""

import time
from gurobipy import Model, GRB, Env
from sppy.utils.sampling import Sampling, generate_scenario_paths
from typing import Any

ini_I = 0
ini_cash = 0
mean_demands = [15, 15, 15, 15]  # [10, 20, 10, 20]
distribution = "poisson"
T = len(mean_demands)
unit_vari_costs = [1 for _ in range(T)]
price = [10 for _ in range(T)]
unit_salvage = 0.5
T = len(mean_demands)
overhead_cost = [50 for t in range(T)]
r0 = 0
r1 = 0.1
r2 = 2  # penalty interest rate for overdraft exceeding the limit
U = 500  # overdraft limit

if T == 4:
    opt = 167.31  # 215.48 #
else:
    opt = 26.68

# sddp settings
sample_num = 10
N = 30  # sampled number of scenarios for forward computing
# sample_num = 10 # change 2
# scenario_forward_num = 10 # change 3
iter_num = 25

sample_nums = [sample_num for t in range(T)]
# detailed samples in each period
sample_details = [[0.0 for _ in range(sample_nums[t])] for t in range(T)]
for t in range(T):
    sampling = Sampling(dist_name=distribution, mu=mean_demands[t])
    sample_details[t] = sampling.generate_samples(sample_nums[t])

iter_ = 0
env = Env(params={"OutputFlag": 0})
models = [Model(env=env) for _ in range(T + 1)]
q = [Any for _ in range(T)]
q_pre = [Any for _ in range(T - 1)]
theta = [Any for _ in range(T)]
I = [Any for _ in range(T)]
B = [Any for _ in range(T)]
cash = [Any for _ in range(T)]
W0 = [Any for _ in range(T)]
W1 = [Any for _ in range(T)]
W2 = [Any for _ in range(T)]
theta_iniValue = -500  # initial theta values (profit) in each period
for t in range(T + 1):
    # no need to change the objetive when setting up the model
    if t < T:
        q[t] = models[t].addVar(
            vtype=GRB.CONTINUOUS, obj=unit_vari_costs[t], name="q_" + str(t + 1)
        )
        W0[t] = models[t].addVar(vtype=GRB.CONTINUOUS, obj=-r0, name="W0_" + str(t + 1))
        W1[t] = models[t].addVar(vtype=GRB.CONTINUOUS, obj=r1, name="W1_" + str(t + 1))
        W2[t] = models[t].addVar(vtype=GRB.CONTINUOUS, obj=r2, name="W2_" + str(t + 1))
        theta[t] = models[t].addVar(
            obj=1,
            lb=theta_iniValue * (T - t),
            vtype=GRB.CONTINUOUS,
            name="theta_" + str(t + 2),
        )
    if t > 0:
        q_pre[t] = models[t].addVar(
            vtype=GRB.CONTINUOUS, obj=unit_vari_costs[t], name="q_pre_" + str(t + 1)
        )
        I[t - 1] = models[t].addVar(vtype=GRB.CONTINUOUS, name="I_" + str(t))
        B[t - 1] = models[t].addVar(
            vtype=GRB.CONTINUOUS, obj=price[t], name="B_" + str(t)
        )
        # noinspection PyTypeChecker
        models[t].addConstr(I[t - 1] - B[t - 1] == 0)
        models[t].addConstr(cash[t - 1] + price[t - 1] * B[t - 1] == 0)
        models[t].update()


slope1_stage = []
intercept1_stage = []
slopes1 = []
slopes2 = []
slopes3 = []
intercepts = []
q_values = []
q_pre_values = []
W0_values = []
W1_values = []
W2_values = []
while iter_ < iter_num:
    slopes1.append([[0 for n in range(N)] for t in range(T)])
    slopes2.append([[0 for n in range(N)] for t in range(T)])
    slopes3.append([[0 for n in range(N)] for t in range(T)])
    intercepts.append([[0 for n in range(N)] for t in range(T - 1)])
    q_values.append([[0 for n in range(N)] for t in range(T)])
    q_pre_values.append([[0 for n in range(N)] for t in range(T)])

    # sample a numer of scenarios from the full scenario tree
    scenario_paths = generate_scenario_paths(N, sample_nums)
    scenario_paths.sort()  # sort to make same numbers together

    # forward
    if iter_ > 0:
        models[0].addConstr(
            theta
            >= slope1_stage[-1][0] * ini_I
            + slope1_stage[-1][1]
            * (
                ini_cash
                - unit_vari_costs[0] * q[0]
                - r1 * W1[0]
                + r0 * W0[0]
                - r2 * W2[0]
            )
            + slope1_stage[-1][2] * q[0]
            + intercept1_stage[-1]
        )
        models[0].update()
    models[0].optimize()

    q_values[-1][0] = [q.x for n in range(N)]
    W0_values.append(W0.x)
    W1_values.append(W1.x)
    W2_values.append(W2.x)
    theta_value = theta.x

    I_forward_values = [[0 for n in range(N)] for t in range(T)]
    B_forward_values = [[0 for n in range(N)] for t in range(T)]
    cash_forward_values = [[0 for n in range(N)] for t in range(T)]
    W0_forward_values = [[0 for n in range(N)] for t in range(T - 1)]
    W1_forward_values = [[0 for n in range(N)] for t in range(T - 1)]
    W2_forward_values = [[0 for n in range(N)] for t in range(T - 1)]

for t in range(T):
        # add the cut constraints
        if iter_ > 0 and t < T - 1:
            for i in range(iter_):
                for nn in range(N):
                    # warnings of an unexpected type by python interpreter for the below line can be ignored
                    models[t + 1].addConstr(
                        theta[t + 1]
                        >= slopes1[i][t][nn] * (I[t] + q_pre[t])
                        + slopes2[i][t][nn]
                        * ((1 + r0) * W0[t] - (1 + r1) * W1[t] - (1 + r2) * W2[t])
                        + slopes3[i][t][nn] * q[t]
                        + intercepts[i][t][nn]
                    )

        for n in range(N):
            index = scenario_paths[n][t]
            demand = sample_details[t][index]
            if t == 0:
                rhs = ini_I + q_values[iter_] - demand
            else:
                rhs = (
                    I_forward_values[t - 1][n]
                    - B_forward_values[t - 1][n]
                    + q_forward_values[t - 1][n]
                    - demand
                )

    iter_ = iter_ + 1
