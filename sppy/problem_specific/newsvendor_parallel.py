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
from typing import Any

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
q = [Any for _ in range(T)]
theta = [Any for _ in range(T)]
I = [Any for _ in range(T + 1)]
B = [Any for _ in range(T + 1)]
for t in range(T + 1):
    if t < T:
        q[t] = models[t].addVar(
            vtype=GRB.CONTINUOUS, obj=unit_vari_costs[t], name="q_" + str(t + 1)
        )
        theta[t] = models[t].addVar(
            lb=theta_iniValue * (T - t),
            vtype=GRB.CONTINUOUS,
            name="theta_" + str(t + 2),
        )
    elif t > 0:
        I[t] = models[t].addVar(
            vtype=GRB.CONTINUOUS, obj=unit_hold_costs[t], name="I_" + str(t)
        )
        B[t] = models[t].addVar(
            vtype=GRB.CONTINUOUS, obj=unit_back_costs[t], name="B_" + str(t)
        )
        # noinspection PyTypeChecker
        models[t].addConstr(I[t] - B[t] == 0)


slope_1st_stage = []
intercept_1st_stage = []
slopes = [
    [[0.0 for _ in range(iter_num)] for n in range(scenario_forward_num)]
    for t in range(T - 1)
]
intercepts = [
    [[0.0 for _ in range(iter_num)] for n in range(scenario_forward_num)]
    for t in range(T - 1)
]
q_values = [0.0 for _ in range(iter_num)]

while iter_ < iter_num:
    # sample a numer of scenarios from the full scenario tree
    scenario_paths = generate_scenario_paths(scenario_forward_num, sample_nums)
    scenario_paths.sort()  # sort to mase same numbers together

    # forward
    if iter_ > 0:
        models[0].addConstr(theta >= slope_1st_stage[-1] * q + intercept_1st_stage[-1])
    models[0].update()
    models[0].Params.OutputFlag = 0
    models[0].optimize()
    # m.write('iter' + str(iter) + '_main1.lp')
    # m.write('iter' + str(iter) + '_main1.sol')

    q_values[iter_] = q[0].x
    theta_value = theta[0].x
    z = models[0].objVal

    q_forward_values = [[0 for n in range(scenario_forward_num)] for t in range(T)]
    I_forward_values = [[0 for n in range(scenario_forward_num)] for t in range(T + 1)]
    B_forward_values = [[0 for n in range(scenario_forward_num)] for t in range(T + 1)]

    # forward loop
    for t in range(1, T + 1):
        # the cuts
        if iter_ > 0 and t < T - 1:
            for i in range(iter_):
                for nn in range(scenario_forward_num):
                    if abs(slopes[t][nn][i]) < 1e-3:
                        break
                    models[t].addConstr(
                        theta[t]
                        >= slopes[t][nn][i] * (I[t] - B[t] + q[t])
                        + intercepts[t][nn][i]
                    )

        # I think parallel computing each time can only be applied at one stage, because
        # consecutive stages have some connections.
        for n in range(scenario_forward_num):  # parallel in this loop
            index = scenario_paths[n][t]
            demand = sample_details[t][index]
            if t == 1:
                rhs = ini_I + q_values[iter_] - demand
            else:
                rhs = (
                    I_forward_values[t - 1][n]
                    - B_forward_values[t - 1][n]
                    + q_forward_values[t - 1][n]
                    - demand
                )
            # noinspection PyTypeChecker
            models[t].setAttr("RHS", models[t].getConstrs()[0], rhs)
            models[t].update()

            # optimize
            models[t].Params.OutputFlag = 0
            models[t].optimize()
            # m_forward[t][n].write('iter' + str(iter) + '_sub_' + str(t+1) + '^' + str(n+1) + '-2.lp')

            I_forward_values[t][n] = I[t].x
            B_forward_values[t][n] = B[t].x
            if t < T - 1:
                q_forward_values[t][n] = q[t].x

    # backward loop
    theta_backward_values = [
        [[0 for s in range(sample_nums[t])] for n in range(scenario_forward_num)]
        for t in range(T)
    ]
    pi_values = [
        [[0 for s in range(sample_nums[t])] for n in range(scenario_forward_num)]
        for t in range(T)
    ]
    pi_rhs_values = [
        [[0 for s in range(sample_nums[t])] for n in range(scenario_forward_num)]
        for t in range(T)
    ]

    for t in range(T - 1, -1, -1):
        for n in range(scenario_forward_num):
            S = sample_nums[t]
            for s in range(S):
                demand = sample_details[t][s]

                if t == 1:
                    rhs = ini_I + q_values[iter_] - demand
                else:
                    rhs = (
                            I_forward_values[t - 1][n]
                            - B_forward_values[t - 1][n]
                            + q_forward_values[t - 1][n]
                            - demand
                    )
                # noinspection PyTypeChecker
                models[t].setAttr("RHS", models[t].getConstrs()[0], rhs)
                models[t].update()
