"""
Python version: 3.12.7
Author: Zhen Chen, chen.zhen5526@gmail.com
Date: 2025/2/20 16:06
Description: 
    Solve newsvendor problem by building less gurobi models.
    Only one gurobi model for each stage is created.
    This method obviously saves computational time compared with building models for each sample

-----
ini_I = 0
vari_cost = 1
unit_back_cost = 10
unit_hold_cost = 2
mean_demands = [10, 20, 10, 20, 10, 20, 10, 20]
----
218.41 for sdp optimal cost, java 0.5s;
110.97 for sdp optimal cost of 4 periods, java 0.119s;

model only for each stage:
194.46 for sddp, 58.968s on a Mac for iter number 30, sample number 20, forward scenario number 10;
211.54 for sddp, 118.37s on a Mac for iter number 30, sample number 10, forward scenario number 20;
221.30 for sddp, 162.43s on a Mac for iter number 25, sample number 10, forward scenario number 30;

model for each uncertainty:
221.48 for sddp, 162.942s on a Mac for iter number 30, sample number 20, forward scenario number 10;
233.50 for sddp, 192.09s on a Mac for iter number 30, sample number 10, forward scenario number 20;
223.95 for sddp, 276.09s on a Mac for iter number 25, sample number 10, forward scenario number 30;

Affecting SDDP performance factors:
scenario_forward_num = 30  # the bigger the better
iter_num = 25 # large enough for a stable solution, 25 for this problem

"""

import time
from gurobipy import Model, GRB, Env
from sppy.utils.sampling import Sampling, generate_scenario_paths
from typing import Any
from sppy.utils.logger import Logger

# problem settings
mean_demands = [10, 20, 10, 20, 10, 20, 10, 20]
distribution = "poisson"
T = len(mean_demands)
# T = 2  # change 1
ini_I = 0
unit_vari_costs = [1 for _ in range(T)]
unit_hold_costs = [2 for _ in range(T)]
unit_back_costs = [10 for _ in range(T)]

# sddp settings
sample_num = 10  # change 2
scenario_forward_num = 30  # change 3
iter_num = 25

sample_nums = [sample_num for t in range(T)]
# detailed samples in each period
sample_details = [[0.0 for _ in range(sample_nums[t])] for t in range(T)]
for t in range(T):
    sampling = Sampling(dist_name=distribution, mu=mean_demands[t])
    sample_details[t] = sampling.generate_samples(sample_nums[t])

# sample_details = [[5, 15], [5, 15], [15, 5], [15, 15]]  # change 4

iter_ = 0
theta_iniValue = 0  # initial theta values (profit) in each period
env = Env(params={"OutputFlag": 0})
models = [Model(env=env) for _ in range(T + 1)]
q = [Any for _ in range(T)]
theta = [Any for _ in range(T)]
I = [Any for _ in range(T)]
B = [Any for _ in range(T)]
for t in range(T + 1):
    if t < T:
        q[t] = models[t].addVar(
            vtype=GRB.CONTINUOUS, obj=unit_vari_costs[t], name="q_" + str(t + 1)
        )
        theta[t] = models[t].addVar(
            obj=1,
            lb=theta_iniValue * (T - t),
            vtype=GRB.CONTINUOUS,
            name="theta_" + str(t + 2),
        )
    if t > 0:
        I[t - 1] = models[t].addVar(
            vtype=GRB.CONTINUOUS, obj=unit_hold_costs[t - 1], name="I_" + str(t)
        )
        B[t - 1] = models[t].addVar(
            vtype=GRB.CONTINUOUS, obj=unit_back_costs[t - 1], name="B_" + str(t)
        )
        # noinspection PyTypeChecker
        models[t].addConstr(I[t - 1] - B[t - 1] == 0)
        models[t].update()


slope_1st_stage = []
intercept_1st_stage = []
# better for not fully initializing, since every iteration requiring adding
# all the available cut constraints
slopes = [[[] for n in range(scenario_forward_num)] for t in range(T - 1)]
intercepts = [[[] for n in range(scenario_forward_num)] for t in range(T - 1)]
q_values = [0.0 for _ in range(iter_num)]

import os

log_file_name = os.path.basename(__file__)
log_file_name = log_file_name.split(".")[0] + ".log"
logger = Logger(log_file_name)
logger.console_header_sddp()
start = time.process_time()
while iter_ < iter_num:
    # sample a numer of scenarios from the full scenario tree
    scenario_paths = generate_scenario_paths(scenario_forward_num, sample_nums)
    scenario_paths.sort()  # sort to mase same numbers together

    # forward
    if iter_ > 0:
        models[0].addConstr(
            theta[0] >= slope_1st_stage[-1] * q[0] + intercept_1st_stage[-1]
        )
    models[0].update()
    models[0].optimize()
    # models[0].write("iter" + str(iter_) + "_main1.lp")
    # models[0].write("iter" + str(iter_) + "_main1.sol")
    # pass

    q_values[iter_] = q[0].x
    theta_value = theta[0].x
    # z = models[0].ObjVal

    q_forward_values = [[0 for n in range(scenario_forward_num)] for t in range(T)]
    I_forward_values = [[0 for n in range(scenario_forward_num)] for t in range(T + 1)]
    B_forward_values = [[0 for n in range(scenario_forward_num)] for t in range(T + 1)]

    # forward loop
    for t in range(0, T):
        # add the cut constraints
        if iter_ > 0 and t < T - 1:
            for i in range(iter_):
                for nn in range(scenario_forward_num):
                    if abs(slopes[t][nn][i]) < 1e-3:
                        break
                    # warnings of an unexpected type by python interpreter for the below line can be ignored
                    models[t + 1].addConstr(
                        theta[t + 1]
                        >= slopes[t][nn][i] * (I[t] - B[t] + q[t + 1])
                        + intercepts[t][nn][i]
                    )

        # I think parallel computing each time can only be applied at one stage, because
        # consecutive stages have some connections.
        for n in range(scenario_forward_num):  # parallel in this loop
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
            # noinspection PyTypeChecker
            models[t + 1].setAttr("RHS", models[t + 1].getConstrs()[0], rhs)

            # optimize
            models[t + 1].optimize()
            I_forward_values[t][n] = I[t].x
            B_forward_values[t][n] = B[t].x
            if t < T - 1:
                q_forward_values[t][n] = q[t + 1].x

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

                if t == 0:
                    rhs = ini_I + q_values[iter_] - demand
                else:
                    rhs = (
                        I_forward_values[t - 1][n]
                        - B_forward_values[t - 1][n]
                        + q_forward_values[t - 1][n]
                        - demand
                    )

                # if iter_ >= 1 and t == 0 and n == 0:
                #     pass

                # noinspection PyTypeChecker
                models[t + 1].setAttr("RHS", models[t + 1].getConstrs()[0], rhs)
                # models[t + 1].update()

                # optimize
                models[t + 1].Params.OutputFlag = 0
                models[t + 1].optimize()
                # if iter_ == 1 and t == 0:
                #     models[t + 1].write(
                #         "iter"
                #         + str(iter_)
                #         + "_sub_"
                #         + str(t + 1)
                #         + "^"
                #         + str(n + 1)
                #         + "_"
                #         + str(s + 1)
                #         + "-back.lp"
                #     )
                #     pass

                pi = models[t + 1].getAttr(GRB.Attr.Pi)
                rhs = models[t + 1].getAttr(GRB.Attr.RHS)
                if t < T - 1:
                    num_con = len(pi)
                    for ss in range(1, num_con):
                        pi_rhs_values[t][n][s] += pi[ss] * rhs[ss]
                    pi_rhs_values[t][n][s] += -pi[0] * demand
                else:
                    pi_rhs_values[t][n][s] = -pi[0] * demand
                pi_values[t][n][s] = pi[0]
                # m_backward[t][n][s].dispose()

            avg_pi = sum(pi_values[t][n]) / S
            avg_pi_rhs = sum(pi_rhs_values[t][n]) / S

            if iter_ >= 1 and t == 0 and n == 0:
                pass

            # recording cuts
            if t == 0 and n == 0:
                slope_1st_stage.append(avg_pi)
                intercept_1st_stage.append(avg_pi_rhs)
            elif t > 0:
                slopes[t - 1][n].append(avg_pi)
                intercepts[t - 1][n].append(avg_pi_rhs)
    logger.console_body_sddp(iter_, models[0].ObjVal)
    iter_ += 1

end = time.process_time()
print("********************************************")
print("final expected total costs is %.2f" % models[0].ObjVal)
Q_0 = q_values[iter_ - 1]
print("ordering Q in the first period is %.2f" % Q_0)
cpu_time = end - start
print("cpu time is %.3f s" % cpu_time)

# the following codes are for outputting information in the log file
description = "This module build gurobi model for each uncertainty realization."
logger.file_header_sddp()
result_txt = ("cpu_time", "objective value", "Q_0")
results_values = (cpu_time, models[0].ObjVal, Q_0)
results = zip(result_txt, results_values)
sddp_parameters_txt = ("sample numbers", "scenario_forward_num", "iter_num")
sddp_parameters_values = (sample_nums, scenario_forward_num, iter_num)
sddp_parameters = zip(sddp_parameters_txt, sddp_parameters_values)
problem_parameters_txt = (
    "ini_I",
    "T",
    "distribution",
    "mean_demands",
    "unit_vari_costs",
    "unit_back_costs",
    "unit_hold_costs",
)
problem_parameters_values = (
    ini_I,
    T,
    distribution,
    mean_demands,
    unit_vari_costs,
    unit_back_costs,
    unit_hold_costs,
)
problem_parameters = zip(problem_parameters_txt, problem_parameters_values)
logger.file_body_sddp(description, results, sddp_parameters, problem_parameters)
