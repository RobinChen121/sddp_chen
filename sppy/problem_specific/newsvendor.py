"""
Python version: 3.12.7
Author: Zhen Chen, chen.zhen5526@gmail.com
Date: 2025/2/17 13:39
Description:
    SDDP codes to solve the classic multi-stage newsvendor problem
    in which the parameters are: unit_vari_cost, unit_back_cost, unit_hold_cost.

-----
ini_I = 0
vari_cost = 1
unit_back_cost = 10
unit_hold_cost = 2
mean_demands = [10, 20, 10, 20, 10, 20, 10, 20]
----
218.41 for sdp optimal cost, java 0.5s;

"""

import time
from gurobipy import Model, GRB
from sppy.utils.sampling import Sampling, generate_scenario_paths
from sppy.utils.logger import Logger

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
m = Model()  # linear model in the first stage
# decision variable in the first stage model
q = m.addVar(vtype=GRB.CONTINUOUS, name="q_1")
theta = m.addVar(lb=theta_iniValue * T, vtype=GRB.CONTINUOUS, name="theta_2")
m.setObjective(unit_vari_costs[0] * q + theta, GRB.MINIMIZE)

slope_1st_stage = []
intercept_1st_stage = []
slopes = [[[] for n in range(scenario_forward_num)] for t in range(T - 1)]
intercepts = [[[] for n in range(scenario_forward_num)] for t in range(T - 1)]
q_values = [0.0 for _ in range(iter_num)]
q_sub_values = [
    [[0.0 for n in range(scenario_forward_num)] for t in range(T - 1)]
    for _ in range(iter_num)
]

z = 0
log_file_name = "newsvendor.log"
logger = Logger(log_file_name)
logger.console_header_sddp()
start = time.process_time()
while iter_ < iter_num:
    # sample a numer of scenarios from the full scenario tree
    scenario_paths = generate_scenario_paths(scenario_forward_num, sample_nums)
    scenario_paths.sort()  # sort to make same numbers together

    # forward
    if iter_ > 0:
        m.addConstr(theta >= slope_1st_stage[-1] * q + intercept_1st_stage[-1])
    m.update()
    m.Params.OutputFlag = 0
    m.optimize()
    # m.write('iter' + str(iter) + '_main1.lp')
    # m.write('iter' + str(iter) + '_main1.sol')

    q_values[iter_] = q.x
    theta_value = theta.x
    z = m.objVal

    m_forward = [[Model() for n in range(scenario_forward_num)] for t in range(T)]
    q_forward = [
        [
            m_forward[t][n].addVar(
                vtype=GRB.CONTINUOUS, name="q_" + str(t + 2) + "^" + str(n)
            )
            for n in range(scenario_forward_num)
        ]
        for t in range(T - 1)
    ]
    I_forward = [
        [
            m_forward[t][n].addVar(
                vtype=GRB.CONTINUOUS, name="I_" + str(t + 1) + "^" + str(n)
            )
            for n in range(scenario_forward_num)
        ]
        for t in range(T)
    ]
    # B is the quantity of lost sale
    B_forward = [
        [
            m_forward[t][n].addVar(
                vtype=GRB.CONTINUOUS, name="B_" + str(t + 1) + "^" + str(n)
            )
            for n in range(scenario_forward_num)
        ]
        for t in range(T)
    ]
    theta_forward = [
        [
            m_forward[t][n].addVar(
                lb=-theta_iniValue * (T - 1 - t),
                vtype=GRB.CONTINUOUS,
                name="theta_" + str(t + 3) + "^" + str(n),
            )
            for n in range(scenario_forward_num)
        ]
        for t in range(T - 1)
    ]

    q_forward_values = [[0 for n in range(scenario_forward_num)] for t in range(T - 1)]
    I_forward_values = [[0 for n in range(scenario_forward_num)] for t in range(T)]
    B_forward_values = [[0 for n in range(scenario_forward_num)] for t in range(T)]
    theta_forward_values = [[0 for n in range(scenario_forward_num)] for t in range(T)]

    # forward loop
    for t in range(T):
        for n in range(scenario_forward_num): # parallel in this loop
            index = scenario_paths[n][t]
            demand = sample_details[t][index]

            if iter_ > 0 and t < T - 1:
                for i in range(iter_):
                    for nn in range(scenario_forward_num):  #
                        if abs(slopes[t][nn][i]) < 1e-3:
                            break
                        m_forward[t][n].addConstr(
                            theta_forward[t][n]
                            >= slopes[t][nn][i]
                            * (I_forward[t][n] - B_forward[t][n] + q_forward[t][n])
                            + intercepts[t][nn][i]
                        )

            if t == T - 1:
                m_forward[t][n].setObjective(
                    unit_hold_costs[t] * I_forward[t][n]
                    + unit_back_costs[t] * B_forward[t][n],
                    GRB.MINIMIZE,
                )
            else:
                m_forward[t][n].setObjective(
                    unit_vari_costs[t] * q_forward[t][n]
                    + unit_hold_costs[t] * I_forward[t][n]
                    + unit_back_costs[t] * B_forward[t][n]
                    + theta_forward[t][n],
                    GRB.MINIMIZE,
                )
            if t == 0:
                m_forward[t][n].addConstr(
                    I_forward[t][n] - B_forward[t][n]
                    == ini_I + q_values[iter_] - demand
                )
            else:
                m_forward[t][n].addConstr(
                    I_forward[t][n] - B_forward[t][n]
                    == I_forward_values[t - 1][n]
                    - B_forward_values[t - 1][n]
                    + q_forward_values[t - 1][n]
                    - demand
                )

            # optimize
            m_forward[t][n].Params.OutputFlag = 0
            m_forward[t][n].optimize()
            # m_forward[t][n].write('iter' + str(iter) + '_sub_' + str(t+1) + '^' + str(n+1) + '-2.lp')

            I_forward_values[t][n] = I_forward[t][n].x
            B_forward_values[t][n] = B_forward[t][n].x
            if t < T - 1:
                q_forward_values[t][n] = q_forward[t][n].x
                q_sub_values[iter_][t][n] = q_forward[t][n].x
                theta_forward_values[t][n] = theta_forward[t][n].x
            # m_forward[t][n].dispose()

    # backward
    m_backward = [
        [[Model() for s in range(sample_nums[t])] for n in range(scenario_forward_num)]
        for t in range(T)
    ]
    q_backward = [
        [
            [
                m_backward[t][n][s].addVar(
                    vtype=GRB.CONTINUOUS, name="q_" + str(t + 2) + "^" + str(n)
                )
                for s in range(sample_nums[t])
            ]
            for n in range(scenario_forward_num)
        ]
        for t in range(T - 1)
    ]
    I_backward = [
        [
            [
                m_backward[t][n][s].addVar(
                    vtype=GRB.CONTINUOUS, name="I_" + str(t + 1) + "^" + str(n)
                )
                for s in range(sample_nums[t])
            ]
            for n in range(scenario_forward_num)
        ]
        for t in range(T)
    ]
    # B is the quantity of lost sale
    B_backward = [
        [
            [
                m_backward[t][n][s].addVar(
                    vtype=GRB.CONTINUOUS, name="B_" + str(t + 1) + "^" + str(n)
                )
                for s in range(sample_nums[t])
            ]
            for n in range(scenario_forward_num)
        ]
        for t in range(T)
    ]
    theta_backward = [
        [
            [
                m_backward[t][n][s].addVar(
                    lb=-theta_iniValue * (T - 1 - t),
                    vtype=GRB.CONTINUOUS,
                    name="theta_" + str(t + 3) + "^" + str(n),
                )
                for s in range(sample_nums[t])
            ]
            for n in range(scenario_forward_num)
        ]
        for t in range(T - 1)
    ]

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

    # it is better t in the first loop
    # backward loop
    for t in range(T - 1, -1, -1):
        for n in range(scenario_forward_num):
            S = sample_nums[t]
            for s in range(S):
                demand = sample_details[t][s]

                if iter_ > 0 and t < T - 1:
                    for i in range(iter_):
                        for nn in range(scenario_forward_num):  # scenario_forward_num
                            if abs(slopes[t][nn][i]) < 1e-3:
                                break
                            m_backward[t][n][s].addConstr(
                                theta_backward[t][n][s]
                                >= slopes[t][nn][i]
                                * (
                                    I_backward[t][n][s]
                                    - B_backward[t][n][s]
                                    + q_backward[t][n][s]
                                )
                                + intercepts[t][nn][i]
                            )

                if t == T - 1:
                    m_backward[t][n][s].setObjective(
                        unit_hold_costs[t] * I_backward[t][n][s]
                        + unit_back_costs[t] * B_backward[t][n][s],
                        GRB.MINIMIZE,
                    )
                else:
                    m_backward[t][n][s].setObjective(
                        unit_vari_costs[t] * q_backward[t][n][s]
                        + unit_hold_costs[t] * I_backward[t][n][s]
                        + unit_back_costs[t] * B_backward[t][n][s]
                        + theta_backward[t][n][s],
                        GRB.MINIMIZE,
                    )
                if t == 0:
                    m_backward[t][n][s].addConstr(
                        I_backward[t][n][s] - B_backward[t][n][s]
                        == ini_I + q_values[iter_] - demand
                    )
                else:
                    m_backward[t][n][s].addConstr(
                        I_backward[t][n][s] - B_backward[t][n][s]
                        == I_forward_values[t - 1][n]
                        - B_forward_values[t - 1][n]
                        + q_forward_values[t - 1][n]
                        - demand
                    )

                # optimize
                m_backward[t][n][s].Params.OutputFlag = 0
                m_backward[t][n][s].optimize()
                # if t == 0 and n == 0 and iter > 0:
                #     m_backward[t][n][s].write('iter' + str(iter) + '_sub_' + str(t+1) + '^' + str(n+1) + '_' + str(s+1) +'-2back.lp')
                # if t > 0:
                #     m_backward[t][n][s].write('iter' + str(iter) + '_sub_' + str(t+1) + '^' + str(n+1) + '_' + str(s+1) +'-2back.lp')

                pi = m_backward[t][n][s].getAttr(GRB.Attr.Pi)
                rhs = m_backward[t][n][s].getAttr(GRB.Attr.RHS)
                if t < T - 1:
                    num_con = len(pi)
                    for ss in range(num_con - 1):
                        pi_rhs_values[t][n][s] += pi[ss] * rhs[ss]
                    pi_rhs_values[t][n][s] += -pi[-1] * demand
                else:
                    pi_rhs_values[t][n][s] = -pi[-1] * demand
                pi_values[t][n][s] = pi[-1]
                # m_backward[t][n][s].dispose()

            avg_pi = sum(pi_values[t][n]) / S
            avg_pi_rhs = sum(pi_rhs_values[t][n]) / S

            # recording cuts
            if t == 0 and n == 0:
                slope_1st_stage.append(avg_pi)
                intercept_1st_stage.append(avg_pi_rhs)
            elif t > 0:
                slopes[t - 1][n].append(avg_pi)
                intercepts[t - 1][n].append(avg_pi_rhs)
    logger.console_body_sddp(iter_, z)
    iter_ += 1

end = time.process_time()
print("********************************************")
print("final expected total costs is %.2f" % z)
Q_0 = q_values[iter_ - 1]
print("ordering Q in the first period is %.2f" % Q_0)
cpu_time = end - start
print("cpu time is %.3f s" % cpu_time)

# the following codes are for outputting information in the log file
description = "This module build gurobi model for each uncertainty realization."
logger.file_header_sddp()
result_txt = ("cpu_time", "objective value", "Q_0")
results_values = (cpu_time, z, Q_0)
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
