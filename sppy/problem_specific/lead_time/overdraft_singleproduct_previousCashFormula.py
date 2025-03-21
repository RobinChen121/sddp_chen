"""
Python version: 3.12.7
Author: Zhen Chen, chen.zhen5526@gmail.com
Date: 2025/3/3 22:41
Description: 
    SDDP for single product with lead time.
    mean_demands = [15, 15, 15, 15].
    Cash flow equation uses the previous formula.

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
prices = [10 for _ in range(T)]
unit_salvage = 0.5
T = 3  # len(mean_demands) # change 1
overhead_costs = [50 for t in range(T)]
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
sample_num = 2  # change 2
N = 8  # sampled number of scenarios for forward computing # change 3
iter_num = 20

sample_nums = [sample_num for t in range(T)]
# detailed samples in each period
sample_details = [[0.0 for _ in range(sample_nums[t])] for t in range(T)]
for t in range(T):
    sampling = Sampling(dist_name=distribution, mu=mean_demands[t])
    sample_details[t] = sampling.generate_samples(sample_nums[t])

sample_details = [[5, 15], [5, 15], [5, 15]]  # change 4

iter_ = 0
env = Env(params={"OutputFlag": 0})
models = [Model(env=env) for _ in range(T + 1)]
q = [Any for _ in range(T)]
q_pre = [Any for _ in range(T)]  # previously T - 1
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
    if t > 0:
        cash[t - 1] = models[t].addVar(
            lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="cash_" + str(t)
        )
        I[t - 1] = models[t].addVar(vtype=GRB.CONTINUOUS, name="I_" + str(t))
        B[t - 1] = models[t].addVar(
            vtype=GRB.CONTINUOUS, name="B_" + str(t)  # obj=prices[t - 1],
        )
        # noinspection PyTypeChecker
        models[t].addConstr(I[t - 1] - B[t - 1] == 0)
        models[t].addConstr(cash[t - 1] + prices[t - 1] * B[t - 1] == 0)
        if t < T:
            q_pre[t - 1] = models[t].addVar(  #
                vtype=GRB.CONTINUOUS, name="q_pre_" + str(t + 1)
            )
            models[t].addConstr(q_pre[t - 1] == 0)
    if t < T:
        q[t] = models[t].addVar(
            vtype=GRB.CONTINUOUS, obj=unit_vari_costs[t], name="q_" + str(t + 1)
        )
        W0[t] = models[t].addVar(
            vtype=GRB.CONTINUOUS, name="W0_" + str(t + 1)
        )  # obj=-r0,
        W1[t] = models[t].addVar(
            vtype=GRB.CONTINUOUS, name="W1_" + str(t + 1)
        )  # obj=r1,
        W2[t] = models[t].addVar(
            vtype=GRB.CONTINUOUS, name="W2_" + str(t + 1)
        )  # obj=r2,
        theta[t] = models[t].addVar(
            # obj=1,
            lb=-GRB.INFINITY,
            vtype=GRB.CONTINUOUS,
            name="theta_" + str(t + 2),
        )
        models[t].addConstr(W1[t] <= U)
        if t == 0:
            models[t].addConstr(
                ini_cash - unit_vari_costs[t] * q[t] - W0[t] + W1[t] + W2[t]
                == overhead_costs[t]
            )
        else:
            models[t].addConstr(
                cash[t - 1] - unit_vari_costs[t] * q[t] - W0[t] + W1[t] + W2[t]
                == overhead_costs[t]
            )
        models[t].addConstr(theta[t] >= theta_iniValue * (T - t))
    if t == 0:
        models[t].setObjective(
            overhead_costs[0]
            + unit_vari_costs[0] * q[0]
            + r2 * W2[0]
            + r1 * W1[0]
            - r0 * W0[0]
            + theta[0],
            GRB.MINIMIZE,
        )
    models[t].update()

slopes1 = [[[0 for _ in range(N)] for _ in range(T)] for _ in range(iter_num)]
slopes2 = [[[0 for _ in range(N)] for _ in range(T)] for _ in range(iter_num)]
slopes3 = [[[0 for _ in range(N)] for _ in range(T)] for _ in range(iter_num)]
intercepts = [[[0 for _ in range(N)] for _ in range(T)] for _ in range(iter_num)]
q_values = [[[0 for _ in range(N)] for _ in range(T)] for _ in range(iter_num)]
q_pre_values = [[[0 for _ in range(N)] for _ in range(T)] for _ in range(iter_num)]
cpu_time = 0
start = time.process_time()
while iter_ < iter_num:
    I_forward_values = [[0 for n in range(N)] for t in range(T)]
    B_forward_values = [[0 for n in range(N)] for t in range(T)]
    cash_forward_values = [[0 for n in range(N)] for t in range(T)]
    W0_forward_values = [[0 for n in range(N)] for t in range(T)]
    W1_forward_values = [[0 for n in range(N)] for t in range(T)]
    W2_forward_values = [[0 for n in range(N)] for t in range(T)]

    # sample a numer of scenarios from the full scenario tree
    scenario_paths = generate_scenario_paths(N, sample_nums)
    scenario_paths.sort()  # sort to make same numbers together
    sample_paths = [
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1],
    ]

    if iter_ > 0:
        models[0].addConstr(
            theta[0]
            >= slopes1[iter_-1][0][0] * ini_I
            + slopes2[iter_-1][0][0]
            * (
                ini_cash
                - unit_vari_costs[0] * q[0]
                - r1 * W1[0]
                + r0 * W0[0]
                - r2 * W2[0]
            )
            # * (
            #         (1+r0) * W0[0]
            #         -(1+r1) * W1[0]
            #         - (1+r2) * W2[0]
            # )
            + slopes3[iter_-1][0][0] * q[0] + intercepts[iter_-1][0][0]
        )
        models[0].update()
    models[0].optimize()
    # if iter_ >= 1:
    #     models[0].write("iter" + str(iter_ + 1) + "_main-1.lp")
    #     models[0].write("iter" + str(iter_ + 1) + "_main-1.sol")
    #     pass

    # forward
    q_values[iter_][0] = [q[0].x for n in range(N)]
    W0_forward_values[0] = [W0[0].x for n in range(N)]
    W1_forward_values[0] = [W1[0].x for n in range(N)]
    W2_forward_values[0] = [W2[0].x for n in range(N)]

    for t in range(1, T + 1):
        # add the cut constraints
        if iter_ > 0 and t < T:
            for nn in range(N):
                if t == 1 and iter_ == 2:
                    pass
                # warnings of an unexpected type by python interpreter for the below line can be ignored
                models[t].addConstr(
                    theta[t]
                    >= slopes1[iter_ - 1][t][nn] * (I[t - 1] + q_pre[t - 1])
                    + slopes2[iter_ - 1][t][nn]
                    # * ((1 + r0) * W0[t] - (1 + r1) * W1[t] - (1 + r2) * W2[t])
                    * (
                        cash[t - 1]
                        - unit_vari_costs[t] * q[t]
                        - r2 * W2[t]
                        - r1 * W1[t]
                        + r0 * W0[t]
                    )
                    + slopes3[iter_ - 1][t][nn] * q[t]
                    + intercepts[iter_ - 1][t][nn]
                )

        for n in range(N):
            index = scenario_paths[n][t - 1]
            demand = sample_details[t - 1][index]
            if t == 1:  # actually the model in the 2nd stage
                rhs1 = ini_I - demand
            else:
                rhs1 = I_forward_values[t - 1][n] + q_pre_values[iter_][t - 2][n] - demand
            if t < T + 1:
                rhs2 = (
                    ini_cash
                    - overhead_costs[t - 1]
                    - unit_vari_costs[t - 1] * q_values[iter_][t - 1][n]
                    - r1 * W1_forward_values[t - 1][n]
                    + r0 * W0_forward_values[t - 1][n]
                    - r2 * W2_forward_values[t - 1][n]
                    + prices[t - 1] * demand
                    if t == 1
                    else cash_forward_values[t - 2][n]
                    - overhead_costs[t - 1]
                    - unit_vari_costs[t - 1] * q_values[iter_][t - 1][n]
                    - r1 * W1_forward_values[t - 1][n]
                    + r0 * W0_forward_values[t - 1][n]
                    - r2 * W2_forward_values[t - 1][n]
                    + prices[t - 1] * demand

                    # prices[t - 1] * demand
                    # + (1 + r0) * W0_forward_values[t - 1][n]
                    # - (1 + r1) * W1_forward_values[t - 1][n]
                    # - (1 + r2) * W2_forward_values[t - 1][n]
                )
                rhs3 = q_values[iter_][t - 1][n]
            if t == T:
                models[t].setObjective(
                    -prices[t - 1] * (demand - B[t - 1]) - unit_salvage * I[t - 1],
                    GRB.MINIMIZE,
                )
            else:
                models[t].setObjective(
                    overhead_costs[t]
                    + unit_vari_costs[t] * q[t]
                    - prices[t - 1] * (demand - B[t - 1])
                    + r2 * W2[t]
                    + r1 * W1[t]
                    - r0 * W0[t]
                    + theta[t],
                    GRB.MINIMIZE,
                )
            # noinspection PyTypeChecker
            models[t].setAttr("RHS", models[t].getConstrs()[0], rhs1)
            if t < T:
                models[t].setAttr("RHS", models[t].getConstrs()[1], rhs2)
                models[t].setAttr("RHS", models[t].getConstrs()[2], rhs3)

            # optimize
            models[t].optimize()
            if iter_ == 2 and t == 1:
                models[t].write(
                    "iter"
                    + str(iter_ + 1)
                    + "_sub_"
                    + str(t)
                    + "^"
                    + str(n + 1)
                    + "-1.lp"
                )
                models[t].write(
                    "iter"
                    + str(iter_ + 1)
                    + "_sub_"
                    + str(t)
                    + "^"
                    + str(n + 1)
                    + "-1.sol"
                )
                pass
            I_forward_values[t - 1][n] = I[t - 1].x
            B_forward_values[t - 1][n] = B[t - 1].x
            cash_forward_values[t - 1][n] = cash[t - 1].x

            if t < T:
                q_values[iter_][t][n] = q[t].x
                q_pre_values[iter_][t - 1][n] = q_pre[t - 1].x
                W1_forward_values[t][n] = W1[t].x
                W0_forward_values[t][n] = W0[t].x
                W2_forward_values[t][n] = W2[t].x

    # backward
    intercept_values = [
        [[0 for s in range(sample_nums[t])] for n in range(N)] for t in range(T)
    ]
    slope1_values = [
        [[0 for s in range(sample_nums[t])] for n in range(N)] for t in range(T)
    ]
    slope2_values = [
        [[0 for s in range(sample_nums[t])] for n in range(N)] for t in range(T)
    ]
    slope3_values = [
        [[0 for s in range(sample_nums[t])] for n in range(N)] for t in range(T)
    ]

    for t in range(T, 0, -1):
        # add the cut constraints

        for n in range(N):
            S = len(sample_details[t - 1])
            for s in range(S):
                demand = sample_details[t - 1][s]
                if t == 1:
                    rhs1 = ini_I - demand
                else:
                    rhs1 = (
                        I_forward_values[t - 1][n] + q_pre_values[iter_][t - 2][n] - demand
                    )
                if t < T + 1:  # test for T
                    rhs2 = (
                        ini_cash
                        - overhead_costs[t - 1]
                        - unit_vari_costs[t - 1] * q_values[iter_][t - 1][n]
                        - r1 * W1_forward_values[t - 1][n]
                        + r0 * W0_forward_values[t - 1][n]
                        - r2 * W2_forward_values[t - 1][n]
                        + prices[t - 1] * demand
                        if t == 1
                        else cash_forward_values[t - 2][n]
                        - overhead_costs[t - 1]
                        - unit_vari_costs[t - 1] * q_values[iter_][t - 1][n]
                        - r1 * W1_forward_values[t - 1][n]
                        + r0 * W0_forward_values[t - 1][n]
                        - r2 * W2_forward_values[t - 1][n]
                        + prices[t - 1] * demand
                        # prices[t] * demand
                        # + (1 + r0) * W0_forward_values[t - 1][n]
                        # - (1 + r1) * W1_forward_values[t - 1][n]
                        # - (1 + r2) * W2_forward_values[t - 1][n]
                    )
                if t < T:
                    rhs3 = q_values[iter_][t - 1][n]
                if t == T:
                    models[t].setObjective(
                        -prices[t - 1] * (demand - B[t - 1]) - unit_salvage * I[t - 1],
                        GRB.MINIMIZE,
                    )
                else:
                    models[t].setObjective(
                        overhead_costs[t]
                        + unit_vari_costs[t] * q[t]
                        - prices[t - 1] * (demand - B[t - 1])
                        + r2 * W2[t]
                        + r1 * W1[t]
                        - r0 * W0[t]
                        + theta[t],
                        GRB.MINIMIZE,
                    )
                # noinspection PyTypeChecker
                models[t].setAttr("RHS", models[t].getConstrs()[0], rhs1)
                if t < T + 1:  # test for T
                    models[t].setAttr("RHS", models[t].getConstrs()[1], rhs2)
                if t < T:
                    models[t].setAttr("RHS", models[t].getConstrs()[2], rhs3)

                # optimize
                models[t].optimize()
                pi = models[t].getAttr(GRB.Attr.Pi)
                rhs = models[t].getAttr(GRB.Attr.RHS)
                if iter_ == 2 and t == 2 and n == 0:
                    models[t].write(
                        "iter"
                        + str(iter_ + 1)
                        + "_sub_"
                        + str(t)
                        + "^"
                        + str(n + 1)
                        + "_"
                        + str(s + 1)
                        + "back-1.lp"
                    )
                    models[t].write(
                        "iter"
                        + str(iter_ + 1)
                        + "_sub_"
                        + str(t)
                        + "^"
                        + str(n + 1)
                        + "_"
                        + str(s + 1)
                        + "back-1.sol"
                    )
                    filename = (
                        "iter"
                        + str(iter_ + 1)
                        + "_sub_"
                        + str(t)
                        + "^"
                        + str(n + 1)
                        + "_"
                        + str(s + 1)
                        + "-1.txt"
                    )
                    with open(filename, "w") as f:
                        f.write("demand=" + str(demand) + "\n")
                        f.write(str(pi) + "\n")
                        f.write(str(rhs))
                    pass

                num_con = len(pi)
                if t < T:
                    intercept_values[t - 1][n][s] += (
                        -pi[0] * demand
                        + pi[1] * prices[t - 1] * demand
                        - pi[1]
                        * overhead_costs[
                            t - 1
                        ]  # previously not include the following 2 lines
                        - prices[t - 1] * demand
                        + overhead_costs[t]
                    )
                else:
                    intercept_values[t - 1][n][s] += (
                        -pi[0] * demand
                        + pi[1] * prices[t - 1] * demand
                        - pi[1]
                        * overhead_costs[
                            t - 1
                        ]  # previously not include the following 1 line
                        - prices[t - 1] * demand
                    )

                for sk in range(3, num_con):  # previously inside the above loop
                    intercept_values[t - 1][n][s] += pi[sk] * rhs[sk]
                slope1_values[t - 1][n][s] = pi[0]
                if t < T:
                    slope2_values[t - 1][n][s] = pi[1]
                    slope3_values[t - 1][n][s] = pi[2]

            avg_intercept = sum(intercept_values[t - 1][n]) / S
            avg_slope1 = sum(slope1_values[t - 1][n]) / S
            avg_slope2 = sum(slope2_values[t - 1][n]) / S
            avg_slope3 = sum(slope3_values[t - 1][n]) / S

            slopes1[iter_][t - 1][n] = avg_slope1
            slopes2[iter_][t - 1][n] = avg_slope2
            slopes3[iter_][t - 1][n] = avg_slope3
            intercepts[iter_][t - 1][n] = avg_intercept

    iter_ = iter_ + 1
    end = time.process_time()
    cpu_time = end - start

print("********************************************")
final_value = -models[0].objVal
Q1 = q_values[iter_ - 1][0][0]
print("after %d iteration: " % iter_)
print("final expected cash balance is %.2f" % final_value)
print("ordering Q in the first period is %.2f" % Q1)
print("cpu time is %.3f s" % cpu_time)
# gap = (-opt + final_value) / opt
# print("optimality gap is %.2f%%" % (100 * gap))
