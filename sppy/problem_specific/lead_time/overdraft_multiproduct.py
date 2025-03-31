"""
Python version: 3.12.7
Author: Zhen Chen, chen.zhen5526@gmail.com
Date: 2025/3/14 18:21
Description: 
    

"""

import itertools
import time
from gurobipy import Model, GRB, Env
from sppy.utils.sampling import Sampling, generate_scenario_paths
from typing import Any


def remove_duplicate_rows(matrix):
    return list(map(list, set(map(tuple, matrix))))  # 先转换成元组去重，再转换回列表


mean_demands1 = [30, 30, 30]  # higher average demand vs lower average demand
mean_demands2 = [i * 0.5 for i in mean_demands1]
distribution = "poisson"
T = len(mean_demands1)

ini_I1 = 0
ini_I2 = 0
ini_cash = 0
unit_vari_costs1 = [1 for _ in range(T)]
unit_vari_costs2 = [2 for _ in range(T)]
prices1 = [5 for _ in range(T)]
prices2 = [10 for _ in range(T)]
product_num = 2
unit_salvage1 = 0.5 * unit_vari_costs1[T - 1]
unit_salvage2 = 0.5 * unit_vari_costs2[T - 1]
overhead_costs = [100 for t in range(T)]

r0 = 0  # when it is 0.01, can largely slow the computational speed
r1 = 0.1
r2 = 2  # penalty interest rate for overdraft exceeding the limit does not affect computation time
U = 500  # overdraft limit

sample_num = 10
N = 20  # sampled number of scenarios for forward computing # change 3
iter_num = 30

sample_nums1 = [sample_num for t in range(T)]
sample_nums2 = [sample_num for t in range(T)]
# detailed samples in each period
sample_details1 = [[0.0 for _ in range(sample_nums1[t])] for t in range(T)]
sample_details2 = [[0.0 for _ in range(sample_nums2[t])] for t in range(T)]
for t in range(T):
    sampling1 = Sampling(dist_name=distribution, mu=mean_demands1[t])
    sample_details1[t] = sampling1.generate_samples(sample_nums1[t])
    sampling2 = Sampling(dist_name=distribution, mu=mean_demands2[t])
    sample_details2[t] = sampling2.generate_samples(sample_nums2[t])

iter_ = 0
env = Env(params={"OutputFlag": 0})
models = [Model(env=env) for _ in range(T + 1)]
q1 = [Any for _ in range(T)]
q2 = [Any for _ in range(T)]
q1_pre = [Any for _ in range(T)]  # previously T - 1
q2_pre = [Any for _ in range(T)]
theta = [Any for _ in range(T)]
I1 = [Any for _ in range(T)]
I2 = [Any for _ in range(T)]
B1 = [Any for _ in range(T)]
B2 = [Any for _ in range(T)]
cash = [Any for _ in range(T)]
W0 = [Any for _ in range(T)]
W1 = [Any for _ in range(T)]
W2 = [Any for _ in range(T)]

theta_iniValue = -1000
for t in range(T + 1):
    if t < T:
        q1[t] = models[t].addVar(vtype=GRB.CONTINUOUS, name="q1_" + str(t + 1))
        q2[t] = models[t].addVar(vtype=GRB.CONTINUOUS, name="q2_" + str(t + 1))
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
                ini_cash
                - unit_vari_costs1[t] * q1[t]
                - unit_vari_costs2[t] * q2[t]
                - W0[t]
                + W1[t]
                + W2[t]
                == overhead_costs[t]
            )
        else:
            models[t].addConstr(
                cash[t - 1]
                - unit_vari_costs1[t] * q1[t]
                - unit_vari_costs2[t] * q2[t]
                - W0[t]
                + W1[t]
                + W2[t]
                == overhead_costs[t]
            )
        models[t].addConstr(theta[t] >= theta_iniValue * (T - t))
    if t > 0:
        cash[t - 1] = models[t].addVar(
            lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="cash_" + str(t)
        )
        I1[t - 1] = models[t].addVar(vtype=GRB.CONTINUOUS, name="I1_" + str(t))
        I2[t - 1] = models[t].addVar(vtype=GRB.CONTINUOUS, name="I1_" + str(t))
        B1[t - 1] = models[t].addVar(
            vtype=GRB.CONTINUOUS, name="B1_" + str(t)  # obj=prices[t - 1],
        )
        # noinspection PyTypeChecker
        models[t].addConstr(I1[t - 1] - B1[t - 1] == 0)
        models[t].addConstr(I2[t - 1] - B2[t - 1] == 0)
        if t < T:
            models[t].addConstr(
                cash[t - 1] + prices1[t - 1] * B1[t - 1] + prices2[t - 1] * B2[t - 1]
                == 0
            )
            q1_pre[t - 1] = models[t].addVar(  #
                vtype=GRB.CONTINUOUS, name="q1_pre_" + str(t + 1)
            )
            models[t].addConstr(q1_pre[t - 1] == 0)
            models[t].addConstr(q2_pre[t - 1] == 0)
    if t == 0:
        models[t].setObjective(
            overhead_costs[0]
            + unit_vari_costs1[0] * q1[0]
            + unit_vari_costs2[0] * q2[0]
            + r2 * W2[0]
            + r1 * W1[0]
            - r0 * W0[0]
            + theta[0],
            GRB.MINIMIZE,
        )
    models[t].update()

slopes1_1 = [[[0 for _ in range(N)] for _ in range(T)] for _ in range(iter_num)]
slopes1_2 = [[[0 for _ in range(N)] for _ in range(T)] for _ in range(iter_num)]
slopes2 = [[[0 for _ in range(N)] for _ in range(T)] for _ in range(iter_num)]
slopes3_1 = [[[0 for _ in range(N)] for _ in range(T)] for _ in range(iter_num)]
slopes3_2 = [[[0 for _ in range(N)] for _ in range(T)] for _ in range(iter_num)]
intercepts = [[[0 for _ in range(N)] for _ in range(T)] for _ in range(iter_num)]

q1_values = [[[0 for _ in range(N)] for _ in range(T)] for _ in range(iter_num)]
q1_pre_values = [[[0 for _ in range(N)] for _ in range(T)] for _ in range(iter_num)]
I1_forward_values = [[[0 for n in range(N)] for t in range(T)] for _ in range(iter_num)]
B1_forward_values = [[[0 for n in range(N)] for t in range(T)] for _ in range(iter_num)]
q2_values = [[[0 for _ in range(N)] for _ in range(T)] for _ in range(iter_num)]
q2_pre_values = [[[0 for _ in range(N)] for _ in range(T)] for _ in range(iter_num)]
I2_forward_values = [[[0 for n in range(N)] for t in range(T)] for _ in range(iter_num)]
B2_forward_values = [[[0 for n in range(N)] for t in range(T)] for _ in range(iter_num)]

cash_forward_values = [
    [[0 for n in range(N)] for t in range(T)] for _ in range(iter_num)
]
W0_forward_values = [[[0 for n in range(N)] for t in range(T)] for _ in range(iter_num)]
W1_forward_values = [[[0 for n in range(N)] for t in range(T)] for _ in range(iter_num)]
W2_forward_values = [[[0 for n in range(N)] for t in range(T)] for _ in range(iter_num)]

objs = [0 for _ in range(iter_num)]
cut_coefficients_cache = [set() for t in range(T)]
cpu_time = 0
start = time.process_time()
while iter_ < iter_num:
    scenario_paths1 = generate_scenario_paths(N, sample_nums1)
    scenario_paths2 = generate_scenario_paths(N, sample_nums2)

    skip_iter = False  # For stage 1, if the added cut constraint is same with previous added, skip solving the model
    if iter_ > 0:
        if iter_ == 1:  # remove the big M constraints at iteration 2
            index = models[0].NumConstrs - 1
            models[0].remove(models[0].getConstrs()[index])
            models[0].update()

        this_coefficient = (
            slopes1_1[iter_ - 1][0][0],
            slopes1_2[iter_ - 1][0][0],
            slopes2[iter_ - 1][0][0],
            slopes3_1[iter_ - 1][0][0],
            slopes3_2[iter_ - 1][0][0],
            intercepts[iter_ - 1][0][0],
        )
        if (
            not cut_coefficients_cache[0]
            or this_coefficient not in cut_coefficients_cache[0]
        ):
            models[0].addConstr(
                theta[0]
                >= slopes1_1[iter_ - 1][0][0] * ini_I1
                + slopes1_2[iter_ - 1][0][0] * ini_I2
                + slopes2[iter_ - 1][0][0]
                * ((1 + r0) * W0[0] - (1 + r1) * W1[0] - (1 + r2) * W2[0])
                + slopes3_1[iter_ - 1][0][0] * q1[0]
                + slopes3_2[iter_ - 1][0][0] * q2[0]
                + intercepts[iter_ - 1][0][0]
            )
            models[0].update()
            cut_coefficients_cache[0].add(this_coefficient)
        else:
            skip_iter = True

    if skip_iter:
        q1_values[iter_][0] = [q1_values[iter_ - 1][0][0] for n in range(N)]
        q2_values[iter_][0] = [q2_values[iter_ - 1][0][0] for n in range(N)]
        W0_forward_values[iter_][0] = [
            W0_forward_values[iter_ - 1][0][0] for n in range(N)
        ]
        W1_forward_values[iter_][0] = [
            W1_forward_values[iter_ - 1][0][0] for n in range(N)
        ]
        W2_forward_values[iter_][0] = [
            W2_forward_values[iter_ - 1][0][0] for n in range(N)
        ]
        pass
    else:
        models[0].optimize()

        # forward
        q1_values[iter_][0] = [q1[0].x for n in range(N)]
        q2_values[iter_][0] = [q2[0].x for n in range(N)]
        W0_forward_values[iter_][0] = [W0[0].x for n in range(N)]
        W1_forward_values[iter_][0] = [W1[0].x for n in range(N)]
        W2_forward_values[iter_][0] = [W2[0].x for n in range(N)]

    for t in range(1, T + 1):
        # add the cut constraints
        if iter_ > 0 and t < T:
            if iter_ == 1:  # remove the big M constraints at iteration 2
                index = models[t].NumConstrs - 1
                models[t].remove(models[t].getConstrs()[index])

            cut_coefficients = [
                [
                    slopes1_1[iter_ - 1][t][nn],
                    slopes1_2[iter_ - 1][t][nn],
                    slopes2[iter_ - 1][t][nn],
                    slopes3_1[iter_ - 1][t][nn],
                    slopes3_2[iter_ - 1][t][nn],
                    intercepts[iter_ - 1][t][nn],
                ]
                for nn in range(N)
            ]
            final_coefficients = remove_duplicate_rows(cut_coefficients)

            for final_coefficient in final_coefficients:
                # warnings of an unexpected type by python interpreter for the below line can be ignored
                final_coefficient = tuple(final_coefficient)
                if (
                    not cut_coefficients_cache[t]
                    or final_coefficient not in cut_coefficients_cache[t]
                ):
                    models[t].addConstr(
                        theta[t]
                        >= final_coefficient[0] * (I1[t - 1] + q1_pre[t - 1])
                        + final_coefficient[1] * (I2[t - 1] + q2_pre[t - 1])
                        + final_coefficient[2]
                        * ((1 + r0) * W0[t] - (1 + r1) * W1[t] - (1 + r2) * W2[t])
                        + final_coefficient[3] * q1[t]
                        + final_coefficient[4] * q2[t]
                        + final_coefficient[5]
                    )
                    cut_coefficients_cache[t].add(final_coefficient)

        for n in range(N):
            index1 = scenario_paths1[n][t - 1]
            index2 = scenario_paths2[n][t - 1]
            demand1 = sample_details1[t - 1][index1]
            demand2 = sample_details2[t - 1][index1]
            rhs1_1 = 0
            rhs1_2 = 0
            if t == 1:  # actually the model in the 2nd stage
                rhs1_1 = ini_I1 - demand1
                rhs1_2 = ini_I2 - demand2
            else:
                rhs1_1 = (
                    I1_forward_values[iter_][t - 1][n]
                    + q1_pre_values[iter_][t - 2][n]
                    - demand1
                )
                rhs1_2 = (
                    I2_forward_values[iter_][t - 1][n]
                    + q2_pre_values[iter_][t - 2][n]
                    - demand2
                )
            if t < T:
                rhs2 = (
                    prices1[t - 1] * demand1
                    + prices2[t - 1] * demand2
                    + (1 + r0) * W0_forward_values[iter_][t - 1][n]
                    - (1 + r1) * W1_forward_values[iter_][t - 1][n]
                    - (1 + r2) * W2_forward_values[iter_][t - 1][n]
                )
                rhs3_1 = q1_values[iter_][t - 1][n]
                rhs3_2 = q2_values[iter_][t - 1][n]
            if t == T:
                models[t].setObjective(
                    -prices1[t - 1] * (demand1 - B1[t - 1])
                    - unit_salvage1 * I1[t - 1]
                    - prices2[t - 1] * (demand2 - B2[t - 1])
                    - unit_salvage2 * I2[t - 1],
                    GRB.MINIMIZE,
                )
            else:
                models[t].setObjective(
                    overhead_costs[t]
                    + unit_vari_costs1[t] * q1[t]
                    - prices1[t - 1] * (demand1 - B1[t - 1])
                    + unit_vari_costs2[t] * q2[t]
                    - prices2[t - 1] * (demand2 - B2[t - 1])
                    + r2 * W2[t]
                    + r1 * W1[t]
                    - r0 * W0[t]
                    + theta[t],
                    GRB.MINIMIZE,
                )
            # noinspection PyTypeChecker
            models[t].setAttr("RHS", models[t].getConstrs()[0], rhs1_1)
            models[t].setAttr("RHS", models[t].getConstrs()[1], rhs1_2)
            if t < T:
                models[t].setAttr("RHS", models[t].getConstrs()[2], rhs2)
                models[t].setAttr("RHS", models[t].getConstrs()[3], rhs3_1)
                models[t].setAttr("RHS", models[t].getConstrs()[4], rhs3_2)

            # set lb and ub for some variables
            this_I1_value = rhs1_1 if rhs1_1 > 0 else 0
            this_B1_value = -rhs1_1 if rhs1_1 < 0 else 0
            this_I2_value = rhs1_2 if rhs1_2 > 0 else 0
            this_B2_value = -rhs1_2 if rhs1_2 < 0 else 0
            I1[t - 1].setAttr(GRB.Attr.LB, this_I1_value)
            I2[t - 1].setAttr(GRB.Attr.UB, this_I2_value)
            B1[t - 1].setAttr(GRB.Attr.LB, this_B1_value)
            B2[t - 1].setAttr(GRB.Attr.UB, this_B2_value)
            if t < T:
                this_end_cash = (
                    rhs2
                    - prices1[t - 1] * this_B1_value
                    - prices2[t - 1] * this_B2_value
                )
                cash[t - 1].setAttr(GRB.Attr.LB, this_end_cash)
                cash[t - 1].setAttr(GRB.Attr.UB, this_end_cash)

            # optimize
            models[t].optimize()

            I1_forward_values[iter_][t - 1][n] = I1[t - 1].x
            B1_forward_values[iter_][t - 1][n] = B1[t - 1].x
            I2_forward_values[iter_][t - 1][n] = I2[t - 1].x
            B2_forward_values[iter_][t - 1][n] = B2[t - 1].x
            cash_forward_values[iter_][t - 1][n] = cash[t - 1].x

            if t < T:
                q1_values[iter_][t][n] = q1[t].x
                q1_pre_values[iter_][t - 1][n] = q1_pre[t - 1].x
                q2_values[iter_][t][n] = q2[t].x
                q2_pre_values[iter_][t - 1][n] = q2_pre[t - 1].x
                W1_forward_values[iter_][t][n] = W1[t].x
                W0_forward_values[iter_][t][n] = W0[t].x
                W2_forward_values[iter_][t][n] = W2[t].x

    # backward
    demands_all = [[] for t in range(T)]
    sample_nums_backward = [0 for t in range(T)]
    for t in range(T):
        demand_temp = [sample_details1[t], sample_details2[t]]
        demands_all[t] = list(itertools.product(*demand_temp))
        sample_nums_backward[t] = len(demands_all[t])
    intercept_values = [
        [[0 for s in range(sample_nums_backward[t])] for n in range(N)]
        for t in range(T)
    ]
    slope1_1values = [
        [[0 for s in range(sample_nums_backward[t])] for n in range(N)]
        for t in range(T)
    ]
    slope1_2values = [
        [[0 for s in range(sample_nums_backward[t])] for n in range(N)]
        for t in range(T)
    ]
    slope2_values = [
        [[0 for s in range(sample_nums_backward[t])] for n in range(N)]
        for t in range(T)
    ]
    slope3_1values = [
        [[0 for s in range(sample_nums_backward[t])] for n in range(N)]
        for t in range(T)
    ]
    slope3_2values = [
        [[0 for s in range(sample_nums_backward[t])] for n in range(N)]
        for t in range(T)
    ]

    for t in range(T, 0, -1):
        for n in range(N):
            S = len(demands_all[t - 1])
            for s in range(S):
                demand1 = demands_all[t - 1][s][0]
                demand2 = demands_all[t - 1][s][1]
                if t == 1:
                    rhs1_1 = ini_I1 - demand1
                    rhs1_2 = ini_I2 - demand2
                else:
                    rhs1_1 = (
                        I1_forward_values[iter_][t - 1][n]
                        + q1_pre_values[iter_][t - 2][n]
                        - demand1
                    )
                    rhs1_2 = (
                        I2_forward_values[iter_][t - 1][n]
                        + q2_pre_values[iter_][t - 2][n]
                        - demand2
                    )
                if t < T + 1:  # test for T
                    rhs2 = (
                        prices1[t - 1] * demand1
                        + prices2[t - 1] * demand2
                        + (1 + r0) * W0_forward_values[iter_][t - 1][n]
                        - (1 + r1) * W1_forward_values[iter_][t - 1][n]
                        - (1 + r2) * W2_forward_values[iter_][t - 1][n]
                    )
                if t < T:
                    rhs3_1 = q1_values[iter_][t - 1][n]
                    rhs3_2 = q2_values[iter_][t - 1][n]
                if t == T:
                    models[t].setObjective(
                        -prices1[t - 1] * (demand1 - B1[t - 1])
                        - unit_salvage1 * I1[t - 1]
                        - prices2[t - 1] * (demand2 - B2[t - 1])
                        - unit_salvage2 * I2[t - 1],
                        GRB.MINIMIZE,
                    )
                else:
                    models[t].setObjective(
                        overhead_costs[t]
                        + unit_vari_costs1[t] * q1[t]
                        + unit_vari_costs2[t] * q2[t]
                        - prices1[t - 1] * (demand1 - B1[t - 1])
                        - prices2[t - 1] * (demand2 - B2[t - 1])
                        + r2 * W2[t]
                        + r1 * W1[t]
                        - r0 * W0[t]
                        + theta[t],
                        GRB.MINIMIZE,
                    )
                # noinspection PyTypeChecker
                models[t].setAttr("RHS", models[t].getConstrs()[0], rhs1_1)
                models[t].setAttr("RHS", models[t].getConstrs()[1], rhs1_2)
                if t < T:  # test for T
                    models[t].setAttr("RHS", models[t].getConstrs()[2], rhs2)
                    models[t].setAttr("RHS", models[t].getConstrs()[3], rhs3_1)
                    models[t].setAttr("RHS", models[t].getConstrs()[4], rhs3_2)

                # de set the lb and ub for some variables
                I1[t - 1].setAttr(GRB.Attr.LB, 0.0)
                I1[t - 1].setAttr(GRB.Attr.UB, GRB.INFINITY)
                B1[t - 1].setAttr(GRB.Attr.LB, 0.0)
                B1[t - 1].setAttr(GRB.Attr.UB, GRB.INFINITY)
                I2[t - 1].setAttr(GRB.Attr.LB, 0.0)
                I2[t - 1].setAttr(GRB.Attr.UB, GRB.INFINITY)
                B2[t - 1].setAttr(GRB.Attr.LB, 0.0)
                B2[t - 1].setAttr(GRB.Attr.UB, GRB.INFINITY)
                if t < T:
                    cash[t - 1].setAttr(GRB.Attr.LB, -GRB.INFINITY)
                    cash[t - 1].setAttr(GRB.Attr.UB, GRB.INFINITY)

                # optimize
                models[t].optimize()
                pi = models[t].getAttr(GRB.Attr.Pi)
                rhs = models[t].getAttr(GRB.Attr.RHS)

                num_con = len(pi)
                if t < T:
                    intercept_values[t - 1][n][s] += (
                            -pi[0] * demand1 -pi[1] * demand1
                            + pi[2] * (prices1[t - 1] * demand1 + prices2[t - 1] * demand2)
                            - prices1[t - 1] * demand1- prices2[t - 1] * demand2
                            + overhead_costs[t]
                    )
                else:
                    intercept_values[t - 1][n][s] += (
                            -pi[0] * demand1 -pi[1] * demand1
                            - prices1[t - 1] * demand1 - prices2[t - 1] * demand2
                    )

                for sk in range(3, num_con):  # previously inside the above loop
                    intercept_values[t - 1][n][s] += pi[sk] * rhs[sk]
                slope1_1values[t - 1][n][s] = pi[0]
                slope1_2values[t - 1][n][s] = pi[1]
                if t < T:
                    slope2_values[t - 1][n][s] = pi[2]
                    slope3_1values[t - 1][n][s] = pi[3]
                    slope3_2values[t - 1][n][s] = pi[4]

            avg_intercept = sum(intercept_values[t - 1][n]) / S
            avg_slope1_1 = sum(slope1_1values[t - 1][n]) / S
            avg_slope1_2 = sum(slope1_2values[t - 1][n]) / S
            avg_slope2 = sum(slope2_values[t - 1][n]) / S
            avg_slope3_1 = sum(slope3_1values[t - 1][n]) / S
            avg_slope3_2 = sum(slope3_2values[t - 1][n]) / S

            slopes1_1[iter_][t - 1][n] = avg_slope1_1
            slopes1_2[iter_][t - 1][n] = avg_slope1_2
            slopes2[iter_][t - 1][n] = avg_slope2
            slopes3_1[iter_][t - 1][n] = avg_slope3_1
            slopes3_2[iter_][t - 1][n] = avg_slope3_2
            intercepts[iter_][t - 1][n] = avg_intercept

    objs[iter_] = -models[0].objVal
    print(f"iteration {iter_}, obj is {objs[iter_]:.2f}")
    iter_ = iter_ + 1
    end = time.process_time()
    cpu_time = end - start

print("********************************************")
final_value = -models[0].objVal
Q1 = q1_values[iter_ - 1][0][0]
Q2 = q2_values[iter_ - 1][0][0]
print("after %d iteration: " % iter_)
print("final expected cash balance is %.2f" % final_value)
print("ordering Q1 in the first period is %.2f" % Q1)
print("ordering Q2 in the first period is %.2f" % Q1)
print("cpu time is %.3f s" % cpu_time)
