"""
Python version: 3.12.7
Author: Zhen Chen, chen.zhen5526@gmail.com
Date: 2025/3/22 12:02
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
T = len(mean_demands1)

# demand1_values = [[20, 30, 40] for _ in range(T)]
# demand2_values = [[i * 0.5 for i in demand1_values[t]] for t in range(T)]
# demand1_weights = [[0.25, 0.5, 0.25] for _ in range(T)]
# demand2_weights = [[0.25, 0.5, 0.25] for _ in range(T)]

distribution = "poisson"
# distribution = "rv_discrete"

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

sample_num = 10  # change 1
scenario_num = 10 # sampled number of scenarios for forward computing # change 2
iter_num = 30

sample_nums1 = [sample_num for t in range(T)]
sample_nums2 = [sample_num for t in range(T)]
# detailed samples in each period
sample_details1 = [[0.0 for _ in range(sample_nums1[t])] for t in range(T)]
sample_details2 = [[0.0 for _ in range(sample_nums2[t])] for t in range(T)]
for t in range(T):
    sampling1 = Sampling(dist_name=distribution, mu=mean_demands1[t])
    sampling2 = Sampling(dist_name=distribution, mu=mean_demands2[t])
    # sampling1 = Sampling(
    #     dist_name=distribution, values=(demand1_values[t], demand1_weights[t])
    # )
    # sampling2 = Sampling(
    #     dist_name=distribution, values=(demand2_values[t], demand2_weights[t])
    # )

    sample_details1[t] = sampling1.generate_samples(sample_nums1[t])
    sample_details2[t] = sampling2.generate_samples(sample_nums2[t])

# sample_details1 = [[10, 30], [10, 30], [10, 30]]  # change 3
# sample_details2 = [[5, 15], [5, 15], [5, 15]]

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
    # variables:
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
    if t > 0:
        q1_pre[t - 1] = models[t].addVar(
            vtype=GRB.CONTINUOUS, name="q1_pre_" + str(t + 1)
        )
        q2_pre[t - 1] = models[t].addVar(
            vtype=GRB.CONTINUOUS, name="q2_pre_" + str(t + 1)
        )
        I1[t - 1] = models[t].addVar(vtype=GRB.CONTINUOUS, name="I1_" + str(t))
        I2[t - 1] = models[t].addVar(vtype=GRB.CONTINUOUS, name="I2_" + str(t))
        cash[t - 1] = models[t].addVar(
            lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="cash_" + str(t)
        )
        B1[t - 1] = models[t].addVar(
            vtype=GRB.CONTINUOUS, name="B1_" + str(t)  # obj=prices[t - 1],
        )
        B2[t - 1] = models[t].addVar(
            vtype=GRB.CONTINUOUS, name="B2_" + str(t)  # obj=prices[t - 1],
        )
    # constraints:
    if t > 0:
        # noinspection PyTypeChecker
        models[t].addConstr(I1[t - 1] - B1[t - 1] == 0)
        models[t].addConstr(I2[t - 1] - B2[t - 1] == 0)
        if t == T:  # not very necessary
            cash[t - 1] = models[t].addVar(
                lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="cash_" + str(t)
            )
            models[t].addConstr(
                cash[t - 1] + prices1[t - 1] * B1[t - 1] + prices2[t - 1] * B2[t - 1]
                == 0
            )
        if t < T:
            models[t].addConstr(
                cash[t - 1] + prices1[t - 1] * B1[t - 1] + prices2[t - 1] * B2[t - 1]
                == 0
            )
            models[t].addConstr(q1_pre[t - 1] == 0)
            models[t].addConstr(q2_pre[t - 1] == 0)
    if t < T:
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
    # if t == T - 1: # not necessary for new formulation
    #     models[t].addConstr(
    #         cash[t - 1]
    #         - unit_vari_costs1[t] * q1[t]
    #         - unit_vari_costs2[t] * q2[t]
    #         - W0[t]
    #         + W1[t]
    #         + W2[t]
    #         == overhead_costs[t]
    #     )
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

slopes1_1 = [[[0 for _ in range(scenario_num)] for _ in range(T)] for _ in range(iter_num)]
slopes1_2 = [[[0 for _ in range(scenario_num)] for _ in range(T)] for _ in range(iter_num)]
slopes2 = [[[0 for _ in range(scenario_num)] for _ in range(T)] for _ in range(iter_num)]
slopes3_1 = [[[0 for _ in range(scenario_num)] for _ in range(T)] for _ in range(iter_num)]
slopes3_2 = [[[0 for _ in range(scenario_num)] for _ in range(T)] for _ in range(iter_num)]
intercepts = [[[0 for _ in range(scenario_num)] for _ in range(T)] for _ in range(iter_num)]

q1_values = [[[0 for _ in range(scenario_num)] for _ in range(T)] for _ in range(iter_num)]
q1_pre_values = [[[0 for _ in range(scenario_num)] for _ in range(T)] for _ in range(iter_num)]
I1_forward_values = [[[0 for n in range(scenario_num)] for t in range(T)] for _ in range(iter_num)]
B1_forward_values = [[[0 for n in range(scenario_num)] for t in range(T)] for _ in range(iter_num)]
q2_values = [[[0 for _ in range(scenario_num)] for _ in range(T)] for _ in range(iter_num)]
q2_pre_values = [[[0 for _ in range(scenario_num)] for _ in range(T)] for _ in range(iter_num)]
I2_forward_values = [[[0 for n in range(scenario_num)] for t in range(T)] for _ in range(iter_num)]
B2_forward_values = [[[0 for n in range(scenario_num)] for t in range(T)] for _ in range(iter_num)]

cash_forward_values = [
    [[0 for n in range(scenario_num)] for t in range(T)] for _ in range(iter_num)
]
W0_forward_values = [[[0 for n in range(scenario_num)] for t in range(T)] for _ in range(iter_num)]
W1_forward_values = [[[0 for n in range(scenario_num)] for t in range(T)] for _ in range(iter_num)]
W2_forward_values = [[[0 for n in range(scenario_num)] for t in range(T)] for _ in range(iter_num)]

objs = [0 for _ in range(iter_num)]
cpu_time = 0
start = time.process_time()
while iter_ < iter_num:
    scenario_paths1 = generate_scenario_paths(scenario_num, sample_nums1)
    scenario_paths2 = generate_scenario_paths(scenario_num, sample_nums2)

    # sample_scenarios1 = [[20, 20, 30], [20, 40, 40], [40, 30, 30], [30, 30, 40], [30, 30, 30], [30, 30, 40],
    #                      [30, 30, 30], [40, 40, 20]]
    # sample_scenarios2 = [[15., 15., 15.], [10., 20., 10.], [10., 15., 10.], [10., 15., 15.], [20., 20., 20.],
    #                      [15., 15., 10.], [10., 15., 15.], [10., 10., 15.]]

    # sample_paths1 = [
    #     [0, 0, 0],
    #     [0, 0, 1],
    #     [0, 1, 0],
    #     [0, 1, 1],
    #     [1, 0, 0],
    #     [1, 0, 1],
    #     [1, 1, 0],
    #     [1, 1, 1],
    # ]
    # sample_paths2 = sample_paths1

    if iter_ > 0:
        # if iter_ == 1:  # remove the big M constraints at iteration 2
        #     index = models[0].NumConstrs - 1
        #     models[0].remove(models[0].getConstrs()[index])
        #     models[0].update()

        this_coefficient = (
            slopes1_1[iter_ - 1][0][0],
            slopes1_2[iter_ - 1][0][0],
            slopes2[iter_ - 1][0][0],
            slopes3_1[iter_ - 1][0][0],
            slopes3_2[iter_ - 1][0][0],
            intercepts[iter_ - 1][0][0],
        )
        models[0].addConstr(
            theta[0]
            >= slopes1_1[iter_ - 1][0][0] * ini_I1
            + slopes1_2[iter_ - 1][0][0] * ini_I2
            + slopes2[iter_ - 1][0][0]
            * ((1 + r0) * W0[0] - (1 + r1) * W1[0] - (1 + r2) * W2[0])
            # * (
            #     ini_cash
            #     - unit_vari_costs1[0] * q1[0]
            #     - unit_vari_costs2[0] * q2[0]
            #     - r1 * W1[0]
            #     + r0 * W0[0]
            #     - r2 * W2[0]
            # )
            + slopes3_1[iter_ - 1][0][0] * q1[0]
            + slopes3_2[iter_ - 1][0][0] * q2[0]
            + intercepts[iter_ - 1][0][0]
        )
        models[0].update()

    models[0].optimize()
    # if iter_ >= 1:
    #     models[0].write("iter" + str(iter_) + "_main.lp")
    #     models[0].write("iter" + str(iter_) + "_main.sol")
    #     pass

    # forward
    q1_values[iter_][0] = [q1[0].x for n in range(scenario_num)]
    q2_values[iter_][0] = [q2[0].x for n in range(scenario_num)]
    W0_forward_values[iter_][0] = [W0[0].x for n in range(scenario_num)]
    W1_forward_values[iter_][0] = [W1[0].x for n in range(scenario_num)]
    W2_forward_values[iter_][0] = [W2[0].x for n in range(scenario_num)]

    for t in range(1, T + 1):
        # add the cut constraints
        if iter_ > 0 and t < T:

            # if iter_ == 1:  # remove the big M constraints at iteration 2
            #     index = models[t].NumConstrs - 1
            #     models[t].remove(models[t].getConstrs()[index])

            cut_coefficients = [
                [
                    slopes1_1[iter_ - 1][t][nn],
                    slopes1_2[iter_ - 1][t][nn],
                    slopes2[iter_ - 1][t][nn],
                    slopes3_1[iter_ - 1][t][nn],
                    slopes3_2[iter_ - 1][t][nn],
                    intercepts[iter_ - 1][t][nn],
                ]
                for nn in range(scenario_num)
            ]
            final_coefficients = remove_duplicate_rows(cut_coefficients) # cut_coefficients
            for final_coefficient in final_coefficients:
                models[t].addConstr(
                    theta[t]
                    >= final_coefficient[0] * (I1[t - 1] + q1_pre[t - 1])
                    + final_coefficient[1] * (I2[t - 1] + q2_pre[t - 1])
                    + final_coefficient[2]
                    * ((1 + r0) * W0[t] - (1 + r1) * W1[t] - (1 + r2) * W2[t])
                    # * (
                    #     cash[t - 1]
                    #     - unit_vari_costs1[t] * q1[t]
                    #     - unit_vari_costs2[t] * q2[t]
                    #     + r0 * W0[t]
                    #     - r1 * W1[t]
                    #     - r2 * W2[t]
                    # )
                    + final_coefficient[3] * q1[t]
                    + final_coefficient[4] * q2[t]
                    + final_coefficient[5])

        for n in range(scenario_num):
            index1 = scenario_paths1[n][t - 1]
            index2 = scenario_paths2[n][t - 1]
            demand1 = sample_details1[t - 1][index1]
            demand2 = sample_details2[t - 1][index1]

            # demand1 = sample_scenarios1[n][t-1]
            # demand2 = sample_scenarios2[n][t-1]

            rhs1_1 = 0
            rhs1_2 = 0
            if t == 1:  # actually the model in the 2nd stage
                rhs1_1 = ini_I1 - demand1
                rhs1_2 = ini_I2 - demand2
            else:
                rhs1_1 = (
                    I1_forward_values[iter_][t - 2][n]
                    + q1_pre_values[iter_][t - 2][n]
                    - demand1
                )
                rhs1_2 = (
                    I2_forward_values[iter_][t - 2][n]
                    + q2_pre_values[iter_][t - 2][n]
                    - demand2
                )
            if t < T:
                rhs2 = (
                    # ini_cash
                    # - overhead_costs[t - 1]
                    # - unit_vari_costs1[t - 1] * q1_values[iter_][t - 1][n]
                    # - unit_vari_costs2[t - 1] * q2_values[iter_][t - 1][n]
                    # + r0 * W0_forward_values[iter_][t - 1][n]
                    # - r1 * W1_forward_values[iter_][t - 1][n]
                    # - r2 * W2_forward_values[iter_][t - 1][n]
                    # + prices1[t - 1] * demand1
                    # + prices2[t - 1] * demand2
                    #
                    # if t == 1
                    # else
                    # cash_forward_values[iter_][t - 2][n]
                    # - overhead_costs[t - 1]
                    # - unit_vari_costs1[t - 1] * q1_values[iter_][t - 1][n]
                    # - unit_vari_costs2[t - 1] * q2_values[iter_][t - 1][n]
                    # - r1 * W1_forward_values[iter_][t - 1][n]
                    # + r0 * W0_forward_values[iter_][t - 1][n]
                    # - r2 * W2_forward_values[iter_][t - 1][n]
                    # + prices1[t - 1] * demand1
                    # + prices2[t - 1] * demand2

                    + prices1[t - 1] * demand1
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
            if t < T:
                models[t].setAttr("RHS", models[t].getConstrs()[2], rhs2)
                models[t].setAttr("RHS", models[t].getConstrs()[3], rhs3_1)
                models[t].setAttr("RHS", models[t].getConstrs()[4], rhs3_2)

            # optimize
            models[t].optimize()
            # if iter_ == 0 and t == 1:
            #     models[t].write('iter' + str(iter_) + '_sub_' + str(t) + '^' + str(n+1) + '.lp')
            #     models[t].write('iter' + str(iter_) + '_sub_' + str(t) + '^' + str(n+1) + '.sol')
            #     pass

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
        [[0 for s in range(sample_nums_backward[t])] for n in range(scenario_num)]
        for t in range(T)
    ]
    slope1_1values = [
        [[0 for s in range(sample_nums_backward[t])] for n in range(scenario_num)]
        for t in range(T)
    ]
    slope1_2values = [
        [[0 for s in range(sample_nums_backward[t])] for n in range(scenario_num)]
        for t in range(T)
    ]
    slope2_values = [
        [[0 for s in range(sample_nums_backward[t])] for n in range(scenario_num)]
        for t in range(T)
    ]
    slope3_1values = [
        [[0 for s in range(sample_nums_backward[t])] for n in range(scenario_num)]
        for t in range(T)
    ]
    slope3_2values = [
        [[0 for s in range(sample_nums_backward[t])] for n in range(scenario_num)]
        for t in range(T)
    ]

    for t in range(T, 0, -1):
        for n in range(scenario_num):
            S = len(demands_all[t - 1])
            for s in range(S):
                demand1 = demands_all[t - 1][s][0]
                demand2 = demands_all[t - 1][s][1]
                if t == 1:
                    rhs1_1 = ini_I1 - demand1
                    rhs1_2 = ini_I2 - demand2
                else:
                    rhs1_1 = (
                        I1_forward_values[iter_][t - 2][n]
                        + q1_pre_values[iter_][t - 2][n]
                        - demand1
                    )
                    rhs1_2 = (
                        I2_forward_values[iter_][t - 2][n]
                        + q2_pre_values[iter_][t - 2][n]
                        - demand2
                    )
                if t < T:  # test for T
                    rhs2 = (
                        # ini_cash
                        # - overhead_costs[t - 1]
                        # - unit_vari_costs1[t - 1] * q1_values[iter_][t - 1][n]
                        # - unit_vari_costs2[t - 1] * q2_values[iter_][t - 1][n]
                        # - r1 * W1_forward_values[iter_][t - 1][n]
                        # + r0 * W0_forward_values[iter_][t - 1][n]
                        # - r2 * W2_forward_values[iter_][t - 1][n]
                        # + prices1[t - 1] * demand1
                        # + +prices2[t - 1] * demand2
                        # if t == 1
                        # else cash_forward_values[iter_][t - 2][n]
                        # - overhead_costs[t - 1]
                        # - unit_vari_costs1[t - 1] * q1_values[iter_][t - 1][n]
                        # - unit_vari_costs2[t - 1] * q2_values[iter_][t - 1][n]
                        # - r1 * W1_forward_values[iter_][t - 1][n]
                        # + r0 * W0_forward_values[iter_][t - 1][n]
                        # - r2 * W2_forward_values[iter_][t - 1][n]
                        # + prices1[t - 1] * demand1
                        # + prices2[t - 1] * demand2

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

                # # de set the lb and ub for some variables
                # I1[t - 1].setAttr(GRB.Attr.LB, 0.0)
                # I1[t - 1].setAttr(GRB.Attr.UB, GRB.INFINITY)
                # B1[t - 1].setAttr(GRB.Attr.LB, 0.0)
                # B1[t - 1].setAttr(GRB.Attr.UB, GRB.INFINITY)
                # I2[t - 1].setAttr(GRB.Attr.LB, 0.0)
                # I2[t - 1].setAttr(GRB.Attr.UB, GRB.INFINITY)
                # B2[t - 1].setAttr(GRB.Attr.LB, 0.0)
                # B2[t - 1].setAttr(GRB.Attr.UB, GRB.INFINITY)
                # if t < T:
                #     cash[t - 1].setAttr(GRB.Attr.LB, -GRB.INFINITY)
                #     cash[t - 1].setAttr(GRB.Attr.UB, GRB.INFINITY)

                # optimize
                models[t].optimize()
                pi = models[t].getAttr(GRB.Attr.Pi)
                rhs = models[t].getAttr(GRB.Attr.RHS)
                # if iter_ == 2 and t == 1 and n == 0 and s == 0:
                #     models[t].write(
                #         "iter"
                #         + str(iter_)
                #         + "_sub_"
                #         + str(t)
                #         + "^"
                #         + str(n + 1)
                #         + "_"
                #         + str(s + 1)
                #         + "back.lp"
                #     )
                #     models[t].write(
                #         "iter"
                #         + str(iter_)
                #         + "_sub_"
                #         + str(t)
                #         + "^"
                #         + str(n + 1)
                #         + "_"
                #         + str(s + 1)
                #         + "back.sol"
                #     )
                #     pass

                num_con = len(pi)
                if t < T:
                    intercept_values[t - 1][n][s] += (
                        -pi[0] * demand1
                        - pi[1] * demand2
                        + overhead_costs[t]
                        - prices1[t - 1] * demand1
                        - prices2[t - 1] * demand2
                        + pi[2]
                        * (
                            prices1[t - 1] * demand1 + prices2[t - 1] * demand2
                        )  # previously not include the following 1 lines
                        # - pi[2] * overhead_costs[t - 1]
                    )
                else:
                    intercept_values[t - 1][n][s] += (
                        -pi[0] * demand1
                        - pi[1] * demand2
                        - prices1[t - 1] * demand1
                        - prices2[t - 1] * demand2
                        # + pi[2] * prices1[t - 1] * demand1  # following 3 lines
                        # + pi[2] * prices2[t - 1] * demand2
                        # - pi[2] * overhead_costs[t - 1]
                    )

                for sk in range(5, num_con):  # previously inside the above loop
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
print("after %d iteration, sample numer %d, scenario %d: " % (iter_, sample_num, scenario_num))
print("final expected cash balance is %.2f" % final_value)
print("ordering Q1 in the first period is %.2f" % Q1)
print("ordering Q2 in the first period is %.2f" % Q2)
print("cpu time is %.3f s" % cpu_time)