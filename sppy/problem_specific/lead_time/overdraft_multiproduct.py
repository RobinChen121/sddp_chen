"""
Python version: 3.12.7
Author: Zhen Chen, chen.zhen5526@gmail.com
Date: 2025/3/14 18:21
Description: 
    

"""

import time
from gurobipy import Model, GRB, Env
from sppy.utils.sampling import Sampling, generate_scenario_paths
from typing import Any

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
unit_salvages1 = [0.5 * unit_vari_costs1[t] for t in range(T)]
unit_salvages2 = [0.5 * unit_vari_costs2[t] for t in range(T)]
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
        q1_values[iter_][0] = [q1[0].x for n in range(N)]
        W0_forward_values[iter_][0] = [W0[0].x for n in range(N)]
        W1_forward_values[iter_][0] = [W1[0].x for n in range(N)]
        W2_forward_values[iter_][0] = [W2[0].x for n in range(N)]

    iter_ += 1
