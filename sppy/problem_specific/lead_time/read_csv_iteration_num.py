"""
Python version: 3.12.7
Author: Zhen Chen, chen.zhen5526@gmail.com
Date: 2025/4/29 11:08
Description: 
    

"""
import pandas as pd

df_iter = pd.read_csv(
    "/Users/zhenchen/Library/CloudStorage/OneDrive-BrunelUniversityLondon/Numerical-tests/overdraft/c++/sddp_singleproduct_iterNum_testing.csv"
)

df_iter = df_iter[['final value', 'iter number', 'gap']]
df_iter = df_iter.apply(pd.to_numeric, errors='coerce')
df_group = df_iter.groupby('iter number').mean()
df_round = df_group.round(2)
pass
