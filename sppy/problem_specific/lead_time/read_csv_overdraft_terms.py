"""
Python version: 3.12.7
Author: Zhen Chen, chen.zhen5526@gmail.com
Date: 2025/4/28 09:49
Description: 
    for the comparison results of different overdraft parameter values

"""
import pandas as pd

def compute_gap1(df: pd.DataFrame) -> list:
    n = 10
    arr = [[0 for j in range(5)] for i in range(n)]
    for i in range(n):
        base_value = df[(df['demand pattern'] == i) & (df['interest rate'] == 0.1) & (df['overdraft_limit'] == 500)]['final value'].tolist()[0]
        for j in range(5):
            rate = round(j * 0.05, 2)
            value = df[(df['demand pattern'] == i) & (df['interest rate'] == rate)]['final value'].tolist()[0]
            arr[i][j] = value # - base_value
    return arr

def compute_gap2(df: pd.DataFrame) -> list:
    n = 10
    arr = [[0 for j in range(5)] for i in range(n)]
    for i in range(n):
        base_value = df[(df['demand pattern'] == i) & (df['interest rate'] == 0.1) & (df['overdraft_limit'] == 500)]['final value'].tolist()[0]
        for j in range(5):
            limit = 300 + j*100
            value = df[(df['demand pattern'] == i) & (df['overdraft_limit'] == limit)]['final value'].tolist()[0]
            arr[i][j] = value # - base_value
    return arr


df_sddp = pd.read_csv(
    "/Users/zhenchen/Library/CloudStorage/OneDrive-BrunelUniversityLondon/Numerical-tests/overdraft/c++/sddp_doubleproduct_overdraft_terms_testing.csv"
)
df_sddp.drop(df_sddp[df_sddp[' demand pattern'] == ' demand pattern'].index, inplace=True)
df_sddp.rename(columns = lambda x : x.strip(), inplace = True)
df_sddp.drop_duplicates(keep='first', inplace=True, ignore_index=True)
df_sddp = df_sddp.apply(pd.to_numeric)
df_group = df_sddp.groupby(['demand pattern', 'interest rate', 'overdraft_limit']).mean().reset_index()
df_group = df_group[['demand pattern', 'interest rate', 'overdraft_limit', 'final value']]
df_limit = df_group[df_group['overdraft_limit'] == 500]
df1 = pd.DataFrame(compute_gap1(df_limit))
df_rate = df_group[df_group['interest rate'] == 0.1]
df2 = pd.DataFrame(compute_gap2(df_rate))
df_final = pd.concat([df1, df2], axis = 1)
# df_percent = df_final.applymap(lambda x: f"{x * 100:.2f}%")
df_final.index = ["STA", "LC1", "LC2", "SIN1", "SIN2", "RAND", "EMP1", "EMP2", "EMP3", "EMP4"]
df_final2 = df_final.T

df_final.to_csv("/Users/zhenchen/Library/CloudStorage/OneDrive-BrunelUniversityLondon/Numerical-tests/overdraft/c++/gap_doubleproduct_overdraft_terms.csv", index=True
    )
df_final2.to_csv("/Users/zhenchen/Library/CloudStorage/OneDrive-BrunelUniversityLondon/Numerical-tests/overdraft/c++/gap_doubleproduct_overdraft_terms2.csv", index=True
    )

pass