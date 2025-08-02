"""
Python version: 3.12.7
Author: Zhen Chen, chen.zhen5526@gmail.com
Date: 2025/4/30 12:39
Description: 
    

"""
import pandas as pd
import numpy as np


def compute_gap1(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    df1['gap'] = 0.0
    df1['abs gap'] = 0.0
    num = df1.shape[0]
    for i in range(num):
        demand_pattern = df1['demand pattern'][i]
        sddp_value = df2[df2['demand pattern'] == demand_pattern]['final value2'].values[0]
        gap = df1['final value'][i] - sddp_value
        df1.loc[i, 'gap'] = gap
        df1.loc[i, 'abs gap'] = abs(df1.loc[i, 'gap'])
    return df1


df_skip = pd.read_csv(
    "/Users/zhenchen/Library/CloudStorage/OneDrive-BrunelUniversityLondon/Numerical-tests/overdraft/c++/sddp_doubleproduct_SKIP_testing.csv"
)
df_skip = df_skip[['demand pattern', 'final value', 'time', 'skip number']]
df_skip.drop_duplicates(keep='first', inplace=True, ignore_index=True)
df_skip.drop(df_skip[df_skip['demand pattern'] == 'demand pattern'].index, inplace=True)
df_skip = df_skip.apply(pd.to_numeric)
df_group_skip = df_skip.groupby(['demand pattern', 'skip number']).mean().reset_index()

df_skip3 = df_group_skip[df_group_skip['skip number'] == 3]
# df_skip3.rename(columns={"skip number": "skip number3", "final value": "final value3", "time": "skip time3"}, inplace=True)

df_skip5 = df_group_skip[df_group_skip['skip number'] == 5]
# df_skip5.rename(columns={"skip number": "skip number5", "final value": "final value5", "time": "skip time5"}, inplace=True)

df_skip7 = df_group_skip[df_group_skip['skip number'] == 7]
# df_skip7.rename(columns={"skip number": "skip number7", "final value": "final value7", "time": "skip time7"}, inplace=True)

df_skip9 = df_group_skip[df_group_skip['skip number'] == 9]
# df_skip9.rename(columns={"skip number": "skip number9", "final value": "final value9", "time": "skip time9"}, inplace=True)

df_skip11 = df_group_skip[df_group_skip['skip number'] == 11]
# df_skip11.rename(columns={"skip number": "skip number11", "final value": "final value11", "time": "skip time11"}, inplace=True)


df_sddp = pd.read_csv(
    "/Users/zhenchen/Library/CloudStorage/OneDrive-BrunelUniversityLondon/Numerical-tests/overdraft/c++/sddp_doubleproduct_overdraft_terms_testing.csv"
)
df_sddp.drop(df_sddp[df_sddp[' demand pattern'] == ' demand pattern'].index, inplace=True)
df_sddp.rename(columns=lambda x: x.strip(), inplace=True)
df_sddp.drop_duplicates(keep='first', inplace=True, ignore_index=True)
df_sddp = df_sddp.apply(pd.to_numeric)
df_enhance1 = df_sddp.groupby(['demand pattern', 'interest rate', 'overdraft_limit']).mean().reset_index()
df_enhance1 = df_enhance1[['demand pattern', 'final value', 'time', 'overdraft_limit', 'interest rate']]
df_enhance1 = df_enhance1[(df_enhance1['overdraft_limit'] == 500) & (df_enhance1['interest rate'] == 0.1)]
df_enhance1 = df_enhance1[['demand pattern', 'final value', 'time']]
# df_enhance1.rename(columns={"final value": "enhance1 final value", "time": "enhance1 time"}, inplace=True)

df_noenhacne = pd.read_csv(
    "/Users/zhenchen/Library/CloudStorage/OneDrive-BrunelUniversityLondon/Numerical-tests/overdraft/c++/sddp_doubleproduct_noenhance.csv"
)
df_noenhacne = df_noenhacne[['demand pattern', 'final value', 'time']]
df_noenhacne = df_noenhacne.apply(pd.to_numeric)
df_noenhacne.rename(columns={"final value": "noenhance final value", "time": "noenhance time"}, inplace=True)

df_final_time = df_noenhacne[['demand pattern', 'noenhance time']]
df_final_time['enhance1 time'] = df_enhance1['time'].values
df_final_time['skip3 time'] = df_skip3['time'].values
df_final_time['skip5 time'] = df_skip5['time'].values
df_final_time['skip7 time'] = df_skip7['time'].values
df_final_time['skip9 time'] = df_skip9['time'].values
df_final_time['skip11time'] = df_skip11['time'].values
df_final_time = df_final_time.round(2)
df_final_time['demand pattern'] = ['STA', 'LC1', 'LC2', "SIN1", 'SIN2', "RAND", 'EMP1', 'EMP2', 'EMP3', 'EMP4']

df_final_gap = df_enhance1[['demand pattern', 'final value']]
df_final_gap['skip3 gap'] = abs(df_skip3['final value'].values - df_final_gap['final value'].values)
df_final_gap['skip5 gap'] = abs(df_skip5['final value'].values - df_final_gap['final value'].values)
df_final_gap['skip7 gap'] = abs(df_skip7['final value'].values - df_final_gap['final value'].values)
df_final_gap['skip9 gap'] = abs(df_skip9['final value'].values - df_final_gap['final value'].values)
df_final_gap['skip11 gap'] = abs(df_skip11['final value'].values - df_final_gap['final value'].values)
df_final_gap= df_final_gap.round(2)
df_final_gap['demand pattern'] = ['STA', 'LC1', 'LC2', "SIN1", 'SIN2', "RAND", 'EMP1', 'EMP2', 'EMP3', 'EMP4']

df_final_time.to_csv(
    "/Users/zhenchen/Library/CloudStorage/OneDrive-BrunelUniversityLondon/Numerical-tests/overdraft/c++/skip_times.csv",
    index=False)
df_final_gap.to_csv(
    "/Users/zhenchen/Library/CloudStorage/OneDrive-BrunelUniversityLondon/Numerical-tests/overdraft/c++/skip_gaps.csv",
    index=False)
pass
