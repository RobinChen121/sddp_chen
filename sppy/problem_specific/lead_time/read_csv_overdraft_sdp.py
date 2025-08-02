"""
Python version: 3.12.7
Author: Zhen Chen, chen.zhen5526@gmail.com
Date: 2025/4/20 10:31
Description: 
    read the csvs of the results in the overdraft problems.

"""
import pandas as pd

df_sdp = pd.read_csv(
    "/Users/zhenchen/Library/CloudStorage/OneDrive-BrunelUniversityLondon/Numerical-tests/overdraft/c++/sdp_testing.csv"
)
df_sdp.drop_duplicates(keep='first', inplace=True, ignore_index=True)
df_sdp.drop(df_sdp[df_sdp['demand pattern'] == 'demand pattern'].index, inplace=True)
df_sdp.rename(columns=lambda x: x.strip(), inplace=True)
# df_sdp['demand pattern'] = df_sdp['demand pattern'].astype(int)
# df_sdp['overhead'] = df_sdp['overhead'].astype(int)
# df_sdp['price'] = df_sdp['price'].astype(int)
# df_sdp['interest rate'] = df_sdp['interest rate'].astype(float)
df_sdp = df_sdp.apply(pd.to_numeric)

df_sddp = pd.read_csv(
    "/Users/zhenchen/Library/CloudStorage/OneDrive-BrunelUniversityLondon/Numerical-tests/overdraft/c++/sddp_singleproduct_testing.csv"
)
df_sddp = df_sddp[[' demand pattern', ' interest rate', ' overhead', ' price', ' final value', ' time', ' Q']]
df_sddp = df_sddp.groupby([' demand pattern', ' interest rate', ' overhead', ' price']).mean().reset_index()
df_sddp.rename(columns={' final value': 'sddp value', ' time': 'sddp time', ' Q': 'sddp Q'}, inplace=True)
df_sddp.rename(columns=lambda x: x.strip(), inplace=True)

df_sddp_enhance = pd.read_csv(
    "/Users/zhenchen/Library/CloudStorage/OneDrive-BrunelUniversityLondon/Numerical-tests/overdraft/c++/sddp_singleproduct_enhance_testing.csv"
)
df_sddp_enhance = df_sddp_enhance[
    [' demand pattern', ' interest rate', ' overhead', ' price', ' final value', ' time', ' Q']]
df_sddp_enhance = df_sddp_enhance.groupby(
    [' demand pattern', ' interest rate', ' overhead', ' price']).mean().reset_index()
df_sddp_enhance.rename(
    columns={' final value': 'sddp enhance value', ' time': 'sddp enhance time', ' Q': 'sddp enhance Q'}, inplace=True)
df_sddp_enhance.rename(columns=lambda x: x.strip(), inplace=True)

df_sddp_further = pd.read_csv(
    "/Users/zhenchen/Library/CloudStorage/OneDrive-BrunelUniversityLondon/Numerical-tests/overdraft/c++/sddp_singleproduct_enhancefurther_testing.csv"
)
df_sddp_further = df_sddp_further[
    [' demand pattern', ' interest rate', ' overhead', ' price', ' final value', ' time', ' Q']]
df_sddp_further = df_sddp_further.groupby(
    [' demand pattern', ' interest rate', ' overhead', ' price']).mean().reset_index()
df_sddp_further.rename(
    columns={' final value': 'sddp further value', ' time': 'sddp further time', ' Q': 'sddp further Q'}, inplace=True)
df_sddp_further.rename(columns=lambda x: x.strip(), inplace=True)

df_sddp_further3 = pd.read_csv(
    "/Users/zhenchen/Library/CloudStorage/OneDrive-BrunelUniversityLondon/Numerical-tests/overdraft/c++/sddp_singleproduct_enhancefurtherSKIP_testing.csv"
)
df_sddp_further3 = df_sddp_further3[
    [' demand pattern', ' interest rate', ' overhead', ' price', ' final value', ' time', ' Q']]
df_sddp_further3 = df_sddp_further3.groupby(
    [' demand pattern', ' interest rate', ' overhead', ' price']).mean().reset_index()
df_sddp_further3.rename(
    columns={' final value': 'sddp further value3', ' time': 'sddp further time3', ' Q': 'sddp further3 Q'}, inplace=True)
df_sddp_further3.rename(columns=lambda x: x.strip(), inplace=True)

df_final = df_sdp.merge(df_sddp, on=['demand pattern', 'interest rate', 'overhead', 'price'], how='outer')
df_final = df_final.merge(df_sddp_enhance, on=['demand pattern', 'interest rate', 'overhead', 'price'], how='outer')
df_final = df_final.merge(df_sddp_further, on=['demand pattern', 'interest rate', 'overhead', 'price'], how='outer')
df_final = df_final.merge(df_sddp_further3, on=['demand pattern', 'interest rate', 'overhead', 'price'], how='outer')

df_final['gap'] = - df_final['final value'] + df_final['sddp value']
df_final['gap enhance'] = -df_final['final value'] + df_final['sddp enhance value']
df_final['gap further'] = -df_final['final value'] + df_final['sddp further value']
df_final['gap further3'] = -df_final['final value'] + df_final['sddp further value3']
df_final['abs gap'] = abs(df_final['final value'] - df_final['sddp value'])
df_final['abs gap enhance'] = abs(df_final['final value'] - df_final['sddp enhance value'])
df_final['abs gap further'] = abs(df_final['final value'] - df_final['sddp further value'])
df_final['abs gap further3'] = abs(df_final['final value'] - df_final['sddp further value3'])
df_final['abs value'] = abs(df_final['final value'])
df_final['abs value sddp'] = abs(df_final['sddp value'])
df_final['abs value enhance'] = abs(df_final['sddp enhance value'])
df_final['abs value further'] = abs(df_final['sddp further value'])
df_final['abs value further3'] = abs(df_final['sddp further value3'])
df_final.to_csv(
    "/Users/zhenchen/Library/CloudStorage/OneDrive-BrunelUniversityLondon/Numerical-tests/overdraft/c++/gap_singleproduct_final.csv", index=False
    )
