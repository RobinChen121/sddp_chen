"""
Python version: 3.12.7
Author: Zhen Chen, chen.zhen5526@gmail.com
Date: 2025/4/21 13:20
Description: 
    read the csv and ouput some result to latex format.

"""
import pandas as pd

df = pd.read_csv(
    "/Users/zhenchen/Library/CloudStorage/OneDrive-BrunelUniversityLondon/Numerical-tests/overdraft/c++/gap_singleproduct_final.csv"
)

df_price = df.groupby(['price']).mean().reset_index()
df_price = df_price[['price', 'time', 'abs value', 'sddp time', 'sddp value', 'abs gap', 'sddp enhance time', 'sddp enhance value' , 'abs gap enhance', 'sddp further time3', 'abs value further3', 'abs gap further3']]
df_price = df_price.applymap(lambda x: round(x, 2) if isinstance(x, (int, float)) else x)

df_overhead= df.groupby(['overhead']).mean().reset_index()
df_overhead= df_overhead[['overhead', 'time', 'abs value', 'sddp time', 'abs value sddp', 'abs gap', 'sddp enhance time', 'abs value enhance', 'abs gap enhance', 'sddp further time3', 'abs value further3', 'abs gap further3']]
df_overhead = df_overhead.applymap(lambda x: round(x, 2) if isinstance(x, (int, float)) else x)

df_interest= df.groupby(['interest rate']).mean().reset_index()
df_interest= df_interest[['interest rate', 'time', 'abs value', 'sddp time', 'abs value sddp', 'abs gap', 'sddp enhance time', 'abs value enhance', 'abs gap enhance', 'sddp further time3', 'abs value further3', 'abs gap further3']]
df_interest = df_interest.applymap(lambda x: round(x, 2) if isinstance(x, (int, float)) else x)

df_demand= df.groupby(['demand pattern']).mean().reset_index()
df_demand= df_demand[['demand pattern', 'time', 'abs value', 'sddp time', 'abs value sddp', 'abs gap', 'sddp enhance time', 'abs value enhance', 'abs gap enhance', 'sddp further time3', 'abs value further3', 'abs gap further3']]
df_demand = df_demand.applymap(lambda x: round(x, 2) if isinstance(x, (int, float)) else x)

df_final =df[['time', 'abs value', 'sddp time', 'abs value sddp', 'abs gap', 'sddp enhance time', 'abs value enhance', 'abs gap enhance', 'sddp further time3', 'abs value further3', 'abs gap further3']]
df_mean = df_final.describe()
df_mean = df_mean.applymap(lambda x: round(x, 2) if isinstance(x, (int, float)) else x)
pass