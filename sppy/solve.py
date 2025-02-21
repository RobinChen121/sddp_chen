"""
Python version: 3.12.7
Author: Zhen Chen, chen.zhen5526@gmail.com
Date: 2025/2/17 13:39
Description: 
    Methods that solve the stochastic programming.

"""


def problem_specific(file_name: str = 'newsvendor.py'):
    import os
    file_name = os.path.dirname(os.path.abspath(__file__)) + '/problem_specific/' + file_name
    import subprocess
    subprocess.run(["python", file_name], check=True)


def general():
    pass
