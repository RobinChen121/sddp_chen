"""
created on 2025/2/15, 08:17
@author: Zhen Chen, chen.zhen5526@gmail.com
@version: 3.10

@description: Test the modules in the utils/ directory.

"""
from utils.sampling import Sampling
import pytest
import numpy as np


@pytest.mark.skip(reason='tested')
def test_raise():
    with pytest.raises(ValueError) as info:
        Sampling(dist_name='normal')
    print(f"\nException message: {info.value}")


# @pytest.mark.skip(reason='tested')
def test_generate_sample():
    # sampling = Sampling(dist_name='rv_discrete')
    # pk = [0.25, 0.5, 0.25]
    # xk = [10, 20, 30]
    # sampling.set_distribution(values=(xk, pk))
    # sampling = Sampling(dist_name='poisson')
    # sampling.set_distribution(mu = 10)
    # sampling = Sampling(dist_name='norm')
    # sampling.set_distribution(loc=10, scale=3)
    sampling = Sampling(dist_name='gamma')
    mean_demand = 10
    beta = 0.5
    sampling.set_distribution(mean_demand * beta, loc=0, scale=1 / beta)
    samples = sampling.generate_samples(10)
    assert isinstance(samples, np.ndarray)
    print('\n')
    print(samples)

    mean_demand = 10
    beta = 0.5
    sampling = Sampling(dist_name='gamma', a=mean_demand * beta, loc=0, scale=1 / beta)
    samples = sampling.generate_samples(10)
    assert isinstance(samples, np.ndarray)
    print('\n')
    print(samples)


@pytest.mark.skip(reason='tested')
def test_generate_sample_path():
    sample_nums = [5, 10, 5]
    paths = Sampling.generate_scenario_paths(5, sample_nums)
    assert isinstance(paths, np.ndarray)
    print('\n')
    print(paths)
