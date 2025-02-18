"""
created on 2025/2/14, 23:17
@author: Zhen Chen, chen.zhen5526@gmail.com
@version: 3.10

@description: Generate samples, scenarios, etc.

"""
import scipy.stats as st
import numpy as np
from numpy.typing import ArrayLike


def generate_scenario_paths(scenario_num: int, sample_nums: list[int]) -> ArrayLike:
    """

    Args:
        scenario_num: The number of scenarios to be generated.
        sample_nums: Given sample numbers at each stage

    Returns:
        An numpy array of scenario paths.
    """
    T = len(sample_nums)
    scenario_paths = [[0 for _ in range(T)] for _ in range(scenario_num)]
    for i in range(scenario_num):
        # np.random.seed(10000)
        for t in range(T):
            rand_index = np.random.randint(0, sample_nums[t])
            scenario_paths[i][t] = rand_index
    return scenario_paths


class Sampling:
    """
    This class is for generating samples and scenarios
    in the stochastic programming.
    Currently, this class is not for multi variate distribution.

    Attributes:
        dist: The distribution object.
    """
    dist = None

    def __init__(self, dist_name: str, *args, **kwargs):
        """
        Args:
            dist_name: Given distribution name (e.g., 'norm', 'poisson').
        """
        self.dist_name = dist_name
        self._check_name()
        if args or kwargs:  # meaning they are not empty
            self.set_distribution(*args, **kwargs)

    def _check_name(self):
        """
            Check whether the distribution name is in the scipy.stats distributions.
        """
        dist_name = self.dist_name.lower()
        # get the distribution attribute
        dist_ = getattr(st.distributions, dist_name, None)
        if dist_ is None:
            raise ValueError(f'unknown distribution type: {dist_name}.\n'
                             'please input the name of the distribution in scipy.stats.\n'
                             'if your distribution is self given, please input "rv_discrete"'
                             )

    def set_distribution(self, *args, **kwargs):
        """
        Dynamically get a SciPy distribution based on its name.


        Returns:
            The corresponding SciPy distribution object
        """
        # Get the distribution object from scipy.stats
        dist_ = getattr(st, self.dist_name)
        # Return the frozen distribution with given parameters
        self.dist = dist_(*args, **kwargs)

    # noinspection PyTypeChecker
    def generate_samples(self, sample_num: int) -> ArrayLike:
        """
            Generate samples for one random variable at one stage.
            Use Latin Hyper Sampling.
        Args:
            sample_num: The number of samples to be generated.

        Returns:
            The samples details in a 1-D list.
        """
        samples = [0.0 for _ in range(sample_num)]
        for i in range(sample_num):
            # np.random.seed(10000)
            rand_p = np.random.uniform(i / sample_num, (i + 1) / sample_num)
            samples[i] = self.dist.ppf(rand_p)
        return samples

    def generate_samples_multi_stage(self, sample_nums: list) -> ArrayLike:
        """
            Generate samples for one random variable at one stage.
            Distribution at each stage are same.
            Use Latin Hyper Sampling.
        Args:
            sample_nums: The number of samples to be generated at each stage.

        Returns:
            The samples details in a 1-D Numpy array.
        """
        T = len(sample_nums)
        samples = [[0.0 for _ in sample_nums[t]] for t in range(T)]
        for t in range(T):
            for i in range(sample_nums[t]):
                # np.random.seed(10000)
                rand_p = np.random.uniform(i / sample_nums[t], (i + 1) / sample_nums[t])
                samples[t][i] = self.dist.ppf(rand_p)
        return samples


if __name__ == '__main__':
    sampling = Sampling(dist_name='poisson')
    sampling.set_distribution(mu=10)
    sampling.generate_samples(10)
