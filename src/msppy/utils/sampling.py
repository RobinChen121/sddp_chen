"""
created on 2025/2/14, 23:17
@author: Zhen Chen, chen.zhen5526@gmail.com
@version: 3.10

@description: Generate samples, scenarios, etc.

"""
import scipy.stats as st
from numpy.typing import ArrayLike


class Sampling:
    """
    This class is for generating samples and scenarios
    in the stochastic programming.

    Attributes:
        dist: The distribution object.
        T: The number of stages.
        params: Parameters for the distributions
    """

    def __init__(self, dist_name: str, T: int, params: ArrayLike):
        """

        :param dist_name: Given distribution name.
               T: The number of stages.
               params: Parameters for the distributions
        """
        self.dist = self.get_dist(dist_name)
        self.T = T
        self.params = params

    @staticmethod
    def get_dist(dist_name):
        dist_name = dist_name.lower()
        # get the distribution object
        dist = getattr(st.distributions, dist_name, None)
        if dist is None:
            raise ValueError(f'unknown distribution type: {dist_name}')
        return dist

    def generate_samples(self):
        pass
