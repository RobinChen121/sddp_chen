"""
created on 2025/2/15, 22:16
@author: Zhen Chen, chen.zhen5526@gmail.com
@version: 3.10

@description:

"""
import scipy.stats as stats


def get_distribution(dist_name, *args, **kwargs):
    """
    Dynamically get a SciPy distribution based on its name.

    :param dist_name: Name of the distribution (e.g., 'norm', 'poisson')
    :param args: Positional arguments for the distribution
    :param kwargs: Keyword arguments for the distribution
    :return: The corresponding SciPy distribution object
    """
    try:
        # Get the distribution object from scipy.stats
        dist = getattr(stats, dist_name)

        # Return the frozen distribution with given parameters
        return dist(*args, **kwargs)
    except AttributeError:
        raise ValueError(f"Distribution '{dist_name}' not found in scipy.stats")


# Example Usage
normal_dist = get_distribution("norm", loc=0, scale=1)  # Standard normal
poisson_dist = get_distribution("poisson", mu=3)  # Poisson with mean 3

print(normal_dist.mean())  # Expected: 0
print(poisson_dist.mean())  # Expected: 3
