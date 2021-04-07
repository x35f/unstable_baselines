import numpy as np
import random
def Bernstein_confidence_bound(sigma, b, n):
    return np.sqrt(2 * (sigma ** 2) / n * np.log(2 / sigma)) + \
        2 * b / (3 * n) * np.log(2 / sigma)


def get_bias