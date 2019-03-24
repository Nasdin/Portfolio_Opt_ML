from collections import namedtuple
from typing import List

import numpy as np

# win rate - The probability to win in a given day
# win_reward_ratio - When it strategy does win, how much larger the wins are from the loss.
# in absolute terms - How much the gains/losses are
# mutation frequency - Markets are bound to change, this determines how stable this strategy in retaining its edge
# mutation std - When a strategy change, by how much will it change.
CreateStrategy = namedtuple("StrategyData", ('win_rate', 'win_rate_std',
                                             'win_reward_ratio', 'win_reward_ratio_std',
                                             'win_size', 'win_size_std',
                                             'mutation_frequency', 'mutation_std'), verbose=True)


def _mutate_strategy(strategy: CreateStrategy, mutations: int) -> np.ndarray:
    pass


def batch(seed: CreateStrategy, size: int):
    """

    :param seed: Strategy that will be used to generate interval returns
    :param size: The number of intervals to return
    :return: numpy array of shape (size,2 )
    0 - Equity return for that interval
    1 - Bool if win or loss
    """

    empty_return_array = np.zeros((size, 2))

    pass


def stream():
    pass


def generate_correlated_strategy(strategy_seeds: List[CreateStrategy]):
    pass
