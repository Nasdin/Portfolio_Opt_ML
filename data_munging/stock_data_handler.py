from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


def sample_stock_data(file_name, context_dim, num_actions, num_contexts,
                      sigma, shuffle_rows=True):
    """Samples linear bandit game from stock prices dataset.
    Args:
      file_name: Route of file containing the stock prices dataset.
      context_dim: Context dimension (i.e. vector with the price of each stock).
      num_actions: Number of actions (different linear portfolio strategies).
      num_contexts: Number of contexts to sample.
      sigma: Vector with additive noise levels for each action.
      shuffle_rows: If True, rows from original dataset are shuffled.
    Returns:
      dataset: Sampled matrix with rows: (context, reward_1, ..., reward_k).
      opt_vals: Vector of expected optimal (reward, action) for each context.
    """

    with tf.gfile.Open(file_name, 'r') as f:
        contexts = np.loadtxt(f, skiprows=1)

    if shuffle_rows:
        np.random.shuffle(contexts)
    contexts = contexts[:num_contexts, :]

    betas = np.random.uniform(-1, 1, (context_dim, num_actions))
    betas /= np.linalg.norm(betas, axis=0)

    mean_rewards = np.dot(contexts, betas)
    noise = np.random.normal(scale=sigma, size=mean_rewards.shape)
    rewards = mean_rewards + noise

    opt_actions = np.argmax(mean_rewards, axis=1)
    opt_rewards = [mean_rewards[i, a] for i, a in enumerate(opt_actions)]
    return np.hstack((contexts, rewards)), (np.array(opt_rewards), opt_actions)
