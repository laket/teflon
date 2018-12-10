"""
This code was modified from https://github.com/sfujim/TD3.
The original code was written for the paper,

Scott Fujimoto, Herke van Hoof, David Meger,
Addressing Function Approximation Error in Actor-Critic Methods, ICML 2018

If you want to use this code, confirm the conditions of use in https://github.com/sfujim/TD3.

"""


import numpy as np
import tensorflow as tf

# Code based on: 
# https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py

NUMPY_DTYPE = np.float32
TF_DTYPE = tf.float32

# Simple replay replay_buffer
class ReplayBuffer(object):
    def __init__(self):
        self.storage = []

    # Expects tuples of (state, next_state, action, reward, done)
    def add(self, data):
        self.storage.append(data)

    def sample(self, batch_size=100):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x, y, u, r, d = [], [], [], [], []

        for i in ind: 
            X, Y, U, R, D = self.storage[i]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))

        return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)


def huber_loss(x, delta=1.0):
    """Reference: https://en.wikipedia.org/wiki/Huber_loss"""
    return tf.where(
        tf.abs(x) < delta,
        tf.square(x) * 0.5,
        delta * (tf.abs(x) - 0.5 * delta)
    )


class PytorchInitializer(tf.keras.initializers.Initializer):
    """PytorchのLinearにあわせたinitializer

    stdv = 1. / math.sqrt(self.weight.size(1))
    self.weight.data.uniform_(-stdv, stdv)

    :param seed:
    :return:
    """
    def __init__(self, scale=1.0, seed=None):
        self.seed = seed
        self.scale = scale

    def __call__(self, shape, dtype=None, partition_info=None):
        if dtype is None:
            dtype = TF_DTYPE

        if len(shape) == 1:
            fan_in = shape[0]
        elif len(shape) == 2:
            fan_in = shape[0]
        else:
            raise ValueError("invalid shape")

        scale = self.scale * fan_in

        stdv = 1. / tf.math.sqrt(scale)

        return tf.random_uniform(
            shape, -stdv, stdv, dtype=dtype, seed=self.seed)
