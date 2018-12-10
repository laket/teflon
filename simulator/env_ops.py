# Copyright 2018 Oiki Tomoaki. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
Tensorflow Operators contains multiple environments
"""

import threading
import numpy as np
import tensorflow as tf


class MultiThreadEnv(object):
    """
    This contains multiple environments.
    When step() is called, all of them forward one-step.

    This serve tensorflow operators to manipulate multiple environments.
    """

    def __init__(self, env_make, batch_size, thread_pool=4):
        """
        :param env_make: function to make environments.
        :param batch_size: batch_size
        ;param thread_pool thread pool size
        """
        assert batch_size % thread_pool == 0

        self.batch_size = batch_size
        self.thread_pool = thread_pool
        self.batch_thread = batch_size // thread_pool
        self.envs = [env_make() for idx_env in range(batch_size)]

        # collects environment information
        sample_env = env_make()
        sample_obs = sample_env.reset()
        self._sample_env = sample_env
        self.observation_shape = sample_obs.shape
        # episode time limit
        self.max_episode_steps = sample_env.spec.max_episode_steps

        # MEMO now all dimension in action space must have the same range.
        assert np.all(self._sample_env.action_space.high == self._sample_env.action_space.high[0])
        assert np.all(self._sample_env.action_space.low == self._sample_env.action_space.low[0])

        self.list_obs = [None] * self.batch_size
        self.list_rewards = [None] * self.batch_size
        self.list_done = [None] * self.batch_size
        self.list_steps = [0] * self.batch_size

        self.py_reset()

    def step(self, action, name=None):
        """take 1-step in all environments.

        :param tf.Tensor action: float32[batch_size, dim_action]
        :param name: Operator名
        :rtype: (tf.Tensor, tf.Tensor, tf.Tensor)
        :return:
           (obs, reward, done)
          obs = [batch_size, dim_obs]
          reward = [batch_size]
          done = [batch_size]
          reach_limit = [batch_size] : whether each environment reached time limit or not.
        """

        with tf.variable_scope(name, default_name="MultiStep"):
            obs, reward, done, reach_limit = tf.py_func(self.py_step, [action], [tf.float32, tf.float32, tf.bool, tf.bool])
            obs.set_shape((self.batch_size, ) + self.observation_shape)
            reward.set_shape((self.batch_size,))
            done.set_shape((self.batch_size,))
            reach_limit.set_shape((self.batch_size,))

        return obs, reward, done, reach_limit

    def observation(self, name=None):
        with tf.variable_scope(name, default_name="MultiObservation"):
            obs = tf.py_func(self.py_observation, [], [tf.float32])[0]
            obs.set_shape((self.batch_size,) + self.observation_shape)

        return obs

    def py_step(self, action):
        """

        :param np.array action: np.float32 [batch_size, dim_action]
        :return:
          (obs, reward, done)

          obs = [batch_size, dim_obs]
          reward = [batch_size]
          done = [batch_size]
        """


        def _process(offset):
            for idx_env in range(offset, offset+self.batch_thread):
                new_obs, reward, done, _ = self.envs[idx_env].step(action[idx_env, :].astype(np.float64))

                # can we use np.float32?
                new_obs = new_obs.astype(np.float32)
                reward = reward.astype(np.float32)

                self.list_obs[idx_env] = new_obs
                self.list_rewards[idx_env] = reward
                self.list_done[idx_env] = done
                self.list_steps[idx_env] += 1

        threads = []
        for i in range(self.thread_pool):
            thread = threading.Thread(target=_process, args=[i*self.batch_thread])
            thread.start()
            threads.append(thread)

        for t in threads:
            t.join()

        obs = np.stack(self.list_obs, axis=0)
        reward = np.stack(self.list_rewards, axis=0)
        done = np.stack(self.list_done, axis=0)
        reach_limit = [False] * self.batch_size

        # TODO reset from multiple threads
        for i in range(self.batch_size):
            if self.list_done[i]:
                reset_obs = self.envs[i].reset()

                if self.list_steps[i] == self.max_episode_steps:
                    reach_limit[i] = True

                self.list_steps[i] = 0
                self.list_obs[i] = reset_obs.astype(np.float32)

        reach_limit = np.stack(reach_limit, axis=0)

        return obs, reward, done, reach_limit

    def py_observation(self):
        obs = np.stack(self.list_obs, axis=0)
        return obs

    def py_reset(self):
        for idx_env, env in enumerate(self.envs):
            obs = env.reset()
            self.list_obs[idx_env] = obs.astype(np.float32)

        return np.stack(self.list_obs, axis=0)

    @property
    def action_dim(self):
        return self._sample_env.action_space.shape[0]

    @property
    def max_action(self):
        return float(self._sample_env.action_space.high[0])

    @property
    def min_action(self):
        return float(self._sample_env.action_space.low[0])

    @property
    def state_dim(self):
        return self._sample_env.observation_space.shape[0]


def py_main():

    import gym
    import time

    env_maker = lambda : gym.make("Ant-v2")
    sample_env = env_maker()
    obs = sample_env.reset()
    print ("shape:", obs.shape)

    total_size = 2048
    batch_size = 256
    env = MultiThreadEnv(env_maker, batch_size=batch_size, thread_pool=4)

    action = np.random.normal(0, 0.1, size=[batch_size, sample_env.action_space.shape[0]]).clip(
        sample_env.action_space.low, sample_env.action_space.high)

    action = tf.constant(action, dtype=tf.float32)
    print ("action generated!")

    list_during = []
    for idx_trial in range(10):
        start_time = time.time()

        for i in range(total_size//batch_size):
            obs, reward, done = env.py_step(action)

        during = time.time() - start_time
        list_during.append(during)
        print("Trial {} Time {}".format(idx_trial, during))

    list_during = list_during[1:]
    total_sample = total_size * len(list_during)
    total_time = np.sum(list_during)

    print ("Total Time {} and {} steps/seconds".format(total_time, total_sample/total_time))


def main():
    import gym
    import time

    env_maker = lambda : gym.make("Ant-v2")
    sample_env = env_maker()
    obs = sample_env.reset()
    print ("shape:", obs.shape)

    total_size = 2048
    batch_size = 256
    env = MultiThreadEnv(env_maker, batch_size=batch_size, thread_pool=4)
    env.py_reset()

    action = np.random.normal(0, 0.1, size=[batch_size, sample_env.action_space.shape[0]]).clip(
        sample_env.action_space.low, sample_env.action_space.high)
    print ("action generated!")

    action = tf.constant(action)
    tf_obs, tf_reward, tf_done, tf_reach = env.step(action)

    list_during = []

    sess = tf.Session()

    for idx_trial in range(10):
        start_time = time.time()

        for i in range(total_size//batch_size):
            obs = sess.run(tf_obs)

        during = time.time() - start_time
        list_during.append(during)
        print("Trial {} Time {}".format(idx_trial, during))

    list_during = list_during[1:]
    total_sample = total_size * len(list_during)
    total_time = np.sum(list_during)

    print ("Total Time {} and {} steps/seconds".format(total_time, total_sample/total_time))


if __name__ == "__main__":
    main()
    #py_main()
