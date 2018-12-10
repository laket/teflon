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
test for MultiThreadEnv
"""

import numpy as np
import tensorflow as tf

from gym.envs.mujoco.ant import AntEnv

from simulator.env_ops import MultiThreadEnv


class DebugAntEnv(AntEnv):
    def get_status(self):
        return (self.sim.data.qpos, self.sim.data.qvel)
    def set_status(self, qpos, qvel):
        self.set_state(qpos, qvel)

    @property
    def spec(self):
        class Dummy(object):
            max_episode_steps = 1000
        return Dummy()

class EnvOpsTest(tf.test.TestCase):
    def test_multi_thread(self):
        """
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        """
        num_envs = 128

        env_maker = lambda: DebugAntEnv()
        sample_env = env_maker()
        sample_env.reset()
        env = MultiThreadEnv(env_maker, batch_size=num_envs, thread_pool=16)

        list_env = []
        for i in range(num_envs):
            qpos, qvel = env.envs[i].get_status()
            cur_env = DebugAntEnv()
            cur_env.set_status(qpos, qvel)
            list_env.append(cur_env)

        action = np.random.normal(0, 0.1, size=[num_envs, sample_env.action_space.shape[0]]).clip(
            sample_env.action_space.low, sample_env.action_space.high)
        tf_step = env.step(action)

        with self.test_session() as sess:
            for idx_step in range(5):
                obs, reward, done, reach_limit = sess.run(tf_step)

                for idx_env in range(num_envs):
                    py_obs, py_reward, py_done, _ = list_env[idx_env].step(action[idx_env, :])

                    self.assertAllClose(obs[idx_env, :], py_obs)
                    self.assertAllClose(reward[idx_env], py_reward)
                    self.assertAllEqual(done[idx_env], py_done)


if __name__ == "__main__":
    tf.test.main()
