#!/usr/bin/python3

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
Performance measure scripts for calling environment from multiple threads.
"""

import time
import argparse
import threading

import numpy as np
import gym

# import simulator.vec_env.subproc_vec_env as subproc_vec_env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="Ant-v2")
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--render", default=False, action="store_true")  # Sets Gym, PyTorch and Numpy seeds
    args = parser.parse_args()

    batch_size = 2048
    trial = 10
    make_env = lambda : gym.make(args.env)

    case = "thread"

    if case == "vec":
        raise NotImplementedError()
        """
        num_env = 8
        make_env = lambda: AutoResetWrapper(gym.make(args.env))
        make_envs = [make_env for i in range(num_env)]
        env = subproc_vec_env.SubprocVecEnv(make_envs)
        env.reset()

        action = np.random.normal(0, 0.1, size=[num_env, env.action_space.shape[0]]).clip(env.action_space.low,
                                                                                          env.action_space.high)

        list_during = []
        for idx_trial in range(trial):
            start_time = time.time()

            for idx_batch in range(batch_size//num_env):
                env.step_async(action)
                # Resetはwrapperでやる
                obs, rews, dones, infos = env.step_wait()

            during = time.time() - start_time
            list_during.append(during)
            print ("Trial {} Time {}".format(idx_trial, during))


        # 1回目はノイズとして無視
        list_during = list_during[1:]
        total_sample = batch_size * len(list_during)
        total_time = np.sum(list_during)

        print ("Total Time {} and {} steps/seconds".format(total_time, total_sample/total_time))


        env.reset()
        print (rews)
        env.close_extras()
        """
    elif case == "single":
        env = make_env()
        env.reset()

        # action is fixed
        action = np.random.normal(0, 0.1, size=[env.action_space.shape[0]]).clip(env.action_space.low,
                                                                                 env.action_space.high)

        list_during = []
        for idx_trial in range(trial):
            start_time = time.time()

            for idx_batch in range(batch_size):
                new_obs, reward, done, _ = env.step(action)
                if done:
                    env.reset()

            during = time.time() - start_time
            list_during.append(during)
            print ("Trial {} Time {}".format(idx_trial, during))

        # first trial is eliminated
        list_during = list_during[1:]
        total_sample = batch_size * len(list_during)
        total_time = np.sum(list_during)

        print ("Total Time {} and {} steps/seconds".format(total_time, total_sample/total_time))

    elif case == "thread":
        num_envs = 4
        env = make_env()
        env.reset()

        action = np.random.normal(0, 0.1, size=[env.action_space.shape[0]]).clip(env.action_space.low,
                                                                                 env.action_space.high)

        def _worker():
            env = make_env()
            env.reset()
            num_work = batch_size // num_envs

            for idx_batch in range(num_work):
                new_obs, reward, done, _ = env.step(action)
                if done:
                    env.reset()

        list_during = []
        for idx_trial in range(trial):
            start_time = time.time()
            threads = []

            for idx_thread in range(num_envs):
                thread = threading.Thread(target=_worker)
                thread.start()
                threads.append(thread)

            for t in threads:
                t.join()

            during = time.time() - start_time
            list_during.append(during)
            print ("Trial {} Time {}".format(idx_trial, during))

        # first trial is eliminated
        list_during = list_during[1:]
        total_sample = batch_size * len(list_during)
        total_time = np.sum(list_during)

        print ("Total Time {} and {} steps/seconds".format(total_time, total_sample/total_time))

if __name__ == "__main__":
    main()
