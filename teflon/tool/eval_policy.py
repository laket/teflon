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
policy evaluation script
This script repeats evaluating during training.

"""

import os
import time
import logging
import argparse

import numpy as np

import gym
import tensorflow as tf

from teflon.policy import DDPG
from teflon.policy import SAC_eager as SAC

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
tf.enable_eager_execution(config=config)

# Runs policy for X episodes and returns average reward
def evaluate_policy(env, policy, eval_episodes=10):
    avg_reward = 0.
    scores = []

    for _ in range(eval_episodes):
        obs = env.reset()

        done = False
        cur_return = 0

        while not done:
            action = policy.select_action(np.array(obs))
            obs, reward, done, _ = env.step(action)
            avg_reward += reward
            cur_return += reward

        scores.append(cur_return)

    print ("Average Return {}".format(np.mean(scores)))

    return scores


def main():
    logging.basicConfig(datefmt="%d/%Y %I:%M:%S", level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] (%(filename)s:%(lineno)s) %(message)s'
                        )
    logger = logging.getLogger(__name__)


    parser = argparse.ArgumentParser()
    parser.add_argument("--parameter", dest="parameter", default="output/parameter", help="parameter directory")
    parser.add_argument("--logdir", dest="logdir", default="output/logdir", help="parameter directory")
    parser.add_argument("--policy", dest="policy", default="SAC", help="policy name(DDPG)")
    parser.add_argument("--interval", dest="interval", default=60, help="evaluation interval (seconds)")
    parser.add_argument("--env", default="Ant-v2")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--onetime", default=False, action="store_true")
    args = parser.parse_args()

    env = gym.make(args.env)

    # Set seeds
    env.seed(args.seed)
    tf.set_random_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    if args.policy == "DDPG":
        policy = DDPG.DDPG(state_dim, action_dim, max_action)
    elif args.policy == "SAC":
        policy = SAC.SAC(state_dim, action_dim, max_action)
    else:
        raise NotImplementedError()

    train_dir = os.path.join(args.logdir, "eval")

    summary_writer = tf.contrib.summary.create_file_writer(
        train_dir, flush_millis=10000)

    checkpoint = tf.train.Checkpoint(policy=policy)
    prev_checkpont = None
    interval_seconds = 30

    while True:
        with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
            latest_checkpoint = tf.train.latest_checkpoint(args.parameter)
            if prev_checkpont == latest_checkpoint:
                time.sleep(interval_seconds)
                continue

            checkpoint.restore(latest_checkpoint)
            learner_steps = int(latest_checkpoint.split("-")[-1])
            prev_checkpont = latest_checkpoint
            logger.info("Found checkpoint {}".format(latest_checkpoint))

            with tf.device("/cpu:0"):
                scores = evaluate_policy(env, policy)
            tf.contrib.summary.scalar("return_mean", np.mean(scores), family="loss", step=learner_steps)
            tf.contrib.summary.scalar("return_median", np.median(scores), family="loss", step=learner_steps)
            tf.contrib.summary.histogram("return_hist", tf.constant(scores), family="loss", step=learner_steps)

            if args.onetime:
                break

            time.sleep(interval_seconds)


if __name__ == "__main__":
    main()

