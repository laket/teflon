#!/usr/bin/python3

import gym
import time
import argparse
import numpy as np
import tensorflow as tf

from teflon.policy import DDPG
from teflon.policy import SAC_eager as SAC

tf.enable_eager_execution()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="Ant-v2")
    # TODO load SavedModel
    parser.add_argument("--policy", default="SAC")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--render", default=False, action="store_true")
    parser.add_argument("--parameter", default="parameter", required=True)
    args = parser.parse_args()

    env = gym.make(args.env)

    render = args.render
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # Set seeds
    env.seed(args.seed)
    tf.set_random_seed(args.seed)
    np.random.seed(args.seed)

    # Initialize policy
    if args.policy == "DDPG":
        policy = DDPG.DDPG(state_dim, action_dim, max_action)
    elif args.policy == "SAC":
        policy = SAC.SAC(state_dim, action_dim, max_action)
    else:
        raise NotImplementedError()

    checkpoint = tf.train.Checkpoint(policy=policy)
    checkpoint.restore(args.parameter)

    eval_episodes = 10

    avg_reward = 0.
    for _ in range(eval_episodes):
        obs = env.reset()

        done = False
        while not done:
            action = policy.select_action(np.array(obs))
            obs, reward, done, _ = env.step(action)

            if render:
                env.render()
                # Tabでカメラを切り替えるのを推奨
                time.sleep(0.02)

            avg_reward += reward

    avg_reward /= eval_episodes

    print ("---------------------------------------")
    print ("Evaluation over %d episodes: %f" % (eval_episodes, avg_reward))
    print ("---------------------------------------")


if __name__ == "__main__":
    main()