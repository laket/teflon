"""
This code was modified from https://github.com/sfujim/TD3.
The original code was written for the paper,

Scott Fujimoto, Herke van Hoof, David Meger,
Addressing Function Approximation Error in Actor-Critic Methods, ICML 2018

If you want to use this code, confirm the conditions of use in https://github.com/sfujim/TD3.

"""

import logging
import numpy as np
import tensorflow as tf
import gym
import argparse
import shutil

from teflon import utils
from teflon.policy import SAC_eager as SAC
from teflon.policy import DDPG

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
tf.enable_eager_execution(config=config)

# Runs policy for X episodes and returns average reward
def evaluate_policy(policy, eval_episodes=1):
    avg_reward = 0.
    for _ in range(eval_episodes):
        obs = env.reset()

        done = False
        while not done:
            # たぶんbatch dimensionが必要
            action = policy.select_action(np.array(obs))
            obs, reward, done, _ = env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes

    print ("---------------------------------------")
    print ("Evaluation over %d episodes: %f" % (eval_episodes, avg_reward))
    print ("---------------------------------------")
    tf.contrib.summary.scalar("eval_return", avg_reward, family="reward")
    return avg_reward


if __name__ == "__main__":
    logging.basicConfig(datefmt="%d/%Y %I:%M:%S", level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] (%(filename)s:%(lineno)s) %(message)s'
                        )

    
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="SAC")  # OpenAI gym environment name
    parser.add_argument("--env", default="Ant-v2")  # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)                    # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=1e4, type=int)        # How many time steps purely random policy is run for
    parser.add_argument("--eval_freq", default=5e3, type=float)            # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e7, type=int)        # Max time steps to run environment for
    parser.add_argument("--expl_noise", default=0.1, type=float)        # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)            # Batch size for both actor and critic
    parser.add_argument("--tau", default=0.001, type=float)                # Target network update rate
    parser.add_argument("--policy_noise", default=0.2, type=float)        # Noise added to target policy during critic update
    parser.add_argument("--logdir", default="./output/logdir", help="log directory")
    parser.add_argument("--parameter", default="./output/parameter", help="parameter directory")
    args = parser.parse_args()

    shutil.rmtree(args.logdir, ignore_errors=True)

    env = gym.make(args.env)
    obs = env.reset()

    # Set seeds
    env.seed(args.seed)
    tf.set_random_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # Initialize policy
    if args.policy == "SAC":
        policy = SAC.SAC(state_dim, action_dim, max_action)
    elif args.policy == "DDPG":
        policy = DDPG.DDPG(state_dim, action_dim, max_action)
    else:
        raise ValueError("invalid policy {}".format(args.policy))

    replay_buffer = utils.ReplayBuffer()
    
    # Evaluate untrained policy
    evaluations = [evaluate_policy(policy)] 

    timesteps_since_eval = 0
    episode_num = 0
    done = False
    episode_reward = 0
    episode_timesteps = 0

    checkpoint = tf.train.Checkpoint(policy=policy)
    checkpoint_manager = tf.contrib.checkpoint.CheckpointManager(checkpoint,
                                                                 directory=args.parameter,
                                                                 max_to_keep=5)


    writer = tf.contrib.summary.create_file_writer(args.logdir)
    writer.set_as_default()
    tf.contrib.summary.initialize()

    logger = logging.getLogger(__name__)

    obs = env.reset()

    total_timesteps = 0
    summary_timesteps = tf.train.create_global_step()

    with tf.contrib.summary.record_summaries_every_n_global_steps(1000):

        while total_timesteps < args.max_timesteps:
            if total_timesteps > args.start_timesteps:
                if args.policy == "SAC":
                    total_actor_loss, total_v_loss, total_q_loss = policy.train(replay_buffer,
                                                                                batch_size=args.batch_size, tau=args.tau)
                    tf.contrib.summary.scalar(name="V_Loss", tensor=total_v_loss, step=total_timesteps, family="loss")
                    tf.contrib.summary.scalar(name="Q_Loss", tensor=total_q_loss, step=total_timesteps, family="loss")
                    tf.contrib.summary.scalar(name="ActorLoss", tensor=total_actor_loss, step=total_timesteps,
                                              family="loss")
                elif args.policy == "DDPG":
                    total_actor_loss, total_critic_loss = policy.train(replay_buffer, batch_size=args.batch_size,
                                                                       tau=args.tau)
                    tf.contrib.summary.scalar(name="ActorLoss", tensor=total_actor_loss, step=total_timesteps,
                                              family="loss")
                    tf.contrib.summary.scalar(name="CriticLoss", tensor=total_critic_loss, step=total_timesteps,
                                              family="loss")
                else:
                    raise ValueError("invalid policy {}".format(policy))

            if done:
                logger.info("Total T: %d Episode Num: %d Episode T: %d Reward: %f" % (total_timesteps, episode_num, episode_timesteps, episode_reward))

                # Evaluate episode
                if timesteps_since_eval >= args.eval_freq:
                    timesteps_since_eval %= args.eval_freq
                    cur_evaluate = evaluate_policy(policy)
                    evaluations.append(cur_evaluate)
                    checkpoint_manager.save()

                    tf.contrib.summary.scalar(name="online_return", tensor=cur_evaluate, step=total_timesteps, family="loss")
                    writer.flush()

                # Reset environment
                obs = env.reset()
                done = False
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1

            # Select action randomly or according to policy
            if total_timesteps < args.start_timesteps:
                action = env.action_space.sample()
            else:
                action = policy.select_action_noise(np.array(obs), noise_level=args.expl_noise)
                action = action.clip(env.action_space.low, env.action_space.high)

            # Perform action
            new_obs, reward, done, _ = env.step(action)

            # 上限ステップ由来のdoneであれば学習上はdoneとみなさない
            if episode_timesteps + 1 == env._max_episode_steps:
                done_bool = 0
            else:
                done_bool = float(done)

            episode_reward += reward

            # Store data in replay replay_buffer
            obs = obs.astype(np.float32)
            new_obs = new_obs.astype(np.float32)
            reward, action = reward.astype(np.float32), action.astype(np.float32)
            done_bool = np.array(done_bool, dtype=np.float32)
            replay_buffer.add((obs, new_obs, action, reward, done_bool))

            obs = new_obs

            episode_timesteps += 1
            total_timesteps += 1
            summary_timesteps.assign_add(1)
            timesteps_since_eval += 1

        # Final evaluation
        cur_evaluate = evaluate_policy(policy)
        tf.contrib.summary.scalar(name="EvalReturn", tensor=cur_evaluate, step=total_timesteps)
        checkpoint_manager.save()

