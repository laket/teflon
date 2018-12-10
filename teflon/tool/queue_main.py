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

import time
import os
import logging
import numpy as np
import tensorflow as tf
import gym
import argparse
import shutil

from replay_buffer.segment_tree import ReplayBuffer
from simulator.env_ops import MultiThreadEnv
from teflon.policy.SAC import SAC as SAC
from teflon.multi_step import MultistepAggregator

tf.enable_resource_variables()


def explorer(env, policy, start_transitions=30000, explorer_noise=0.1, initial_random=True):
    """
    explorer makes transitions from environment (called Actor in Ape-X paper).

    :param MultiThreadEnv env:
    :param SAC policy:
    :param int start_transitions: transition numbers collected from random actions.
    :param bool initial_random: whether it uses random actions at first or not
                                During finetuning this is set to False.
    :return:
    """
    explorer_size = env.batch_size

    with tf.variable_scope('explorer'):
        explorer_steps = tf.get_variable('explorer_steps', shape=(), dtype=tf.int64)
        state = env.observation()

        def _initial_action():
            cur_noised_action = tf.random.uniform(shape=[explorer_size, env.action_dim],
                                              minval=env.min_action,
                                              maxval=env.max_action)
            return cur_noised_action

        def _noised_action():
            cur_noised_action = policy.explorer_action(state, noise_level=explorer_noise)
            return cur_noised_action

        if initial_random:
            noised_action = tf.cond(explorer_steps * explorer_size < start_transitions, _initial_action, _noised_action)
        else:
            noised_action = _noised_action()

        with tf.control_dependencies([state]):
            with tf.control_dependencies([explorer_steps.assign_add(1)]):
                next_state, reward, done, reach_limit = env.step(noised_action, name="explorer_step")


    # Summary
    tf.summary.scalar("explorer_steps", explorer_steps, family="speed")
    explorer_transitions = explorer_steps * explorer_size
    tf.summary.scalar("explorer_transitions", explorer_transitions, family="speed")

    transition = state, noised_action, reward, next_state, done, reach_limit
    return transition


def learner(learner_steps, policy, replay_buffer, batch_size):
    """Define learner

    :param tf.Tensor learner_steps: learner step (global step)
    :param replay_buffer:
    :param tf_DDPG.DDPG policy:
    :param batch_size:
    :return:
    """
    with tf.variable_scope("Learner"):

        # TODO Use prefetching and StagingArea
        idx, weights, components = replay_buffer.sample_proportional_from_buffer(batch_size)
        state, action, reward, next_state, discount_rate = components

        with tf.control_dependencies([learner_steps.assign_add(1)]):
            # updates, policy_loss, vf_loss_t, td_loss1
            train_op, policy_loss, vf_loss, qf_loss = policy.train(state, action, reward, next_state, discount_rate, weights)

        # Update TD-Error
        with tf.control_dependencies([train_op]):
            td_error = policy.td_error(state, action, reward, next_state, discount_rate)

        new_priorities = tf.abs(td_error)
        update_priority = replay_buffer.assign_with_eps(idx, new_priorities)

        train_op = tf.group([train_op, update_priority])

    tf.summary.histogram("importance_weights", weights, family="weight")
    tf.summary.scalar("policy_loss", policy_loss, family="loss")
    tf.summary.scalar("vf_loss", vf_loss, family="loss")
    tf.summary.scalar("qf_loss", qf_loss, family="loss")
    tf.summary.scalar('learner_steps', learner_steps, family="speed")
    learner_transitions = learner_steps * batch_size
    tf.summary.scalar('learner_transitions', learner_transitions, family="speed")

    return train_op, policy_loss, vf_loss, qf_loss


def main():
    logging.basicConfig(datefmt="%d/%Y %I:%M:%S", level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] (%(filename)s:%(lineno)s) %(message)s'
                        )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="SAC")                    # Policy name
    parser.add_argument("--env", default="Ant-v2")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--max_timesteps", default=1e7, type=float)
    parser.add_argument("--logdir", default="./output/logdir", help="log directory")
    parser.add_argument("--parameter", default="./output/parameter", help="parameter directory")
    parser.add_argument("--finetune", help="finetune parameter")
    parser.add_argument("--multi", default=1, type=int, help="multi step")
    args = parser.parse_args()

    shutil.rmtree(args.logdir, ignore_errors=True)
    logdir = os.path.join(args.logdir, "train")
    shutil.rmtree(args.parameter, ignore_errors=True)
    os.makedirs(args.parameter)

    # Set seeds
    # env.seed(args.seed)
    tf.set_random_seed(args.seed)
    np.random.seed(args.seed)

    # ========Constants=============
    lr = 3e-4
    num_envs = 2
    explorer_size = 64
    learner_size = 256
    gamma = 0.99

    is_finetune = args.finetune is not None

    multi_step_n = args.multi
    assert multi_step_n != 0 and multi_step_n is not None

    # ========Environment=============
    # consumed transitions by learner
    learner_steps = tf.train.create_global_step()

    env_maker = lambda: gym.make(args.env)
    env = MultiThreadEnv(env_maker, batch_size=explorer_size, thread_pool=num_envs)

    # make policy
    policy = SAC(env.state_dim, env.action_dim, env.max_action, lr=lr)

    transition = explorer(env, policy, initial_random=(not is_finetune))

    with tf.variable_scope("ReplayBuffer"):
        prioritized_replay_alpha = 0.6
        prioritized_replay_beta0 = 0.4
        prioritized_replay_eps = 1e-6
        replay_buffer_size = 1000000

        # state, action, reward, next_state, discount_rate
        buffer_shapes = [t.shape[1:] for t in transition[:-1]]
        buffer_dtypes = [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32]

        replay_buffer = ReplayBuffer(replay_buffer_size,
                                     shapes=buffer_shapes,
                                     dtypes=buffer_dtypes,
                                     alpha=prioritized_replay_alpha,
                                     beta=prioritized_replay_beta0,
                                     priority_eps=prioritized_replay_eps)

    # Multistep mode
    if multi_step_n > 1:
        state, action, reward, next_state, done, reach_limit = transition
        step_agg = MultistepAggregator(multi_step_n=multi_step_n)
        is_valid, history = step_agg.add_step(state, action, reward, done, reach_limit, name="explorer_step")

        def add_to_replay():
            state, action, discounted_reward, next_state, discount_rate = step_agg.make_multistep_transition(history, gamma)
            td_err = policy.explorer_td_error(state, action, discounted_reward, next_state, discount_rate)
            priority = tf.abs(td_err)
            transition = state, action, discounted_reward, next_state, discount_rate
            enqueue_op = replay_buffer.enqueue_many(transition, priority)
            return enqueue_op

        def nothing():
            return tf.no_op()

        enqueue_op = tf.cond(is_valid, add_to_replay, nothing)

    else:
        state, action, reward, next_state, done, reach_limit = transition

        # If both reach_limit and done are True, done is invalidated.
        done = tf.logical_xor(done, reach_limit)
        discount_rate = (1 - tf.cast(done, tf.float32)) * gamma

        td_error = policy.explorer_td_error(state, action, reward, next_state, discount_rate)
        priority = tf.abs(td_error)

        # TODO try original TD Error definition(max TD Error in each episode)
        transition = state, action, reward, next_state, discount_rate
        enqueue_op = replay_buffer.enqueue_many(transition, priority)

    # Enqueing threads are bound to replay_buffer.
    qr = tf.train.QueueRunner(replay_buffer, [enqueue_op])
    tf.train.add_queue_runner(qr)

    train_op, policy_loss, vf_loss, qf_loss = learner(learner_steps, policy, replay_buffer, learner_size)

    config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False,
        gpu_options=tf.GPUOptions(
            allow_growth=True
        )
    )

    summary_writer = tf.summary.FileWriterCache.get(logdir)
    summary_op = tf.summary.merge_all()

    hooks = [tf.train.StepCounterHook(every_n_steps=None, every_n_secs=10, summary_writer=summary_writer),
             tf.train.SummarySaverHook(save_secs=30, summary_writer=summary_writer, summary_op=summary_op),
             tf.train.StopAtStepHook(last_step=args.max_timesteps)]

    checkpoint = tf.train.Checkpoint(policy=policy)

    if args.finetune is not None:
        restore_status = checkpoint.restore(args.finetune)
    else:
        restore_status = None


    save_interval = 10000
    checkpoint_manager = tf.contrib.checkpoint.CheckpointManager(checkpoint,
                                                                 directory=args.parameter,
                                                                 max_to_keep=10,
                                                                 keep_checkpoint_every_n_hours=1)

    # Dummy Session to make graph definition for checkpoint_manager
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        checkpoint_manager.save()

    policy_init_op = policy.init_op()

    with tf.train.MonitoredTrainingSession(
        save_checkpoint_secs=600,
        save_summaries_secs=None,
        save_summaries_steps=None,
        log_step_count_steps=None,
        config=config,
        hooks=hooks) as sess:
        logger.info('Training started')

        if is_finetune:
            sess.run_step_fn(
                lambda step_context: restore_status.assert_consumed().run_restore_ops(step_context.session))
        else:
            sess.run_step_fn(lambda step_context: step_context.session.run(policy_init_op))

        # wait to fill the Experience Buffer
        # TODO DEBUG when this sleep is removed.
        time.sleep(30)

        while not sess.should_stop():

            cur_step, _, cur_policy_loss, cur_vf_loss, cur_qf_loss = sess.run([learner_steps, train_op, policy_loss, vf_loss, qf_loss])

            if cur_step % 100 == 0:
                print ("Step : {}  PolicyLoss: {:.3f} VLoss: {:.2f} QLoss: {:.2f}".format(
                    cur_step, cur_policy_loss, cur_vf_loss, cur_qf_loss))

            if cur_step % save_interval == 0:
                with sess._tf_sess().as_default():
                    checkpoint_manager.save(checkpoint_number=cur_step)

if __name__ == "__main__":
    main()