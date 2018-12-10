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
This is implementation of Soft Actor Critic.

Tuomas Haarnoja · Aurick Zhou · Pieter Abbeel · Sergey Levine,
Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor, ICML2018

The detail of implementation is modified to improve score.

"""

import numpy as np
from teflon.utils import huber_loss
import tensorflow as tf

import trfl.target_update_ops as target_update


layers = tf.keras.layers
regularizers = tf.keras.regularizers
losses = tf.keras.losses
tfe = tf.contrib.eager
tds = tf.contrib.distributions


# Actor Network
class GaussianActor(tf.keras.Model):
    LOG_SIG_CAP_MAX = 2
    LOG_SIG_CAP_MIN = -20
    EPS = 1e-6

    def __init__(self, state_dim, action_dim, max_action, name='gaussian_policy'):
        super().__init__(name=name)

        # Unit number is equal to the paper.
        self.l1 = layers.Dense(256, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                               bias_initializer=tf.contrib.layers.xavier_initializer(),
                               name="L1")
        self.l2 = layers.Dense(256, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                               bias_initializer=tf.contrib.layers.xavier_initializer(),
                               name="L2")
        self.out_mean = layers.Dense(action_dim, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     bias_initializer=tf.contrib.layers.xavier_initializer(),
                                     name="L_mean")
        self.out_sigma = layers.Dense(action_dim, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                      bias_initializer=tf.contrib.layers.xavier_initializer(),
                                      name="L_sigma")

        self._max_action = max_action

        dummy_state = tf.constant(np.zeros(shape=[1, state_dim], dtype=np.float32))
        self(dummy_state)

    def _dist_from_states(self, states):
        """make action distributions from states

        :param tf.Tensor states:
        :rtype: tds.MultivariateNormalDiag
        :return:
        """
        features = self.l1(states)
        features = tf.nn.relu(features)
        features = self.l2(features)
        features = tf.nn.relu(features)
        mu_t = self.out_mean(features)

        log_sigma_t = self.out_sigma(features)
        log_sigma_t = tf.clip_by_value(log_sigma_t, self.LOG_SIG_CAP_MIN, self.LOG_SIG_CAP_MAX)

        dist = tds.MultivariateNormalDiag(loc=mu_t, scale_diag=tf.exp(log_sigma_t))

        return dist

    def call(self, states):
        dist = self._dist_from_states(states)
        raw_actions = dist.sample()
        log_pis = dist.log_prob(raw_actions)

        actions = tf.tanh(raw_actions)

        # for variable replacement
        diff = tf.reduce_sum(tf.log(1 - actions ** 2 + self.EPS), axis=1)
        log_pis -= diff

        actions = actions * self._max_action
        return actions, log_pis

    def mean_action(self, states):
        dist = self._dist_from_states(states)
        raw_actions = dist.mean()
        actions = tf.tanh(raw_actions) * self._max_action

        return actions

    def log_pis_for(self, states, actions):
        """calculates log pis for actions (used to calculate priority)

        :param states:
        :param actions:
        :return:
        """
        dist = self._dist_from_states(states)
        raw_actions = tf.atanh(actions)

        log_pis = dist.log_prob(raw_actions)
        diff = tf.reduce_sum(tf.log(1 - actions ** 2 + self.EPS), axis=1)
        log_pis -= diff

        return log_pis


class CriticV(tf.keras.Model):
    def __init__(self, state_dim, name='vf'):
        super().__init__(name=name)

        self.l1 = layers.Dense(256, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                               name="L1")
        self.l2 = layers.Dense(256, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                               name="L2")

        self.l3 = layers.Dense(1, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                               name="L2")

        dummy_state = tf.constant(np.zeros(shape=[1, state_dim], dtype=np.float32))
        self(dummy_state)

    def call(self, states):
        features = self.l1(states)
        features = tf.nn.relu(features)
        features = self.l2(features)
        features = tf.nn.relu(features)
        values = self.l3(features)

        values = tf.squeeze(values, axis=1, name="values")
        return values

class CriticQ(tf.keras.Model):
    def __init__(self, state_dim, action_dim, name='vq'):
        super().__init__(name=name)

        self.l1 = layers.Dense(256, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                               name="L1")
        self.l2 = layers.Dense(256, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                               name="L2")

        self.l3 = layers.Dense(1, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                               name="L2")

        dummy_state = tf.constant(np.zeros(shape=[1, state_dim], dtype=np.float32))
        dummy_action = tf.constant(np.zeros(shape=[1, action_dim], dtype=np.float32))
        self([dummy_state, dummy_action])


    def call(self, inputs):
        [states, actions] = inputs
        features = tf.concat([states, actions], axis=1)
        features = self.l1(features)
        features = tf.nn.relu(features)
        features = self.l2(features)
        features = tf.nn.relu(features)
        values = self.l3(features)

        values = tf.squeeze(values, axis=1)
        return values



class SAC(tf.contrib.checkpoint.Checkpointable):
    def __init__(self, state_dim, action_dim, max_action, lr=3e-4):
        # in the paper, this value was used for Ant-v1, HalfCheetah-v1
        self.scale_reward = 5.0

        self.actor = GaussianActor(state_dim, action_dim, max_action)
        self.actor_optimizer = tf.train.AdamOptimizer(learning_rate=lr)

        self.vf = CriticV(state_dim)
        self.vf_target = CriticV(state_dim)
        self.vf_optimizer = tf.train.AdamOptimizer(learning_rate=lr)

        vf_init = target_update.update_target_variables(self.vf_target.weights, self.vf.weights)
        self.list_init_assign = [vf_init]

        self.qf1 = CriticQ(state_dim, action_dim, name="vq1")
        self.qf2 = CriticQ(state_dim, action_dim, name="vq2")

        self.qf1_optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.qf2_optimizer = tf.train.AdamOptimizer(learning_rate=lr)

        # critical session to prevent from accessing concurrent access from explorer
        self.explorer_lock = tf.contrib.framework.CriticalSection()

    def init_op(self):
        """returns initialization operator

        :return:
        """
        return tf.group(self.list_init_assign)

    def explorer_action(self, state, noise_level=None):
        """returns actions for state to explore state space.

        :param state:
        :param update_period:
        :return:
        """
        # TODO try delay update
        def _action():
            cur_action, _ = self.actor(state)
            return cur_action

        action = self.explorer_lock.execute(_action)

        return action

    def train(self, states, actions, rewards, next_states, discount_rate, weights, tau=0.005):
        assert len(rewards.shape) == 1
        assert len(discount_rate.shape) == 1
        assert len(weights.shape) == 1

        # Critic Update
        with tf.device("/gpu:0"):
            q1 = self.qf1([states, actions])
            q2 = self.qf2([states, actions])
            vf_next_target_t = self.vf_target(next_states)

            # Equation (7, 8)
            ys = tf.stop_gradient(
                self.scale_reward * rewards + discount_rate * vf_next_target_t
            )

            td_loss1 = tf.reduce_mean(huber_loss(ys - q1) * weights)
            td_loss2 = tf.reduce_mean(huber_loss(ys - q2) * weights)

            # Equation (9)
            q1_grad = tf.gradients(td_loss1, self.qf1.trainable_variables)
            update_q1 = self.qf1_optimizer.apply_gradients(zip(q1_grad, self.qf1.trainable_variables))
            q2_grad = tf.gradients(td_loss2, self.qf2.trainable_variables)
            update_q2 = self.qf2_optimizer.apply_gradients(zip(q2_grad, self.qf2.trainable_variables))

            update_q = tf.group([update_q1, update_q2])

        # Actor Update
        with tf.device("/gpu:0"):
            vf_t = self.vf(states)
            sample_actions, log_pi = self.actor(states)

            tf.contrib.summary.scalar(name="log_pi_min", tensor=tf.reduce_min(log_pi))
            tf.contrib.summary.scalar(name="log_pi_max", tensor=tf.reduce_max(log_pi))

            # TODO lock for explorer_td_error
            with tf.control_dependencies([update_q]):
                q1 = self.qf1([states, sample_actions])
                q2 = self.qf2([states, sample_actions])
            min_q = tf.minimum(q1, q2)

            # Equation (12)
            policy_loss = tf.reduce_mean((log_pi - q1) * weights)

            # Equation (5)
            target_vf = tf.stop_gradient(min_q - log_pi)

            #vf_loss_t = 0.5 * tf.reduce_mean((target_vf - vf_t)**2 * weights)
            vf_loss_t = tf.reduce_mean(huber_loss(target_vf - vf_t) * weights)

            # Equation (6)
            vf_grad = tf.gradients(vf_loss_t, self.vf.trainable_variables)
            update_vf = self.vf_optimizer.apply_gradients(zip(vf_grad, self.vf.trainable_variables))

            # Equation (13)
            actor_grad = tf.gradients(policy_loss, self.actor.trainable_variables)

            # Actor can be accessed from explorer.
            def _update_actor():
                update_actor = self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor.trainable_variables))
                return update_actor

            update_actor = self.explorer_lock.execute(_update_actor)

            with tf.control_dependencies([update_vf]):
                update_vf_target = target_update.update_target_variables(self.vf_target.weights, self.vf.weights,
                                                                         tau)

        updates = tf.group([update_q, update_vf, update_actor, update_vf_target])

        return updates, policy_loss, vf_loss_t, td_loss1


    def td_error(self, states, actions, rewards, next_states, discount_rate):
        """calculates TD error (used for priority)

        :param states:
        :param actions:
        :param rewards:
        :param next_states:
        :param tf.Tensor discount_rate: Tensor<batch_size>. (depends on how far next_state is).
                                       If done, discount_rate = 0
        :return:
          tf.Tensor (batch_size)
        """
        assert len(rewards.shape) == 1
        assert len(discount_rate.shape) == 1

        with tf.device("/gpu:0"):
            q1 = self.qf1([states, actions])
            # TODO try TD3 like TD-error calculation
            # q2 = self.qf2([states, actions])
            vf_next_target_t = self.vf_target(next_states)

            # Equation (7, 8)
            ys = tf.stop_gradient(
                self.scale_reward * rewards + discount_rate * vf_next_target_t
            )

            td_error = ys - q1

        return td_error


    def explorer_td_error(self, state, action, reward, next_state, discount):
        """calculates TD error (used for priority) (used from explorer)

        :return:
        """
        assert len(reward.shape) == 1

        def call_td_error():
            return self.td_error(state, action, reward, next_state, discount)

        td_error = self.explorer_lock.execute(call_td_error)
        return td_error


