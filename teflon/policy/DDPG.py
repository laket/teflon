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
DDPG implementation in Tensorflow Eager Execution
"""

import numpy as np
import tensorflow as tf
from teflon.utils import PytorchInitializer

layers = tf.keras.layers
regularizers = tf.keras.regularizers
losses = tf.keras.losses


class Actor(tf.keras.Model):
    def __init__(self, state_dim, action_dim, max_action, name="Actor"):
        super().__init__(name=name)

        self.l1 = layers.Dense(400, kernel_initializer=PytorchInitializer(),
                               name="L1")
        self.l2 = layers.Dense(300, kernel_initializer=PytorchInitializer(),
                               name="L2")
        self.l3 = layers.Dense(action_dim, kernel_initializer=PytorchInitializer(),
                               name="L3")

        self.max_action = max_action

        # 後段の処理のために早めにshapeを確定させる
        dummy_state = tf.constant(np.zeros(shape=[1, state_dim], dtype=np.float32))
        self(dummy_state)

    def call(self, inputs):
        with tf.device("/gpu:0"):
            features = tf.nn.relu(self.l1(inputs))
            features = tf.nn.relu(self.l2(features))
            features = self.l3(features)
            action = self.max_action * tf.nn.tanh(features)
        return action


class Critic(tf.keras.Model):
    def __init__(self, state_dim, action_dim, wd=1e-2, name="Critic"):
        super().__init__(name=name)

        self.l1 = layers.Dense(400, kernel_initializer=PytorchInitializer(),
                               kernel_regularizer=regularizers.l2(wd), bias_regularizer=regularizers.l2(wd),
                               name="L1")
        self.l2 = layers.Dense(300, kernel_initializer=PytorchInitializer(),
                               kernel_regularizer=regularizers.l2(wd), bias_regularizer=regularizers.l2(wd),
                               name="L2")
        self.l3 = layers.Dense(1, kernel_initializer=PytorchInitializer(),
                               kernel_regularizer=regularizers.l2(wd), bias_regularizer=regularizers.l2(wd),
                               name="L3")

        dummy_state = tf.constant(np.zeros(shape=[1, state_dim], dtype=np.float32))
        dummy_action = tf.constant(np.zeros(shape=[1, action_dim], dtype=np.float32))
        self([dummy_state, dummy_action])

    def call(self, inputs):
        with tf.device("/gpu:0"):
            x, u = inputs

            x = tf.nn.relu(self.l1(x))
            inner_feat = tf.concat([x, u], axis=1)
            x = tf.nn.relu(self.l2(inner_feat))
            x = self.l3(x)
        return x


class DDPG(tf.contrib.checkpoint.Checkpointable):
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = Actor(state_dim, action_dim, max_action)
        self.actor_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)

        # initialize target network
        for param, target_param in zip(self.actor.weights, self.actor_target.weights):
            target_param.assign(param)

        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.critic_optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)

        # initialize target network
        for param, target_param in zip(self.critic.weights, self.critic_target.weights):
            target_param.assign(param)

    def select_action(self, state):
        """

        :param np.ndarray state:
        :return:
        """
        assert isinstance(state, np.ndarray)
        assert len(state.shape) == 1

        state = np.expand_dims(state, axis=0).astype(np.float32)
        action = self._select_action_body(tf.constant(state))

        return action.numpy()[0]

    @tf.contrib.eager.defun
    def _select_action_body(self, state):
        """

        :param np.ndarray state:
        :return:
        """
        action = self.actor(state)
        return action

    def select_action_noise(self, state, noise_level):
        """

        :param np.ndarray state:
        :return:
        """
        assert isinstance(state, np.ndarray)
        assert len(state.shape) == 1

        state = np.expand_dims(state, axis=0).astype(np.float32)
        action = self._select_action_body(tf.constant(state))
        action = action.numpy()[0]

        action = (action + np.random.normal(0, noise_level, size=action.shape))

        return action

    @tf.contrib.eager.defun
    def _select_action_body(self, state):
        """

        :param np.ndarray state:
        :return:
        """
        action = self.actor(state)
        return action


    def train(self, replay_buffer, batch_size=64, discount=0.99, tau=0.001):
        state, next_state, action, reward, done = replay_buffer.sample(batch_size)
        state = np.array(state, dtype=np.float32)
        next_state = np.array(next_state, dtype=np.float32)
        action = np.array(action, dtype=np.float32)
        reward = np.array(reward, dtype=np.float32)
        done = np.array(done, dtype=np.float32)
        not_done = 1 - done
        critic_loss, actor_loss = self._train_body(state, next_state, action, reward, not_done, discount, tau)

        return actor_loss, critic_loss

    @tf.contrib.eager.defun
    def _train_body(self, state, next_state, action, reward, not_done, discount, tau):
        with tf.device("/gpu:0"):

            with tf.GradientTape() as tape:
                target_Q = self.critic_target([next_state, self.actor_target(next_state)])
                target_Q = reward + (not_done * discount * target_Q)
                # detach => stop_gradient
                target_Q = tf.stop_gradient(target_Q)

                current_Q = self.critic([state, action])

                # Compute critic loss + L2 loss
                critic_loss = tf.reduce_mean(losses.MSE(current_Q, target_Q)) + 0.5*tf.add_n(self.critic.losses)

            critic_grad = tape.gradient(critic_loss, self.critic.trainable_variables)
            self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic.trainable_variables))

            with tf.GradientTape() as tape:
                next_action = self.actor(state)
                actor_loss = -tf.reduce_mean(self.critic([state, next_action]))

            actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor.trainable_variables))

            # Update target networks
            for param, target_param in zip(self.critic.weights, self.critic_target.weights):
                target_param.assign(tau * param + (1 - tau) * target_param)

            for param, target_param in zip(self.actor.weights, self.actor_target.weights):
                target_param.assign(tau * param + (1 - tau) * target_param)

            return actor_loss, critic_loss
