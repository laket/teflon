"""
SAC policy to reproduce the paper's result.

Tuomas Haarnoja · Aurick Zhou · Pieter Abbeel · Sergey Levine,
Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor, ICML2018

"""

import tensorflow as tf

losses = tf.keras.losses

from.SAC import GaussianActor, CriticV, CriticQ

class SAC(tf.contrib.checkpoint.Checkpointable):

    def __init__(self, state_dim, action_dim, max_action):
        # in the paper, this value was used for Ant-v1, HalfCheetah-v1
        self.scale_reward = 5.0

        self.actor = GaussianActor(state_dim, action_dim, max_action)
        lr = 3e-4
        self.actor_optimizer = tf.train.AdamOptimizer(learning_rate=lr)

        self.vf = CriticV(state_dim)
        self.vf_target = CriticV(state_dim)
        self.vf_optimizer = tf.train.AdamOptimizer(learning_rate=lr)

        for param, target_param in zip(self.vf.weights, self.vf_target.weights):
            target_param.assign(param)

        self.qf1 = CriticQ(state_dim, action_dim, name="vq1")
        self.qf2 = CriticQ(state_dim, action_dim, name="vq2")

        self.qf1_optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.qf2_optimizer = tf.train.AdamOptimizer(learning_rate=lr)

    def select_action(self, state):
        """

        :param state:
        :return:
        """
        assert len(state.shape) == 1

        state = tf.cast(state, tf.float32)
        state = tf.expand_dims(state, axis=0)
        action = self._select_action_body(state)
        action = tf.squeeze(action, axis=0)

        return action

    @tf.contrib.eager.defun
    def _select_action_body(self, state):
        """

        :param np.ndarray state:
        :return:
        """
        action = self.actor.mean_action(state)
        return action


    def select_action_noise(self, state, noise_level):
        """

        TODO Use noise_level

        :param state:
        :return:
        """
        assert len(state.shape) == 1

        state = tf.cast(state, tf.float32)
        state = tf.expand_dims(state, axis=0)
        action = self._select_action_noise_body(state)

        action = tf.squeeze(action, axis=0)

        return action.numpy()

    @tf.contrib.eager.defun
    def _select_action_noise_body(self, state):
        """

        :param state:
        :return:
        """
        action, _ = self.actor(state)
        return action

    @tf.contrib.eager.defun
    def train_for_batch(self, states, actions, rewards, next_states, done, discount=0.99):
        """

        :param states:
        :param actions:
        :param rewards:
        :param next_states:
        :param done:
        :param discount:
        :return:
        """
        assert len(done.shape) == 2
        assert len(rewards.shape) == 2
        done = tf.squeeze(done, axis=1)
        rewards = tf.squeeze(rewards, axis=1)

        not_done = 1 - tf.cast(done, dtype=tf.float32)

        # Critic Update
        with tf.GradientTape(persistent=True) as tape:
            q1 = self.qf1([states, actions])
            q2 = self.qf2([states, actions])
            vf_next_target_t = self.vf_target(next_states)

            # Equation (7, 8)
            ys = tf.stop_gradient(
                self.scale_reward * rewards + not_done * discount * vf_next_target_t
            )

            td_loss1 = 0.5 * losses.MSE(ys, q1)
            td_loss2 = 0.5 * losses.MSE(ys, q2)

        # Equation (9)
        q1_grad = tape.gradient(td_loss1, self.qf1.trainable_variables)
        self.qf1_optimizer.apply_gradients(zip(q1_grad, self.qf1.trainable_variables))
        q2_grad = tape.gradient(td_loss2, self.qf2.trainable_variables)
        self.qf2_optimizer.apply_gradients(zip(q2_grad, self.qf2.trainable_variables))

        del tape

        # Actor Update
        with tf.GradientTape(persistent=True) as tape:
            vf_t = self.vf(states)
            sample_actions, log_pi = self.actor(states)

            tf.contrib.summary.scalar(name="log_pi_min", tensor=tf.reduce_min(log_pi))
            tf.contrib.summary.scalar(name="log_pi_max", tensor=tf.reduce_max(log_pi))

            q1 = self.qf1([states, sample_actions])
            q2 = self.qf2([states, sample_actions])
            min_q = tf.minimum(q1, q2)

            # Equation (12)
            policy_kl_loss = tf.reduce_mean(log_pi - q1)

            #policy_loss = policy_kl_loss + reg_loss
            policy_loss = policy_kl_loss

            # Equation (5)
            vf_loss_t = 0.5 * losses.MSE(tf.stop_gradient(min_q - log_pi), vf_t)

        # Equation (6)
        vf_grad = tape.gradient(vf_loss_t, self.vf.trainable_variables)
        self.vf_optimizer.apply_gradients(zip(vf_grad, self.vf.trainable_variables))

        # Equation (13)
        actor_grad = tape.gradient(policy_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor.trainable_variables))

        del tape

        return policy_loss, vf_loss_t, td_loss1


    def train(self, replay_buffer, batch_size=256, discount=0.99, tau=0.001):
        """

        :param replay_buffer:
        :param batch_size: initial value comes from the paper
        :param discount: initial value comes from the paper
        :param tau: initial value comes from the paper
        :return:
        """
        # Sample replay replay_buffer
        states, next_states, actions, rewards, dones = replay_buffer.sample(batch_size)

        with tf.device("/gpu:0"):
            policy_loss, vf_loss_t, td_loss1 = self.train_for_batch(states, actions, rewards, next_states, dones, discount=discount)
            total_actor_loss = policy_loss
            total_vf_loss = vf_loss_t
            total_q_loss = td_loss1

            for param, target_param in zip(self.vf.weights, self.vf_target.weights):
                target_param.assign(tau * param + (1 - tau) * target_param)

        return total_actor_loss, total_vf_loss, total_q_loss
