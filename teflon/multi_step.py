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


import tensorflow as tf
from replay_buffer.segment_tree import ShortTermBuffer


class MultistepAggregator(object):
    """Make multistep transition
    (s_t, a_t, cumulative reward, s_(t+n), discount ratio for Q_(t+n))

    n-step Bellman Equation:

    Q(s_t, a_t)
        = cumulative_reward + discount_ratio * Q(s_(t+n), policy(s_(t+n)))
    """

    def __init__(self, multi_step_n):
        """
        :param int multi_step_n: if n==1, it is not multistep.
        """
        self.multi_step_n = multi_step_n
        self.transition_buffer = None

    def add_step(self, state, action, reward, done, reach_limit, name=None):
        """add 1 step

        :param reach_limit: whether transition reached to time limit for the env.
        :return:
        """
        with tf.name_scope(name, default_name="add_step"):
            current_transition = state, action, reward, done, reach_limit

            if self.transition_buffer is None:
                self.transition_buffer = ShortTermBuffer(shapes=[v.get_shape() for v in current_transition],
                                                         dtypes=[v.dtype for v in current_transition],
                                                         multi_step=self.multi_step_n)

            # add transition to ShortTermBuffer
            # transition = state, action, reward, done, reach_limit
            # history = [transition1, transition2, ...]
            is_valid, history = self.transition_buffer.enqueue(current_transition)

            # historyの例外ケース
            """
            if is_valid is False, don't use the paired history. 
            """
            return is_valid, history

    def make_multistep_transition(self, history, gamma):
        """

        :param history:
        :param float gamma: discount rate
        :rtype:
        :return:
          (state, action, cumulative_reward, next_state, discounted_rate)

         Q(s_t, a_t)
           = cumulative_reward + discount_ratio * Q(s_(t+n), policy(s_(t+n)))
        """

        multi_step_n = self.multi_step_n
        assert len(history) == multi_step_n+1

        # multi_step_n後(一番最後)の情報はstateしか使わない
        old = history[0]
        new = history[-1]

        # frame = state, action, reward, done, reach_limit
        rewards = list(list(zip( * history))[2])
        dones = list(list(zip( * history))[3])
        reach_limits = list(list(zip( * history))[4])
        batch_size = int(rewards[0].shape.as_list()[0])

        # N-step後のdiscounted_rateを計算する
        # これはepisodeの途中でtimelimitが来た場合の措置である。
        # ここで生成されるtransitionを元にmulti-step TD-errorを計算する。
        # その際にn-step後のQ-valueを足すが、n-stepの手前でtimelimitが来た場合は、
        # そのstep先のQ-valueを足す。その際の係数をこの関数では返す
        # なお、通常のdoneのときは0をsetすればよい
        discounted_reward = tf.zeros([batch_size], dtype=tf.float32)
        discounted_rate = tf.ones([batch_size], dtype=tf.float32)
        already_done = tf.zeros([batch_size], dtype=tf.bool)
        cur_gamma = 1.0

        # rewardはmulti_step_n+1の長さ
        # TODO テスト
        for step_n in range(multi_step_n):
            not_done = (1 - tf.cast(already_done, dtype=tf.float32))
            next_reward = cur_gamma * not_done * rewards[step_n]
            discounted_reward = discounted_reward + next_reward

            cur_gamma = cur_gamma * gamma
            discounted_rate = tf.where(already_done, discounted_rate, cur_gamma * discounted_rate)

            # このstepで突然死したかどうか
            # TODO: doneしてからはじまった別のシナリオで突然死すると関係ないのに0にされる (レアなので放置)
            is_dead = tf.logical_xor(dones[step_n], reach_limits[step_n])
            discounted_rate = tf.where(is_dead, tf.zeros([batch_size]), discounted_rate)

            already_done = tf.logical_or(already_done, dones[step_n])

        # frame = state, action, reward, done, reach_limit
        state = old[0]
        action = old[1]
        next_state = new[0]

        return state, action, discounted_reward, next_state, discounted_rate
