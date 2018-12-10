'''
Copyright (c) 2018 Uber Technologies, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

"""
This code was partially modified by Oiki Tomoaki.
"""

import os

import tensorflow as tf

custom_modules = tf.load_op_library(os.path.join(os.path.dirname(__file__), './../buffer_ops.so'))


class ReplayBuffer(object):
    # Custom Tensorflow ops that implemented Prioritized Experience Replay
    # TODO: Perform tests beyond learning
    def __init__(self, capacity, shapes, alpha, beta, priority_eps, dtypes=[tf.float32]):
        self.dtypes = dtypes
        self.shapes = shapes

        self._alpha = alpha
        self._beta = beta
        self._priority_eps = priority_eps

        self._buffer = custom_modules.experience_buffer(capacity, component_types=self.dtypes, shapes=shapes)

    def size(self):
        return custom_modules.experience_buffer_size(self._buffer)

    def enqueue(self, components):
        raise NotImplementedError()
        #components = [tf.convert_to_tensor(v, dtype=dt) for v, dt in zip(components, self.dtypes)]
        #return custom_modules.experience_buffer_enqueue(self._buffer, components)

    def enqueue_many(self, components, priorities=None):
        # Modified by Oiki Tomoaki
        priorities = (priorities + self._priority_eps) ** self._alpha
        return custom_modules.replay_buffer_priority_enqueue_many(self._buffer,
                                                                  priorities,
                                                                  components)

    def dequeue(self, indices):
        return custom_modules.experience_buffer_dequeue(self._buffer, indices, Tcomponents=self.dtypes)

    def dequeue_many(self, indices):
        return custom_modules.experience_buffer_dequeue_many(self._buffer, indices, Tcomponents=self.dtypes)

    @property
    def name(self):
        # Modified by Oiki Tomoaki
        return "ReplayBuffer"

    def close(self, cancel_pending_enqueues=None):
        # Modified by Oiki Tomoaki
        return custom_modules.experience_buffer_close(self._buffer)

    def update(self, indices, values):
        queue = tf.FIFOQueue(32, [tf.int64, tf.float32], [indices.get_shape()] * 2)
        enqueue_op = queue.enqueue([indices, values])

        indices, values = queue.dequeue()
        dequeue_op = self.assign(indices, values)

        qr = tf.train.QueueRunner(queue, [dequeue_op])
        tf.train.add_queue_runner(qr)

        return enqueue_op

    def assign_with_eps(self, indices, values):
        values = (values + self._priority_eps) ** self._alpha
        return custom_modules.replay_buffer_update_priority(self._buffer,
                                                            indices,
                                                            values)

    def sample_proportional_from_buffer(self, size, minimum_sample_size=1):
        with tf.variable_scope("sample_buffer"):
            prefixsum = tf.random_uniform((size,), dtype=tf.float32)

            idxes, weights, total, p_min, _num_elements, components = custom_modules.replay_buffer_priority_sample(handle=self._buffer,
                                                                   prefixsum=prefixsum,
                                                                   Tcomponents=self.dtypes,
                                                                   minimum_sample_size=minimum_sample_size)
            idxes.set_shape((size,))
            weights.set_shape((size,))

            neg_beta = -self._beta
            p_min = p_min / total
            _num_elements = tf.cast(_num_elements, tf.float32)
            max_weight = (p_min * _num_elements) ** neg_beta

            weights = weights * (_num_elements / total)
            weights = weights ** neg_beta
            weights = weights * (1.0 / max_weight)

            for shape, c in zip(self.shapes, components):
                c.set_shape((size,) + tuple(shape))
            idxes.set_shape((size,))
            weights.set_shape((size,))
            # return (idxes, weights) + tuple(components)
            # Modified by Oiki Tomoaki
            return (idxes, weights, tuple(components))


class ShortTermBuffer(object):
    # This class is modified by Oiki Tomoaki to specialize for multistep.
    # This replay_buffer is a helper class to create a stack of frames as well as short term history (n-step transitions)
    def __init__(self, shapes, dtypes=[tf.float32], multi_step=1):
        """

        :param list[tuple] shapes:
        :param list[dtype] dtypes:
        :param int multi_step:
        """

        self.dtypes = dtypes
        self.shapes = shapes
        self._length = multi_step
        # _buffer is handler to manipulate an experience replay_buffer.
        self._buffer = custom_modules.experience_buffer(self._length, component_types=self.dtypes, shapes=shapes)
        self.blank_components = [tf.zeros(s,t) for s, t in zip(self.shapes, self.dtypes)]

    def size(self):
        return custom_modules.experience_buffer_size(self._buffer)

    def encode_history(self):
        """get a history

        :rtype: (bool, list[Tensor])
        :return:
          (valid, history)
          valid: whether history is filled with valid data.
          history: contains a _length+1 transition. (current, future1, future2...)
        """

        # observation = (A, B) =>
        # history = [A1, B1, A2, B2, ....]
        valid, history=custom_modules.experience_buffer_encode_recent(self._buffer,
                                                                      blank_components=self.blank_components,
                                                                      Tout_components=self.dtypes * self._length)

        # split history to [[A1,B1], [A2,B2], [A3,B3], ...]
        history = list(zip( * ([iter(history)] * len(self.dtypes))))
        for step in history:
            for s, t, c in zip(self.shapes, self.dtypes, step):
                c.set_shape(s)
                assert c.dtype == t
        return valid, history

    def enqueue(self, components):
        components = [tf.convert_to_tensor(v, dtype=dt) for v, dt in zip(components, self.dtypes)]
        valid, history=custom_modules.experience_buffer_enqueue_recent(self._buffer, components=components,
                                                                       blank_components=self.blank_components,
                                                                       Tout_components=self.dtypes * (self._length+1))
        history = list(zip( * ([iter(history)] * len(self.dtypes))))
        for step in history:
            for s, t, c in zip(self.shapes, self.dtypes, step):
                c.set_shape(s)
                assert c.dtype == t
        return valid, history

