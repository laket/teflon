teflon is a Tensorflow implementation to achieve high score for Ant-v2 (but not specialized for Ant-v2).

|Methods|Ant-v2 Score(Roughly)|
|-------|-----|
|[TD3](https://arxiv.org/pdf/1802.09477.pdf)    |4400|
|[SAC](https://arxiv.org/abs/1801.01290)    | 6100|
|[Ref-ER](https://openreview.net/pdf?id=Bye9LiR9YX) | 6900|
|teflon|8400|

teflon was implemented to show a Tensorflow Graph mode example that tensorflow users 
can read and extend easily. If you are interested in Eager Execution DDPG implementation,
see also [DDPG_Eager](https://github.com/laket/DDPG_Eager).

This implementation is based on [Soft Actor Critic](https://arxiv.org/abs/1801.01290). 
In addition to this, the following methods were added, 

- Prioritized Experiment Replay
- multi-step return
- Huber Loss  (for updating state value function and action value function)
 
Many Tensorflow implementations of reinforcement learning contains many feed_dict, 
placeholder, and variable_scope. These make codes too complex to read.

teflon eliminates most of them with tf.keras.layers and using 
Tensorflow operators to manipulate environments and Replay Buffer.

teflon was greatly inspired by [Uber's blog post](https://eng.uber.com/accelerated-neuroevolution/)
 and [their ape-x implementation](https://github.com/uber-research/ape-x).
Their implementation is awesome! The codes contained in replay_buffer folder come from them. (slightly modified)


# Requirement

- gym
- tensorflow (>=1.12)
- mujoco-py (You must slightly modify to increase performance)
 
Although teflon called mujoco-py function from multiple threads, 
GIL(Glocal Interpreter Lock) prevents parallel execution.
So you have to add some codes in mjsim.pyx(mujoco-py) to release GIL.

The following three functions acquire GIL for a relatively long time

- reset
- forward
- step
 
 Releasing GIL code is not difficult, which only requires 1 line
 
 ```python
    def forward(self):
        """
        Computes the forward kinematics. Calls ``mj_forward`` internally.
        """
        with nogil:
            mj_forward(self.model.ptr, self.data.ptr)
```
 
Add nogil to the above 3 functions. Then build and install.
 
 # Usage
First you must build buffer_ops.so by
```bash
make
```

Then call the training script (queue_main.py) and eval_policy.py concurrently.
 
```bash
python3 teflon/tool/queue_main.py --multi 4 &
python3 teflon/tool/eval_policy.py
tensorboard --logdir ./output
```
 
 
There is also a Soft Actor Critic implementation in Eager Execution.
 
 ```bash
python3 teflon/tool/eager_main.py --policy SAC --env Ant-v2 --logdir ./eager_output/log --parameter ./eager_output/parameter
``` 
 
 # Learning Curve
 
"queue_main.py --multi 4" outputs the following learning curve on Intel i5-4440 and GeForce GTX 1070.
 
 ![learning curvehttps://user-images.githubusercontent.com/1290076/49698737-275c4b80-fc0b-11e8-8917-7e8ee485bbd3.png](https://user-images.githubusercontent.com/1290076/49698737-275c4b80-fc0b-11e8-8917-7e8ee485bbd3.png)
 