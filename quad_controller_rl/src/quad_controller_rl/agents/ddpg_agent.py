"""DDPG Agent"""

import time
import os

import tensorflow as tf
import numpy as np
import rospy

from quad_controller_rl.agents.base_agent import BaseAgent
from quad_controller_rl.agents.memory import Memory
from quad_controller_rl.agents.actor_critic import ActorCritic
from quad_controller_rl.agents.ou_noise import OUNoise

class DDPGAgent(BaseAgent):
    """Agent implementing the DDPG algorithm, largely
    as described in https://arxiv.org/pdf/1509.02971.pdf"""

    def __init__(self, task):
        self.episode = 1
        self.name = rospy.get_param('agentname')
        self.test_mode = rospy.get_param('test') == 'true'
        if self.name == '':
            self.name = str(os.getpid())
        self.task = task
        self.state_shape = self.task.observation_space.shape
        # assumes a pre-flattened action space
        self.action_shape = self.task.action_space.shape[0]
        self.action_scale = (self.task.action_space.high - self.task.action_space.low) / 2.0
        self.action_offset = np.zeros(self.action_shape)
        # scale the z force
        # self.action_offset[2] = 20.0
        self.action_scale[2] = 1.0

        # self.action_scale *= [1.0, 1.0, 1.0, 0.1, 0.1, 0.1]
        # remove the torque actions
        self.action_shape = 3
        self.action_scale = self.action_scale[0:3]
        self.action_offset = self.action_offset[0:3]
        # z only
        # self.action_shape = 1
        # self.action_scale = [5.0]
        # self.action_offset = [20.0]

        # hyperparams
        self.gamma = 0.99
        self.tau = 0.001
        self.actor_lr =  0.0001
        self.critic_lr = 0.001
        self.batch_size = 128
        self.memory_size = 100000

        task = rospy.get_param('task')
        self.state_dir = "/home/robond/catkin_ws/src/ddpg_state/{}/{}".format(task, self.name)
        if not os.path.exists(self.state_dir):
            os.mkdir(self.state_dir)
        rewards_logname = "{}/rewards.log".format(self.state_dir)
        self.rewards_log = open(rewards_logname, 'a', buffering=1)
        print("writing rewards to " + rewards_logname)

        self._build_model()
        self._start_session()
        self._reset_episode()

    def _reset_episode(self):
        self.last_state = self.last_action = None
        self.total_reward = 0.0
        self.episode += 1
        self.noise.reset()

    @property
    def is_test(self):
        return self.test_mode or self.episode % 10 == 0

    def _build_model(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self._step = tf.train.create_global_step()
            tf.summary.scalar('global_step', self._step)
            with tf.variable_scope("network"):
                self.network = self._build_actor_critic()
            with tf.variable_scope("target_network"):
                self.target_network = self._build_actor_critic()
                tvs = self.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="target_network/")
                lvs = self.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="network/")
                with tf.variable_scope("initializer"):
                    self.initialize_target_network = tf.group(*[tf.assign(tv, lv) for tv, lv in zip(tvs, lvs)])
                with tf.variable_scope("updater"):
                    self.update_target_network = tf.group(*[tf.assign(tv, self.tau * lv + (1 - self.tau) * tv) for tv, lv in zip(tvs, lvs)])
                diff = tf.add_n([tf.reduce_mean(tf.squared_difference(tv, lv)) for tv, lv in zip(tvs, lvs)])
                tf.summary.scalar('diff', diff)
            with tf.variable_scope("training"):
                # add the batch norm ops
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope="network/")
                with tf.variable_scope("critic"):
                    self.critic_y = tf.placeholder(tf.float32, shape=[self.batch_size], name="y")
                    mse = tf.reduce_mean(tf.squared_difference(self.network.value, self.critic_y), name="mse")
                    tf.summary.scalar('loss', mse)
                    critic_vars = self.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="network/critic/")
                    with tf.control_dependencies(update_ops):
                        self.critic_train = tf.train.AdamOptimizer(self.critic_lr).minimize(mse, global_step=self._step, var_list=critic_vars)
                with tf.variable_scope("actor"):
                    critic_grad = tf.gradients(self.network.value, self.network.action)[0]
                    loss = tf.reduce_mean(critic_grad * self.network.action)
                    tf.summary.scalar('loss', loss)
                    actor_vars = self.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="network/actor/")
                    with tf.control_dependencies(update_ops):
                        self.actor_train = tf.train.AdamOptimizer(self.actor_lr).minimize(loss, var_list=actor_vars)

            self.memory = Memory(self.memory_size, self.state_shape, (self.action_shape,))
            self.noise = OUNoise(self.action_shape)

    def _build_actor_critic(self):
        return ActorCritic(self.state_shape, self.action_shape)

    def _start_session(self):
        with self.graph.as_default():
            self.saver = tf.train.Saver()
            self.session = tf.Session()
            self.summary_data = tf.summary.merge_all()
            self.writer = tf.summary.FileWriter(self.state_dir, self.graph)
            self.session.run(tf.global_variables_initializer())
            self.graph.finalize()
            self.session.run(self.initialize_target_network)
            checkpoint = tf.train.latest_checkpoint(self.state_dir)
            if checkpoint:
                self._restore(checkpoint)
    
    def _scale_action(self, action):
        action = action * self.action_scale + self.action_offset
        action = np.pad(action, (0, 6 - len(action)), 'constant')
        if action[2] > 0:
            action[2] = 21.0 + action[2] * 4
        else:
            action[2] = 21.0 + action[2] * 40
        return action
        # return np.array([0, 0, action[0], 0, 0, 0])

    def step(self, state, reward, done):
        self.total_reward += reward

        if not self.is_test and self.last_state is not None:
            self.remember(self.last_state, self.last_action, reward, done)
            self.train()

        if done:
            print(("TEST " if self.is_test else "train ") + "score: {:.2f} step: {}".format(self.total_reward, self.get_global_step()))
            self.rewards_log.write("{}\n".format(self.total_reward))
            self._reset_episode()
            return None

        actions, _values = self.session.run(
            [self.network.action, self.network.value],
            {self.network.state: [state], self.network.is_training: False},
        )
        action = actions[0]
        if not self.is_test:
            action = np.clip(action + self.noise.sample(), -1.0, 1.0)

        self.last_state = state
        self.last_action = action

        step = self.get_global_step()
        if step > 0 and step % 2000 == 0:
            self._save()

        return self._scale_action(action)

    def get_global_step(self):
        return tf.train.global_step(self.session, self._step)

    def remember(self, state, action, reward, done):
        self.memory.remember(state, action, reward, done)

    def train(self):
        rows = self.memory.sample(self.batch_size)
        if not rows:
            return
        states, actions, rewards, dones, next_states = rows
        tn = self.target_network
        pred_next_actions = self.session.run(tn.action, {tn.state: next_states, tn.is_training: False})
        pred_next_values = self.session.run(tn.value,
            {tn.state: next_states, tn.action: pred_next_actions, tn.is_training: False})
        ys = rewards + self.gamma * pred_next_values * (1 - dones)

        step = self.get_global_step()
        do_summary = (step % 20 == 0)
        ops = [self.critic_train]
        if do_summary:
            ops += [self.summary_data]
        res = self.session.run(ops,
            {self.network.state: states, self.network.action: actions, self.critic_y: ys, self.network.is_training: True})
        if do_summary:
            self.writer.add_summary(res[-1], step)
        self.session.run(self.actor_train,
            {self.network.state: states, self.network.is_training: True})

        self.session.run(self.update_target_network)

    def _restore(self, checkpoint):
        self.saver.restore(self.session, checkpoint)
        self.memory = self.memory.restore(checkpoint)

    def _save(self):
        snapshot_name = "{}/snap".format(self.state_dir)
        save_path = self.saver.save(
            self.session, snapshot_name, global_step=self._step)
        self.memory.save(save_path)
        print("saved model")
