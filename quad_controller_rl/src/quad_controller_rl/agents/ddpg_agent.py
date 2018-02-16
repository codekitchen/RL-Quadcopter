"""DDPG Agent"""

import time
import os
import math
from sys import stdout

import tensorflow as tf
from tensorflow.python import debug as tf_debug
import numpy as np
import rospy

import json

from quad_controller_rl.agents.base_agent import BaseAgent
from quad_controller_rl.agents.memory import Memory
from quad_controller_rl.agents.actor_critic import ActorCritic
from quad_controller_rl.agents.ou_noise import OUNoise

class DDPGAgent(BaseAgent):
    """Agent implementing the DDPG algorithm, largely
    as described in https://arxiv.org/pdf/1509.02971.pdf"""

    def __init__(self, task):
        self.episode = 0
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
        self.action_offset[2] = 20.0
        self.action_scale[2] = 5.0

        # self.action_scale *= [1.0, 1.0, 1.0, 0.1, 0.1, 0.1]
        # remove the torque actions
        self.action_shape = 3
        self.action_scale = self.action_scale[0:3]
        self.action_offset = self.action_offset[0:3]
        # z only
        self.action_shape = 1
        self.action_scale = [self.action_scale[2]]
        self.action_offset = [self.action_offset[2]]

        # hyperparams
        self.gamma = 0.99
        self.tau = 0.001
        self.actor_lr =  0.0001
        self.critic_lr = 0.001
        self.batch_size = 64
        self.memory_size = 1000000
        self.steps_per_noise = 20

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
        stdout.write("\n")
        self.last_state = self.last_action = None
        self.total_reward = 0.0
        self.episode += 1
        self.noise.reset()
        self.rand_action = not self.is_test and np.random.rand() > 0.3
        self.noise.sigma = np.random.randn() * 0.3
        self.noise.theta = (0.6 + np.random.rand() * 0.2) * self.noise.sigma

    @property
    def is_test(self):
        return self.test_mode or self.episode % 10 == 0

    def _build_model(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self._step = tf.train.create_global_step()
            tf.summary.scalar('global_step', self._step)
            with tf.variable_scope("network"):
                self.network = self._build_actor_critic(summarize=True)
                # tf.summary.histogram('action', self.network.raw_action)
                tf.summary.histogram('value', self.network.raw_value)
            with tf.variable_scope("target_network"):
                self.target_network = self._build_actor_critic()
                tvs = self.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="target_network/")
                lvs = self.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="network/")
                with tf.variable_scope("initializer"):
                    self.initialize_target_network = tf.group(*[tf.assign(tv, lv) for tv, lv in zip(tvs, lvs)])
                with tf.variable_scope("updater"):
                    self.update_target_network = tf.group(*[tf.assign(tv, self.tau * lv + (1 - self.tau) * tv) for tv, lv in zip(tvs, lvs)])
                diff = tf.add_n([tf.reduce_sum(tf.abs(tv-lv)) for tv, lv in zip(tvs, lvs)])
                tf.summary.scalar('diff', diff)
            with tf.variable_scope("training"):
                # add the batch norm ops
                with tf.variable_scope("critic"):
                    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope="network/critic/")
                    self.critic_y = tf.placeholder(tf.float32, shape=[self.batch_size, 1], name="y")
                    tf.summary.histogram('ys', self.critic_y)
                    regularization_losses = tf.get_collection(
                        tf.GraphKeys.REGULARIZATION_LOSSES, scope="network/critic/")
                    mse = tf.reduce_mean(tf.squared_difference(self.network.value, self.critic_y), name="mse")
                    loss = tf.add_n([mse] + regularization_losses, name="total_loss")
                    tf.summary.scalar('loss', loss)
                    critic_vars = self.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="network/critic/")
                    with tf.control_dependencies(update_ops):
                        opt = tf.train.AdamOptimizer(self.critic_lr)
                        self.critic_train = minimize_with_clipping(opt, loss, critic_vars, step=self._step, stop_gradients=[self.network.critic_state, self.network.critic_action])
                with tf.variable_scope("actor"):
                    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope="network/actor/")
                    # critic_grad = tf.gradients(self.network.value, self.network.action)[0]
                    actor_vars = self.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="network/actor/")
                    # grads = tf.gradients(self.network.action, actor_vars, grad_ys=-critic_grad)
                    # loss = -tf.reduce_mean(self.network.value)
                    loss = self.network.actor_loss
                    # loss = tf.reduce_mean(-critic_grad * self.network.action)
                    tf.summary.scalar('loss', loss)
                    with tf.control_dependencies(update_ops):
                        opt = tf.train.AdamOptimizer(self.actor_lr)
                        self.actor_train = minimize_with_clipping(opt, loss, actor_vars)
            with tf.variable_scope('stats'):
                self.score_placeholder = tf.placeholder(
                    tf.float32, shape=[], name='score_input')
                score_1 = tf.Variable(0., trainable=False, name='score_1')
                tf.summary.scalar('score_1', score_1)
                score_10 = tf.Variable(0., trainable=False, name='score_10')
                tf.summary.scalar('score_10', score_10)
                score_100 = tf.Variable(
                    0., trainable=False, name='score_100')
                tf.summary.scalar('score_100', score_100)

                self.set_scores = tf.group(
                    tf.assign(score_1, self.score_placeholder),
                    tf.assign(
                        score_10,
                        score_10 + (self.score_placeholder / 10.0) - (score_10 / 10.0)),
                    tf.assign(
                        score_100,
                        score_100 + (self.score_placeholder / 100.0) - (score_100 / 100.0)),
                )

            self.memory = Memory(self.memory_size, self.state_shape, (self.action_shape,))
            self.noise = OUNoise(self.action_shape, steps=self.steps_per_noise)

    def _build_actor_critic(self, summarize=False):
        return ActorCritic(self.state_shape, self.action_shape, summarize=summarize)

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
            # self.session = tf_debug.LocalCLIDebugWrapperSession(self.session)
    
    def _scale_action(self, action):
        action = action * self.action_scale + self.action_offset
        action = np.pad(action, (2, 3), 'constant')
        # if action[2] > 0:
        #     action[2] = 21.0 + action[2] * 4
        # else:
        #     action[2] = 21.0 + action[2] * 40
        return action

    def step(self, state, reward, done):
        self.total_reward += reward

        if not self.is_test and self.last_state is not None:
            self.remember(self.last_state, self.last_action, reward, done)
            self.train()

        if done:
            self.rewards_log.write("{}\n".format(self.total_reward))
            self.session.run(self.set_scores, {self.score_placeholder: self.total_reward})
            self._reset_episode()
            return None

        step = self.get_global_step()
        actions = self.session.run(
            self.network.action,
            {self.network.state: [state], self.network.is_training: False},
        )
        base_action = actions[0]
        action = base_action
        if self.rand_action:
            action = np.clip(action + self.noise.sample(), -1.0, 1.0)

        stdout.write("\r{} tgt: {:2.0f} score: {:9.2f} ep: {:6d} step: {:10d} {} {:5.2f} {:5.2f} {:5.2f}".format("TEST " if self.is_test else "train", self.task.target_z, self.total_reward, self.episode, step, "rand  " if self.rand_action else "norand", base_action[0], self._scale_action(base_action)[2], self._scale_action(action)[2]))

        self.last_state = state
        self.last_action = action

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
            {tn.critic_state: next_states, tn.critic_action: pred_next_actions, tn.is_training: False})
        ys = rewards + self.gamma * pred_next_values * (1 - dones)
        ys = ys[:, None]

        step = self.get_global_step()
        if math.isnan(ys[0]):
            json.dump({'ys': ys.tolist(), 'rewards': rewards.tolist(), 'pred': pred_next_values.tolist(), 'next_states': next_states.tolist(), 'pred_next_actions': pred_next_actions.tolist(), 'step': step}, self.rewards_log)
            self.rewards_log.write("\n")
        do_summary = False#(step % 20 == 0)
        ops = [self.critic_train, self.network.critic_action_gradients]
        if do_summary:
            ops += [self.summary_data]
        res = self.session.run(ops,
            {self.network.critic_state: states, self.network.critic_action: actions, self.critic_y: ys, self.network.is_training: True})
        if do_summary:
            self.writer.add_summary(res[-1], step)
        action_grads = res[1]
        self.session.run(self.actor_train,
            {self.network.state: states, self.network.action_gradients: action_grads, self.network.is_training: True})

        self.session.run(self.update_target_network)

    def _restore(self, checkpoint):
        self.saver.restore(self.session, checkpoint)
        self.memory = self.memory.restore(checkpoint)

    def _save(self):
        snapshot_name = "{}/snap".format(self.state_dir)
        save_path = self.saver.save(
            self.session, snapshot_name, global_step=self._step)
        self.memory.save(save_path)

def minimize_with_clipping(opt, loss, vars, norm=0.5, stop_gradients=None, step=None):
    grads = tf.gradients(loss, vars, stop_gradients=stop_gradients)
    # grads, _ = tf.clip_by_global_norm(grads, norm)
    grads_and_vars = list(zip(grads, vars))
    for grad, var in grads_and_vars:
        tf.summary.histogram("grad/"+var.name.replace(":0", ""), grad)
    return opt.apply_gradients(grads_and_vars, global_step=step)
