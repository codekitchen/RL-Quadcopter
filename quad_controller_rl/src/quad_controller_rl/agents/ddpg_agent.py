"""DDPG Agent"""

# pylint: disable=E1129,E0401

import os
from sys import stdout

import tensorflow as tf
import numpy as np
import rospy

from quad_controller_rl.agents.base_agent import BaseAgent
from quad_controller_rl.agents.memory import Memory
from quad_controller_rl.agents.actor_critic import ActorCritic
from quad_controller_rl.agents.ou_noise import OUNoise
from quad_controller_rl.z_only_task import ZOnlyTask

class DDPGAgent(BaseAgent):
    """Agent implementing the DDPG algorithm, largely
    as described in https://arxiv.org/pdf/1509.02971.pdf"""

    def __init__(self, task, random_seed=1234):
        self.task = task
        if hasattr(self.task, 'target_z'):
            self.task = ZOnlyTask(task)

        self.random_seed = random_seed
        np.random.seed(self.random_seed)
        tf.set_random_seed(self.random_seed)
        print("random seed:", self.random_seed)

        self.episode = 0
        self.name = rospy.get_param('agentname')
        self.test_mode = rospy.get_param('test')
        if self.name == '':
            self.name = str(os.getpid())
        self.state_size = self.task.observation_space.shape[0]
        self.action_size = self.task.action_space.shape[0]

        assert self.task.action_space.high == -self.task.action_space.low, "action space assumed to be symmetric"
        self.action_scale = self.task.action_space.high

        # hyperparams
        self.gamma = 0.99
        self.tau = 0.001
        self.actor_lr = 0.00001
        self.critic_lr = 0.001
        self.batch_size = 64
        self.memory_size = 1000000
        self.steps_per_noise = 3

        task = self.task.name
        self.state_dir = "ddpg_state/{}/{}".format(task, self.name)
        if os.path.exists("/home/robond"):
            self.state_dir = "/home/robond/catkin_ws/src/"+self.state_dir
        if not os.path.exists(self.state_dir):
            os.makedirs(self.state_dir)
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
        self.ep_steps = 0
        self.ep_q = 0.0
        self.noise.reset()
        # self.noise.sigma = np.random.randn() * 0.3
        # self.noise.theta = (0.6 + np.random.rand() * 0.2) * self.noise.sigma

    @property
    def is_test(self):
        return self.test_mode or self.episode % 20 == 0

    def _build_model(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self._step = tf.train.create_global_step()
            with tf.variable_scope("network"):
                self.network = self._build_actor_critic(summarize=True)

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
                    critic_vars = self.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="network/critic/")
                    tf.summary.histogram('ys', self.network.ys)
                    loss = tf.reduce_mean(tf.squared_difference(self.network.value, self.network.ys), name="mse")
                    tf.summary.scalar('loss', loss)
                    opt = tf.train.AdamOptimizer(self.critic_lr)
                    self.critic_train = opt.minimize(loss, var_list=critic_vars)
                with tf.variable_scope("actor"):
                    actor_vars = self.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="network/actor/")
                    # grads = tf.gradients(self.network.action, actor_vars, grad_ys=-critic_grad)
                    # loss = -tf.reduce_mean(self.network.value)
                    # loss = self.network.actor_loss
                    loss = tf.reduce_mean(-self.network.actor_gradients * self.network.action)
                    tf.summary.scalar('loss', loss)
                    opt = tf.train.AdamOptimizer(self.actor_lr)
                    self.actor_train = opt.minimize(loss, var_list=actor_vars)
                self.train_ops = tf.group(self.critic_train, self.actor_train, tf.assign_add(self._step, 1, name="global_step"))
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

            self.memory = Memory(self.memory_size, (self.state_size,), (self.action_size,), self.random_seed)
            self.noise = OUNoise(self.action_size, steps=self.steps_per_noise)

    def _build_actor_critic(self, summarize=False):
        return ActorCritic(self.state_size, self.action_size, summarize=summarize)

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
        return action * self.action_scale

    def step(self, state, reward, done):
        if hasattr(self.task, 'filter_state'):
            state = self.task.filter_state(state)

        self.ep_steps += 1
        self.total_reward += reward

        if not self.is_test and self.last_state is not None:
            self.remember(self.last_state, self.last_action, reward, done)
            self.train()

        step = self.get_global_step()
        actions = self.session.run(
            self.network.action,
            {self.network.actor_state: [state], self.network.is_training: False},
        )
        base_action = actions[0]
        action = base_action
        if not self.is_test:
            action = np.clip(action + self.noise.sample(), -1.0, 1.0)

        dbg = "\r{} score: {:9.2f} ep: {:6d} step: {:5d} act: {:6.2f} {:6.2f} {:6.2f} avg_Q: {:5.2f}".format("TEST " if self.is_test else "train", self.total_reward, self.episode, self.ep_steps, base_action[0], action[0], self._scale_action(action)[0], (self.ep_q / self.ep_steps))
        if hasattr(self.task, 'target_z'):
            dbg += " tgt: {:5.2f}".format(self.task.target_z)
        stdout.write(dbg)

        self.last_state = state
        self.last_action = action

        if not self.test_mode and step > 0 and step % 2000 == 0:
            self._save()

        if done:
            self.rewards_log.write("{}\n".format(self.total_reward))
            self.session.run(self.set_scores, {self.score_placeholder: self.total_reward})
            self._reset_episode()
            return None

        action = self._scale_action(action)
        if hasattr(self.task, 'filter_action_response'):
            action = self.task.filter_action_response(action)
        return action

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
        pred_next_actions = self.session.run(tn.action, {tn.actor_state: next_states, tn.is_training: False})
        pred_next_values = self.session.run(tn.value,
            {tn.critic_state: next_states, tn.critic_action: pred_next_actions, tn.is_training: False})
        ys = rewards + self.gamma * pred_next_values * (1 - dones)

        pred_actions = self.session.run(self.network.action, {self.network.actor_state: states})
        action_grads = self.session.run(self.network.critic_gradients, {self.network.critic_state: states, self.network.critic_action: pred_actions})

        step = self.get_global_step()
        do_summary = (step % 20 == 0)
        ops = [self.network.value, self.train_ops]
        if do_summary:
            ops += [self.summary_data]
        res = self.session.run(ops,
            {self.network.critic_state: states, self.network.critic_action: actions, self.network.ys: ys, self.network.actor_state: states, self.network.actor_gradients: action_grads, self.network.is_training: True})
        qvals = res[0]
        self.ep_q += np.amax(qvals)
        if do_summary:
            self.writer.add_summary(res[-1], step)

        self.session.run(self.update_target_network)

    def _restore(self, checkpoint):
        self.saver.restore(self.session, checkpoint)
        self.memory = self.memory.restore(checkpoint)

    def _save(self):
        snapshot_name = "{}/snap".format(self.state_dir)
        save_path = self.saver.save(
            self.session, snapshot_name, global_step=self._step)
        self.memory.save(save_path)
