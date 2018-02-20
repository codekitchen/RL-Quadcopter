from sys import stdout

import numpy as np
from keras import layers, models, optimizers, regularizers, initializers, activations
from keras import backend as K

from quad_controller_rl.agents.base_agent import BaseAgent
from quad_controller_rl.agents.memory import Memory
from quad_controller_rl.agents.ou_noise import OUNoise
from quad_controller_rl.z_only_task import ZOnlyTask

def relu6(x):
    return activations.relu(x, max_value=6)

class DDPGKeras(BaseAgent):
    """Reinforcement Learning agent that learns using DDPG."""

    def __init__(self, task, random_seed = 12345):
        self.task = task
        if hasattr(self.task, 'target_z'):
            self.task = ZOnlyTask(task)

        self.random_seed = random_seed
        np.random.seed(self.random_seed)
        K.tf.set_random_seed(self.random_seed)
        self.episode = 0
        self._step = 0
        self.state_size = self.task.observation_space.shape[0]
        self.action_size = self.task.action_space.shape[0]

        # Actor (Policy) Model
        self.action_low = self.task.action_space.low
        self.action_high = self.task.action_space.high
        self.actor_local = Actor(
            self.state_size, self.action_size, self.action_low, self.action_high)
        self.actor_target = Actor(
            self.state_size, self.action_size, self.action_low, self.action_high)

        # Critic (Value) Model
        self.critic_local = Critic(self.state_size, self.action_size)
        self.critic_target = Critic(self.state_size, self.action_size)

        # Initialize target model parameters with local model parameters
        self.critic_target.model.set_weights(
            self.critic_local.model.get_weights())
        self.actor_target.model.set_weights(
            self.actor_local.model.get_weights())

        # Noise process
        self.noise = OUNoise(self.action_size)
        if hasattr(self.task, 'target_z'):
            self.noise.sigma = 0.6

        # Replay memory
        self.buffer_size = 1000000
        self.batch_size = 64
        self.memory = Memory(self.buffer_size, (self.state_size,), (self.action_size,), self.random_seed)

        # Algorithm parameters
        self.gamma = 0.99  # discount factor
        self.tau = 0.001  # for soft update of target parameters

        print("random seed: ", self.random_seed)
        self._reset_episode()

    def _reset_episode(self):
        stdout.write("\n")
        self.last_state = self.last_action = None
        self.rand_action = True
        self.total_reward = 0.0
        self.ep_q = 0.0
        self.episode += 1
        self.ep_step = 0
        self.is_test = self.episode % 20 == 0
        self.add_noise = not self.is_test
        self.noise.reset()
        # self.noise.mu = np.zeros(self.noise.size)
        # if np.random.rand() > 0.7:
        #     self.noise.mu = 30 * (np.random.random_sample(self.noise.size) - 0.5)

    def step(self, state, reward, done):
        if hasattr(self.task, 'filter_state'):
            state = self.task.filter_state(state)

        self.total_reward += reward
        self._step += 1
        self.ep_step += 1

        # Save experience / reward
        if self.last_state is not None and self.last_action is not None:
            self.memory.remember(self.last_state, self.last_action, reward, done)

        if done:
            self._reset_episode()
            return None

        # Choose an action
        action = self.act(state)

        # Learn, if enough samples are available in memory
        if not self.is_test:
            experiences = self.memory.sample(self.batch_size)
            if experiences:
                self.learn(experiences)

        stdout.write("\r{} score: {:9.2f} ep: {:6d} step: {:5d} act: {:6.2f} avg_Q: {:5.2f}".format("TEST " if self.is_test else "train", self.total_reward, self.episode, self.ep_step, action[0], (self.ep_q / self.ep_step)))

        self.last_state = state
        self.last_action = action

        if hasattr(self.task, 'filter_action_response'):
            action = self.task.filter_action_response(action)
        return action

    def act(self, states):
        """Returns actions for given state(s) as per current policy."""
        states = np.reshape(states, [-1, self.state_size])
        actions = self.actor_local.model.predict(states)[0]
        assert -25.1 <= actions[0] <= 25.1, "action {} out of range".format(actions[0])
        if self.add_noise:
            # add some noise for exploration
            actions = actions + self.noise.sample()
        return actions

    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience tuples."""

        states, actions, rewards, dones, next_states = experiences

        # Get predicted next-state actions and Q values from target models
        #     Q_targets_next = critic_target(next_state, actor_target(next_state))
        actions_next = self.actor_target.model.predict_on_batch(next_states)
        Q_targets_next = self.critic_target.model.predict_on_batch(
            [next_states, actions_next])

        # Compute Q targets for current states and train critic model (local)
        Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)
        self.critic_local.model.train_on_batch(
            x=[states, actions], y=Q_targets)
        Qs = self.critic_local.model.predict_on_batch(x=[states, actions])
        self.ep_q += np.amax(Qs)

        # Train actor model (local)
        pred_actions = self.actor_local.model.predict_on_batch(states)
        action_gradients = self.critic_local.get_action_gradients([states, pred_actions, 0])
        # custom training function
        self.actor_local.train_fn([states, action_gradients[0], 1])

        # Soft-update target models
        self.soft_update(self.critic_local.model, self.critic_target.model)
        self.soft_update(self.actor_local.model, self.actor_target.model)

    def soft_update(self, local_model, target_model):
        """Soft update model parameters."""
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())

        new_weights = self.tau * local_weights + (1. - self.tau) * target_weights
        target_model.set_weights(new_weights)

class Actor:
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, action_low, action_high):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            action_low (array): Min value of each action dimension
            action_high (array): Max value of each action dimension
        """
        self.state_size = state_size
        self.action_size = action_size
        self.action_low = action_low
        self.action_high = action_high
        assert self.action_high == -self.action_low, "action space assumed to be symmetric"

        # Initialize any other variables here

        self.build_model()

    def build_model(self):
        """Build an actor (policy) network that maps states -> actions."""
        # Define input layer (states)
        states = layers.Input(shape=(self.state_size,), name='states')

        # Add hidden layers
        net = layers.Dense(units=400)(states)
        net = bn()(net)
        net = layers.Activation('relu')(net)
        net = layers.Dense(units=300)(net)
        net = bn()(net)
        net = layers.Activation('relu')(net)

        # Try different layer sizes, activations, add batch normalization, regularizers, etc.

        # Add final output layer with tanh activation
        init = initializers.random_uniform(-3e-3, 3e-3)
        raw_actions = layers.Dense(units=self.action_size, activation='tanh', name='raw_actions', kernel_initializer=init)(net)

        # Scale [-1, 1] output for each action dimension to proper range
        actions = layers.Lambda(lambda x: x * self.action_high, name='actions')(raw_actions)

        # Create Keras model
        self.model = models.Model(inputs=states, outputs=actions)

        # Define loss function using action value (Q value) gradients
        action_gradients = layers.Input(shape=(self.action_size,))
        loss = K.mean(-action_gradients * actions)

        # Incorporate any additional losses here (e.g. from regularizers)

        # Define optimizer and training function
        optimizer = optimizers.Adam(lr=0.0001)
        updates_op = optimizer.get_updates(
            params=self.model.trainable_weights, loss=loss)
        self.train_fn = K.function(
            inputs=[self.model.input, action_gradients, K.learning_phase()],
            outputs=[],
            updates=updates_op)

class Critic:
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
        """
        self.state_size = state_size
        self.action_size = action_size

        # Initialize any other variables here

        self.build_model()

    def build_model(self):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        # Define input layers
        states = layers.Input(shape=(self.state_size,), name='states')
        actions = layers.Input(shape=(self.action_size,), name='actions')

        # Add hidden layer(s) for state pathway
        net_states = layers.Dense(units=400)(states)
        net_states = bn()(net_states)
        net_states = layers.Activation('relu')(net_states)
        net_states = layers.Dense(units=300)(net_states)

        # Add hidden layer(s) for action pathway
        net_actions = layers.Dense(units=300)(actions)

        # Try different layer sizes, activations, add batch normalization, regularizers, etc.

        # Combine state and action pathways
        net = layers.Add()([net_states, net_actions])
        net = bn()(net)
        net = layers.Activation('relu')(net)

        # Add more layers to the combined network if needed

        # Add final output layer to prduce action values (Q values)
        init = initializers.random_uniform(-3e-3, 3e-3)
        Q_values = layers.Dense(units=1, kernel_initializer=init)(net)

        # Create Keras model
        self.model = models.Model(inputs=[states, actions], outputs=Q_values)

        # Define optimizer and compile model for training with built-in loss function
        optimizer = optimizers.Adam(lr=0.001)
        self.model.compile(optimizer=optimizer, loss='mse')

        # Compute action gradients (derivative of Q values w.r.t. to actions)
        action_gradients = K.gradients(Q_values, actions)

        # Define an additional function to fetch action gradients (to be used by actor model)
        self.get_action_gradients = K.function(
            inputs=[*self.model.input, K.learning_phase()],
            outputs=action_gradients)

def bn():
    # return layers.BatchNormalization()
    return lambda x: x
