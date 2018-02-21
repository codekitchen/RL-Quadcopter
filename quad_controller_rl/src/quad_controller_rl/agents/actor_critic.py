import tensorflow as tf

def scaled_tanh(tensor):
    return tf.nn.tanh(tensor * 0.1)

class ActorCritic:
    def __init__(self, state_size, action_size, summarize=False):
        self.summarize = summarize
        self.random_init = tf.random_uniform_initializer(-3e-3, 3e-3)
        self.is_training = tf.placeholder(tf.bool)
        self._build_actor(state_size, action_size)
        self._build_critic(state_size, action_size)

    def _build_actor(self, state_size, action_size):
        with tf.variable_scope("actor"):
            self.actor_state = tf.placeholder(tf.float32, shape=(None,state_size), name="state")
            layer = self.actor_state
            layer = self._dense(layer, 400, name="dense1")
            layer = self._dense(layer, 300, name="dense2")
            layer = self._dense(layer, action_size, name="action",
                kernel_initializer=self.random_init,
                activation=scaled_tanh)
            self.action = layer
            self.actor_gradients = tf.placeholder(tf.float32, shape=(None, action_size), name="actor_gradients")

    def _build_critic(self, state_size, action_size):
        with tf.variable_scope("critic"):
            self.critic_state = tf.placeholder(tf.float32, shape=(None,state_size), name="state")
            self.critic_action = tf.placeholder(tf.float32, shape=(None, action_size), name="action")
            with tf.variable_scope("state"):
                state_layer = self.critic_state
                state_layer = self._dense(state_layer, 400, name="dense1")
                state_layer = self._dense(state_layer, 300, activation=None, name="dense2")
            with tf.variable_scope("action"):
                action_layer = self.critic_action
                action_layer = self._dense(action_layer, 300, activation=None, name="dense1")

            layer = tf.nn.relu(state_layer+action_layer, name="combined")

            self.value = self._dense(layer, 1, activation=None, kernel_initializer=self.random_init, name="value")
            self.ys = tf.placeholder(tf.float32, shape=(None, 1), name="ys")
            self.critic_gradients = tf.gradients(self.value, self.critic_action)[0]

    def _dense(self, layer, units, name, kernel_initializer=None, activation=tf.nn.relu):
        layer = tf.layers.dense(layer, units, kernel_initializer=kernel_initializer, name=name)
        if activation:
            layer = activation(layer)
        with tf.variable_scope(name, reuse=True):
            weights = tf.get_variable('kernel')
        self._histogram(name+"/weights", weights)
        return layer
    
    def _histogram(self, name, values):
        if self.summarize:
            tf.summary.histogram(name, values)