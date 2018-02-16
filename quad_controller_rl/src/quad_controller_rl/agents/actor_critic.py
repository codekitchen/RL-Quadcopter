import tensorflow as tf

# "penalized tanh" similar to https://arxiv.org/pdf/1602.05980.pdf
def penalized_tanh(tensor):
    # return 0.1 * tensor
    return tf.nn.tanh(0.1 * tensor)
    # th = tf.nn.tanh(tensor)
    # return tf.maximum(th, 0.25 * th)

class ActorCritic:
    def __init__(self, state_shape, action_shape, summarize=False):
        self.summarize = summarize
        self.random_init = tf.random_uniform_initializer(-3e-3, 3e-3)
        # interesting paper on what L2 regularization is actually doing in a network that also uses batch normalization: https://arxiv.org/pdf/1706.05350.pdf
        self.regularizer = tf.contrib.layers.l2_regularizer(1e-2)
        self.is_training = tf.placeholder(tf.bool)
        self._build_actor(state_shape, action_shape)
        self._build_critic(state_shape, action_shape)

    def _build_actor(self, state_shape, action_shape):
        self.state = tf.placeholder(
            tf.float32, shape=((None,) + state_shape), name="state")

        with tf.variable_scope("actor"):
            layer = self.state
            layer = tf.layers.batch_normalization(layer, training=self.is_training)
            layer = self._dense(layer, 16, norm=True, name="dense1")
            layer = self._dense(layer, 16, norm=True, name="dense2")
            layer = self._dense(layer, 16, norm=True, name="dense3")
            layer = self._dense(layer, action_shape, norm=True, name="action",
                kernel_initializer=self.random_init,
                activation=penalized_tanh)
            self.action = layer

            self.action_gradients = tf.placeholder(tf.float32, shape=((None,action_shape)), name="action_gradients")
            self.actor_loss = tf.reduce_mean(-self.action_gradients * self.action)

    def _build_critic(self, state_shape, action_shape):
        with tf.variable_scope("critic"):
            with tf.variable_scope("state"):
                self.critic_state = tf.placeholder(tf.float32, shape=((None,) + state_shape), name="critic_state")
                state_layer = self.critic_state
                state_layer = self._dense(state_layer, 16, norm=True, kernel_regularizer=self.regularizer, name="dense1")
            self.critic_action = tf.placeholder(tf.float32, shape=((None,action_shape)), name="critic_action")
            action_layer = self.critic_action
            layer = self._dense(tf.concat([state_layer, action_layer], 1), 16, activation=tf.nn.relu, kernel_regularizer=self.regularizer, name="dense1")

            self.raw_value = self._dense(layer, 1, activation=None, kernel_initializer=self.random_init, kernel_regularizer=self.regularizer, name="value")
            self.value = self.raw_value[:, 0] # grab the first value for each input in the batch

            self.critic_action_gradients = tf.gradients(self.value, self.critic_action)[0]

    def _dense(self, layer, units, name, kernel_initializer=None, kernel_regularizer=None, norm=False, activation=tf.nn.relu):
        layer = tf.layers.dense(layer, units, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer, name=name)
        if norm:
            layer = tf.layers.batch_normalization(layer, training=self.is_training)
        if activation:
            layer = activation(layer)
        with tf.variable_scope(name, reuse=True):
            weights = tf.get_variable('kernel')
        self._histogram(name+"/weights", weights)
        return layer
    
    def _histogram(self, name, values):
        if self.summarize:
            tf.summary.histogram(name, values)