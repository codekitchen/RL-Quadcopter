import tensorflow as tf

# "penalized tanh" similar to https://arxiv.org/pdf/1602.05980.pdf
def penalized_tanh(tensor):
    return 0.1 * tensor
    # return tf.nn.tanh(0.5 * tensor)
    # th = tf.nn.tanh(tensor)
    # return tf.maximum(th, 0.25 * th)

class ActorCritic:
    def __init__(self, state_shape, action_shape, summarize=False):
        self.summarize = summarize
        self.random_init = tf.random_uniform_initializer(-3e-5, 3e-5)
        self.is_training = tf.placeholder(tf.bool)
        self._build_actor(state_shape, action_shape)
        self._build_critic()

    def _build_actor(self, state_shape, action_shape):
        assert(len(state_shape) == 2)
        self.state = tf.placeholder(
            tf.float32, shape=((None,) + state_shape), name="state")
        self.state_flat = tf.reshape(self.state, [-1, state_shape[0] * state_shape[1]], name="flatten")

        with tf.variable_scope("actor"):
            layer = self.state_flat
            layer = tf.layers.batch_normalization(layer, training=self.is_training)
            layer = self._dense_norm(layer, 400, "dense1")
            layer = self._dense_norm(layer, 300, "dense2")
            layer = self._dense_norm(layer, action_shape, name="action",
                kernel_initializer=self.random_init,
                activation=penalized_tanh)
            self.raw_action = layer
            self.action = self.raw_action

    def _build_critic(self):
        with tf.variable_scope("critic"):
            with tf.variable_scope("state"):
                state_layer = self.state_flat
                state_layer = self._dense_norm(state_layer, 400, name="dense1")
            action_layer = self.action
            layer = tf.layers.dense(tf.concat([state_layer, action_layer], 1), 300, activation=tf.nn.relu, name="dense1")

            self.raw_value = tf.layers.dense(layer, 1, activation=None, kernel_initializer=self.random_init, name="value")
            self.value = self.raw_value[:, 0] # grab the first value for each input in the batch

    def _dense_norm(self, layer, units, name, kernel_initializer=None, activation=tf.nn.relu):
        layer = tf.layers.dense(layer, units, kernel_initializer=kernel_initializer, name=name)
        layer = tf.layers.batch_normalization(layer, training=self.is_training)
        layer = activation(layer)
        return layer
    
    def _histogram(self, name, values):
        if self.summarize:
            tf.summary.histogram(name, values)
