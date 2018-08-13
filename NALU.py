import tensorflow as tf
class NALU:
    def __init__(self, in_size=1, out_size=1,
                 mode='NALU', epsil=1e-7, name=''):
        self.in_size = in_size
        self.out_size = out_size
        self.mode = mode
        self.epsil = epsil
        self.name = name
        
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            shape = (in_size, out_size)
            self.W_hat = tf.get_variable("W_hat", shape=shape)
            self.M_hat = tf.get_variable("M_hat", shape=shape)
            self.G = tf.get_variable("G", shape=shape)
    
    def __call__(self, x):
        W = tf.nn.tanh(self.W_hat) * tf.nn.sigmoid(self.M_hat)
        a = tf.matmul(x, W)
        if self.mode == 'NAC':
            return a
        m = tf.exp(tf.matmul(tf.log(tf.abs(x)+self.epsil), W))
        g = tf.sigmoid(tf.matmul(x, self.G))
        return g*a + (1-g)*m