import numpy as np
import tensorflow as tf
import collections
from tensorflow.python.layers.core import Dense as Dense_

conv1d = tf.layers.conv1d

tfversion_ = tf.VERSION.split(".")
global tfversion
if int(tfversion_[0]) < 1:
    raise EnvironmentError("TF version should be above 1.0!!")
if int(tfversion_[1]) < 1:
    print("Working in TF version 1.0....")
    from tensorflow.python.ops.rnn_cell_impl import _RNNCell as RNNCell

    tfversion = "old"
else:
    print("Working in TF version 1.%d...." % int(tfversion_[1]))
    from tensorflow.python.ops.rnn_cell_impl import RNNCell

    tfversion = "new"

# gconvLSTM
_LSTMStateTuple = collections.namedtuple("LSTMStateTuple", ('c', 'h'))


class LSTMStateTuple(_LSTMStateTuple):
    __slots__ = ()

    @property
    def dtype(self):
        (c, h) = self
        if not c.dtype == h.dtype:
            raise TypeError("Inconsistent internal state")
        return c.dtype


# Time aware LSTM

class TimeRNNCell(RNNCell):
    def __init__(self, num_units, batch_size=None, feat_in=None, state_is_tuple=True, activation=None, reuse=None):
        if tfversion == 'new':
            super(TimeRNNCell, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._batch_size = batch_size
        self._state_is_tuple = state_is_tuple
        self._feat_in = feat_in

        if activation is None:
            self.activation = lambda x: 1.7159 * tf.tanh(2 / 3 * x)
            # from: LeCun et al. 2012: Efficient backprop
        else:
            self.activation = activation

    @property
    def state_size(self):
        return (LSTMStateTuple((self._num_units), (self._num_units))
                if self._state_is_tuple else 2 * self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def zero_state(self, batch_size, dtype):
        with tf.name_scope(type(self).__name__ + "myZeroState"):
            zero_state_c = tf.zeros([self._batch_size, self._num_units], name='c')
            zero_state_h = tf.zeros([self._batch_size, self._num_units], name='h')
            return (zero_state_c, zero_state_h)

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            if self._state_is_tuple:
                c, h = state
            else:
                c, h = tf.split(value=state, num_or_size_splits=2, axis=1)

            feat_in = self._feat_in
            batch_size = self._batch_size
            feat_out = self._num_units

            scope = tf.get_variable_scope()
            with tf.variable_scope(scope) as scope:
                try:
                    # orthogonal_initializer
                    Wzxt = tf.get_variable("Wzxt", [feat_in, feat_out], dtype=tf.float32,
                                           initializer=tf.orthogonal_initializer())
                    Wixt = tf.get_variable("Wixt", [feat_in, feat_out], dtype=tf.float32,
                                           initializer=tf.orthogonal_initializer())
                    Wfxt = tf.get_variable("Wfxt", [feat_in, feat_out], dtype=tf.float32,
                                           initializer=tf.orthogonal_initializer())
                    Woxt = tf.get_variable("Woxt", [feat_in, feat_out], dtype=tf.float32,
                                           initializer=tf.orthogonal_initializer())
                    Wtxt = tf.get_variable("Wtxt", [feat_in, feat_out], dtype=tf.float32,
                                           initializer=tf.orthogonal_initializer())

                    Wzht = tf.get_variable("Wzht", [feat_out, feat_out], dtype=tf.float32,
                                           initializer=tf.orthogonal_initializer())
                    Wiht = tf.get_variable("Wiht", [feat_out, feat_out], dtype=tf.float32,
                                           initializer=tf.orthogonal_initializer())
                    Wfht = tf.get_variable("Wfht", [feat_out, feat_out], dtype=tf.float32,
                                           initializer=tf.orthogonal_initializer())
                    Woht = tf.get_variable("Woht", [feat_out, feat_out], dtype=tf.float32,
                                           initializer=tf.orthogonal_initializer())
                    Wttt = tf.get_variable("Wttt", [1, feat_out], dtype=tf.float32,
                                           initializer=tf.orthogonal_initializer())
                    Wott = tf.get_variable("Wott", [1, feat_out], dtype=tf.float32,
                                           initializer=tf.orthogonal_initializer())

                except ValueError:
                    scope.reuse_variables()
                    # orthogonal_initializer
                    Wzxt = tf.get_variable("Wzxt", [feat_in, feat_out], dtype=tf.float32,
                                           initializer=tf.orthogonal_initializer())
                    Wixt = tf.get_variable("Wixt", [feat_in, feat_out], dtype=tf.float32,
                                           initializer=tf.orthogonal_initializer())
                    Wfxt = tf.get_variable("Wfxt", [feat_in, feat_out], dtype=tf.float32,
                                           initializer=tf.orthogonal_initializer())
                    Woxt = tf.get_variable("Woxt", [feat_in, feat_out], dtype=tf.float32,
                                           initializer=tf.orthogonal_initializer())
                    Wtxt = tf.get_variable("Wtxt", [feat_in, feat_out], dtype=tf.float32,
                                           initializer=tf.orthogonal_initializer())

                    Wzht = tf.get_variable("Wzht", [feat_out, feat_out], dtype=tf.float32,
                                           initializer=tf.orthogonal_initializer())
                    Wiht = tf.get_variable("Wiht", [feat_out, feat_out], dtype=tf.float32,
                                           initializer=tf.orthogonal_initializer())
                    Wfht = tf.get_variable("Wfht", [feat_out, feat_out], dtype=tf.float32,
                                           initializer=tf.orthogonal_initializer())
                    Woht = tf.get_variable("Woht", [feat_out, feat_out], dtype=tf.float32,
                                           initializer=tf.orthogonal_initializer())
                    Wttt = tf.get_variable("Wttt", [1, feat_out], dtype=tf.float32,
                                           initializer=tf.orthogonal_initializer())
                    Wott = tf.get_variable("Wott", [1, feat_out], dtype=tf.float32,
                                           initializer=tf.orthogonal_initializer())

                bzt = tf.get_variable("bzt", [feat_out], dtype=tf.float32, initializer=tf.constant_initializer(0))
                bit = tf.get_variable("bit", [feat_out], dtype=tf.float32, initializer=tf.constant_initializer(0))
                bft = tf.get_variable("bft", [feat_out], dtype=tf.float32, initializer=tf.constant_initializer(0))
                bot = tf.get_variable("bot", [feat_out], dtype=tf.float32, initializer=tf.constant_initializer(0))
                b_weight = tf.get_variable("b_weight", [feat_out], dtype=tf.float32,
                                           initializer=tf.constant_initializer(0))

                x = inputs[:, :feat_in]  # feature
                time_interval = inputs[:, feat_in:]  # time index
                # print(time_interval.get_shape)

                # time gate
                txt = tf.matmul(x, Wtxt)
                ttt = tf.sigmoid(tf.matmul(time_interval, Wttt))
                tt = txt + ttt + b_weight
                tt = tf.sigmoid(tt)

                # candidate
                zxt = tf.matmul(x, Wzxt)
                zht = tf.matmul(h, Wzht)
                zt = zxt + zht + bzt
                zt = tf.tanh(zt)  # * time_coef

                # input gate
                ixt = tf.matmul(x, Wixt)  # [batch_size,feat_out]
                iht = tf.matmul(h, Wiht)
                it = ixt + iht + bit
                it = tf.sigmoid(it)

                # forget gate
                fxt = tf.matmul(x, Wfxt)
                fht = tf.matmul(h, Wfht)
                ft = fxt + fht + bft
                ft = tf.sigmoid(ft)

                new_c = ft * c + it * tt * zt

                # output gate
                oxt = tf.matmul(x, Woxt)
                oht = tf.matmul(h, Woht)
                ott = tf.matmul(time_interval, Wott)
                ot = oxt + ott + oht + bot
                ot = tf.sigmoid(ot)

                # h
                new_h = ot * tf.tanh(new_c)

                if self._state_is_tuple:
                    new_state = LSTMStateTuple(new_c, new_h)
                else:
                    new_state = tf.concat([new_c, new_h], 1)
                return new_h, new_state

flags = tf.app.flags
FLAGS = flags.FLAGS


def weight_variable_glorot(input_dim, output_dim, name=""):
    """Create a weight variable with Glorot & Bengio (AISTATS 2010)
    initialization.
    """
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = tf.random_uniform([input_dim, output_dim], minval=-init_range,
                                maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def build_sparse_matrix(L):
    L = L.tocoo()
    indices = np.column_stack((L.row, L.col))
    L = tf.SparseTensor(indices, L.data, L.shape)
    return tf.sparse_reorder(L)


def calculate_random_walk_matrix(adj_mx):
    # adj = adj
    d = tf.reduce_sum(adj_mx, axis=1)
    d_inv = tf.pow(d, -1)
    zero = tf.zeros_like(d_inv)
    d_inv = tf.where(tf.is_finite(d_inv), d_inv, zero)
    d_mat_inv = tf.matrix_diag(d_inv)
    random_walk_mx = d_mat_inv * adj_mx
    return random_walk_mx


def uniform(shape, scale=0.05, name=None):
    """Uniform init."""
    initial = tf.random_uniform(shape, minval=-scale, maxval=scale, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def glorot(shape, name=None):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def zeros(shape, name=None):
    """All zeros."""
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def ones(shape, name=None):
    """All ones."""
    initial = tf.ones(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1. / keep_prob)


# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res


class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    Implementation inspired by keras (http://keras.io).

    # Properties
        name: String, defines the variable scope of the layer.
        logging: Boolean, switches Tensorflow histogram logging on/off

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])


class Diffusion_GCN(Layer):
    """Graph convolution layer."""

    def __init__(self, input_dim, output_dim, len_sup=2, dropout=0.,
                 act=tf.nn.relu, bias=False, **kwargs):
        super(Diffusion_GCN, self).__init__(**kwargs)
        self.dropout = dropout
        self.act = act
        self.bias = bias

        with tf.variable_scope(self.name + '_vars'):
            # for i in range(len_sup):
            #     self.vars['weights_' + str(i)] = glorot([input_dim * 3, output_dim],
            #                                             name='weights')  # + str(i))
            self.vars['weights'] = glorot([input_dim * 3, output_dim],
                                          name='weights')
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

    def _call(self, inputs):
        x, adj_mxs = inputs  # [B, N, F]

        # dropout
        x = tf.nn.dropout(x, 1 - self.dropout)

        batch_size, num_nodes, input_size = x.get_shape()
        adj_mx = tf.unstack(adj_mxs, batch_size, 0)  # B[N, N]

        supports_list = list()

        for adj in adj_mx:
            support = list()
            support.append(tf.transpose(calculate_random_walk_matrix(adj)))
            support.append(tf.transpose(calculate_random_walk_matrix(tf.transpose(adj))))
            supports_list.append(support)

        X0 = tf.unstack(x, batch_size, 0)  # B[ N, F]

        out_ = list()
        for supports, x0 in zip(supports_list, X0):

            outs = list()
            for i in range(len(supports)):
                # pre_sup = dot(x0, self.vars['weights_' + str(i)],
                #               sparse=self.sparse_inputs)
                support = dot(supports[i], x0, sparse=False)
                outs.append(support)  # 2[N, F']
            # out
            outs.append(x0)  # Adds for x itself.
            outs = tf.stack(outs)  # [3, N, E]
            out_.append(outs)

        output = tf.stack(out_)  # [B, 3, N, E]
        # bias
        output = tf.transpose(output, [0, 2, 3, 1])
        # print(output.shape)
        output = tf.reshape(output, [batch_size * num_nodes, input_size * 3])  # [32*200, 50*3]
        output = dot(output, self.vars['weights'])
        if self.bias:
            output += self.vars['bias']
        # out_.append(output)  # B[N, F']
        # out_ = tf.stack(out_)  # [B, N, F']
        output = tf.reshape(output, [batch_size, num_nodes, -1])  # [B, N, F']

        return self.act(output)


### graph capsule
epsilon = 1e-11

"""capsule network"""


def weight_variable(shape, dtype=tf.float32, name=None, lamda=0.0001):
    var = tf.get_variable(name, shape, dtype=dtype, initializer=tf.contrib.layers.xavier_initializer())
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(lamda)(var))
    return var


def l2(v_j):
    # # ?, 1, A, 1
    vec_squared_norm = tf.square(v_j)  # (?, 1, A, 1)
    v_l2 = tf.sqrt(vec_squared_norm + epsilon)
    return v_l2


"""attention"""


def get_caps_alpha(inputs_poses, input_dim, att_hid_size, iter=3):
    inputs_shape = inputs_poses.get_shape()
    b_IJ = tf.zeros(shape=[inputs_shape[0], inputs_shape[1], 1, 1],
                    dtype=np.float32)  # (?, N, 1, 1)
    F_reshape = tf.reshape(inputs_poses, [-1, input_dim])  # B*N, d
    W_F = weight_variable([input_dim, att_hid_size], name='capsuel_proj')
    F_p = tf.matmul(F_reshape, W_F)
    F_p = tf.reshape(F_p, [-1, inputs_shape[1], att_hid_size])
    F_p = tf.expand_dims(F_p, axis=3)  # ?, N, A, 1
    F_p_stop = tf.stop_gradient(F_p, name='F_p_stop_gradient')  # ?, N, A, 1

    c_IJ_list = []

    with tf.variable_scope('routing'):
        for i in range(iter):
            with tf.variable_scope('iter_' + str(i)):
                # obtaining coupling coefficients
                c_IJ = tf.nn.softmax(b_IJ, dim=1) # (?, N, 1, 1)
                c_IJ_list.append(c_IJ)

                if i == iter - 1:
                    f_v = tf.multiply(c_IJ, F_p)  # ?, N, A, 1
                    f_v = tf.reduce_sum(f_v, axis=1, keep_dims=True)  # batch * seq_per_img, 1, h_dim, 1

                else:
                    # obtaining weighted-sum feature
                    f_v = tf.multiply(c_IJ, F_p_stop)  # ?, N, A, 1
                    f_v = tf.reduce_mean(f_v, axis=1, keep_dims=True)  # ?, 1, A, 1
                    f_v = l2(f_v)
                    v_j = tf.tile(f_v, [1, inputs_shape[1], 1, 1])  # (?, N, A, 1)
                    u_produce_v = tf.reduce_sum(F_p_stop * v_j, axis=2,
                                                keep_dims=True)  # (?, N, 1, 1)
                    # u_produce_v = tf.matmul(F_p_stop, v_j, transpose_a=True)
                    b_IJ += u_produce_v
    h_out = tf.reshape(f_v, [-1, att_hid_size])
    alpha = tf.reshape(c_IJ, [-1, inputs_shape[1]])

    return h_out, alpha, c_IJ_list


def self_attention(inputs_a, inputs_v, inputs_t, name="_Unshare"):
    """
    :param inputs_a: audio input (B, T, dim)
    :param inputs_v: video input (B, T, dim)
    :param inputs_t: text input (B, T, dim)
    :param mask: (B, T, 1)
    :param name: scope name
    :return:
    """

    inputs_a = tf.expand_dims(inputs_a, axis=1)
    inputs_v = tf.expand_dims(inputs_v, axis=1)
    inputs_t = tf.expand_dims(inputs_t, axis=1)
    # inputs = (B, 3, T, dim)
    inputs = tf.concat([inputs_a, inputs_v, inputs_t], axis=1)
    t = inputs.get_shape()[2].value
    B = inputs.get_shape()[0].value
    share_param = True
    hidden_size = inputs.shape[-1].value  # D value - hidden size of the RNN layer
    kernel_init1 = tf.glorot_uniform_initializer(seed=1234, dtype=tf.float32)
    dense = Dense_(hidden_size, kernel_initializer=kernel_init1)
    if share_param:
        scope_name = 'self_attn'
    else:
        scope_name = 'self_attn' + name
    # print(scope_name)
    inputs = tf.transpose(inputs, [2, 0, 1, 3])
    # [T,B,3,dim]
    with tf.variable_scope(scope_name):
        outputs = []
        atten = []
        for x in range(t):
            t_x = inputs[x, :, :, :]
            # t_x => B, 3, dim
            den = True
            if den:
                x_proj = dense(t_x)
                x_proj = tf.nn.tanh(x_proj)
            else:
                x_proj = t_x
            u_w = tf.Variable(tf.random_normal([hidden_size, 1], stddev=0.01, seed=1234))
            x = tf.tensordot(x_proj, u_w, axes=1)
            alphas = tf.nn.softmax(x, axis=1)  # [B, 3, 1]
            output = tf.matmul(tf.transpose(t_x, [0, 2, 1]), alphas)  # [B, 3, d]
            atten.append(tf.squeeze(alphas, axis=-1))
            output = tf.squeeze(output, axis=-1)
            outputs.append(output)
        atten = tf.stack(atten, axis=1)
        final_output = tf.stack(outputs, axis=1)  # * mask
        return final_output, atten


def squeeze_excitation_layer(input_x, ratio, residual):
    kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
    bias_initializer = tf.constant_initializer(value=0.0)
    channel = input_x.get_shape()[-1]
    # Global average pooling
    squeeze = tf.reduce_mean(input_x, axis=[1, 2], keepdims=True)

    bottleneck_fc = tf.layers.dense(inputs=squeeze,
                                    units=channel // ratio,
                                    activation=tf.tanh,
                                    kernel_initializer=kernel_initializer,
                                    bias_initializer=bias_initializer,
                                    name='bottleneck_fc')
    recover_fc = tf.layers.dense(inputs=bottleneck_fc,
                                 units=channel,
                                 activation=tf.nn.softmax,
                                 kernel_initializer=kernel_initializer,
                                 bias_initializer=bias_initializer,
                                 name='recover_fc')
    scale = input_x * recover_fc
    if residual:
        output = input_x + scale
    else:
        output = scale
    atten = tf.squeeze(recover_fc)
    return atten, output


