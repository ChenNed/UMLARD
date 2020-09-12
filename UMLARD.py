""" This file is the implementation of UMLARD """
# import package
import tensorflow as tf
from model import Layer as layers
from keras_radam.training import RAdamOptimizer


class Model(object):
    """
    Defined:
        Placeholder
        Model architecture
        Train / Test function
    """

    def __init__(self, config, word_embedding, sess):
        self.name = config.name
        self.batch_size = config.batch_size_train  # bach size
        self.num_nodes = config.num_nodes  # number of user for diffusion network
        self.n_sequence = config.n_sequence  # user number in diffusion path
        self.total_nodes = config.total_nodes  # total number of user

        # content feature
        self.word_embedding = word_embedding  # word embedding matrix
        self.word_num = config.word_num  # words number in tweet

        # user feature
        self.d_user_fea = config.d_user_fea  # dimension of the user feature
        self.ratio = config.ratio
        self.residual = True

        # diffusion structure
        self.insTancenorm = True
        self.feat_in = config.feat_in  # number of feature
        self.z_size_1 = config.z_size_1
        self.z_size_2 = config.z_size_2

        # CTLSTM continuous-time LSTM
        self.embeding_size = config.embedding_size
        self.emb_learning_rate = config.emb_learning_rate
        self.num_ctlstm = config.num_ctlstm

        # view size
        self.equal = False
        self.view_size = config.view_size

        # content
        self.kernel_sizes = [3, 4, 5]

        # attention capsule
        self.capsule_size = config.capsule_size
        # iteration num for routing
        self.iter_routing = config.iter_routing

        # prediction
        self.num_label = 4
        self.h1 = config.h1
        self.sess = sess

        self.max_grad_norm = config.max_grad_norm

        self.learning_rate = config.learning_rate

        self.n_time_interval = config.n_time_interval  # 时间间隔 6

        self.scale1 = config.l1  # 正则化方法
        self.scale2 = config.l2
        self.stddev = config.stddev
        self.initializer = tf.random_normal_initializer(stddev=self.stddev)

        self._build_placeholders()
        self._build_var()
        self._build_model()

        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        learning_rate = tf.compat.v1.train.exponential_decay(self.learning_rate,
                                                             self.global_step, 20000, 0.9,
                                                             staircase=True)  # linear decay over time

        var_list_all = [var for var in tf.compat.v1.trainable_variables() if not 'embedding' in var.name]
        var_list_dy = [var for var in tf.compat.v1.trainable_variables() if
                       'embedding' in var.name]  # user dynamic embedding

        lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in var_list_all]) * self.scale2

        opt1 = RAdamOptimizer(learning_rate=learning_rate)  # all

        opt2 = RAdamOptimizer(learning_rate=self.emb_learning_rate)  # for user embedding

        train_op1 = opt1.minimize(self.loss + lossL2, var_list=var_list_all, global_step=self.global_step)
        train_op2 = opt2.minimize(self.loss, var_list=var_list_dy)
        self.train_op = tf.group(train_op1, train_op2)

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

    def _build_placeholders(self):
        # Temporal information
        """[v1, v2, v3, ..., vn]"""
        # words sequence for content feature learning
        self.content = tf.placeholder(tf.int32, shape=[self.batch_size, self.word_num],
                                      name="content")

        # diffusion path
        self.x = tf.placeholder(tf.int32, shape=[self.batch_size, self.n_sequence],
                                name="x")  # user sequence (user index)
        self.pos = tf.placeholder(tf.int32, shape=[self.batch_size, self.n_sequence],
                                  name="pos")  # user position sequence (position index)
        self.p_e = tf.placeholder(tf.float32, shape=[self.n_sequence, self.embeding_size],
                                  name="pos_emb")  # position embedding
        self.time_interval = tf.placeholder(tf.float32, shape=[self.batch_size, self.n_sequence],
                                            name="time_interval")

        # record the actual length of the sequence
        self.rnn_seq = tf.placeholder(tf.float32, shape=[self.batch_size, self.n_sequence],
                                      name="rnn_seq")

        # User profile
        self.user_fea = tf.placeholder(tf.float32, shape=[self.batch_size, self.n_sequence, self.d_user_fea],
                                       name="user_fea")

        # Structural information
        self.feature = tf.placeholder(tf.float32, shape=[self.batch_size, self.num_nodes, 40],
                                      name="feature")

        self.adj = tf.placeholder(tf.float32, shape=[self.batch_size, self.num_nodes, self.num_nodes],
                                  name="adj")

        # rumor label placeholder
        self.y_rumor = tf.placeholder(tf.float32, shape=[self.batch_size, 4],
                                      name="y_rumor")  # non-rumor rumor(true,false,unverified)

        self.attn_drop = tf.placeholder(dtype=tf.float32, shape=(),
                                        name="attn_drop")

        self.ffd_drop = tf.placeholder(dtype=tf.float32, shape=(),
                                       name="ffd_drop")

        self.gcn_drop = tf.placeholder(dtype=tf.float32, shape=(),
                                       name="gcn_drop")

        self.is_training = tf.placeholder(dtype=tf.bool, shape=(), name="is_training")

    def _build_var(self):
        with tf.variable_scope(self.name) as scope:
            with tf.variable_scope('embedding'):
                self.embedding = tf.get_variable('embedding', [self.total_nodes + 1, 32],
                                                 initializer=tf.random_uniform_initializer(minval=0, maxval=1),
                                                 dtype=tf.float32)  # user dynamic embedding

    def _build_model(self):
        with tf.variable_scope(self.name) as scope:
            with tf.variable_scope('user_feature'):
                print("User characteristic modeling (Dimensional-wise attention)!--------")
                user_ = tf.expand_dims(self.user_fea, axis=2)
                self.atten1, o_mlp = layers.squeeze_excitation_layer(user_, self.ratio, self.residual)
                user_fea = tf.reshape(o_mlp, [self.batch_size, self.n_sequence, -1])
                print("the shape of user profile attention: ", self.atten1.shape)
                user_fea = tf.layers.dense(user_fea, units=self.view_size, activation=tf.nn.relu, name='user_fea')

            with tf.variable_scope('structure_z'):
                print('Begin to model structural view ===> Diffusion Graph')
                # Diffusion Network
                node_index = tf.reshape(self.rnn_seq, [self.batch_size, self.n_sequence, 1])

                if self.insTancenorm:
                    print("Instance normalization!-------")
                    feature = tf.expand_dims(self.feature, -1)
                    feature = tf.contrib.layers.instance_norm(feature)
                    feature = tf.squeeze(feature, -1)
                else:
                    feature = self.feature

                print("Modeling through Diffusion GCN!-------")
                self.z_mean_1 = layers.Diffusion_GCN(input_dim=self.feat_in,
                                                     output_dim=self.z_size_1,
                                                     len_sup=2,
                                                     act=tf.nn.relu,
                                                     dropout=self.gcn_drop,
                                                     )((feature, self.adj))
                self.z_mean = layers.Diffusion_GCN(input_dim=self.z_size_1,
                                                   output_dim=self.z_size_2,
                                                   len_sup=2,
                                                   act=tf.nn.relu,
                                                   dropout=self.gcn_drop,
                                                   )((self.z_mean_1, self.adj))
                str_fea = self.z_mean * node_index  # [B,N,Z]
                print('the shape of user structural information...', str_fea.shape)

            with tf.variable_scope('embedding'):
                # n1,n2,n3,...,nn
                x_dy = tf.nn.dropout(tf.nn.embedding_lookup(self.embedding, self.x), 1 - self.ffd_drop)

                print('the shape of dynamic embedding x_dy...', x_dy.shape)  # [B, N, E]

            with tf.variable_scope('temporal_view_all'):
                print('Begin to model temporal view ===> Nodes Sequence')
                x_dy = tf.layers.dense(x_dy, units=self.embeding_size, activation=tf.nn.relu)
                pos = tf.nn.embedding_lookup(self.p_e, self.pos)  # [?, N, E]
                x_vector = pos + x_dy
                print('the shape of x_vector...', x_vector.shape)  # [B, N, E+8]
                # use time-decay lstm"""
                time_interval = tf.reshape(self.time_interval, shape=[self.batch_size, self.n_sequence, -1])
                print("the shape of time_interval...", time_interval.shape)
                input_con = tf.concat([x_vector, time_interval], axis=2)
                print("the shape of input_con...", input_con.shape)
                inp = tf.unstack(input_con, self.n_sequence, 1)
                cell = layers.TimeRNNCell(num_units=self.num_ctlstm,
                                          batch_size=self.batch_size,
                                          feat_in=self.embeding_size)

                outputs, _ = tf.contrib.rnn.static_rnn(cell,
                                                       inp,
                                                       dtype=tf.float32)
                hidden_states = tf.transpose(tf.stack(outputs), [1, 0, 2])  # batch_size,num_steps,num_seq,feat_in]

                print("------temporal output------")
                print('the shape of hidden_states...', hidden_states.get_shape())
                hidden_states = hidden_states * node_index
                temp_fea = hidden_states

            with tf.variable_scope("equal_dim"):
                if self.equal:
                    # use mlp to transform view vectors into the same dimension whether need activation func
                    str_fea = tf.layers.dense(str_fea, units=self.view_size, use_bias=False, activation=None,
                                              name="user_stru")
                    # str_fea = tf.nn.dropout(str_fea, 1 - self.ffd_drop)
                    print("The shape of str_fea...", str_fea.shape)

                    # use mlp to transform view vectors into the same dimision
                    temp_fea = tf.layers.dense(temp_fea, units=self.view_size, use_bias=False, activation=None,
                                               name="user_temp")
                    print("the shape of temp_fea...", temp_fea.get_shape())

                    # use mlp to transform view vectors into the same dimision

                    user_fea = tf.layers.dense(user_fea, units=self.view_size, use_bias=False, activation=None,
                                               name="user_pro")
                    print("the shape of user_fea...", user_fea.get_shape())
                else:
                    print('without equal_dims!--------')

            with tf.variable_scope('view_attention_layer'):
                print('view-attention!---------')
                final_input, self.atten2 = layers.self_attention(user_fea, temp_fea, str_fea)

            with tf.variable_scope('attention_caps_layer'):
                print('capsule attention!------------')
                fi, self.atten3, self.atten3_list = layers.get_caps_alpha(final_input, self.view_size,
                                                                          3 * self.capsule_size, iter=self.iter_routing)
                print("attenion in attention caps: ", self.atten3.shape)
                print("after attention cap", fi.get_shape())

            with tf.variable_scope('content_all'):
                content = tf.nn.dropout(tf.nn.embedding_lookup(self.word_embedding, self.content),
                                        1)
                pooled_outputs = []
                for kernel_size in self.kernel_sizes:
                    # CNN layer
                    conv = tf.layers.conv1d(content, 100, kernel_size,
                                            name='conv-%s' % kernel_size)
                    # global max pooling layer
                    gmp = tf.reduce_max(conv, reduction_indices=[1], name='gmp')
                    pooled_outputs.append(gmp)
                h_pool = tf.concat(pooled_outputs, 1)  # 池化后进行拼接
                print(h_pool.shape)
                f = tf.concat((fi, h_pool), axis=1)

            with tf.variable_scope('predict_layer_all'):
                print('classification layer')

                self.pred1 = tf.layers.dense(f, units=self.h1, use_bias=True, activation=tf.nn.relu,
                                             name="pred1")
                self.pred = tf.layers.dense(self.pred1, units=self.num_label, use_bias=True, activation=None,
                                            name="pred")
                print(self.pred.shape)

            with tf.variable_scope("loss_layer_all"):
                self.loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.y_rumor, logits=self.pred)
                self.pred1_c = tf.nn.softmax(self.pred)
                self.predictions = tf.argmax(self.pred1_c, 1, name="predictions")
                truth = tf.argmax(self.y_rumor, axis=1)
                self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.predictions, truth), dtype=tf.float32))

    def train_batch(self, x, time_interval, rnn_seq, user_fea, feature, adj, y_rumor, atten_drop, ffd_drop, gcn_drop,
                    pos, p_e, is_train, content):
        self.sess.run(self.train_op,
                      feed_dict={
                          self.x: x,
                          self.user_fea: user_fea,
                          self.feature: feature,
                          self.adj: adj,
                          self.y_rumor: y_rumor,
                          self.time_interval: time_interval,
                          self.rnn_seq: rnn_seq,
                          self.attn_drop: atten_drop,
                          self.ffd_drop: ffd_drop,
                          self.gcn_drop: gcn_drop,
                          self.pos: pos,
                          self.p_e: p_e,
                          self.is_training: is_train,
                          self.content: content
                      })

    def get_error(self, x, time_interval, rnn_seq, user_fea, feature, adj, y_rumor, atten_drop, ffd_drop, gcn_drop, pos,
                  p_e, is_train, content):
        loss = self.sess.run(self.loss,
                             feed_dict={self.x: x,
                                        self.user_fea: user_fea,
                                        self.feature: feature,
                                        self.adj: adj,
                                        self.y_rumor: y_rumor,
                                        self.time_interval: time_interval,
                                        self.rnn_seq: rnn_seq,
                                        self.attn_drop: atten_drop,
                                        self.ffd_drop: ffd_drop,
                                        self.gcn_drop: gcn_drop,
                                        self.pos: pos,
                                        self.p_e: p_e,
                                        self.is_training: is_train,
                                        self.content: content
                                        })
        return loss

    def predict(self, x, time_interval, rnn_seq, user_fea, feature, adj, y_rumor, atten_drop, ffd_drop, gcn_drop, pos,
                p_e, is_train, content):
        pred = self.sess.run(self.predictions,
                             feed_dict={self.x: x,
                                        self.user_fea: user_fea,
                                        self.feature: feature,
                                        self.adj: adj,
                                        self.y_rumor: y_rumor,
                                        self.time_interval: time_interval,
                                        self.rnn_seq: rnn_seq,
                                        self.attn_drop: atten_drop,
                                        self.ffd_drop: ffd_drop,
                                        self.gcn_drop: gcn_drop,
                                        self.pos: pos,
                                        self.p_e: p_e,
                                        self.is_training: is_train,
                                        self.content: content

                                        })
        return pred

    def predict_(self, x, time_interval, rnn_seq, user_fea, feature, adj, y_rumor, atten_drop, ffd_drop, gcn_drop, pos,
                 p_e, is_train, content):
        pred = self.sess.run(self.pred1_c,
                             feed_dict={self.x: x,
                                        self.user_fea: user_fea,
                                        self.feature: feature,
                                        self.adj: adj,
                                        self.y_rumor: y_rumor,
                                        self.time_interval: time_interval,
                                        self.rnn_seq: rnn_seq,
                                        self.attn_drop: atten_drop,
                                        self.ffd_drop: ffd_drop,
                                        self.gcn_drop: gcn_drop,
                                        self.pos: pos,
                                        self.p_e: p_e,
                                        self.is_training: is_train,
                                        self.content: content
                                        })
        return pred

    def acc(self, x, time_interval, rnn_seq, user_fea, feature, adj, y_rumor, atten_drop, ffd_drop, gcn_drop, pos, p_e,
            is_train, content):
        acc = self.sess.run(self.accuracy,
                            feed_dict={self.x: x,
                                       self.user_fea: user_fea,
                                       self.feature: feature,
                                       self.adj: adj,
                                       self.y_rumor: y_rumor,
                                       self.time_interval: time_interval,
                                       self.rnn_seq: rnn_seq,
                                       self.attn_drop: atten_drop,
                                       self.ffd_drop: ffd_drop,
                                       self.gcn_drop: gcn_drop,
                                       self.pos: pos,
                                       self.p_e: p_e,
                                       self.is_training: is_train,
                                       self.content: content

                                       })
        return acc
