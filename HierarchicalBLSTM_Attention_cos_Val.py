import numpy as np
import tensorflow as tf
from NN import DNN
from NN import RNN
from utils import tf_utils
from utils import utils 
import os

class Hier_Att_cos_Val(object):
    """docstring for Hier_GAN"""
    def __init__(self, args):
        super(Hier_Att_cos_Val, self).__init__()

        self.story_nsent = 4
        self.model_name = 'Hier_Att_cos_Val'
        self.word_rnn_units = map(int, args.word_rnn_setting.split('x')) 
        self.sent_rnn_units = map(int, args.sent_rnn_setting.split('x'))

        # self.mlp_units = [int(elem) for elem in mlp_setting.split('x')]
        self.batchsize = int(args.batchsize)

        self.delta = float(args.delta)
       
        self.wemb_trainable = bool(int(args.wtrainable))
        self.discrit_lr = float(args.discrit_lr)

        self.mode = args.rnn_output_mode

        self.bias = 0.001
        
        self.dnn_discriminator_setting = map(int, args.score_func_setting.split('x'))

        self.reasoning_type = args.reasoning_type

        self.loss_type = args.loss_type
        self.score_func = args.score_func
        self.discrim_regularization_level = args.drl 
        self.top_k = args.top_k

        # self.random_input_type = random_input_type
        self.discrim_regularization_dict = {0:"no regularization on discriminator",
                                            1:"L2 on discriminator DNN",
                                            2:"L2 on discriminator word level RNN + DNN",
                                            3:"L2 on discriminator word level RNN",
                                            4:"L2 on discriminator all RNN",
                                            5:"L2 on discriminator all level"}
        
        self.regularization_factor = float(args.reg_factor)

        if self.loss_type == 'log':
            assert self.dnn_discriminator_setting[-1] == 2
            assert self.score_func != 'cos'
            assert self.sent_rnn_units[-1] > 1
            assert self.nonlin_func == None
        else:
            if self.score_func == 'DNN':
                assert self.dnn_discriminator_setting[-1] == 1
            elif self.score_func == 'cos':
                assert self.sent_rnn_units[-1] > 2
            else:
                assert self.sent_rnn_units[-1] == 1
        
        if self.score_func != "DNN":
            assert self.discrim_regularization_level != 1 and self.discrim_regularization_level != 2


    def word_level_encoding(self):
        self.sent_seqs = []

        sent_seq = self.word_level_rnn.BLSTM_encoder(self.inputs_variables[0], 
                                                     self.sent_seq_len[0], 
                                                     reuse = False, 
                                                     variable_scope = 'word_level_rnn')

        self.sent_seqs.append(sent_seq)

        for i in xrange(1, self.story_nsent+2):
            sent_seq = self.word_level_rnn.BLSTM_encoder(self.inputs_variables[i], 
                                                         self.sent_seq_len[i], 
                                                         reuse = True, 
                                                         variable_scope = 'word_level_rnn')

            self.sent_seqs.append(sent_seq)

    def attention(self):
        self.ends = []
        self.ends.append(tf.reduce_sum(sent_seqs[-2] * tf.expand_dims(self.inputs_masks[-2], -1), axis = 1, name='sum_over'+str(-2)) / tf.reduce_sum(self.inputs_masks[-2], axis = 1, keep_dims = True))
        self.ends.append(tf.reduce_sum(sent_seqs[-1] * tf.expand_dims(self.inputs_masks[-1], -1), axis = 1, name='sum_over'+str(-1)) / tf.reduce_sum(self.inputs_masks[-1], axis = 1, keep_dims = True))
        sent_reps1 = []
        sent_reps2 = []

        for i in xrange(self.story_nsent):
            broad_bilinear_matrix = tf.tile(tf.expand_dims(self.bilinear_att_matrix, 0), tf.stack([self.current_batch_size,1,1]))
            bilinear_part1 = tf.matmul(self.sent_seqs[i], broad_bilinear_matrix)
            weights1 = tf.matmul(bilinear_part1, tf.expand_dims(self.ends[0], -1)) # weight.shape = [batch, seq, 1]
            seq_len_matrix = tf.reduce_sum(self.inputs_masks[i], axis = 1, keep_dims = True)

            sent_reps1.append(tf.reduce_sum(self.sent_seqs[i] * weights1, axis = 1) / seq_len_matrix)

            weights2 = tf.matmul(bilinear_part1, tf.expand_dims(self.ends[1], -1)) # weight.shape = [batch, seq, 1]

            sent_reps2.append(tf.reduce_sum(self.sent_seqs[i] * weights2, axis = 1) / seq_len_matrix)


    def sent_level_encoding(self):
        self.ends_reps = []

        plot1_combine = tf.stack(self.sent_reps1[:4], axis = 1, name = 'make_plot1_seq')
        plot2_combine = tf.stack(self.sent_reps2[:4], axis = 1, name = 'make_plot2_seq')

        self.story_plot1 = self.sent_level_rnn.BLSTM_encoder(plot1_combine, 
                                                            4 * tf.ones(shape = [self.current_batch_size], dtype=tf.int32, name='plot_sent_len'),
                                                            reuse = False,
                                                            variable_scope = 'sent_level_rnn')

        self.story_plot2 = self.sent_level_rnn.BLSTM_encoder(plot2_combine, 
                                                            4 * tf.ones(shape = [self.current_batch_size], dtype=tf.int32, name='plot_sent_len'),
                                                            reuse = True,
                                                            variable_scope = 'sent_level_rnn')      
        
        for i in xrange(2):
            self.ends_reps.append(self.sent_level_rnn.BLSTM_encoder(tf.expand_dims(self.ends[i], 1), 
                                                                    tf.ones( shape = [self.current_batch_size], dtype=tf.int32, name='end_sent_len'+str(i)),
                                                                    reuse = True,
                                                                    variable_scope = 'sent_level_rnn'))




    def build_graph(self, wemb):
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        # with tf.Graph().as_default() as self.g:
        self.inputs_variables = []
        self.inputs_masks = []
        self.sent_seq_len = []

        #Xavier Init Bilinear
        (fan_in, fan_out) = (self.sent_rnn_units[-1], self.sent_rnn_units[-1])
        low = -1*np.sqrt(6.0/(fan_in + fan_out)) # {sigmoid:4, tanh:1} 
        high = 1*np.sqrt(6.0/(fan_in + fan_out))

        self.bilinear_att_matrix = tf.Variable(initial_value=tf.random_uniform([self.sent_rnn_units[-1], self.sent_rnn_units[-1]], 
            minval=low, maxval=high, dtype=tf.float32), trainable=True, name='bilinear_att_matrix', expected_shape=[self.sent_rnn_units[-1], self.sent_rnn_units[-1]])

        """=================Start======================
        | 4 input placeholders for 4 story plot sent  |
        | 2 input placeholders for 2 possible endings |
        ============================================="""
        for i in xrange(self.story_nsent+2):
            self.inputs_variables.append(tf.placeholder(dtype = tf.int32, 
                                                        shape = [None, None], 
                                                        name = 'input'+str(i+1)))

            self.inputs_masks.append(tf.placeholder(dtype = tf.float32, 
                                                    shape = [None, None],
                                                    name = 'mask'+str(i+1)))

            self.sent_seq_len.append(tf.cast(tf.reduce_sum(self.inputs_masks[i], axis = 1), tf.int32))

        self.current_batch_size = tf.placeholder(tf.int32, name = 'batchsize')

        """=======================Word Level=========================
        | go through word level encdoing process for each sentence. |
        =========================================================="""

        self.wemb = tf.Variable(wemb, name = 'wemb', trainable = self.wemb_trainable, dtype = tf.float32)
        self.word_level_rnn = RNN.single_layer_BLSTM_withWemb(wemb = self.wemb,
                                                              hidden_unit = self.word_rnn_units[0], 
                                                              output_style = 'sequence')

        self.sent_seqs = tf_utils.word_level_encoding(word_level_rnn=self.word_level_rnn, sents_num=self.story_nsent+2, 
            inputs_variables=self.inputs_variables, sent_seq_len=self.sent_seq_len)


        """=======================Attention =========================
        | go through word level encdoing process for each sentence. |
        =========================================================="""
        self.ends = []
        self.ends.append(tf.reduce_sum(self.sent_seqs[-2] * tf.expand_dims(self.inputs_masks[-2], -1), axis = 1, name='sum_over'+str(-2)) / tf.reduce_sum(self.inputs_masks[-2], axis = 1, keep_dims = True))
        self.ends.append(tf.reduce_sum(self.sent_seqs[-1] * tf.expand_dims(self.inputs_masks[-1], -1), axis = 1, name='sum_over'+str(-1)) / tf.reduce_sum(self.inputs_masks[-1], axis = 1, keep_dims = True))

        self.sent_reps1, self.sent_reps2 = tf_utils.attention(ends = self.ends, sent_seqs=self.sent_seqs, inputs_masks=self.inputs_masks, 
            current_batch_size=self.current_batch_size, story_nsent=self.story_nsent, 
            bilinear_att_matrix=self.bilinear_att_matrix)

        """======================Sentence Level=======================
        | go through sent level encoding process for all the story in|
        | a minibatch                                                |
        ==========================================================="""
        
        self.sent_level_rnn = RNN.single_layer_BLSTM(hidden_unit = self.sent_rnn_units[0], 
                                                     output_style = 'last')

        self.story_plot1 = tf_utils.sent_level_encoding_plot(sent_level_rnn=self.sent_level_rnn, sent_reps=self.sent_reps1, 
            current_batch_size=self.current_batch_size)

        self.story_plot2 = tf_utils.sent_level_encoding_plot(sent_level_rnn=self.sent_level_rnn, sent_reps=self.sent_reps2,
            current_batch_size=self.current_batch_size, reuse=True)

        self.ends_reps = tf_utils.sent_level_encoding_end(sent_level_rnn=self.sent_level_rnn, end_reps=self.ends,
            current_batch_size=self.current_batch_size)


        """==============cosine similarity hinge max================
        | construct story representation by concatenate story plot | 
        | representation and ending rep                            |
        ==========================================================="""
        self.score1 = tf_utils.batch_cos(self.story_plot1, self.ends_reps[0])
        self.score2 = tf_utils.batch_cos(self.story_plot2, self.ends_reps[1])


        """==============Hinge loss func & prediction===============
        | create hinge loss objective function to be minimized     |
        ========================================================="""
        # add regularization on score DNN
        self.eval_answer = tf.placeholder(tf.int32, name='eval_answer')

        score1_indicator = 2 * self.eval_answer - 1
        score2_indicator = -2 * self.eval_answer + 1

        self.discrit_ls = tf.cast(score1_indicator, tf.float32)* self.score1 + tf.cast(score2_indicator, tf.float32)* self.score2 + self.delta

        self.critic_loss = tf.reduce_sum(tf.cast(tf.greater(self.discrit_ls, 
            tf.zeros_like(self.discrit_ls)), tf.float32)* self.discrit_ls) #+ \
            #self.regularization_factor * self.regularization_loss


        self.abs_eva_diff = tf.abs(tf.cast(tf.argmax(tf.concat([self.score1, self.score2], axis = 1), axis = 1), tf.int32) - self.eval_answer)

        self.eva_correct_count = self.current_batch_size - tf.reduce_sum(self.abs_eva_diff)


        """===================Params Update=======================
        | create hinge loss objective function to be minimized   |
        ======================================================="""          

        self.critic_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='word_level_rnn') + \
                             tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='sent_level_rnn') + \
                             tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='bilinear_att_matrix')


        self.critic_optimizer = tf.train.AdamOptimizer(learning_rate = self.discrit_lr)

        self.critic_optimization = tf_utils.gradient_clip_update(self.critic_optimizer, 
            self.critic_loss, var_list = self.critic_params, global_step = self.global_step, name = "critic_optimization")

        self.critic_summary = tf.summary.scalar("critic_loss", self.critic_loss)

        # merge all summaries into a single "operation" which we can execute in a session 

        """===================Params Update=======================
        | name wrapper for reloading model and training purpose  |
        ======================================================="""

        # name_train_correct_count = tf.identity(self.train_correct_count, name="train_correct_count")
        name_eva_coorect_count = tf.identity(self.eva_correct_count, name="eva_correct_count")
        name_critic_loss = tf.identity(self.critic_loss, name="critic_loss")
        name_score1 = tf.identity(self.score1, name="score_1")
        name_score2 = tf.identity(self.score2, name="score_2")


        
