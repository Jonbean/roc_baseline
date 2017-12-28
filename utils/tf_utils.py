import tensorflow as tf
def get_weights(scope):
    return [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = scope) if v.name.endswith('kernel:0')]


def gradient_clip_update(optimizer, loss, var_list, global_step, name, no_grad_clip = False, clip_norm = 40.0):

    # re-scale the loss to be the sum of losses of all time steps
    grads = tf.gradients(loss, var_list)
    grad_norm = tf.global_norm(grads)

    if no_grad_clip:
        normed_grads = grads
        clipped_norm = grad_norm
    else:
        # gradient clipping
        normed_grads, _ = tf.clip_by_global_norm(grads, clip_norm, grad_norm)
        clipped_norm = tf.minimum(clip_norm, grad_norm)

    update_op = optimizer.apply_gradients(zip(normed_grads, var_list), global_step, name = name)
    return update_op, grads, grad_norm, clipped_norm

def gradient_clip_update_withgradients(optimizer, loss, var_list, global_step, name, clip_min = -10.0, clip_max = 10.0):
    gvs = optimizer.compute_gradients(loss, var_list)
    capped_gvs = [(tf.clip_by_value(grad, clip_min, clip_max), var) for grad, var in gvs]
    return optimizer.apply_gradients(capped_gvs, global_step = global_step, name = name), gvs


def matrix_DNN(score_func, current_batch_size, feature_size, batch_rep1, batch_rep2, reuse = False, training_mode = False, dropout_rate = 0.0, variable_scope = "DNN_score_func"):
    batch_rep1_broad = batch_rep1 + tf.zeros((current_batch_size, current_batch_size, feature_size), dtype = tf.float32)
    
    batch_rep2_broad = batch_rep2 + tf.zeros((current_batch_size, current_batch_size, feature_size), dtype = tf.float32)        
    batch_rep1_reshape = tf.reshape(tf.transpose(batch_rep1_broad, perm=[1,0,2]), shape=[-1, feature_size])
    batch_rep2_reshape = tf.reshape(batch_rep2_broad, shape = [-1, feature_size])
    
    batch_concate_input = tf.concat([batch_rep1_reshape, batch_rep2_reshape], axis = 1)        
    score_matrix = score_func.build(batch_concate_input, 
        reuse = reuse, variable_scope = variable_scope, dropout_rate=dropout_rate, 
        training_mode = training_mode)
    return tf.reshape(score_matrix, shape = [current_batch_size, current_batch_size])

def dot_matrix_DNN(score_func, current_batch_size, feature_size, batch_rep1, batch_rep2, reuse = False, training_mode = False, dropout_rate = 0.0, variable_scope = "DNN_score_func"):
    batch_rep1_broad = batch_rep1 + tf.zeros((current_batch_size, current_batch_size, feature_size), dtype = tf.float32)
    
    batch_rep2_broad = batch_rep2 + tf.zeros((current_batch_size, current_batch_size, feature_size), dtype = tf.float32)

    batch_rep1_reshape = tf.reshape(tf.transpose(batch_rep1_broad, perm=[1,0,2]), shape=[-1, feature_size])

    batch_rep2_reshape = tf.reshape(batch_rep2_broad, shape = [-1, feature_size])
    
    batch_elementwise_dot = batch_rep1_reshape * batch_rep2_reshape
       
    score_matrix = score_func.build(batch_elementwise_dot, 
        reuse = reuse, variable_scope = variable_scope, dropout_rate=dropout_rate, 
        training_mode = training_mode)
    return tf.reshape(score_matrix, shape = [current_batch_size, current_batch_size])

def feature_matrix_DNN(score_func, current_batch_size, feature_size, batch_rep1, batch_rep2, reuse = False, training_mode = False, dropout_rate = 0.0, variable_scope = "DNN_score_func"):

    batch_rep1_broad = batch_rep1 + tf.zeros((current_batch_size, current_batch_size, feature_size), dtype = tf.float32)
    
    batch_rep2_broad = batch_rep2 + tf.zeros((current_batch_size, current_batch_size, feature_size), dtype = tf.float32)

    batch_rep1_reshape = tf.reshape(tf.transpose(batch_rep1_broad, perm=[1,0,2]), shape=[-1, feature_size])

    batch_rep2_reshape = tf.reshape(batch_rep2_broad, shape = [-1, feature_size])
    
    cos_feature = batch_cos(batch_rep1_reshape, batch_rep2_reshape)

    dot_feature = batch_rep1_reshape * batch_rep2_reshape

    batch_concate_input = tf.concat([cos_feature, dot_feature, batch_rep1_reshape, batch_rep2_reshape], axis = 1)

    score_matrix = score_func.build(batch_concate_input, 
        reuse = reuse, variable_scope = variable_scope, dropout_rate=dropout_rate, 
        training_mode = training_mode)
    return tf.reshape(score_matrix, shape = [current_batch_size, current_batch_size])

def matrix_cos(current_batch_size, feature_size, batch_rep1, batch_rep2):
    batch_rep1_broad = batch_rep1 + tf.zeros((current_batch_size, current_batch_size, feature_size), dtype = tf.float32)
    
    batch_rep2_broad = batch_rep2 + tf.zeros((current_batch_size, current_batch_size, feature_size), dtype = tf.float32)        
    batch_rep1_reshape = tf.reshape(tf.transpose(batch_rep1_broad, perm=[1,0,2]), shape=[-1, feature_size])
    batch_rep2_reshape = tf.reshape(batch_rep2_broad, shape = [-1, feature_size])
    
    batch_rep1_norm = tf.sqrt(tf.reduce_sum(tf.square(batch_rep1_reshape), axis = 1, keep_dims = True))
    batch_rep2_norm = tf.sqrt(tf.reduce_sum(tf.square(batch_rep2_reshape), axis = 1, keep_dims = True))
    prod = tf.reduce_sum(tf.multiply(batch_rep1_reshape, batch_rep2_reshape), axis = 1, keep_dims = True)

    prod_norm = batch_rep1_norm * batch_rep2_norm
    score_matrix = tf.div(prod, prod_norm)
    return tf.reshape(score_matrix, shape = [current_batch_size, current_batch_size])

def get_elements(data, indices):
    # the indices shall be the same 
    indeces = tf.range(0, tf.shape(indices)[0])*data.shape[1] + indices
    return tf.gather(tf.reshape(data, [-1]), indeces)


def batch_cos(batch_rep1, batch_rep2):
    batch_rep1_norm = tf.sqrt(tf.reduce_sum(tf.square(batch_rep1), axis = 1, keep_dims = True))
    batch_rep2_norm = tf.sqrt(tf.reduce_sum(tf.square(batch_rep2), axis = 1, keep_dims = True))
    prod = tf.reduce_sum(tf.multiply(batch_rep1, batch_rep2), axis = 1, keep_dims = True)

    prod_norm = batch_rep1_norm * batch_rep2_norm

    batch_score = tf.div(prod, prod_norm)

    return batch_score

def minibatch_max_score(score_matrix, current_batch_size, k):

    min_score_onrow = tf.reduce_min(score_matrix, axis = 1)
    all_other_score = score_matrix * (tf.ones_like(score_matrix, dtype = tf.float32) - tf.eye(current_batch_size)) - min_score_onrow * tf.eye(current_batch_size)
    top_k_score_matrix, top_k_score_index_matrix = tf.nn.top_k(all_other_score, k=k, sorted=True, name=None)

    return top_k_score_matrix, top_k_score_index_matrix


def word_level_encoding(word_level_rnn, sents_num, inputs_variables, sent_seq_len):
    '''Arguments : 
                    word_level_rnn ==> a RNN object 
                    inputs_variables ==> tf.Variables
                    sent_seq_len ==> tf.Variables
       Return : 
                    sent_seqs ==> list of tf.Variables
       Require:   
                    inputs_variables[i].shape[0] == sent_seq_len[i].shape[0]
                    sent_seq_len should indicate the length of corresponding inputs_variables 
                    on axis = 1
    '''
    sent_seqs = []
    states = []

    sent_seq, state = word_level_rnn.BLSTM_encoder(inputs_variables[0], 
                                            sent_seq_len[0], 
                                            reuse = False, 
                                            variable_scope = 'word_level_rnn')

    sent_seqs.append(sent_seq)
    states.append(state)



    for i in xrange(1, sents_num):
        sent_seq, state = word_level_rnn.BLSTM_encoder(inputs_variables[i], 
                                                sent_seq_len[i], 
                                                reuse = True, 
                                                variable_scope = 'word_level_rnn')

        sent_seqs.append(sent_seq)
        states.append(state)

    return sent_seqs, states


def sent_level_encoding_plot(sent_level_rnn, sent_reps, current_batch_size, reuse=False):
    plot_combine = tf.stack(sent_reps[:4], axis = 1, name = 'make_plot_seq')

    story_plot = sent_level_rnn.BLSTM_encoder(plot_combine,
        4 * tf.ones(shape = [current_batch_size], dtype=tf.int32, name='plot_sent_len'),
        reuse = reuse, variable_scope = 'sent_level_rnn')
    return story_plot

def sent_level_encoding_end(sent_level_rnn, end_reps, current_batch_size):
    ends_reps = []
    for i in xrange(2):
        ends_reps.append(sent_level_rnn.BLSTM_encoder(tf.expand_dims(end_reps[i], 1), 
            tf.ones(shape = [current_batch_size], dtype=tf.int32, name='end_sent_len'+str(i)),
            reuse = True, variable_scope = 'sent_level_rnn'))

    return ends_reps

def attention(ends, sent_seqs, inputs_masks, current_batch_size, story_nsent, bilinear_att_matrix):

    sent_reps1 = []
    sent_reps2 = []


    for i in xrange(story_nsent):
        

        bilinear_part1 = tf.tensordot(ends[0], bilinear_att_matrix, [[-1],[0]])
        bilinear_part2 = tf.tensordot(ends[1], bilinear_att_matrix, [[-1],[0]])

        attention1_score_matrix = tf.reduce_sum(tf.expand_dims(bilinear_part1, 1) * sent_seqs[i], axis = -1)# attention1_score_tensor.shape = [batch, seq]
        attention2_score_matrix = tf.reduce_sum(tf.expand_dims(bilinear_part2, 1) * sent_seqs[i], axis = -1) # attention2_score_tensor.shape = [batch, seq]
        #make sure weights are masked
        numerator1 = inputs_masks[i] * tf.exp(attention1_score_matrix - tf.reduce_max(attention1_score_matrix, axis = 1, keep_dims = True))
        numerator2 = inputs_masks[i] * tf.exp(attention2_score_matrix - tf.reduce_max(attention2_score_matrix, axis = 1, keep_dims = True))

        attention1_weight_matrix = tf.div(numerator1, tf.reduce_sum(numerator1, axis = 1, keep_dims = True))
        attention2_weight_matrix = tf.div(numerator2, tf.reduce_sum(numerator2, axis = 1, keep_dims = True))

        # broad_weight_matrix1 = tf.expand_dims(attention1_weight_matrix, axis = -1) + tf.zeros_like()
        attentioned_sent_seq1 = sent_seqs[i] * (tf.tile(tf.expand_dims(attention1_weight_matrix, -1), multiples = tf.stack([1,1,tf.shape(ends[0])[-1]])))
        attentioned_sent_seq2 = sent_seqs[i] * (tf.tile(tf.expand_dims(attention2_weight_matrix, -1), multiples = tf.stack([1,1,tf.shape(ends[1])[-1]])))
        attentioned_sent_rep1 = tf.div(tf.reduce_sum(attentioned_sent_seq1, axis = 1), tf.reduce_sum(inputs_masks[i], axis = 1, keep_dims = True))
        attentioned_sent_rep2 = tf.div(tf.reduce_sum(attentioned_sent_seq2, axis = 1), tf.reduce_sum(inputs_masks[i], axis = 1, keep_dims = True))
        sent_reps1.append(attentioned_sent_rep1)
        sent_reps2.append(attentioned_sent_rep2)

    return sent_reps1, sent_reps2
