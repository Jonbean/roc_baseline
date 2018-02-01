import tensorflow as tf
import DNN

def last_step(output, length = None):
    """
    last_step returns picked entries from the output of BLSTM 
    for the use of masking.

    Arguments: 

    output -- (tf.Variable, dtype == tf.float32) the sequence outputs of BLSTM
    length -- (tf.Variable, dtype == tf.int32) the same length vector provided 
              dynamic_rnn sequence_length, if not provided, will return -1 time
              step.

    Return:

    last -- last hidden vector of RNN
    """
    if length == None:
        return output[:,-1,:]
    batch_size = tf.shape(output)[0]
    max_length = tf.shape(output)[1]
    output_size = int(output.get_shape()[2])
    index = tf.range(0, batch_size) * max_length + (length - 1)
    flat = tf.reshape(output, [-1, output_size])
    last = tf.gather(flat, index)
    return last



class single_layer_BLSTM_withWemb(object):
    """docstring for single_layer_BLSTM_withWemb"""
    def __init__(self, wemb, hidden_unit, output_style):
        """
        constructor collects 3 model hyperparameters

        Arguments: 
        wemb -- (tf.Variable, dtype == tf.float32) the word embedding matrix
        hidden_unit -- (int) the hidden unit size of RNN cell
        output_style -- (string) choices: "last", "sequence"

        """
        super(single_layer_BLSTM_withWemb, self).__init__()
        self.wemb = wemb
        self.hidden_unit = hidden_unit
        self.output_style = output_style



    def BLSTM_encoder(self, input_variable, seq_len = None, reuse = False, variable_scope = "BLSTM_encoder"):
        """
        Arguments: 

        input_variable -- (tf.Variable, dtype == tf.int32) the input tensor of LSTM, 
                          input_variable.shape == (batch_size, steps, feature_size)
        seq_len -- (tf.Variable, dtype == tf.int32) the length of each sequence, 
                   len(seq_len) == batch_size
        reuse -- (Optional)(boolean) if the LM is reused
        variable_scope -- (Optional)(string) will raise error if reused is False and LM been called after the first
                          multiple times

        Return:

        output -- (tf.Variable, dtype == tf.float32) the output of the final or sequence of hidden vectors
        """
        with tf.variable_scope(variable_scope) as scope:
            if reuse:
                scope.reuse_variables()        

            input_tensor = tf.nn.embedding_lookup(params=self.wemb, ids = input_variable)

            forward_cell = tf.contrib.rnn.LSTMCell( num_units=self.hidden_unit, 
                                                    use_peepholes=False, 
                                                    cell_clip=None, 
                                                    initializer=tf.contrib.layers.xavier_initializer(dtype = tf.float32), 
                                                    num_proj=None, 
                                                    proj_clip=None, 
                                                    num_unit_shards=None, 
                                                    num_proj_shards=None, 
                                                    forget_bias=1.0, 
                                                    state_is_tuple=True, 
                                                    activation=tf.tanh,
                                                    reuse = reuse)

            backward_cell = tf.contrib.rnn.LSTMCell(num_units=self.hidden_unit, 
                                                    use_peepholes=False, 
                                                    cell_clip=None, 
                                                    initializer=tf.contrib.layers.xavier_initializer(dtype = tf.float32), 
                                                    num_proj=None, 
                                                    proj_clip=None, 
                                                    num_unit_shards=None, 
                                                    num_proj_shards=None, 
                                                    forget_bias=1.0, 
                                                    state_is_tuple=True, 
                                                    activation=tf.tanh, 
                                                    reuse = reuse)


            output, state = tf.nn.bidirectional_dynamic_rnn(cell_fw=forward_cell, 
                                                            cell_bw=backward_cell, 
                                                            inputs=input_tensor, 
                                                            sequence_length=tf.cast(seq_len, tf.int32), 
                                                            initial_state_fw=None, 
                                                            initial_state_bw=None, 
                                                            dtype=tf.float32, 
                                                            parallel_iterations=None, 
                                                            swap_memory=False, 
                                                            time_major=False, 
                                                            scope=scope)


            output_fw, output_bw = output
            # output_fw in shape [batch, maxlen, n_feature]    

            if self.output_style == "last":
                relavent_output_fw = last_step(output_fw, seq_len)
                relavent_output_bw = last_step(output_bw, seq_len)
                
                return relavent_output_fw + relavent_output_bw
            else:
                return output_fw + output_bw


class single_layer_BLSTM(object):
    """
    single_layer_BLSTM is a wrapper class for more specific settings
    of single layer Bidirectional LSTM
    """
    def __init__(self, hidden_unit, output_style):
        """
        constructor collects two model hyperparameters

        Arguments: 
        hidden_unit -- (int) the hidden unit size of RNN cell
        output_style -- (string) choices: "last", "sequence"

        """
        super(single_layer_BLSTM, self).__init__()

        self.hidden_unit = hidden_unit
        self.output_style = output_style


    def BLSTM_encoder(self, input_variable, seq_len, reuse = False, variable_scope = "BLSTM_encoder"):
        """
        Arguments: 

        input_variable -- (tf.Variable, dtype == tf.float32) the input tensor of LSTM, 
                          input_variable.shape == (batch_size, steps, feature_size)
        seq_len -- (tf.Variable, dtype == tf.int32) the length of each sequence, 
                   len(seq_len) == batch_size
        reuse -- (Optional)(boolean) if the LM is reused
        variable_scope -- (Optional)(string) will raise error if reused is False and LM been called after the first
                          multiple times

        Return:

        output -- (tf.Variable, dtype == tf.float32) the output of the final or sequence of hidden vectors
        """
        with tf.variable_scope(variable_scope) as scope:
            if reuse:
                scope.reuse_variables()        
            forward_cell = tf.contrib.rnn.LSTMCell( num_units=self.hidden_unit, 
                                                    use_peepholes=False, 
                                                    cell_clip=None, 
                                                    initializer=tf.contrib.layers.xavier_initializer(dtype = tf.float32), 
                                                    num_proj=None, 
                                                    proj_clip=None, 
                                                    num_unit_shards=None, 
                                                    num_proj_shards=None, 
                                                    forget_bias=1.0, 
                                                    state_is_tuple=True, 
                                                    activation=tf.tanh,
                                                    reuse = reuse)

            backward_cell = tf.contrib.rnn.LSTMCell(num_units=self.hidden_unit, 
                                                    use_peepholes=False, 
                                                    cell_clip=None, 
                                                    initializer=tf.contrib.layers.xavier_initializer(dtype = tf.float32), 
                                                    num_proj=None, 
                                                    proj_clip=None, 
                                                    num_unit_shards=None, 
                                                    num_proj_shards=None, 
                                                    forget_bias=1.0, 
                                                    state_is_tuple=True, 
                                                    activation=tf.tanh,
                                                    reuse = reuse)


            output, state = tf.nn.bidirectional_dynamic_rnn(cell_fw=forward_cell, 
                                                            cell_bw=backward_cell, 
                                                            inputs=input_variable, 
                                                            sequence_length=seq_len, 
                                                            initial_state_fw=None, 
                                                            initial_state_bw=None, 
                                                            dtype=tf.float32, 
                                                            parallel_iterations=None, 
                                                            swap_memory=False, 
                                                            time_major=False, 
                                                            scope=scope)

            output_fw, output_bw = output
            # output_fw in shape [batch, maxlen, n_feature]    

            if self.output_style == "last":
                relavent_output_fw = last_step(output_fw, None)
                relavent_output_bw = last_step(output_bw, None)
                
                return relavent_output_fw + relavent_output_bw
            else:
                return output_fw + output_bw




        