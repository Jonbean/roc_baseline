'''
Author: Jon Tsai    
Created: May 29 2016
'''

import numpy as np 
from time import sleep
import sys
import cPickle as pickle
import os

def progress_bar(percent, speed):
    i = int(percent)/2
    sys.stdout.write('\r')
    # the exact output you're looking for:
    sys.stdout.write("[%-50s] %d%% %f instances/s" % ('='*i, percent, speed))
    sys.stdout.flush()
    


def combine_sents(sent_set):
    '''
    parameter: sent_set ==> 2D sentences set
                        ==> type: list[list[list]]

    return: sents1D ==> 1D sentences set
                    ==> type: list[list]

    This function will combine 2D sentence set 
    into 1D sentence set. 
    e.g.
    [
        [[sent1], [sent2], [sent3], ..., [sentn]]
        ...
        [[sent1], [sent2], [sent3], ..., [sentn]]
    ]
    ==> 
    [
        [sentences1],
        ...
        [sentencesn]
    ]
    '''
    sents1D = []
    sent_num = len(sent_set)
    batch_num = len(sent_set[0])

    for i in xrange(batch_num):
        sent = []
        for j in xrange(sent_num):
            sent += sent_set[j][i]
        sents1D.append(sent)       
    return sents1D

def padding(batch_input_list):
    '''
    ----------
    parameter: 
    ----------
    batch_input_list: type = list(list) 

    ----------
    return: 
    ----------
    numpy.ndarray: shape == (n_batch, max_time_step) 
    '''
    n_batch = len(batch_input_list)
    max_time_step = max([len(batch_input_list[i]) for i in xrange(n_batch)])

    padding_result = np.zeros((n_batch, max_time_step))
    for batch in range(n_batch):
        padding_result[batch] = np.concatenate((np.asarray(batch_input_list[batch]),
                                                np.zeros(max_time_step - len(batch_input_list[batch]))))
    return padding_result.astype('int32')



def mask_generator(indices_matrix):
    '''
    ----------
    parameter: 
    ----------
    indices_matrix: type = list[list] 

    ----------
    return: 
    ----------
    mask : type = np.ndarray
    a mask matrix of a batch of varied length instances
    '''

    n_batch = len(indices_matrix)
    len_ls = [len(sent) for sent in indices_matrix]
    max_len = max(len_ls)
    mask = np.zeros((n_batch, max_len))
    for i in range(n_batch):
        for j in range(len(indices_matrix[i])):
            mask[i][j] = 1 

    return mask

def mlp_mask_generator(indices_matrix, wemb_size):
    '''
    ----------
    parameter: 
    ----------
    indices_matrix: type = list[list] 

    ----------
    return: 
    ----------
    mask : type = np.ndarray
           mask.shape = (n_batch, wemb_size)
    '''

    n_batch = len(indices_matrix)
    len_ls = [len(sent) for sent in indices_matrix]
    
    mask = np.ones((n_batch, wemb_size))
    for i in range(n_batch):
        mask[i] = mask[i] * len_ls[i]

    return mask



class ROC_data(object):
    """docstring for ROC_data"""
    def __init__(self, use_train=True, val_train_num = 1500):
        super(ROC_data, self).__init__()
        
        self.train_set_path = './data/pickles/train_index_corpus.pkl'
        self.val_set_path = './data/pickles/val_index_corpus.pkl'
        self.test_set_path = './data/pickles/test_index_corpus.pkl' 
        self.wemb_matrix_path = './data/pickles/index_wemb_matrix.pkl'
        self.index2word_dict_path = './data/pickles/index2word_dict.pkl'
        self.use_train = use_train
        self.val_train_num = val_train_num

    def loading_data(self):
        '''======Train Set====='''
        if self.use_train:
            with open(self.train_set_path) as f:
                train_set = pickle.load(f)
            self.train_story = train_set[0]
            self.train_ending = train_set[1]
            self.n_train = len(self.train_ending)
            
        '''=====Val Set====='''
        with open(self.val_set_path,'r') as f:
            val_set = pickle.load(f)

        self.val_set = {'story': val_set[0], 
                        'ending1': val_set[1], 
                        'ending2': val_set[2], 
                        'answer': val_set[3],
                        'size': len(val_set[3])}

        '''=====Test Set====='''
        with open(self.test_set_path, 'r') as f:
            test_set = pickle.load(f)
        self.test_set = {'story': test_set[0], 
                        'ending1': test_set[1], 
                        'ending2': test_set[2], 
                        'answer': test_set[3],
                        'size': len(test_set[3])}



        ''''=====Wemb====='''
        with open(self.wemb_matrix_path,'r') as f:
            self.wemb = pickle.load(f)
            self.wemb_size = self.wemb.shape[0]

        '''=====Peeping Preparation====='''
        self.index2word_dict = pickle.load(open(self.index2word_dict_path))

        if not self.use_train:
            valtrain_list = np.random.permutation(self.val_set['size'])

            self.val_train_set = {'story': [val_set[0][i] for i in valtrain_list[:self.val_train_num]], 
                                  'ending1': [val_set[1][i] for i in valtrain_list[:self.val_train_num]], 
                                  'ending2': [val_set[2][i] for i in valtrain_list[:self.val_train_num]], 
                                  'answer': [val_set[3][i] for i in valtrain_list[:self.val_train_num]],
                                  'size': self.val_train_num}
            self.val_val_set = {'story': [val_set[0][i] for i in valtrain_list[self.val_train_num:]], 
                                'ending1': [val_set[1][i] for i in valtrain_list[self.val_train_num:]], 
                                'ending2': [val_set[2][i] for i in valtrain_list[self.val_train_num:]], 
                                'answer': [val_set[3][i] for i in valtrain_list[self.val_train_num:]],
                                'size': self.val_set['size'] - self.val_train_num}
            self.n_train = self.val_train_num
    def debugging(self):
        '''======Train Set====='''
        

        self.train_story = [[np.random.randint(100, size=(np.random.randint(1, 12))) for j in range(50)]for i in range(4)]
        self.train_ending = [np.random.randint(100, size=(np.random.randint(1, 12))) for j in range(50)]
        self.n_train = 50
            
        '''=====Val Set====='''

        eva_size = 120
        self.val_set = {'story': [[np.random.randint(100, size=(np.random.randint(1,12)))for i in range(4)]for j in range(eva_size)], 
                        'ending1': [np.random.randint(100, size=(np.random.randint(1,12)))for j in range(eva_size)], 
                        'ending2': [np.random.randint(100, size=(np.random.randint(1,12)))for j in range(eva_size)], 
                        'answer': np.random.randint(2, size=eva_size),
                        'size': eva_size}

        '''=====Test Set====='''


        self.test_set = {'story': [[np.random.randint(100, size=(np.random.randint(1,12)))for i in range(4)] for j in range(eva_size)], 
                        'ending1': [np.random.randint(100, size=(np.random.randint(1,12)))for j in range(eva_size)], 
                        'ending2': [np.random.randint(100, size=(np.random.randint(1,12)))for j in range(eva_size)], 
                        'answer': np.random.randint(2, size=eva_size),
                        'size': eva_size}


        ''''=====Wemb====='''
        self.wemb = np.random.rand(100, 300)
        self.wemb_size = self.wemb.shape[0]

        # '''=====Peeping Preparation====='''
        # self.index2word_dict = pickle.load(open(self.index2word_dict_path))


def evaluation(sess, eva_set_dict, input_placeholders, eval_values):
    correct = 0.
    n_eva = None
    eva_story = None
    eva_ending1 = None
    eva_ending2 = None
    eva_answer = None
    story_nsent = 4

    minibatch_n = 50
    eva_story = eva_set_dict['story']
    eva_ending1 = eva_set_dict['ending1']
    eva_ending2 = eva_set_dict['ending2']
    eva_answer = eva_set_dict['answer']
    n_eva = eva_set_dict['size']

               
    max_batch_n = n_eva / minibatch_n
    residue = n_eva % minibatch_n
    prediction_answer = []
    for i in range(max_batch_n):

        story_ls = [[eva_story[index][j] for index in range(i*minibatch_n, (i+1)*minibatch_n)] for j in range(story_nsent)]
        story_matrix = [padding(batch_sent) for batch_sent in story_ls]
        story_mask = [mask_generator(batch_sent) for batch_sent in story_ls]

        ending1_ls = [eva_ending1[index] for index in range(i*minibatch_n, (i+1)*minibatch_n)]
        ending1_matrix = padding(ending1_ls)
        ending1_mask = mask_generator(ending1_ls)


        ending2_ls = [eva_ending2[index] for index in range(i*minibatch_n, (i+1)*minibatch_n)]
        ending2_matrix = padding(ending2_ls)
        ending2_mask = mask_generator(ending2_ls)

        batch_answer = eva_answer[i*minibatch_n:(i+1)*minibatch_n]

        feed_dict_seq = [element for element in story_matrix] + [ending1_matrix, ending2_matrix] + \
                        [element for element in story_mask] + [ending1_mask, ending2_mask] + \
                        [batch_answer,minibatch_n, False]

        feed_dictionary = {key:value for key, value in zip(input_placeholders, feed_dict_seq)}
        eva_results = sess.run(eval_values, feed_dict = feed_dictionary)

        correct += eva_results[0]
        if len(eval_values) > 1:
            

            prediction_answer.append(eva_results[1])

    story_ls = [[eva_story[index][j] for index in range(-residue, 0)] for j in range(story_nsent)]
    story_matrix = [padding(batch_sent) for batch_sent in story_ls]
    story_mask = [mask_generator(batch_sent) for batch_sent in story_ls]

    ending1_ls = [eva_ending1[index] for index in range(-residue, 0)]
    ending1_matrix = padding(ending1_ls)
    ending1_mask = mask_generator(ending1_ls)


    ending2_ls = [eva_ending2[index] for index in range(-residue, 0)]
    ending2_matrix = padding(ending2_ls)
    ending2_mask = mask_generator(ending2_ls)

    batch_answer = eva_answer[-residue:]


    feed_dict_seq = [element for element in story_matrix] + [ending1_matrix, ending2_matrix] + \
                    [element for element in story_mask] + [ending1_mask, ending2_mask] + \
                    [batch_answer,residue, False]

    feed_dictionary = {key:value for key, value in zip(input_placeholders, feed_dict_seq)}
    eva_results = sess.run(eval_values, feed_dict = feed_dictionary)

    correct += eva_results[0]
    if len(eval_values) > 1:
        
        prediction_answer.append(eva_results[1])

        return correct/(n_eva), prediction_answer
    elif len(eval_values) == 1:

        return correct/(n_eva)

def evaluation_nomask(sess, eva_set_dict, input_placeholders, eval_values):
    correct = 0.
    n_eva = None
    eva_story = None
    eva_ending1 = None
    eva_ending2 = None
    eva_answer = None
    story_nsent = 4

    minibatch_n = 50
    eva_story = eva_set_dict['story']
    eva_ending1 = eva_set_dict['ending1']
    eva_ending2 = eva_set_dict['ending2']
    eva_answer = eva_set_dict['answer']
    n_eva = eva_set_dict['size']

               
    max_batch_n = n_eva / minibatch_n
    residue = n_eva % minibatch_n
    prediction_answer = []
    for i in range(max_batch_n):

        story_ls = [[eva_story[index][j] for index in range(i*minibatch_n, (i+1)*minibatch_n)] for j in range(story_nsent)]
        story_matrix = [padding(batch_sent) for batch_sent in story_ls]
        # print len(story_matrix)
        # print story_matrix[0].shape
        story_lengths = [[len(sent) for sent in batch_sent] for batch_sent in story_ls]
        ending1_ls = [eva_ending1[index] for index in range(i*minibatch_n, (i+1)*minibatch_n)]
        ending1_matrix = padding(ending1_ls)
        ending1_length = [len(sent) for sent in ending1_ls]

        ending2_ls = [eva_ending2[index] for index in range(i*minibatch_n, (i+1)*minibatch_n)]
        ending2_matrix = padding(ending2_ls)
        ending2_length = [len(sent) for sent in ending2_ls]
        batch_answer = eva_answer[i*minibatch_n:(i+1)*minibatch_n]

        feed_dict_seq = [element for element in story_matrix] + [ending1_matrix, ending2_matrix] + [element for element in story_lengths] + [ending1_length, ending2_length] + [batch_answer,minibatch_n, False]

        feed_dictionary = {key:value for key, value in zip(input_placeholders, feed_dict_seq)}
        eva_results = sess.run(eval_values, feed_dict = feed_dictionary)

        correct += eva_results[0]
        if len(eval_values) > 1:
            

            prediction_answer.append(eva_results[1])

    story_ls = [[eva_story[index][j] for index in range(-residue, 0)] for j in range(story_nsent)]
    story_matrix = [padding(batch_sent) for batch_sent in story_ls]
    story_lengths = [[len(sent) for sent in batch_sent] for batch_sent in story_ls]

    ending1_ls = [eva_ending1[index] for index in range(-residue, 0)]
    ending1_matrix = padding(ending1_ls)
    ending1_length = [len(sent) for sent in ending1_ls]


    ending2_ls = [eva_ending2[index] for index in range(-residue, 0)]
    ending2_matrix = padding(ending2_ls)
    ending2_length = [len(sent) for sent in ending2_ls]

    batch_answer = eva_answer[-residue:]


    feed_dict_seq = [element for element in story_matrix] + [ending1_matrix, ending2_matrix] + [element for element in story_lengths] + [ending1_length, ending2_length] + [batch_answer,residue, False]

    feed_dictionary = {key:value for key, value in zip(input_placeholders, feed_dict_seq)}
    eva_results = sess.run(eval_values, feed_dict = feed_dictionary)

    correct += eva_results[0]
    if len(eval_values) > 1:
        
        prediction_answer.append(eva_results[1])

        return correct/(n_eva), prediction_answer
    elif len(eval_values) == 1:

        return correct/(n_eva)

def evaluation_nomask_mergeplot(sess, eva_set_dict, input_placeholders, eval_values):
    correct = 0.
    n_eva = None
    eva_story = None
    eva_ending1 = None
    eva_ending2 = None
    eva_answer = None
    story_nsent = 4

    minibatch_n = 50
    eva_story = eva_set_dict['story']
    eva_ending1 = eva_set_dict['ending1']
    eva_ending2 = eva_set_dict['ending2']
    eva_answer = eva_set_dict['answer']
    n_eva = eva_set_dict['size']

               
    max_batch_n = n_eva / minibatch_n
    residue = n_eva % minibatch_n
    prediction_answer = []
    for i in range(max_batch_n):

        story_ls = []
        for index in range(i*minibatch_n, (i+1)*minibatch_n):
            story = []
            for j in xrange(story_nsent):
                story += list(eva_story[index][j])
            story_ls.append(story)

        story_matrix = padding(story_ls)
        # print len(story_matrix)
        # print story_matrix[0].shape
        story_lengths = [len(sent) for sent in story_ls]

        ending1_ls = [eva_ending1[index] for index in range(i*minibatch_n, (i+1)*minibatch_n)]

        ending1_matrix = padding(ending1_ls)
        ending1_length = [len(sent) for sent in ending1_ls]

        ending2_ls = [eva_ending2[index] for index in range(i*minibatch_n, (i+1)*minibatch_n)]
        ending2_matrix = padding(ending2_ls)
        ending2_length = [len(sent) for sent in ending2_ls]
        batch_answer = eva_answer[i*minibatch_n:(i+1)*minibatch_n]
        feed_dict_seq = [story_matrix, ending1_matrix, ending2_matrix, story_lengths,ending1_length, ending2_length, batch_answer,minibatch_n, False]

        feed_dictionary = {key:value for key, value in zip(input_placeholders, feed_dict_seq)}
        eva_results = sess.run(eval_values, feed_dict = feed_dictionary)

        correct += eva_results[0]
        if len(eval_values) > 1:
            

            prediction_answer.append(eva_results[1])

    story_ls = []
    for index in range(-residue, 0):
        story = []
        for j in range(story_nsent):
            story += list(eva_story[index][j])
        story_ls.append(story)

    story_matrix = padding(story_ls)

    story_lengths = [len(sent) for sent in story_ls]

    ending1_ls = [eva_ending1[index] for index in range(-residue, 0)]
    ending1_matrix = padding(ending1_ls)
    ending1_length = [len(sent) for sent in ending1_ls]


    ending2_ls = [eva_ending2[index] for index in range(-residue, 0)]
    ending2_matrix = padding(ending2_ls)
    ending2_length = [len(sent) for sent in ending2_ls]

    batch_answer = eva_answer[-residue:]


    feed_dict_seq = [story_matrix, ending1_matrix, ending2_matrix, story_lengths,ending1_length, ending2_length, batch_answer,minibatch_n, False]

    feed_dictionary = {key:value for key, value in zip(input_placeholders, feed_dict_seq)}
    eva_results = sess.run(eval_values, feed_dict = feed_dictionary)

    correct += eva_results[0]
    if len(eval_values) > 1:
        
        prediction_answer.append(eva_results[1])

        return correct/(n_eva), prediction_answer
    elif len(eval_values) == 1:

        return correct/(n_eva)

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def data_incubator(model, N_BATCH, batch, shuffled_index_list, data, story_nsent, single_ending = True):
    if single_ending:

        batch_index_list = [shuffled_index_list[i] for i in range(batch * N_BATCH, (batch+1) * N_BATCH)]
        train_story = [[data.train_story[index][i] for index in batch_index_list] for i in range(1, story_nsent+1)]
        train_ending = [data.train_ending[index] for index in batch_index_list]

        train_story_matrices = [padding(batch_sent) for batch_sent in train_story]
        train_end_matrix = padding(train_ending)

        train_story_mask = [mask_generator(batch_sent) for batch_sent in train_story]
        train_end_mask = mask_generator(train_ending)

        feed_dict_seq = [element for element in train_story_matrices] + [train_end_matrix] + \
                        [element for element in train_story_mask] + [train_end_mask, train_end_mask.shape[0], True]

        feed_dictionary = {key: value for key, value in zip(model.inputs_variables[:-1]+\
                                                            model.inputs_masks[:-1]+\
                                                            [model.current_batch_size, model.train_mode_ph], 
                                                            feed_dict_seq)}
    else:
        batch_index_list = [shuffled_index_list[i] for i in range(batch * N_BATCH, (batch+1) * N_BATCH)]

        train_story = [[data.train_story[index][i] for index in batch_index_list] for i in range(1, story_nsent+1)]

        high_score_end_index = np.random.permutation(batch_index_list)
        train_ending = [data.train_ending[index] for index in batch_index_list]
        fake_index = np.random.randint(low=0, high=N_TRAIN_INS, size=(N_BATCH,))
        while(np.any(fake_index - np.asarray(batch_index_list) == 0)):
            fake_index = np.random.randint(low=0, high=N_TRAIN_INS, size=(N_BATCH,))

        train_ending_false = [data.train_ending[index] for index in fake_index]
        # train_ending2 = [data.train_ending[index] for index in batch_index_list]
        
        train_story_matrices = [padding(batch_sent) for batch_sent in train_story]
        train_end_matrix = padding(train_ending)
        train_end_fake_matrix = padding(train_ending_false)

        train_story_mask = [mask_generator(batch_sent) for batch_sent in train_story]
        train_end_mask = mask_generator(train_ending)
        train_end_fake_mask = mask_generator(train_ending_false)


        feed_dict_seq = [element for element in train_story_matrices] + \
                        [train_end_matrix, train_end_fake_matrix] + \
                        [element for element in train_story_mask] + \
                        [train_end_mask, train_end_fake_mask, train_end_mask.shape[0], True]

        feed_dictionary = {key: value for key, value in zip(model.inputs_variables+\
                                                            model.inputs_masks+\
                                                            [model.current_batch_size, model.train_mode_ph], 
                                                            feed_dict_seq)}
    return feed_dictionary

def decoder_monitor(model, train_story_sets, train_end, merge):
    if merge:
        train_story = []
        for index in xrange(len(train_story_sets)):
            story = []
            for i in xrange(1, story_nsent + 1):
                story += train_story_sets[index][i]
            train_story.append(story)

        train_story_matrix = padding(train_story)
        train_end_matrix = padding(train_ending)

        train_story_lengths = [len(story) for story in train_story]
        train_end_lengths = [len(ending) for ending in train_ending]

        feed_dict_seq = [train_story_matrix, train_end_matrix, train_story_lengths, train_end_lengths, train_story_matrix.shape[0], False]

        feed_dictionary = {key: value for key, value in zip(model.inputs_variables[:-1] + model.sent_seq_len[:-1] + [model.current_batch_size, model.train_mode_ph], feed_dict_seq)}


        eva_results = sess.run(model.test_logits, model.test_GAN_logits, model.fake_end_rep, feed_dict = feed_dictionary)

        test_logits = eva_results[0]
        GAN_logits = eva_results[1]
        fake_end_rep = eva_results[2]
    else:
        print "haven't implemented...will do..."
        # train_story = train_story
    return test_logits, GAN_logits, fake_end_rep

def data_incu_mergePlot(model, N_BATCH, batch, shuffled_index_list, data, story_nsent = None, single_ending = True):

    if single_ending:

        batch_index_list = [shuffled_index_list[i] for i in range(batch * N_BATCH, (batch+1) * N_BATCH)]
        story_nsent = 4
        train_story = []
        for index in batch_index_list:
            story = []
            for i in xrange(1, story_nsent + 1):
                story += data.train_story[index][i]
            train_story.append(story)


        train_ending = [data.train_ending[index] for index in batch_index_list]

        train_story_matrix = padding(train_story)
        train_end_matrix = padding(train_ending)

        train_story_lengths = [len(story) for story in train_story]
        train_end_lengths = [len(ending) for ending in train_ending]

        feed_dict_seq = [train_story_matrix, train_end_matrix, train_story_lengths, train_end_lengths, train_story_matrix.shape[0], True]
        print train_story_matrix.shape
        print train_end_matrix.shape

        print len(train_story_lengths)
        print len(train_end_lengths)

        feed_dictionary = {key: value for key, value in zip(model.inputs_variables[:-1] + model.sent_seq_len[:-1] + [model.current_batch_size, model.train_mode_ph], feed_dict_seq)}
    else:
        print "haven't implemented this sample strategy... sigh..."

    return feed_dictionary

def data_incubator_nomask(model, N_BATCH, batch, shuffled_index_list, data, story_nsent, single_ending = True):
    if single_ending:

        batch_index_list = [shuffled_index_list[i] for i in range(batch * N_BATCH, (batch+1) * N_BATCH)]
        train_story = [[data.train_story[index][i] for index in batch_index_list] for i in range(1, story_nsent+1)]
        train_ending = [data.train_ending[index] for index in batch_index_list]

        train_story_matrices = [padding(batch_sent) for batch_sent in train_story]
        train_end_matrix = padding(train_ending)

        train_story_lengths = [[len(sent) for sent in batch_sent] for batch_sent in train_story]
        train_end_lengths = [len(sent) for sent in train_ending]

        feed_dict_seq = [element for element in train_story_matrices] + [train_end_matrix] + [element for element in train_story_lengths] + [train_end_lengths, train_end_matrix.shape[0], True]

        feed_dictionary = {key: value for key, value in zip(model.inputs_variables[:-1]+\
                                                            model.sent_seq_len[:-1]+\
                                                            [model.current_batch_size, model.train_mode_ph], 
                                                            feed_dict_seq)}
    else:
        batch_index_list = [shuffled_index_list[i] for i in range(batch * N_BATCH, (batch+1) * N_BATCH)]

        train_story = [[data.train_story[index][i] for index in batch_index_list] for i in range(1, story_nsent+1)]

        high_score_end_index = np.random.permutation(batch_index_list)
        train_ending = [data.train_ending[index] for index in batch_index_list]
        fake_index = np.random.randint(low=0, high=N_TRAIN_INS, size=(N_BATCH,))
        while(np.any(fake_index - np.asarray(batch_index_list) == 0)):
            fake_index = np.random.randint(low=0, high=N_TRAIN_INS, size=(N_BATCH,))

        train_ending_false = [data.train_ending[index] for index in fake_index]
        # train_ending2 = [data.train_ending[index] for index in batch_index_list]
        
        train_story_matrices = [padding(batch_sent) for batch_sent in train_story]
        train_end_matrix = padding(train_ending)
        train_end_fake_matrix = padding(train_ending_false)

        train_story_mask = [mask_generator(batch_sent) for batch_sent in train_story]
        train_end_mask = mask_generator(train_ending)
        train_end_fake_mask = mask_generator(train_ending_false)


        feed_dict_seq = [element for element in train_story_matrices] + \
                        [train_end_matrix, train_end_fake_matrix] + \
                        [element for element in train_story_mask] + \
                        [train_end_mask, train_end_fake_mask, train_end_mask.shape[0], True]

        feed_dictionary = {key: value for key, value in zip(model.inputs_variables+\
                                                            model.inputs_masks+\
                                                            [model.current_batch_size, model.train_mode_ph], 
                                                            feed_dict_seq)}
    return feed_dictionary
