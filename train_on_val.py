from utils import utils
import argparse
import tensorflow as tf 
import HierarchicalBLSTM_Attention_cos_Val
import HierarchicalBLSTM_Attention_DNN_Val
import numpy as np
import time 
import os 

if __name__ == '__main__':
    start_time = time.time()
    #===========================================#
    #                preparation                #
    #===========================================#
    parser = argparse.ArgumentParser()
    parser.add_argument("--word_rnn_setting", type=str, default = '300', help="word level rnn hidden units size")
    parser.add_argument("--sent_rnn_setting", type=str, default = '300', help="sentence level rnn hidden units size")
    parser.add_argument("--batchsize", type=int, default = 50, help="minibatch size")
    parser.add_argument("--reasoning_type", type=str, default = 'concatenate', choices=['concatenate','gothrough'], help="input type of score function")
    parser.add_argument("--dropout_rate", type =float, default = 0.1, help="dropout rate on DNN part")
    parser.add_argument("--wtrainable", type=bool, default = False, help="word embedding matrix trainable option")
    parser.add_argument("--discrit_lr", type=float, default = 0.001, help="discriminator's learning rate")
    parser.add_argument("--delta", type=float, default = 1.0, help="hinge loss margin value")
    parser.add_argument("--rnn_output_mode", type=str, default = 'last', choices=['last', 'sequence'], help="last output or average over sequence of rnn output")
    parser.add_argument("--nonlin_func", type=str, default = 'default', choices=['default','relu', 'sigmoid'], help="nonlinearity function in the neural networks")
    parser.add_argument("--score_func", type=str, default = 'DNN', choices=['DNN','RNN', 'cos'], help="score function for classification")
    parser.add_argument("--loss_type", type=str, default = 'hinge', choices=['hinge','log'], help="hinge loss or log loss")
    parser.add_argument("--score_func_setting", type=str, default = '512x1', help="DNN score function setting")
    parser.add_argument("--drl", type=int, default = 1, help="discrit DNN regularization level")
    parser.add_argument("--top_k", type=int, default = 1, help="hingemax top k score from a minibatch")
    parser.add_argument("--reg_factor", type=str, default='1E-4', help="coefficient of the regularization terms")
    parser.add_argument("--saving", type=str, default='local', choices=['local','gouda'], help="if gouda, then save to /share/data/... path")
    parser.add_argument("--run_index", type=int, default=1, help="label the execution run by index")
    parser.add_argument("--reload", type=int, default=0, help="whether reload model and restart training")
    parser.add_argument("--path", type=str, default='./tf_checkpoints/', help="base directory")
    parser.add_argument("--logdir", type=str, default='./tf_log/',help="log directory")
    parser.add_argument("--model_name", type=str, default='Hier_Att_DNN_Val',help="make checkpoint easier to lookup")
    parser.add_argument("--num_threads", type=int, default=1, help="specify theads number when run on CPU")
    parser.add_argument("--gpu_num", type=int, default=0, help="how many GPUs to use")
    parser.add_argument("--saving_time", type=float, default=3.5, help="hours start saving model")


    args = parser.parse_args()

    print "loading data"
    data = utils.ROC_data(use_train=False, val_train_num = 1500)
    data.loading_data()
    model = None
    if args.model_name == 'Hier_Att_cos_Val':
        model = HierarchicalBLSTM_Attention_cos_Val.Hier_Att_cos_Val(args)
    elif args.model_name == 'Hier_Att_DNN_Val':
        model = HierarchicalBLSTM_Attention_DNN_Val.Hier_Att_DNN_Val(args)

    print "building graph"
    model.build_graph(data.wemb)


    #===========================================#
    #               training part               #
    #===========================================#

    N_EPOCHS = 100
    N_BATCH = args.batchsize
    N_TRAIN_INS = data.n_train
    max_batch = N_TRAIN_INS / N_BATCH
    best_val_accuracy = 0
    best_test_accuracy = 0
    story_nsent = 4
    saver = tf.train.Saver(max_to_keep = 4)

    continue_train = False
    checkpoints_save_path = args.path+args.model_name+'/'+str(args.run_index)+'/'
    
    logdir = args.logdir+args.model_name+'/'+str(args.run_index)+'/'

    utils.ensure_dir(logdir)
    utils.ensure_dir(checkpoints_save_path)

    config = tf.ConfigProto(device_count = {'GPU':args.gpu_num}, intra_op_parallelism_threads=args.num_threads)
    sess = tf.Session(config = config)
    
    
    if not os.path.exists(checkpoints_save_path):
        os.makedirs(checkpoints_save_path)
    else:
        if not os.listdir(checkpoints_save_path) == []:
            continue_train = True

            checkps = [checkp.split('.')[0] for checkp in os.listdir(checkpoints_save_path) if(checkp.endswith('.meta'))]
    
            steps = map(int, [checkp.split('-')[-1] for checkp in checkps])
            latest_step = max(steps)
    
            filename = 'cp-'+str(latest_step)
            saver.restore(sess, checkpoints_save_path + filename)
 
    with sess.as_default():




        #===========================================#
        #         create summary variables          #
        #===========================================#
        if continue_train == False:
            sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

        train_acc_ph = tf.placeholder(tf.float32, name = 'train_accuracy')
        t_acc = tf.summary.scalar('train_accuracy', train_acc_ph)

        val_acc_ph = tf.placeholder(tf.float32, name = 'val_accuracy')
        v_acc = tf.summary.scalar('val_accuracy', val_acc_ph)

        test_acc_ph = tf.placeholder(tf.float32, name = 'test_accuracy')
        te_acc = tf.summary.scalar('test_accuracy', test_acc_ph)

        global_step = model.global_step
        #===========================================#
        #                initial test               #
        #===========================================#

        # test acc of the init state of the model
        print "initial test..."
        val_accuracy = utils.evaluation(sess, 
                                        data.val_val_set, 
                                        model.inputs_variables + \
                                        model.inputs_masks + \
                                        [model.eval_answer, model.current_batch_size, model.train_mode_ph],
                                        [model.eva_correct_count])

        val_summary = sess.run(v_acc, feed_dict = {val_acc_ph: val_accuracy})
        writer.add_summary(val_summary, global_step.eval())

        print "val set accuracy: ", val_accuracy*100, "%"
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy

        test_accuracy = utils.evaluation(sess, 
                                         data.test_set, 
                                         model.inputs_variables + \
                                         model.inputs_masks + \
                                         [model.eval_answer, model.current_batch_size, model.train_mode_ph],
                                         [model.eva_correct_count])

        test_summary = sess.run(te_acc, feed_dict = {test_acc_ph: test_accuracy})
        writer.add_summary(test_summary, global_step.eval())

        print "test set accuracy: ", test_accuracy * 100, "%"
        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy

        #===========================================#
        #               training begin              #
        #===========================================#

        for epoch in xrange(global_step.eval()/ N_BATCH / max_batch, N_EPOCHS):
            print "epoch ", epoch,":"
            shuffled_index_list = np.random.permutation(N_TRAIN_INS)

            max_batch = N_TRAIN_INS/N_BATCH


            total_disc_cost = 0.0
            total_correct_count = 0.0
            train_acc = 0.0


            for batch in xrange(max_batch):

                batch_index_list = [shuffled_index_list[i] for i in range(batch * N_BATCH, (batch+1) * N_BATCH)]
                train_story = [[data.val_train_set['story'][index][i] for index in batch_index_list] for i in range(story_nsent)]
                train_ending1 = [data.val_train_set['ending1'][index] for index in batch_index_list]
                train_ending2 = [data.val_train_set['ending2'][index] for index in batch_index_list]

                train_story_matrices = [utils.padding(batch_sent) for batch_sent in train_story]
                train_end1_matrix = utils.padding(train_ending1)
                train_end2_matrix = utils.padding(train_ending2)

                train_story_mask = [utils.mask_generator(batch_sent) for batch_sent in train_story]
                train_end1_mask = utils.mask_generator(train_ending1)
                train_end2_mask = utils.mask_generator(train_ending2)

                answer = [data.val_train_set['answer'][index] for index in batch_index_list]

                feed_dict_seq = [element for element in train_story_matrices] + \
                                [train_end1_matrix, train_end2_matrix] + \
                                [element for element in train_story_mask] + \
                                [train_end1_mask, train_end2_mask, train_end1_mask.shape[0], answer, True]

                feed_dictionary = {key: value for key, value in zip(model.inputs_variables+\
                                                                    model.inputs_masks+\
                                                                    [model.current_batch_size, model.eval_answer, model.train_mode_ph], 
                                                                    feed_dict_seq)}

                result_list = sess.run([model.critic_loss, 
                                        model.score1, 
                                        model.score2,
                                        model.eva_correct_count,
                                        model.critic_summary,
                                        model.critic_optimization], 
                                        feed_dict = feed_dictionary)


                discrim_cost = result_list[0]
                score1 = result_list[1]
                score2 = result_list[2]
                correct_count = result_list[3]
                summary = result_list[4]



                # print gvs
                writer.add_summary(summary, global_step.eval())

                total_correct_count += correct_count

                total_disc_cost += discrim_cost

                print discrim_cost

            print "======================================="
            print "epoch summary:"
            acc = total_disc_cost / max_batch
            train_acc = total_correct_count / (max_batch * N_BATCH)*100.0

            train_accuracy = sess.run(t_acc, feed_dict = {train_acc_ph: train_acc})
            writer.add_summary(train_accuracy, global_step.eval())

            print "accuracy on training set: ", train_acc, "%"
            print "test on val set..."
            val_result = utils.evaluation(sess, 
                                          data.val_val_set, 
                                          model.inputs_variables + \
                                          model.inputs_masks + \
                                          [model.eval_answer, model.current_batch_size, model.train_mode_ph],
                                          [model.eva_correct_count])

            val_summary = sess.run(v_acc, feed_dict = {val_acc_ph: val_result})

            writer.add_summary(val_summary, global_step.eval())

            print "accuracy is: ", val_result*100, "%"
            if val_result > best_val_accuracy:
                print "new best! test on test set..."
                best_val_accuracy = val_result

                test_accuracy, test_predict_scores = utils.evaluation(sess, 
                                                     data.test_set, 
                                                     model.inputs_variables + \
                                                     model.inputs_masks + \
                                                    [model.eval_answer, model.current_batch_size, model.train_mode_ph],
                                                    [model.eva_correct_count, model.score_pair])

                test_summary = sess.run(te_acc, feed_dict = {test_acc_ph: test_accuracy})

                writer.add_summary(test_summary, global_step.eval())

                print "test set accuracy: ", test_accuracy*100, "%"
                
            print "discriminator cost per instances:", total_disc_cost/(batch+1)

            
            print "======================================="

            end_time = time.time()
            if (end_time - start_time)/3600 > args.saving_time:
                saver.save(sess, checkpoints_save_path+'cp', global_step=global_step)





