from posixpath import join
import numpy
from numpy.lib.npyio import save
from script.data_iterator import DataIterator
import tensorflow as tf
from script.model import *
import time
import random
import sys
from script.utils import *
from tensorflow.python.framework import ops
from tensorflow.python.client import timeline
import argparse
import os
import json

EMBEDDING_DIM = 18
HIDDEN_SIZE = 18 * 2
ATTENTION_SIZE = 18 * 2
best_auc = 0.0
best_case_acc = 0.0


def prepare_data(input, target, maxlen=None, return_neg=False):
    # x: a list of sentences
    lengths_x = [len(s[4]) for s in input]
    seqs_mid = [inp[3] for inp in input]
    seqs_cat = [inp[4] for inp in input]
    noclk_seqs_mid = [inp[5] for inp in input]
    noclk_seqs_cat = [inp[6] for inp in input]

    if maxlen is not None:
        new_seqs_mid = []
        new_seqs_cat = []
        new_noclk_seqs_mid = []
        new_noclk_seqs_cat = []
        new_lengths_x = []
        for l_x, inp in zip(lengths_x, input):
            if l_x > maxlen:
                new_seqs_mid.append(inp[3][l_x - maxlen:])
                new_seqs_cat.append(inp[4][l_x - maxlen:])
                new_noclk_seqs_mid.append(inp[5][l_x - maxlen:])
                new_noclk_seqs_cat.append(inp[6][l_x - maxlen:])
                new_lengths_x.append(maxlen)
            else:
                new_seqs_mid.append(inp[3])
                new_seqs_cat.append(inp[4])
                new_noclk_seqs_mid.append(inp[5])
                new_noclk_seqs_cat.append(inp[6])
                new_lengths_x.append(l_x)
        lengths_x = new_lengths_x
        seqs_mid = new_seqs_mid
        seqs_cat = new_seqs_cat
        noclk_seqs_mid = new_noclk_seqs_mid
        noclk_seqs_cat = new_noclk_seqs_cat

        if len(lengths_x) < 1:
            return None, None, None, None

    n_samples = len(seqs_mid)
    maxlen_x = numpy.max(lengths_x)
    neg_samples = len(noclk_seqs_mid[0][0])

    mid_his = numpy.zeros((n_samples, maxlen_x)).astype('int64')
    cat_his = numpy.zeros((n_samples, maxlen_x)).astype('int64')
    noclk_mid_his = numpy.zeros(
        (n_samples, maxlen_x, neg_samples)).astype('int64')
    noclk_cat_his = numpy.zeros(
        (n_samples, maxlen_x, neg_samples)).astype('int64')
    mid_mask = numpy.zeros((n_samples, maxlen_x)).astype('float32')
    for idx, [s_x, s_y, no_sx, no_sy] in enumerate(
            zip(seqs_mid, seqs_cat, noclk_seqs_mid, noclk_seqs_cat)):
        mid_mask[idx, :lengths_x[idx]] = 1.
        mid_his[idx, :lengths_x[idx]] = s_x
        cat_his[idx, :lengths_x[idx]] = s_y
        noclk_mid_his[idx, :lengths_x[idx], :] = no_sx
        noclk_cat_his[idx, :lengths_x[idx], :] = no_sy

    uids = numpy.array([inp[0] for inp in input])
    mids = numpy.array([inp[1] for inp in input])
    cats = numpy.array([inp[2] for inp in input])

    if return_neg:
        return uids, mids, cats, mid_his, cat_his, mid_mask, numpy.array(
            target), numpy.array(lengths_x), noclk_mid_his, noclk_cat_his

    else:
        return uids, mids, cats, mid_his, cat_his, mid_mask, numpy.array(
            target), numpy.array(lengths_x)


def eval(sess, test_data, model, model_path):

    loss_sum = 0.
    accuracy_sum = 0.
    aux_loss_sum = 0.
    nums = 0
    stored_arr = []
    for src, tgt in test_data:
        nums += 1
        
        uids, mids, cats, mid_his, cat_his, mid_mask, target, sl, noclk_mids, noclk_cats = prepare_data(
            src, tgt, return_neg=True)



        prob, loss, acc, aux_loss = model.calculate(sess, [
            uids, mids, cats, mid_his, cat_his, mid_mask, target, sl,
            noclk_mids, noclk_cats
        ])
        loss_sum += loss
        aux_loss_sum = aux_loss
        accuracy_sum += acc
        prob_1 = prob[:, 0].tolist()
        target_1 = target[:, 0].tolist()
        for p, t in zip(prob_1, target_1):
            stored_arr.append([p, t])
    test_auc = calc_auc(stored_arr)
    accuracy_sum = accuracy_sum / nums
    loss_sum = loss_sum / nums
    aux_loss_sum / nums
    global best_auc
    global best_case_acc
    if best_auc < test_auc:
        best_auc = test_auc
        best_case_acc = accuracy_sum
        model.save(sess, model_path)
    return test_auc, loss_sum, accuracy_sum, aux_loss_sum


def train(data_location='data',
          output_dir='result',
          batch_size=128,
          maxlen=100,
          steps=0,
          timeline_iter=0,
          test_iter=100,
          save_iter=100,
          seed=2,
          bf16=False,
          ev=False):
    train_file = os.path.join(data_location, "local_train_splitByUser")
    test_file = os.path.join(data_location, "local_test_splitByUser")
    uid_voc = os.path.join(data_location, "uid_voc.pkl")
    mid_voc = os.path.join(data_location, "mid_voc.pkl")
    cat_voc = os.path.join(data_location, "cat_voc.pkl")
    model_type = 'DIEN'
    timestamp = str(int(time.time()))
    model_dir = os.path.join(output_dir, timestamp, "dnn_save_path")
    best_model_dir = os.path.join(output_dir, timestamp, "dnn_best_model")
    model_path = os.path.join(model_dir,
                              "ckpt_noshuff" + model_type + str(seed))
    best_model_path = os.path.join(best_model_dir,
                                   "ckpt_noshuff" + model_type + str(seed))
    if (save_iter > 0 or timeline_iter > 0):
        os.makedirs(model_dir, exist_ok=True)
    if test_iter > 0:
        os.makedirs(best_model_dir, exist_ok=True)

    with tf.Session() as sess:
        train_data = DataIterator(train_file,
                                  uid_voc,
                                  mid_voc,
                                  cat_voc,
                                  batch_size,
                                  maxlen,
                                  data_location=data_location,
                                  shuffle_each_epoch=False)
        test_data = DataIterator(test_file,
                                 uid_voc,
                                 mid_voc,
                                 cat_voc,
                                 batch_size,
                                 maxlen,
                                 data_location=data_location)
        n_uid, n_mid, n_cat = train_data.get_n()
    
        if bf16:
            model = Model_DIN_V2_Gru_Vec_attGru_Neg_bf16(
                n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE,
                ATTENTION_SIZE)
        else:
            model = Model_DIN_V2_Gru_Vec_attGru_Neg(n_uid, n_mid, n_cat,
                                                    EMBEDDING_DIM, HIDDEN_SIZE,
                                                    ATTENTION_SIZE)

        if ev:
            sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
            sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()

        start_time = time.time()
        iter = 1
        lr = 0.001
        for itr in range(3):
            loss_sum = 0.0
            accuracy_sum = 0.
            aux_loss_sum = 0.
            for src, tgt in train_data:
                if (steps > 0 and iter > steps):
                    break

                uids, mids, cats, mid_his, cat_his, mid_mask, target, sl, noclk_mids, noclk_cats = prepare_data(
                    src, tgt, maxlen, return_neg=True)

                save_flag = save_iter > 0 and iter % save_iter == 0
                timeline_flag = timeline_iter > 0 and iter % timeline_iter == 0

                loss, acc, aux_loss, events = model.train(
                    sess, [
                        uids, mids, cats, mid_his, cat_his, mid_mask, target,
                        sl, lr, noclk_mids, noclk_cats
                    ],
                    iter,
                    save_flag,
                    options=(options if timeline_flag else None),
                    run_metadata=(run_metadata if timeline_flag else None))

                loss_sum += loss
                accuracy_sum += acc
                aux_loss_sum += aux_loss
                sys.stdout.flush()

                if (iter % 100 == 0):
                    end_time = time.time()
                    cost_time = end_time - start_time
                    global_step_sec = 100 / cost_time
                    print("global_step/sec: %0.4f" % global_step_sec)
                    print('iter: %d ----> train_loss: %.4f ---- train_accuracy: %.4f ---- tran_aux_loss: %.4f' % \
                                          (iter, loss_sum / 100, accuracy_sum / 100, aux_loss_sum / 100))
                    start_time = time.time()
                if test_iter > 0 and iter % test_iter == 0:
                    print(
                        '                                                                                          test_auc: %.4f ----test_loss: %.4f ---- test_accuracy: %.4f ---- test_aux_loss: %.4f'
                        % eval(sess, test_data, model, best_model_path))
                    loss_sum = 0.0
                    accuracy_sum = 0.0
                    aux_loss_sum = 0.0
                if save_flag:
                    print('save model iter: %d' % (iter))
                    model.save(sess, model_path + "--" + str(iter))
                    model.summary(sess, model_dir, iter, events)
                if timeline_flag:
                    fetched_timeline = timeline.Timeline(
                        run_metadata.step_stats)
                    chrome_trace = fetched_timeline.generate_chrome_trace_format(
                    )
                    print("Save timeline to %s" % model_dir)
                    with open(
                            os.path.join(model_dir, 'timeline-%d.json' % iter),
                            'w') as f:
                        f.write(chrome_trace)

                iter += 1
            lr *= 0.5

        if test_iter > 0:
            print(
                '                                                                                          test_auc: %.4f ----test_loss: %.4f ---- test_accuracy: %.4f ---- test_aux_loss: %.4f'
                % eval(sess, test_data, model, best_model_path))
            global best_auc
            global best_case_acc
            print(
                '==== Best save model ====:\n||| AUC: \t%.4f |||\n||| Accuracy: \t%.4f |||'
                % (best_auc, best_case_acc))
            print('==========================')


def test(data_location='data',
         model_dir=None,
         batch_size=128,
         maxlen=100,
         seed=2,
         bf16=False):
    train_file = os.path.join(data_location, "local_train_splitByUser")
    test_file = os.path.join(data_location, "local_test_splitByUser")
    uid_voc = os.path.join(data_location, "uid_voc.pkl")
    mid_voc = os.path.join(data_location, "mid_voc.pkl")
    cat_voc = os.path.join(data_location, "cat_voc.pkl")
    model_type = 'DIEN'
    model_path = os.path, join(model_dir,
                               "ckpt_noshuff" + model_type + str(seed))

    with tf.Session() as sess:
        train_data = DataIterator(train_file,
                                  uid_voc,
                                  mid_voc,
                                  cat_voc,
                                  batch_size,
                                  maxlen,
                                  data_location=data_location)
        test_data = DataIterator(test_file,
                                 uid_voc,
                                 mid_voc,
                                 cat_voc,
                                 batch_size,
                                 maxlen,
                                 data_location=data_location)
        n_uid, n_mid, n_cat = train_data.get_n()

        if bf16:
            model = Model_DIN_V2_Gru_Vec_attGru_Neg_bf16(
                n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE,
                ATTENTION_SIZE)
        else:
            model = Model_DIN_V2_Gru_Vec_attGru_Neg(n_uid, n_mid, n_cat,
                                                    EMBEDDING_DIM, HIDDEN_SIZE,
                                                    ATTENTION_SIZE)

        model.restore(sess, model_path)
        print(
            'test_auc: %.4f ----test_loss: %.4f ---- test_accuracy: %.4f ---- test_aux_loss: %.4f'
            % eval(sess, test_data, model, model_path))


def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job',
                        help='train or test model',
                        type=str,
                        choices=["train", "test"],
                        default='train')
    parser.add_argument('--seed', help='set random seed', type=int, default=3)
    parser.add_argument('--data_location',
                        help='Full path of train data',
                        required=False,
                        default='./data')
    parser.add_argument('--output_dir',
                        help='Full path to logs & model output directory',
                        required=False,
                        default='./result')
    parser.add_argument('--model_dir',
                        help='Full path to test model directory',
                        required=False)
    parser.add_argument('--save_steps',
                        help='set the number of steps on saving checkpoints',
                        type=int,
                        default=0)
    parser.add_argument('--test_steps',
                        help='set the number of steps on test training model',
                        type=int,
                        default=100)
    parser.add_argument('--steps',
                        help='set the number of steps on train dataset',
                        type=int,
                        default=0)
    parser.add_argument('--timeline',
                        help='number of steps on saving timeline. Default 0',
                        type=int,
                        default=0)
    parser.add_argument('--bf16',
                        help='enable DeepRec BF16 in deep model. Default FP32',
                        action='store_true')
    parser.add_argument('--no_eval',
                        help='not evaluate trained model by eval dataset.',
                        action='store_true')
    return parser


if __name__ == '__main__':
    parser = get_arg_parser()
    args = parser.parse_args()

    SEED = args.seed
    tf.set_random_seed(SEED)
    numpy.random.seed(SEED)
    random.seed(SEED)

    if args.job == 'train':
        train(data_location=args.data_location,
              output_dir=args.output_dir,
              seed=SEED,
              steps=args.steps,
              timeline_iter=args.timeline,
              save_iter=args.save_steps,
              test_iter=0 if args.no_eval else args.test_steps,
              bf16=args.bf16)

    elif args.job == 'test':
        if args.model_dir:
            test(data_location=args.data_location,
                 model_dir=args.model_dir,
                 seed=SEED,
                 bf16=args.bf16)
        else:
            print('Please input correct model path.')
    else:
        print('do nothing...')
