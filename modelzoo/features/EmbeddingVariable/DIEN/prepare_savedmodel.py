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
import pickle as pkl

EMBEDDING_DIM = 18
HIDDEN_SIZE = 18 * 2
ATTENTION_SIZE = 18 * 2
best_auc = 0.0
best_case_acc = 0.0
batch_size = 128

def unicode_to_utf8(d):
    return dict((key.encode("UTF-8"), value) for (key, value) in d.items())


def load_dict(filename):
    try:
        with open(filename, 'rb') as f:
            return unicode_to_utf8(json.load(f))
    except:
        with open(filename, 'rb') as f:
            return pkl.load(f)


def main(bf16,n_uid,n_mid,n_cat):

    with tf.Session() as sess1:

        if bf16:
            model = Model_DIN_V2_Gru_Vec_attGru_Neg_bf16(
                n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE,
                ATTENTION_SIZE)
            
            # Initialize saver
            folder_dir = args.checkpoint
            saver = tf.train.Saver()

            sess1.run(tf.global_variables_initializer())
            sess1.run(tf.local_variables_initializer())
            # Restore from checkpoint

            saver.restore(sess1,tf.train.latest_checkpoint(folder_dir))

            # Get save directory
            dir = "./savedmodels"
            os.makedirs(dir,exist_ok=True)
            cc_time = int(time.time())
            saved_path = os.path.join(dir,str(cc_time))
            os.mkdir(saved_path)
            
            
            tf.saved_model.simple_save(
                sess1,
                saved_path,
                inputs = {"mid_his_batch_ph":model.mid_his_batch_ph,"cat_his_batch_ph":model.cat_his_batch_ph,
                        "uid_batch_ph":model.uid_batch_ph,"mid_batch_ph":model.mid_batch_ph,"cat_batch_ph":model.cat_batch_ph,
                        "mask":model.mask,"seq_len_ph":model.seq_len_ph,"target_ph":model.target_ph},
                outputs = {"predict":model.y_hat}
            )

        else:
            model = Model_DIN_V2_Gru_Vec_attGru_Neg(n_uid, n_mid, n_cat,
                                                    EMBEDDING_DIM, HIDDEN_SIZE,
                                                    ATTENTION_SIZE)
            
            # Initialize saver
            folder_dir = args.checkpoint
            saver = tf.train.Saver()

            sess1.run(tf.global_variables_initializer())
            sess1.run(tf.local_variables_initializer())
            # Restore from checkpoint
            saver.restore(sess1,tf.train.latest_checkpoint(folder_dir))
            
            # Get save directory
            dir = "./savedmodels"
            os.makedirs(dir,exist_ok=True)
            cc_time = int(time.time())
            saved_path = os.path.join(dir,str(cc_time))
            os.mkdir(saved_path)
            
            
            tf.saved_model.simple_save(
                sess1,
                saved_path,
                inputs = {"mid_his_batch_ph":model.mid_his_batch_ph,"cat_his_batch_ph":model.cat_his_batch_ph,
                        "uid_batch_ph":model.uid_batch_ph,"mid_batch_ph":model.mid_batch_ph,"cat_batch_ph":model.cat_batch_ph,
                        "mask":model.mask,"seq_len_ph":model.seq_len_ph,"target_ph":model.target_ph,"noclk_mid_batch_ph":model.noclk_mid_batch_ph,
                        "noclk_cat_batch_ph":model.noclk_cat_batch_ph},
                outputs = {"predict":model.y_hat}
            )

        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--checkpoint',
                        help='ckpt path',
                        required=False,
                        default='./data')
    parser.add_argument('--bf16',
                        help='enable DeepRec BF16 in deep model. Default FP32',
                        action='store_true')
    parser.add_argument('--data_location',
                        help='Full path of train data',
                        required=False,
                        default='./data')
    args = parser.parse_args()

    uid_voc = os.path.join(args.data_location, "uid_voc.pkl")
    mid_voc = os.path.join(args.data_location, "mid_voc.pkl")
    cat_voc = os.path.join(args.data_location, "cat_voc.pkl")

    uid_d = load_dict(uid_voc)
    mid_d = load_dict(mid_voc)
    cat_d = load_dict(cat_voc)


    main(args.bf16,len(uid_d),len(mid_d),len(cat_d))

   
