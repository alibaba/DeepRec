from posixpath import join
import numpy
from numpy.lib.npyio import save
from script.data_iterator import DataIterator
import tensorflow as tf
import time
import random
import sys
from script.utils import *
from tensorflow.python.framework import ops
import os
import json

EMBEDDING_DIM = 18
HIDDEN_SIZE = 18 * 2
ATTENTION_SIZE = 18 * 2
best_auc = 0.0
best_case_acc = 0.0
batch_size=1
maxlen=100

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


data_location='data'
test_file = os.path.join(data_location, "local_test_splitByUser")
uid_voc = os.path.join(data_location, "uid_voc.pkl")
mid_voc = os.path.join(data_location, "mid_voc.pkl")
cat_voc = os.path.join(data_location, "cat_voc.pkl")

test_data = DataIterator(test_file,
                        uid_voc,
                        mid_voc,
                        cat_voc,
                        batch_size,
                        maxlen,
                        data_location=data_location)

f = open("./test_data.csv","w")
counter = 0

for src, tgt in test_data:
        uids, mids, cats, mid_his, cat_his, mid_mask, target, sl = prepare_data(src, tgt)
        
        all_data = [uids, mids, cats, mid_his, cat_his, mid_mask, target, sl]

        for cur_data in all_data:
            cur_data = numpy.squeeze(cur_data).reshape(-1)
            
            for col in range(cur_data.shape[0]):
                uid = cur_data[col]
                # print(uid)
                if col == cur_data.shape[0]-1:
                    f.write(str(uid)+",k,")
                    break
                f.write(str(uid)+",")
        
        f.write("\n");
        if counter >= 1:
            break
        counter += 1
            

f.close()
            
     
        


        

