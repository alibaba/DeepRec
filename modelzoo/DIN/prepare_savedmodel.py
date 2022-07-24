# Copyright (c) 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import time
import argparse
import tensorflow as tf
import os
import sys
import math
import collections
from tensorflow.python.client import timeline
import json

from tensorflow.python.ops import partitioned_variables

from tensorflow.contrib.rnn.python.ops.core_rnn_cell import _Linear
from tensorflow.python.feature_column.feature_column import _LazyBuilder
from tensorflow.python.feature_column import utils as fc_utils

# Set to INFO for tracking training, default is WARN. ERROR for least messages
tf.logging.set_verbosity(tf.logging.INFO)
print("Using TensorFlow version %s" % (tf.__version__))

# Definition of some constants
UNSEQ_COLUMNS = ['UID', 'ITEM', 'CATEGORY']
HIS_COLUMNS = ['HISTORY_ITEM', 'HISTORY_CATEGORY']
SEQ_COLUMNS = HIS_COLUMNS
LABEL_COLUMN = ['CLICKED']
TRAIN_DATA_COLUMNS = LABEL_COLUMN + UNSEQ_COLUMNS + SEQ_COLUMNS

EMBEDDING_DIM = 18
HIDDEN_SIZE = 18 * 2
ATTENTION_SIZE = 18 * 2
MAX_SEQ_LENGTH = 50


class DIN():
    def __init__(self,
                 feature_column=None,
                 learning_rate=0.001,
                 embedding_dim=18,
                 hidden_size=36,
                 attention_size=36,
                 cur_bs = 1,
                 inputs=None,
                 optimizer_type='adam',
                 bf16=False,
                 stock_tf=None,
                 emb_fusion=None,
                 ev=None,
                 ev_elimination=None,
                 ev_filter=None,
                 adaptive_emb=None,
                 dynamic_ev=None,
                 ev_opt=None,
                 multihash=None,
                 input_layer_partitioner=None,
                 dense_layer_partitioner=None):
        if not inputs:
            raise ValueError('Dataset is not defined.')
        if not feature_column:
            raise ValueError('Dense column or sparse column is not defined.')
        self._feature = inputs

        self._uid_emb_column = feature_column['uid_emb_column']
        self._item_cate_column = feature_column['item_cate_column']
        self._his_item_cate_column = feature_column['his_item_cate_column']
        self._category_cate_column = feature_column['category_cate_column']
        self._his_category_cate_column = feature_column[
            'his_category_cate_column']

        self.tf = stock_tf
        self.bf16 = False if self.tf else bf16
        self.is_training = True
        self._emb_fusion = emb_fusion
        self._adaptive_emb = adaptive_emb
        self._ev = ev
        self._ev_elimination = ev_elimination
        self._ev_filter = ev_filter
        self._dynamic_ev = dynamic_ev
        self._ev_opt = ev_opt
        self._multihash = multihash
        self.bs = cur_bs
        self._learning_rate = learning_rate
        self._optimizer_type = optimizer_type
        self._input_layer_partitioner = input_layer_partitioner
        self._dense_layer_partitioner = dense_layer_partitioner

        self._embedding_dim = embedding_dim
        self._hidden_size = hidden_size
        self._attention_size = attention_size
        self._data_type = tf.bfloat16 if self.bf16 else tf.float32

        self._create_model()
      

    # used to add summary in tensorboard
    def _add_layer_summary(self, value, tag):
        tf.summary.scalar('%s/fraction_of_zero_values' % tag,
                          tf.nn.zero_fraction(value))
        tf.summary.histogram('%s/activation' % tag, value)

    def _assert_all_equal_and_return(self, tensors, name=None):
        '''Asserts that all tensors are equal and returns the first one.'''
        with tf.name_scope(name, 'assert_all_equal', values=tensors):
            if len(tensors) == 1:
                return tensors[0]
            assert_equal_ops = []
            for t in tensors[1:]:
                assert_equal_ops.append(
                    tf.debugging.assert_equal(tensors[0], t))
            with tf.control_dependencies(assert_equal_ops):
                return tf.identity(tensors[0])

    def _prelu(self, x, scope=''):
        '''parametric ReLU activation'''
        with tf.variable_scope(name_or_scope=scope, default_name='prelu'):
            alpha = tf.get_variable('prelu_' + scope,
                                    shape=x.get_shape()[-1],
                                    dtype=x.dtype,
                                    initializer=tf.constant_initializer(0.1))
            pos = tf.nn.relu(x)
            neg = alpha * (x - abs(x)) * tf.constant(0.5, dtype=x.dtype)
            return pos + neg

    def _attention(self,
                   query,
                   facts,
                   attention_size,
                   mask,
                   stag='null',
                   mode='SUM',
                   softmax_stag=1,
                   time_major=False,
                   return_alphas=False,
                   forCnn=False):
        if isinstance(facts, tuple):
            # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
            facts = tf.concat(facts, 2)
        if len(facts.get_shape().as_list()) == 2:
            facts = tf.expand_dims(facts, 1)

        if time_major:  # (T,B,D) => (B,T,D)
            facts = tf.array_ops.transpose(facts, [1, 0, 2])
        # Trainable parameters
        # mask = tf.equal(mask, tf.ones_like(mask))
        facts_size = facts.get_shape().as_list()[
            -1]  # D value - hidden size of the RNN layer
        querry_size = query.get_shape().as_list()[-1]
        query = tf.layers.dense(query,
                                facts_size,
                                activation=None,
                                name='f1' + stag)
        query = self._prelu(query)
        queries = tf.tile(query, [1, tf.shape(facts)[1]])
        queries = tf.reshape(queries, tf.shape(facts))
        din_all = tf.concat([queries, facts, queries - facts, queries * facts],
                            axis=-1)
        d_layer_1_all = tf.layers.dense(din_all,
                                        80,
                                        activation=tf.nn.sigmoid,
                                        name='f1_att' + stag)
        d_layer_2_all = tf.layers.dense(d_layer_1_all,
                                        40,
                                        activation=tf.nn.sigmoid,
                                        name='f2_att' + stag)
        d_layer_3_all = tf.layers.dense(d_layer_2_all,
                                        1,
                                        activation=None,
                                        name='f3_att' + stag)
        d_layer_3_all = tf.reshape(d_layer_3_all, [-1, 1, tf.shape(facts)[1]])
        scores = d_layer_3_all
        # Mask
        # key_masks = tf.sequence_mask(facts_length, tf.shape(facts)[1])   # [B, T]
        key_masks = tf.expand_dims(mask, 1)  # [B, 1, T]
        paddings = tf.ones_like(scores) * (-2**32 + 1)
        if not forCnn:
            scores = tf.where(key_masks, scores, paddings)  # [B, 1, T]

        # Scale
        # scores = scores / (facts.get_shape().as_list()[-1] ** 0.5)

        # Activation
        if softmax_stag:
            scores = tf.nn.softmax(scores)  # [B, 1, T]

        # Weighted sum
        if mode == 'SUM':
            output = tf.matmul(scores, facts)  # [B, 1, H]
            # output = tf.reshape(output, [-1, tf.shape(facts)[-1]])
        else:
            scores = tf.reshape(scores, [-1, tf.shape(facts)[1]])
            output = facts * tf.expand_dims(scores, -1)
            output = tf.reshape(output, tf.shape(facts))
        if return_alphas:
            return output, scores
        return output

    def _top_fc_layer(self, inputs):
        bn1 = tf.layers.batch_normalization(inputs=inputs, name='bn1')
        dnn1 = tf.layers.dense(bn1, 200, activation=None, name='dnn1')
        dnn1 = tf.nn.relu(dnn1)

        dnn2 = tf.layers.dense(dnn1, 80, activation=None, name='dnn2')
        dnn2 = tf.nn.relu(dnn2)

        dnn3 = tf.layers.dense(dnn2, 2, activation=None, name='dnn3')
        logits = tf.layers.dense(dnn3, 1, activation=None, name='logits')
        self._add_layer_summary(dnn1, 'dnn1')
        self._add_layer_summary(dnn2, 'dnn2')
        self._add_layer_summary(dnn3, 'dnn3')
        return logits

    def _get_embedding_input(self,
                             builder,
                             feature_column,
                             embedding_table,
                             get_seq_len=False):
        sparse_tensors = feature_column._get_sparse_tensors(builder)
        sparse_tensors_ids = sparse_tensors.id_tensor
        sparse_tensors_weights = sparse_tensors.weight_tensor

        if self._emb_fusion and not self.tf:
            from tensorflow.python.ops.embedding_ops import fused_safe_embedding_lookup_sparse
            embedding = fused_safe_embedding_lookup_sparse(
                embedding_weights=embedding_table,
                sparse_ids=sparse_tensors_ids,
                sparse_weights=sparse_tensors_weights)
        else:
            embedding = tf.nn.safe_embedding_lookup_sparse(
                embedding_weights=embedding_table,
                sparse_ids=sparse_tensors_ids,
                sparse_weights=sparse_tensors_weights)

        if get_seq_len:
            sequence_length = fc_utils.sequence_length_from_sparse_tensor(
                sparse_tensors_ids)
            return embedding, sequence_length
        else:
            return embedding

    def _embedding_input_layer(self):
        for key in SEQ_COLUMNS:
            self._feature[key] = tf.strings.split(self._feature[key], '')
            self._feature[key] = tf.sparse.slice(
                self._feature[key], [0, 0], [self.bs, MAX_SEQ_LENGTH])

        # get uid embeddings
        if self._adaptive_emb and not self.tf:
            '''Adaptive Embedding Feature Part 1 of 2'''
            adaptive_mask_tensors = {
                'UID': tf.ones([args.batch_size], tf.int32)
            }
            uid_emb = tf.feature_column.input_layer(
                self._feature,
                self._uid_emb_column,
                adaptive_mask_tensors=adaptive_mask_tensors)
        else:
            uid_emb = tf.feature_column.input_layer(self._feature,
                                                    self._uid_emb_column)
        # get embeddings of item and category
        # create embedding table
        if self._ev and not self.tf:
            '''Embedding Variable Feature with get embedding variable API'''
            item_embedding_var = tf.get_embedding_variable(
                'item_embedding_var',
                self._embedding_dim,
                ev_option=self._ev_opt)
            category_embedding_var = tf.get_embedding_variable(
                'category_embedding_var',
                self._embedding_dim,
                ev_option=self._ev_opt)
        elif self._multihash and not self.tf:
            '''Multi-Hash Variable'''
            item_embedding_var = tf.get_multihash_variable(
                'item_embedding_var',
                [[
                    int(self._item_cate_column._num_buckets**0.5),
                    self._embedding_dim
                ],
                 [
                     int(self._item_cate_column._num_buckets /
                         int(self._item_cate_column._num_buckets**0.5)),
                     self._embedding_dim
                 ]])
            category_embedding_var = tf.get_multihash_variable(
                'category_embedding_var',
                [[
                    int(self._category_cate_column._num_buckets**0.5),
                    self._embedding_dim
                ],
                 [
                     int(self._category_cate_column._num_buckets /
                         int(self._category_cate_column._num_buckets**0.5)),
                     self._embedding_dim
                 ]])
        else:
            item_embedding_var = tf.get_variable(
                'item_embedding_var',
                [self._item_cate_column._num_buckets, self._embedding_dim])
            category_embedding_var = tf.get_variable(
                'category_embedding_var',
                [self._category_cate_column._num_buckets, self._embedding_dim])

        builder = _LazyBuilder(self._feature)
        # get item embedding concat [item_id,category]
        item_embedding = self._get_embedding_input(builder,
                                                   self._item_cate_column,
                                                   item_embedding_var)
        category_embedding = self._get_embedding_input(
            builder, self._category_cate_column, category_embedding_var)
        item_emb = tf.concat([item_embedding, category_embedding], 1)

        # get history item embedding concat [history_item_id,history_category] and sequence length
        his_item_embedding, his_item_sequence_length = self._get_embedding_input(
            builder,
            self._his_item_cate_column,
            item_embedding_var,
            get_seq_len=True)
        his_category_embedding, his_category_sequence_length = self._get_embedding_input(
            builder,
            self._his_category_cate_column,
            category_embedding_var,
            get_seq_len=True)
        sequence_lengths = [
            his_item_sequence_length, his_category_sequence_length
        ]
        his_item_emb = tf.concat([his_item_embedding, his_category_embedding],
                                 2)
        sequence_length = self._assert_all_equal_and_return(sequence_lengths)

        return uid_emb, item_emb, his_item_emb, sequence_length

    # create model
    def _create_model(self):
        # input layer to get embedding of features
        with tf.variable_scope('input_layer',
                               partitioner=self._input_layer_partitioner,
                               reuse=tf.AUTO_REUSE):
            uid_emb, item_emb, his_item_emb, sequence_length = self._embedding_input_layer(
            )

            if self.bf16:
                uid_emb = tf.cast(uid_emb, tf.bfloat16)
                item_emb = tf.cast(item_emb, tf.bfloat16)
                his_item_emb = tf.cast(his_item_emb, tf.bfloat16)

            item_his_eb_sum = tf.reduce_sum(his_item_emb, 1)
            mask = tf.sequence_mask(sequence_length)

        # Attention layer
        attention_scope = tf.variable_scope('attention_layer')
        with attention_scope.keep_weights(dtype=tf.float32) if self.bf16 \
            else attention_scope:
            attention_output = self._attention(item_emb, his_item_emb,
                                               self._attention_size, mask)
            att_fea = tf.reduce_sum(attention_output, 1)
            tf.summary.histogram('alpha_outputs', att_fea)

        top_input = tf.concat([
            uid_emb, item_emb, item_his_eb_sum, item_emb * item_his_eb_sum,
            att_fea
        ], 1)

        # Top MLP layer
        top_mlp_scope = tf.variable_scope(
            'top_mlp_layer',
            partitioner=self._dense_layer_partitioner,
            reuse=tf.AUTO_REUSE)
        with top_mlp_scope.keep_weights(dtype=tf.float32) if self.bf16 \
            else top_mlp_scope:
            self._logits = self._top_fc_layer(top_input)
        if self.bf16:
            self._logits = tf.cast(self._logits, dtype=tf.float32)

        self.probability = tf.math.sigmoid(self._logits)
        self.output = tf.round(self.probability)

   



# generate feature columns
def build_feature_columns(data_location=None):
    # uid_file
    uid_file = os.path.join(data_location, 'uid_voc.txt')
    mid_file = os.path.join(data_location, 'mid_voc.txt')
    cat_file = os.path.join(data_location, 'cat_voc.txt')
    if (not os.path.exists(uid_file)) or (not os.path.exists(mid_file)) or (
            not os.path.exists(cat_file)):
        print(
            "uid_voc.txt, mid_voc.txt or cat_voc does not exist in data file.")
        sys.exit()
    # uid
    uid_cate_column = tf.feature_column.categorical_column_with_vocabulary_file(
        'UID', uid_file, default_value=0)
    ev_opt = None
    if not args.tf:
        '''Feature Elimination of EmbeddingVariable Feature'''
        if args.ev_elimination == 'gstep':
            # Feature elimination based on global steps
            evict_opt = tf.GlobalStepEvict(steps_to_live=4000)
        elif args.ev_elimination == 'l2':
            # Feature elimination based on l2 weight
            evict_opt = tf.L2WeightEvict(l2_weight_threshold=1.0)
        else:
            evict_opt = None
        '''Feature Filter of EmbeddingVariable Feature'''
        if args.ev_filter == 'cbf':
            # CBF-based feature filter
            filter_option = tf.CBFFilter(filter_freq=3,
                                         max_element_size=2**30,
                                         false_positive_probability=0.01,
                                         counter_type=tf.int64)
        elif args.ev_filter == 'counter':
            # Counter-based feature filter
            filter_option = tf.CounterFilter(filter_freq=3)
        else:
            filter_option = None
        ev_opt = tf.EmbeddingVariableOption(evict_option=evict_opt,
                                            filter_option=filter_option)

        if args.ev:
            '''Embedding Variable Feature with feature_column API'''
            uid_cate_column = tf.feature_column.categorical_column_with_embedding(
                'UID', dtype=tf.string, ev_option=ev_opt)
        elif args.adaptive_emb:
            '''            Adaptive Embedding Feature Part 2 of 2
            Expcet the follow code, a dict, 'adaptive_mask_tensors', is need as the input of 
            'tf.feature_column.input_layer(adaptive_mask_tensors=adaptive_mask_tensors)'.
            For column 'COL_NAME',the value of adaptive_mask_tensors['$COL_NAME'] is a int32
            tensor with shape [batch_size].
            '''
            uid_cate_column = tf.feature_column.categorical_column_with_adaptive_embedding(
                'UID',
                hash_bucket_size=100000,
                dtype=tf.string,
                ev_option=ev_opt)
        elif args.dynamic_ev:
            '''Dynamic-dimension Embedding Variable'''
            print(
                "Dynamic-dimension Embedding Variable isn't really enabled in model now."
            )
            sys.exit()

    uid_emb_column = tf.feature_column.embedding_column(
        uid_cate_column, dimension=EMBEDDING_DIM)

    # item
    item_cate_column = tf.feature_column.categorical_column_with_vocabulary_file(
        'ITEM', mid_file, default_value=0)
    category_cate_column = tf.feature_column.categorical_column_with_vocabulary_file(
        'CATEGORY', cat_file, default_value=0)

    # history behavior
    his_item_cate_column = tf.feature_column.sequence_categorical_column_with_vocabulary_file(
        'HISTORY_ITEM', mid_file, default_value=0)
    his_category_cate_column = tf.feature_column.sequence_categorical_column_with_vocabulary_file(
        'HISTORY_CATEGORY', cat_file, default_value=0)

    return {
        'uid_emb_column': uid_emb_column,
        'item_cate_column': item_cate_column,
        'category_cate_column': category_cate_column,
        'his_item_cate_column': his_item_cate_column,
        'his_category_cate_column': his_category_cate_column
    }, ev_opt


def main(tf_config=None, server=None):
   
    # set batch size, eporch & steps
    batch_size = math.ceil(
        args.batch_size / args.micro_batch
    ) if args.micro_batch and not args.tf else args.batch_size

   
    # set fixed random seed
    tf.set_random_seed(args.seed)

    
    # create feature column
    feature_column, ev_opt = build_feature_columns(args.data_location)

    # create variable partitioner for distributed training
    num_ps_replicas = len(tf_config['ps_hosts']) if tf_config else 0
    input_layer_partitioner = partitioned_variables.min_max_variable_partitioner(
        max_partitions=num_ps_replicas,
        min_slice_size=args.input_layer_partitioner <<
        20) if args.input_layer_partitioner else None
    dense_layer_partitioner = partitioned_variables.min_max_variable_partitioner(
        max_partitions=num_ps_replicas,
        min_slice_size=args.dense_layer_partitioner <<
        10) if args.dense_layer_partitioner else None

    # Session config
    sess_config = tf.ConfigProto()
    sess_config.inter_op_parallelism_threads = args.inter
    sess_config.intra_op_parallelism_threads = args.intra

    
    final_input = {}
    all_fea = UNSEQ_COLUMNS + SEQ_COLUMNS
    for col in all_fea:
        final_input[col] =  tf.placeholder(tf.string,[None], name=col)

    # create model
    model = DIN(feature_column=feature_column,
                learning_rate=args.learning_rate,
                embedding_dim=EMBEDDING_DIM,
                hidden_size=HIDDEN_SIZE,
                attention_size=ATTENTION_SIZE,
                optimizer_type=args.optimizer,
                bf16=args.bf16,
                stock_tf=args.tf,
                emb_fusion=args.emb_fusion,
                ev=args.ev,
                ev_elimination=args.ev_elimination,
                ev_filter=args.ev_filter,
                adaptive_emb=args.adaptive_emb,
                dynamic_ev=args.dynamic_ev,
                ev_opt=ev_opt,
                cur_bs = batch_size,
                inputs=final_input,
                multihash=args.multihash,
                input_layer_partitioner=input_layer_partitioner,
                dense_layer_partitioner=dense_layer_partitioner)


    with tf.Session() as sess1:
        
        # Initialize saver
        folder_dir = args.checkpoint
        saver = tf.train.Saver()

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
            inputs = model._feature,
            outputs = {"predict":model.output}
        )

   


def boolean_string(string):
    low_string = string.lower()
    if low_string not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return low_string == 'true'


# Get parse
def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_location',
                        help='Full path of train data',
                        required=False,
                        default='./data')
    parser.add_argument('--steps',
                        help='set the number of steps on train dataset',
                        type=int,
                        default=0)
    parser.add_argument('--batch_size',
                        help='Batch size to train. Default is 512',
                        type=int,
                        default=512)
    parser.add_argument('--output_dir',
                        help='Full path to model output directory. \
                            Default to ./result. Covered by --checkpoint. ',
                        required=False,
                        default='./result')
    parser.add_argument('--checkpoint',
                        help='Full path to checkpoints input/output. \
                            Default to ./result/$MODEL_TIMESTAMP',
                        required=False)
    parser.add_argument('--save_steps',
                        help='set the number of steps on saving checkpoints',
                        type=int,
                        default=0)
    parser.add_argument('--seed',
                        help='set the random seed for tensorflow',
                        type=int,
                        default=2021)
    parser.add_argument('--optimizer',
                        type=str, \
                        choices=['adam', 'adamasync', 'adagraddecay'],
                        default='adamasync')
    parser.add_argument('--learning_rate',
                        help='Learning rate for model',
                        type=float,
                        default=0.001)
    parser.add_argument('--keep_checkpoint_max',
                        help='Maximum number of recent checkpoint to keep',
                        type=int,
                        default=1)
    parser.add_argument('--timeline',
                        help='number of steps on saving timeline. Default 0',
                        type=int,
                        default=0)
    parser.add_argument('--protocol',
                        type=str,
                        choices=['grpc', 'grpc++', 'star_server'],
                        default='grpc')
    parser.add_argument('--inter',
                        help='set inter op parallelism threads.',
                        type=int,
                        default=0)
    parser.add_argument('--intra',
                        help='set inter op parallelism threads.',
                        type=int,
                        default=0)
    parser.add_argument('--input_layer_partitioner', \
                        help='slice size of input layer partitioner, units MB. Default 8MB',
                        type=int,
                        default=8)
    parser.add_argument('--dense_layer_partitioner', \
                        help='slice size of dense layer partitioner, units KB. Default 16KB',
                        type=int,
                        default=16)
    parser.add_argument('--bf16',
                        help='enable DeepRec BF16 in deep model. Default FP32',
                        action='store_true')
    parser.add_argument('--no_eval',
                        help='not evaluate trained model by eval dataset.',
                        action='store_true')
    parser.add_argument('--tf', \
                        help='Use TF 1.15.5 API and disable DeepRec feature to run a baseline.',
                        action='store_true')
    parser.add_argument('--smartstaged', \
                        help='Whether to enable smart staged feature of DeepRec, Default to True.',
                        type=boolean_string,
                        default=True)
    parser.add_argument('--emb_fusion', \
                        help='Whether to enable embedding fusion, Default to True.',
                        type=boolean_string,
                        default=True)
    parser.add_argument('--ev', \
                        help='Whether to enable DeepRec EmbeddingVariable. Default False.',
                        type=boolean_string,
                        default=False)
    parser.add_argument('--ev_elimination', \
                        help='Feature Elimination of EmbeddingVariable Feature. Default closed.',
                        type=str,
                        choices=[None, 'l2', 'gstep'],
                        default=None)
    parser.add_argument('--ev_filter', \
                        help='Feature Filter of EmbeddingVariable Feature. Default closed.',
                        type=str,
                        choices=[None, 'counter', 'cbf'],
                        default=None)
    parser.add_argument('--op_fusion', \
                        help='Whether to enable Auto graph fusion feature. Default to True',
                        type=boolean_string,
                        default=True)
    parser.add_argument('--micro_batch',
                        help='Set num for Auto Mirco Batch. Default close.',
                        type=int,
                        default=0)  #TODO: Defautl to True
    parser.add_argument('--adaptive_emb', \
                        help='Whether to enable Adaptive Embedding. Default to False.',
                        type=boolean_string,
                        default=False)
    parser.add_argument('--dynamic_ev', \
                        help='Whether to enable Dynamic-dimension Embedding Variable. Default to False.',
                        type=boolean_string,
                        default=False)#TODO
    parser.add_argument('--incremental_ckpt', \
                        help='Set time of save Incremental Checkpoint. Default 0 to close.',
                        type=int,
                        default=0)
    parser.add_argument('--workqueue', \
                        help='Whether to enable Work Queue. Default to False.',
                        type=boolean_string,
                        default=False)
    parser.add_argument('--multihash', \
                        help='Whether to enable Multi-Hash Variable. Default to False.',
                        type=boolean_string,
                        default=False)#TODO
    return parser


# Parse distributed training configuration and generate cluster information
def generate_cluster_info(TF_CONFIG):
    print(TF_CONFIG)
    tf_config = json.loads(TF_CONFIG)
    cluster_config = tf_config.get('cluster')
    ps_hosts = []
    worker_hosts = []
    chief_hosts = []
    for key, value in cluster_config.items():
        if 'ps' == key:
            ps_hosts = value
        elif 'worker' == key:
            worker_hosts = value
        elif 'chief' == key:
            chief_hosts = value
    if chief_hosts:
        worker_hosts = chief_hosts + worker_hosts

    if not ps_hosts or not worker_hosts:
        print('TF_CONFIG ERROR')
        sys.exit()
    task_config = tf_config.get('task')
    task_type = task_config.get('type')
    task_index = task_config.get('index') + (1 if task_type == 'worker'
                                             and chief_hosts else 0)

    if task_type == 'chief':
        task_type = 'worker'

    is_chief = True if task_index == 0 else False
    cluster = tf.train.ClusterSpec({'ps': ps_hosts, 'worker': worker_hosts})
    server = tf.distribute.Server(cluster,
                                  job_name=task_type,
                                  task_index=task_index,
                                  protocol=args.protocol)
    if task_type == 'ps':
        server.join()
    elif task_type == 'worker':
        tf_config = {
            'ps_hosts': ps_hosts,
            'worker_hosts': worker_hosts,
            'type': task_type,
            'index': task_index,
            'is_chief': is_chief
        }
        tf_device = tf.device(
            tf.train.replica_device_setter(
                worker_device='/job:worker/task:%d' % task_index,
                cluster=cluster))
        return tf_config, server, tf_device
    else:
        print("Task type or index error.")
        sys.exit()


# Some DeepRec's features are enabled by ENV.
# This func is used to set ENV and enable these features.
# A triple quotes comment is used to introduce these features and play an emphasizing role.
def set_env_for_DeepRec():
    '''
    Set some ENV for these DeepRec's features enabled by ENV. 
    More Detail information is shown in https://deeprec.readthedocs.io/zh/latest/index.html.
    START_STATISTIC_STEP & STOP_STATISTIC_STEP: On CPU platform, DeepRec supports memory optimization
        in both stand-alone and distributed trainging. It's default to open, and the 
        default start and stop steps of collection is 1000 and 1100. Reduce the initial 
        cold start time by the following settings.
    MALLOC_CONF: On CPU platform, DeepRec can use memory optimization with the jemalloc library.
        Please preload libjemalloc.so by `LD_PRELOAD=./libjemalloc.so.2 python ...`
    '''
    os.environ['START_STATISTIC_STEP'] = '100'
    os.environ['STOP_STATISTIC_STEP'] = '110'
    os.environ['MALLOC_CONF']= \
        'background_thread:true,metadata_thp:auto,dirty_decay_ms:20000,muzzy_decay_ms:20000'


if __name__ == '__main__':
    parser = get_arg_parser()
    args = parser.parse_args()

    if not args.tf:
        set_env_for_DeepRec()

    TF_CONFIG = os.getenv('TF_CONFIG')
    if not TF_CONFIG:
        main()
    else:
        tf_config, server, tf_device = generate_cluster_info(TF_CONFIG)
        main(tf_config, server)
