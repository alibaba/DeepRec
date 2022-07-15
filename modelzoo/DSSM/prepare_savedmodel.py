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

# Set to INFO for tracking training, default is WARN. ERROR for least messages
tf.logging.set_verbosity(tf.logging.INFO)
print("Using TensorFlow version %s" % (tf.__version__))

# Definition of some constants
USER_COLUMN = [
    'user_id', 'cms_segid', 'cms_group_id', 'age_level', 'pvalue_level',
    'shopping_level', 'occupation', 'new_user_class_level',
    'tag_category_list', 'tag_brand_list'
]
ITEM_COLUMN = [
    'adgroup_id', 'cate_id', 'campaign_id', 'customer', 'brand', 'price', 'pid'
]
INPUT_FEATURES = [
    'pid', 'adgroup_id', 'cate_id', 'campaign_id', 'customer', 'brand',
    'user_id', 'cms_segid', 'cms_group_id', 'final_gender_code', 'age_level',
    'pvalue_level', 'shopping_level', 'occupation', 'new_user_class_level',
    'tag_category_list', 'tag_brand_list', 'price'
]
LABEL_COLUMN = ['clk']
BUY_COLUMN = ['buy']
TAG_COLUMN = ['tag_category_list', 'tag_brand_list']
INPUT_COLUMN = LABEL_COLUMN + BUY_COLUMN + INPUT_FEATURES
EMBEDDING_DIM = 16
HASH_BUCKET_SIZES = {
    'pid': 10,
    'adgroup_id': 100000,
    'cate_id': 10000,
    'campaign_id': 100000,
    'customer': 100000,
    'brand': 100000,
    'user_id': 100000,
    'cms_segid': 100,
    'cms_group_id': 100,
    'final_gender_code': 10,
    'age_level': 10,
    'pvalue_level': 10,
    'shopping_level': 10,
    'occupation': 10,
    'new_user_class_level': 10,
    'tag_category_list': 100000,
    'tag_brand_list': 100000,
    'price': 50
}


class DSSM():
    def __init__(self,
                 user_column=None,
                 item_column=None,
                 dnn_hidden_units=[256, 128, 64, 32],
                 optimizer_type='adam',
                 batch_size=0,
                 learning_rate=0.001,
                 use_bn=True,
                 inputs=None,
                 bf16=False,
                 stock_tf=None,
                 adaptive_emb=False,
                 input_layer_partitioner=None,
                 dense_layer_partitioner=None):
        if not inputs:
            raise ValueError("Dataset is not defined.")
        self._feature = inputs

        self._user_column = user_column
        self._item_column = item_column
        if not user_column or not item_column:
            raise ValueError('User column or item column is not defined.')

        self.tf = stock_tf
        self.bf16 = False if self.tf else bf16
        self.is_training = True
        self._adaptive_emb = adaptive_emb

        self._dnn_hidden_units = dnn_hidden_units
        self._dnn_last_hidden_units = self._dnn_hidden_units.pop()
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._use_bn = use_bn
        self._optimizer_type = optimizer_type
        self._input_layer_partitioner = input_layer_partitioner
        self._dense_layer_partitioner = dense_layer_partitioner

        self._create_model()
      

    # used to add summary in tensorboard
    def _add_layer_summary(self, value, tag):
        tf.summary.scalar('%s/fraction_of_zero_values' % tag,
                          tf.nn.zero_fraction(value))
        tf.summary.histogram('%s/activation' % tag, value)

    def _dnn_tower(self, net, name):
        with tf.variable_scope(name + '_layer',
                               partitioner=self._dense_layer_partitioner,
                               reuse=tf.AUTO_REUSE):
            for layer_id, num_hidden_units in enumerate(
                    self._dnn_hidden_units):
                with tf.variable_scope(name + '_%d' % layer_id,
                                       reuse=tf.AUTO_REUSE) as dnn_layer_scope:
                    net = tf.layers.dense(net,
                                          units=num_hidden_units,
                                          activation=None)
                    # BN
                    if self._use_bn:
                        net = tf.layers.batch_normalization(
                            net, training=self.is_training, trainable=True)
                    # activate func
                    net = tf.nn.relu(net)
                    self._add_layer_summary(net, dnn_layer_scope.name)

            # last dnn layer
            with tf.variable_scope(name + '_%d' % len(self._dnn_hidden_units),
                                   reuse=tf.AUTO_REUSE) as dnn_layer_scope:
                net = tf.layers.dense(net, units=self._dnn_last_hidden_units)
                self._add_layer_summary(net, dnn_layer_scope.name)

        return net

    # create model
    def _create_model(self):
        # input embeddings of user & item features
        with tf.variable_scope('input_layer',
                               partitioner=self._input_layer_partitioner,
                               reuse=tf.AUTO_REUSE):
            if self._adaptive_emb and not self.tf:
                '''Adaptive Embedding Feature Part 1 of 2'''
                adaptive_mask_tensors = {}
                for col in INPUT_FEATURES:
                    adaptive_mask_tensors[col] = tf.ones([self._batch_size],
                                                         tf.int32)
                user_emb = tf.feature_column.input_layer(
                    self._feature,
                    self._user_column,
                    adaptive_mask_tensors=adaptive_mask_tensors)
                item_emb = tf.feature_column.input_layer(
                    self._feature,
                    self._item_column,
                    adaptive_mask_tensors=adaptive_mask_tensors)
            else:
                for key in TAG_COLUMN:
                    self._feature[key] = tf.strings.split(
                        self._feature[key], '|')
                user_emb = tf.feature_column.input_layer(
                    self._feature, self._user_column)
                item_emb = tf.feature_column.input_layer(
                    self._feature, self._item_column)

        if self.bf16:
            user_emb = tf.cast(user_emb, dtype=tf.bfloat16)
            item_emb = tf.cast(item_emb, dtype=tf.bfloat16)

        # user dnn network
        user_scope = tf.variable_scope('user_dnn_layer', \
            partitioner=self._dense_layer_partitioner,
            reuse=tf.AUTO_REUSE)
        with user_scope.keep_weights(dtype=tf.float32) if self.bf16 \
            else user_scope:
            user_emb = self._dnn_tower(user_emb, 'user_dnn')

        # item dnn network
        item_scope = tf.variable_scope('item_dnn_layer', \
            partitioner=self._dense_layer_partitioner,
            reuse=tf.AUTO_REUSE)
        with item_scope.keep_weights(dtype=tf.float32) if self.bf16 \
            else item_scope:
            item_emb = self._dnn_tower(item_emb, 'item_dnn')

        if self.bf16:
            user_emb = tf.cast(user_emb, dtype=tf.float32)
            item_emb = tf.cast(item_emb, dtype=tf.float32)

        # norm
        user_emb = tf.math.l2_normalize(user_emb, axis=1)
        item_emb = tf.math.l2_normalize(item_emb, axis=1)

        user_item_sim = tf.reduce_sum(tf.multiply(user_emb, item_emb),
                                      axis=1,
                                      keep_dims=True)

        sim_w = tf.get_variable('sim_w',
                                dtype=tf.float32,
                                shape=(1, 1),
                                initializer=tf.ones_initializer())
        sim_b = tf.get_variable('sim_b',
                                dtype=tf.float32,
                                shape=(1),
                                initializer=tf.zeros_initializer())
        self._logits = tf.matmul(user_item_sim, tf.abs(sim_w)) + sim_b
        self._logits = tf.reshape(self._logits, [-1])
        self.probability = tf.math.sigmoid(self._logits)
        self.output = tf.round(self.probability)



# generate feature columns
def build_feature_columns():
    user_column = []
    item_column = []
    for column_name in INPUT_FEATURES:
        categorical_column = tf.feature_column.categorical_column_with_hash_bucket(
            column_name,
            hash_bucket_size=HASH_BUCKET_SIZES[column_name],
            dtype=tf.string)

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
                '''Embedding Variable Feature'''
                categorical_column = tf.feature_column.categorical_column_with_embedding(
                    column_name, dtype=tf.string, ev_option=ev_opt)
            elif args.adaptive_emb:
                '''                 Adaptive Embedding Feature Part 2 of 2
                Expcet the follow code, a dict, 'adaptive_mask_tensors', is need as the input of 
                'tf.feature_column.input_layer(adaptive_mask_tensors=adaptive_mask_tensors)'.
                For column 'COL_NAME',the value of adaptive_mask_tensors['$COL_NAME'] is a int32
                tensor with shape [batch_size].
                '''
                categorical_column = tf.feature_column.categorical_column_with_adaptive_embedding(
                    column_name,
                    hash_bucket_size=HASH_BUCKET_SIZES[column_name],
                    dtype=tf.string,
                    ev_option=ev_opt)
            elif args.dynamic_ev:
                '''Dynamic-dimension Embedding Variable'''
                print(
                    "Dynamic-dimension Embedding Variable isn't really enabled in model."
                )
                sys.exit()

        if args.tf or not args.emb_fusion:
            embedding_column = tf.feature_column.embedding_column(
                categorical_column, dimension=EMBEDDING_DIM, combiner='mean')
        else:
            '''Embedding Fusion Feature'''
            embedding_column = tf.feature_column.embedding_column(
                categorical_column,
                dimension=EMBEDDING_DIM,
                combiner='mean',
                do_fusion=args.emb_fusion)

        if column_name in USER_COLUMN:
            user_column.append(embedding_column)
        else:
            item_column.append(embedding_column)

    return user_column, item_column


def main(tf_config=None, server=None):

    # set batch size, eporch & steps
    batch_size = math.ceil(
        args.batch_size / args.micro_batch
    ) if args.micro_batch and not args.tf else args.batch_size

    # set fixed random seed
    tf.set_random_seed(args.seed)

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
    
    
    # create feature column
    user_column, item_column = build_feature_columns()

    # Session config
    sess_config = tf.ConfigProto()
    sess_config.inter_op_parallelism_threads = args.inter
    sess_config.intra_op_parallelism_threads = args.intra

    final_input = {}
    for fea in INPUT_FEATURES:
        final_input[fea] = tf.placeholder(tf.string,[None],name=fea)
    

    # create model
    model = DSSM(user_column=user_column,
                 item_column=item_column,
                 batch_size=batch_size,
                 learning_rate=args.learning_rate,
                 optimizer_type=args.optimizer,
                 bf16=args.bf16,
                 stock_tf=args.tf,
                 adaptive_emb=args.adaptive_emb,
                 inputs=final_input,
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
                        help='Batch size to train. Default is 4096',
                        type=int,
                        default=4096)
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
                        default='adamasync') # TODO: change to adam or adamsync
    parser.add_argument('--learning_rate',
                        help='Learning rate for model',
                        type=float,
                        default=0.001)
    parser.add_argument('--deep_dropout',
                        help='Dropout regularization for deep model',
                        type=float,
                        default=0.0)
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
                        default=False)#TODO:enable
    parser.add_argument('--incremental_ckpt', \
                        help='Set time of save Incremental Checkpoint. Default 0 to close.',
                        type=int,
                        default=0)
    parser.add_argument('--workqueue', \
                        help='Whether to enable Work Queue. Default to False.',
                        type=boolean_string,
                        default=False)
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
