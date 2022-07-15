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
CONTINUOUS_COLUMNS = ['I' + str(i) for i in range(1, 14)]  # 1-13 inclusive
CATEGORICAL_COLUMNS = ['C' + str(i) for i in range(1, 27)]  # 1-26 inclusive
LABEL_COLUMN = ['clicked']
TRAIN_DATA_COLUMNS = LABEL_COLUMN + CONTINUOUS_COLUMNS + CATEGORICAL_COLUMNS
FEATURE_COLUMNS = CONTINUOUS_COLUMNS + CATEGORICAL_COLUMNS
HASH_BUCKET_SIZES = {
    'C1': 2500,
    'C2': 2000,
    'C3': 5000000,
    'C4': 1500000,
    'C5': 1000,
    'C6': 100,
    'C7': 20000,
    'C8': 4000,
    'C9': 20,
    'C10': 100000,
    'C11': 10000,
    'C12': 5000000,
    'C13': 40000,
    'C14': 100,
    'C15': 100,
    'C16': 3000000,
    'C17': 50,
    'C18': 10000,
    'C19': 4000,
    'C20': 20,
    'C21': 4000000,
    'C22': 100,
    'C23': 100,
    'C24': 250000,
    'C25': 400,
    'C26': 100000
}


class DLRM():
    def __init__(self,
                 dense_column=None,
                 sparse_column=None,
                 mlp_bot=[512, 256, 64, 16],
                 mlp_top=[512, 256],
                 optimizer_type='adam',
                 learning_rate=0.1,
                 inputs=None,
                 interaction_op='dot',
                 bf16=False,
                 stock_tf=None,
                 adaptive_emb=False,
                 input_layer_partitioner=None,
                 dense_layer_partitioner=None):
        if not inputs:
            raise ValueError('Dataset is not defined.')
        self._feature = inputs

        if not dense_column or not sparse_column:
            raise ValueError('Dense column or sparse column is not defined.')
        self._dense_column = dense_column
        self._sparse_column = sparse_column

        self.tf = stock_tf
        self.bf16 = False if self.tf else bf16
        self.is_training = True
        self._adaptive_emb = adaptive_emb

        self._mlp_bot = mlp_bot
        self._mlp_top = mlp_top
        self._learning_rate = learning_rate
        self._input_layer_partitioner = input_layer_partitioner
        self._dense_layer_partitioner = dense_layer_partitioner
        self._optimizer_type = optimizer_type
        self.interaction_op = interaction_op
        if self.interaction_op not in ['dot', 'cat']:
            print("Invaild interaction op, must be 'dot' or 'cat'.")
            sys.exit()

        self._create_model()
      

    # used to add summary in tensorboard
    def _add_layer_summary(self, value, tag):
        tf.summary.scalar('%s/fraction_of_zero_values' % tag,
                          tf.nn.zero_fraction(value))
        tf.summary.histogram('%s/activation' % tag, value)

    def _dot_op(self, features):
        batch_size = tf.shape(features)[0]
        matrixdot = tf.matmul(features, features, transpose_b=True)
        feature_dim = matrixdot.shape[-1]

        ones_mat = tf.ones_like(matrixdot)
        lower_tri_mat = ones_mat - tf.linalg.band_part(ones_mat, 0, -1)
        lower_tri_mask = tf.cast(lower_tri_mat, tf.bool)
        result = tf.boolean_mask(matrixdot, lower_tri_mask)
        output_dim = feature_dim * (feature_dim - 1) // 2

        return tf.reshape(result, (batch_size, output_dim))

    # create model
    def _create_model(self):
        # input dense feature and embedding of sparse features
        with tf.variable_scope('input_layer', reuse=tf.AUTO_REUSE):
            with tf.variable_scope('dense_input_layer',
                                   partitioner=self._input_layer_partitioner,
                                   reuse=tf.AUTO_REUSE):
                dense_inputs = tf.feature_column.input_layer(
                    self._feature, self._dense_column)
            with tf.variable_scope('sparse_input_layer', reuse=tf.AUTO_REUSE):
                column_tensors = {}
                if self._adaptive_emb and not self.tf:
                    '''Adaptive Embedding Feature Part 1 of 2'''
                    adaptive_mask_tensors = {}
                    for col in CATEGORICAL_COLUMNS:
                        adaptive_mask_tensors[col] = tf.ones([args.batch_size],
                                                             tf.int32)
                    sparse_inputs = tf.feature_column.input_layer(
                        features=self._feature,
                        feature_columns=self._sparse_column,
                        cols_to_output_tensors=column_tensors,
                        adaptive_mask_tensors=adaptive_mask_tensors)
                else:
                    sparse_inputs = tf.feature_column.input_layer(
                        features=self._feature,
                        feature_columns=self._sparse_column,
                        cols_to_output_tensors=column_tensors)

        # MLP behind dense inputs
        mlp_bot_scope = tf.variable_scope(
            'mlp_bot_layer',
            partitioner=self._dense_layer_partitioner,
            reuse=tf.AUTO_REUSE)
        with mlp_bot_scope.keep_weights(dtype=tf.float32) if self.bf16 \
                else mlp_bot_scope:
            if self.bf16:
                dense_inputs = tf.cast(dense_inputs, dtype=tf.bfloat16)

            for layer_id, num_hidden_units in enumerate(self._mlp_bot):
                with tf.variable_scope(
                        'mlp_bot_hiddenlayer_%d' % layer_id,
                        reuse=tf.AUTO_REUSE) as mlp_bot_hidden_layer_scope:
                    dense_inputs = tf.layers.dense(
                        dense_inputs,
                        units=num_hidden_units,
                        activation=tf.nn.relu,
                        name=mlp_bot_hidden_layer_scope)
                    dense_inputs = tf.layers.batch_normalization(
                        dense_inputs,
                        training=self.is_training,
                        trainable=True)
                    self._add_layer_summary(dense_inputs,
                                            mlp_bot_hidden_layer_scope.name)
            if self.bf16:
                dense_inputs = tf.cast(dense_inputs, dtype=tf.float32)

        # interaction_op
        if self.interaction_op == 'dot':
            # dot op
            with tf.variable_scope('Op_dot_layer', reuse=tf.AUTO_REUSE):
                mlp_input = [dense_inputs]
                for cols in self._sparse_column:
                    mlp_input.append(column_tensors[cols])
                mlp_input = tf.stack(mlp_input, axis=1)
                mlp_input = self._dot_op(mlp_input)
                mlp_input = tf.concat([dense_inputs, mlp_input], 1)
        elif self.interaction_op == 'cat':
            mlp_input = tf.concat([dense_inputs, sparse_inputs], 1)

        # top MLP before output
        if self.bf16:
            mlp_input = tf.cast(mlp_input, dtype=tf.bfloat16)
        mlp_top_scope = tf.variable_scope(
            'mlp_top_layer',
            partitioner=self._dense_layer_partitioner,
            reuse=tf.AUTO_REUSE)
        with mlp_top_scope.keep_weights(dtype=tf.float32) if self.bf16 \
                else mlp_top_scope:
            for layer_id, num_hidden_units in enumerate(self._mlp_top):
                with tf.variable_scope(
                        'mlp_top_hiddenlayer_%d' % layer_id,
                        reuse=tf.AUTO_REUSE) as mlp_top_hidden_layer_scope:
                    mlp_logits = tf.layers.dense(mlp_input,
                                          units=num_hidden_units,
                                          activation=tf.nn.relu,
                                          name=mlp_top_hidden_layer_scope)

                self._add_layer_summary(mlp_logits, mlp_top_hidden_layer_scope.name)

        if self.bf16:
            mlp_logits = tf.cast(mlp_logits, dtype=tf.float32)

        with tf.variable_scope('logits', reuse=tf.AUTO_REUSE) as logits_scope:
            self._logits = tf.layers.dense(mlp_logits,
                                           units=1,
                                           activation=None,
                                           name=logits_scope)
            self.probability = tf.math.sigmoid(self._logits)
            self.output = tf.round(self.probability)

            self._add_layer_summary(self.probability, logits_scope.name)

   

# generate feature columns
def build_feature_columns():
    dense_column = []
    sparse_column = []
    for column_name in FEATURE_COLUMNS:
        if column_name in CATEGORICAL_COLUMNS:
            categorical_column = tf.feature_column.categorical_column_with_hash_bucket(
                column_name,
                hash_bucket_size=10000,
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
                    filter_option = tf.CBFFilter(
                        filter_freq=3,
                        max_element_size=2**30,
                        false_positive_probability=0.01,
                        counter_type=tf.int64)
                elif args.ev_filter == 'counter':
                    # Counter-based feature filter
                    filter_option = tf.CounterFilter(filter_freq=3)
                else:
                    filter_option = None
                ev_opt = tf.EmbeddingVariableOption(
                    evict_option=evict_opt, filter_option=filter_option)

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
                    categorical_column,
                    dimension=16,
                    combiner='mean')
            else:
                '''Embedding Fusion Feature'''
                embedding_column = tf.feature_column.embedding_column(
                    categorical_column,
                    dimension=16,
                    combiner='mean',
                    do_fusion=args.emb_fusion)

            sparse_column.append(embedding_column)
        else:
            column = tf.feature_column.numeric_column(column_name, shape=(1, ))
            dense_column.append(column)

    return dense_column, sparse_column




def main(tf_config=None, server=None):
    
    # set batch size, eporch & steps
    batch_size = math.ceil(
        args.batch_size / args.micro_batch
    ) if args.micro_batch and not args.tf else args.batch_size

    # set fixed random seed
    tf.set_random_seed(args.seed)


    # create feature column
    dense_column, sparse_column = build_feature_columns()

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
    # CONTINUOUS_COLUMNS = ['I' + str(i) for i in range(1, 14)]  # 1-13 inclusive
    # CATEGORICAL_COLUMNS = ['C' + str(i) for i in range(1, 27)]  # 1-26 inclusive 
    for fea in CONTINUOUS_COLUMNS:
        final_input[fea] = tf.placeholder(tf.float32,[None], name=fea)
    for fea in CATEGORICAL_COLUMNS:
        final_input[fea] = tf.placeholder(tf.string, [None], name=fea)

    # create model
    model = DLRM(dense_column=dense_column,
                 sparse_column=sparse_column,
                 learning_rate=args.learning_rate,
                 optimizer_type=args.optimizer,
                 bf16=args.bf16,
                 stock_tf=args.tf,
                 adaptive_emb=args.adaptive_emb,
                 interaction_op=args.interaction_op,
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
                        help='Batch size to train. Default is 512',
                        type=int,
                        default=512)
    parser.add_argument('--output_dir',
                        help='Full path to logs & model output directory',
                        required=False,
                        default='./result')
    parser.add_argument('--checkpoint',
                        help='Full path to checkpoints input/output directory',
                        required=False)
    parser.add_argument('--deep_dropout',
                        help='Dropout regularization for deep model',
                        type=float,
                        default=0.0)
    parser.add_argument('--seed',
                        help='set the random seed for tensorflow',
                        type=int,
                        default=2021)
    parser.add_argument('--optimizer',
                        type=str,
                        choices=['adam', 'adamasync', 'adagraddecay',
                                 'adagrad', 'gradientdescent'],
                        default='adamasync')
    parser.add_argument('--learning_rate',
                        help='Learning rate for model',
                        type=float,
                        default=0.01)
    parser.add_argument('--save_steps',
                        help='set the number of steps on saving checkpoints',
                        type=int,
                        default=0)
    parser.add_argument('--keep_checkpoint_max',
                        help='Maximum number of recent checkpoint to keep',
                        type=int,
                        default=1)
    parser.add_argument('--bf16',
                        help='enable DeepRec BF16 in deep model. Default FP32',
                        action='store_true')
    parser.add_argument('--no_eval',
                        help='not evaluate trained model by eval dataset.',
                        action='store_true')
    parser.add_argument('--timeline',
                        help='number of steps on saving timeline. Default 0',
                        type=int,
                        default=0)
    parser.add_argument('--interaction_op',
                        type=str,
                        choices=['dot', 'cat'],
                        default='cat')
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
    parser.add_argument('--input_layer_partitioner',
                        help='slice size of input layer partitioner. units MB',
                        type=int,
                        default=8)
    parser.add_argument('--dense_layer_partitioner',
                        help='slice size of dense layer partitioner. units KB',
                        type=int,
                        default=16)
    parser.add_argument('--tf',
                        help='Use TF 1.15.5 API and disable DeepRec feature to run a baseline.',
                        action='store_true')
    parser.add_argument('--smartstaged',
                        help='Whether to enable smart staged feature of DeepRec, Default to True.',
                        type=boolean_string,
                        default=True)
    parser.add_argument('--emb_fusion',
                        help='Whether to enable embedding fusion, Default to True.',
                        type=boolean_string,
                        default=True)
    parser.add_argument('--ev',
                        help='Whether to enable DeepRec EmbeddingVariable. Default False.',
                        type=boolean_string,
                        default=False)
    parser.add_argument('--ev_elimination',
                        help='Feature Elimination of EmbeddingVariable Feature. Default closed.',
                        type=str,
                        choices=[None, 'l2', 'gstep'],
                        default=None)
    parser.add_argument('--ev_filter',
                        help='Feature Filter of EmbeddingVariable Feature. Default closed.',
                        type=str,
                        choices=[None, 'counter', 'cbf'],
                        default=None)
    parser.add_argument('--op_fusion',
                        help='Whether to enable Auto graph fusion feature. Default to True',
                        type=boolean_string,
                        default=True)
    parser.add_argument('--micro_batch',
                        help='Set num for Auto Mirco Batch. Default close.',
                        type=int,
                        default=0) # TODO enable
    parser.add_argument('--adaptive_emb',
                        help='Whether to enable Adaptive Embedding. Default to False.',
                        type=boolean_string,
                        default=False)
    parser.add_argument('--dynamic_ev',
                        help='Whether to enable Dynamic-dimension Embedding Variable. Default to False.',
                        type=boolean_string,
                        default=False) # TODO enable
    parser.add_argument('--incremental_ckpt',
                        help='Set time of save Incremental Checkpoint. Default 0 to close.',
                        type=int,
                        default=0)
    parser.add_argument('--workqueue',
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
    os.environ['MALLOC_CONF'] = \
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
