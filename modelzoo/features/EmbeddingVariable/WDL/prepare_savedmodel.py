import time
import argparse
import tensorflow as tf
import os
import sys
import math
import collections
from tensorflow.python.client import timeline
import json
from tensorflow.python.training import incremental_saver as tf_incr_saver
from tensorflow.core.framework.embedding import config_pb2
from tensorflow.python.ops import variables


CONTINUOUS_COLUMNS = ["I" + str(i) for i in range(1, 14)]  # 1-13 inclusive
CATEGORICAL_COLUMNS = ["C" + str(i) for i in range(1, 27)]  # 1-26 inclusive
LABEL_COLUMN = ["clicked"]
TRAIN_DATA_COLUMNS = LABEL_COLUMN + CONTINUOUS_COLUMNS + CATEGORICAL_COLUMNS
FEATURE_COLUMNS = CONTINUOUS_COLUMNS + CATEGORICAL_COLUMNS
IDENTIY_COLUMNS = ["I10"]
HASH_BUCKET_SIZES = {
    'C1': 2500,
    'C2': 2000,
    'C3': 300000,
    'C4': 250000,
    'C5': 1000,
    'C6': 100,
    'C7': 20000,
    'C8': 4000,
    'C9': 20,
    'C10': 100000,
    'C11': 10000,
    'C12': 250000,
    'C13': 40000,
    'C14': 100,
    'C15': 100,
    'C16': 200000,
    'C17': 50,
    'C18': 10000,
    'C19': 4000,
    'C20': 20,
    'C21': 250000,
    'C22': 100,
    'C23': 100,
    'C24': 250000,
    'C25': 400,
    'C26': 100000
}

IDENTITY_NUM_BUCKETS = {'I10': 10}

EMBEDDING_DIMENSIONS = {
    'C1': 64,
    'C2': 64,
    'C3': 128,
    'C4': 128,
    'C5': 64,
    'C6': 64,
    'C7': 64,
    'C8': 64,
    'C9': 64,
    'C10': 128,
    'C11': 64,
    'C12': 128,
    'C13': 64,
    'C14': 64,
    'C15': 64,
    'C16': 128,
    'C17': 64,
    'C18': 64,
    'C19': 64,
    'C20': 64,
    'C21': 128,
    'C22': 64,
    'C23': 64,
    'C24': 128,
    'C25': 64,
    'C26': 128
}


def add_layer_summary(value, tag):
    tf.summary.scalar('%s/fraction_of_zero_values' % tag,
                      tf.nn.zero_fraction(value))
    tf.summary.histogram('%s/activation' % tag, value)


def build_feature_cols(train_file_path, test_file_path):
    # Statistics of Kaggle's Criteo Dataset has been calculated in advance to save time
    print('****Computing statistics of train dataset*****')
    # with open(train_file_path, 'r') as f, open(test_file_path, 'r') as f1:
    #     nums = [line.strip('\n').split(',') for line in f.readlines()
    #             ] + [line.strip('\n').split(',') for line in f1.readlines()]
    #     numpy_arr = np.array(nums)
    #     mins_list, max_list, range_list = [], [], []
    #     for i in range(len(TRAIN_DATA_COLUMNS)):
    #         if TRAIN_DATA_COLUMNS[i] in CONTINUOUS_COLUMNS:
    #             col_min = numpy_arr[:, i].astype(np.float32).min()
    #             col_max = numpy_arr[:, i].astype(np.float32).max()
    #             mins_list.append(col_min)
    #             max_list.append(col_max)
    #             range_list.append(col_max - col_min)
    mins_list = [
        0.0, -3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    ]
    range_list = [
        1539.0, 22069.0, 65535.0, 561.0, 2655388.0, 233523.0, 26297.0, 5106.0,
        24376.0, 9.0, 181.0, 1807.0, 6879.0
    ]

    def make_minmaxscaler(min, range):
        def minmaxscaler(col):
            return (col - min) / range

        return minmaxscaler

    deep_columns = []
    wide_columns = []
    for column_name in FEATURE_COLUMNS:
        if column_name in IDENTITY_NUM_BUCKETS:
            categorical_column = tf.feature_column.categorical_column_with_identity(
                column_name, num_buckets=IDENTITY_NUM_BUCKETS[column_name])
            wide_columns.append(categorical_column)
            deep_columns.append(
                tf.feature_column.indicator_column(categorical_column))
        elif column_name in CATEGORICAL_COLUMNS:
            ev_option = None
            if(args.ev_storage == "pmem_libpmem"):
                ev_option = tf.EmbeddingVariableOption(storage_option=variables.StorageOption(
                    storage_type=config_pb2.StorageType.PMEM_LIBPMEM,
                    storage_path=args.ev_storage_path,
                    storage_size=[args.ev_storage_size_gb*1024*1024*1024]))
            elif(args.ev_storage == "pmem_memkind"):
                ev_option = tf.EmbeddingVariableOption(storage_option=variables.StorageOption(
                    storage_type=config_pb2.StorageType.PMEM_MEMKIND))
            elif(args.ev_storage == "dram_pmem"):
                ev_option = tf.EmbeddingVariableOption(storage_option=variables.StorageOption(
                    storage_type=config_pb2.StorageType.DRAM_PMEM,
                    storage_path=args.ev_storage_path,
                    storage_size=[args.ev_storage_size_gb*1024*1024*1024, args.ev_storage_size_gb*1024*1024*1024]))
            else:
                ev_option = tf.EmbeddingVariableOption(storage_option=variables.StorageOption(
                    storage_type=config_pb2.StorageType.DRAM))
            categorical_column = tf.feature_column.categorical_column_with_embedding(
                column_name,
                dtype=tf.string,
                ev_option=ev_option)
            wide_columns.append(categorical_column)

            deep_columns.append(
                tf.feature_column.embedding_column(
                    categorical_column,
                    dimension=EMBEDDING_DIMENSIONS[column_name],
                    combiner='mean'))
        else:
            normalizer_fn = None
            i = CONTINUOUS_COLUMNS.index(column_name)
            col_min = mins_list[i]
            col_range = range_list[i]
            normalizer_fn = make_minmaxscaler(col_min, col_range)
            column = tf.feature_column.numeric_column(
                column_name, normalizer_fn=normalizer_fn, shape=(1, ))
            wide_columns.append(column)
            deep_columns.append(column)

    return wide_columns, deep_columns


class WDL():
    def __init__(self,
                 wide_column=None,
                 deep_column=None,
                 dnn_hidden_units=[1024, 512, 256],
                 linear_learning_rate=0.2,
                 deep_learning_rate=0.01,
                 inputs=None,
                 bf16=False,
                 input_layer_partitioner=None,
                 dense_layer_partitioner=None,
                 saved_model=False):

        if not inputs:
            raise ValueError('Dataset is not defined.')
        self.wide_column = wide_column
        self.deep_column = deep_column
        if not wide_column or not deep_column:
            raise ValueError('Wide column or Deep column is not defined.')
        self.dnn_hidden_units = dnn_hidden_units
        self.linear_learning_rate = linear_learning_rate
        self.deep_learning_rate = deep_learning_rate
        self.input_layer_partitioner = input_layer_partitioner
        self.dense_layer_partitioner = dense_layer_partitioner
        self.global_step = tf.train.get_or_create_global_step()

        self.feature = inputs
        self.bf16 = bf16

        self._is_training = False

        self.predict = self.prediction()
        
    def dnn(self, dnn_input, dnn_hidden_units=None, layer_name=''):
        for layer_id, num_hidden_units in enumerate(dnn_hidden_units):
            with tf.variable_scope(layer_name + "_%d" % layer_id,
                                   partitioner=self.dense_layer_partitioner,
                                   reuse=tf.AUTO_REUSE) as dnn_layer_scope:
                dnn_input = tf.layers.dense(
                    dnn_input,
                    units=num_hidden_units,
                    activation=tf.nn.relu,
                    kernel_initializer=tf.glorot_uniform_initializer(),
                    name=dnn_layer_scope)

                add_layer_summary(dnn_input, dnn_layer_scope.name)

        return dnn_input

    def prediction(self):
        # input features
        self.dnn_parent_scope = 'dnn'
        with tf.variable_scope(self.dnn_parent_scope):
            with tf.variable_scope("input_from_feature_columns",
                                   partitioner=self.input_layer_partitioner,
                                   reuse=tf.AUTO_REUSE) as dnn_inputs_scope:
                net = tf.feature_column.input_layer(
                    features=self.feature, feature_columns=self.deep_column)

                add_layer_summary(net, dnn_inputs_scope.name)

            if self.bf16:
                net = tf.cast(net, dtype=tf.bfloat16)
                with tf.variable_scope(
                        'dnn_layers',
                        partitioner=self.dense_layer_partitioner,
                        reuse=tf.AUTO_REUSE).keep_weights():
                    net = self.dnn(net, self.dnn_hidden_units, "hiddenlayer")

                    with tf.variable_scope(
                            "logits",
                            values=(net, )).keep_weights() as dnn_logits_scope:
                        dnn_logits = tf.layers.dense(net,
                                                     units=1,
                                                     activation=None,
                                                     name=dnn_logits_scope)
                    add_layer_summary(dnn_logits, dnn_logits_scope.name)
                dnn_logits = tf.cast(dnn_logits, dtype=tf.float32)

            else:
                with tf.variable_scope(
                        'dnn_layers',
                        partitioner=self.dense_layer_partitioner,
                        reuse=tf.AUTO_REUSE):
                    net = self.dnn(net, self.dnn_hidden_units, "hiddenlayer")

                with tf.variable_scope("logits",
                                       values=(net, )) as dnn_logits_scope:
                    dnn_logits = tf.layers.dense(net,
                                                 units=1,
                                                 activation=None,
                                                 name=dnn_logits_scope)
                add_layer_summary(dnn_logits, dnn_logits_scope.name)

        self.linear_parent_scope = 'linear'
        with tf.variable_scope(
                self.linear_parent_scope,
                partitioner=self.input_layer_partitioner) as scope:
            linear_logits = tf.feature_column.linear_model(
                units=1,
                features=self.feature,
                feature_columns=self.wide_column,
                sparse_combiner='sum',
                weight_collections=None,
                trainable=True)

            add_layer_summary(linear_logits, scope.name)

        self.logits = tf.add_n([dnn_logits, linear_logits])
        predict = tf.math.sigmoid(self.logits)

        return predict

   

def get_arg_parser():
    parser = argparse.ArgumentParser()
   
    parser.add_argument('--batch_size',
                        help='Batch size to train. Default is 512',
                        type=int,
                        default=512)

    parser.add_argument('--bf16',
                        help='enable DeepRec BF16 in deep model. Default FP32',
                        action='store_true')
   
    parser.add_argument('--checkpoint',
                        type=str,
                        help='ckpt path',
                        default="")
    parser.add_argument('--data_location',
                        help='Full path of train data',
                        required=False,
                        default='./data')
    parser.add_argument('--ev_storage',
                        type=str,
                        choices=['dram', 'pmem_libpmem', 'pmem_memkind', 'dram_pmem'],
                        default='dram')
    parser.add_argument('--linear_learning_rate',
                        help='Learning rate for linear model',
                        type=float,
                        default=0.2)
    parser.add_argument('--deep_learning_rate',
                        help='Learning rate for deep model',
                        type=float,
                        default=0.01)
    return parser


def main(tf_config=None, server=None):
   
    # set batch size & steps
    batch_size = args.batch_size
    train_file = args.data_location + '/train.csv'
    test_file = args.data_location + '/eval.csv'
   
    # set fixed random seed
    tf.set_random_seed(2021)
  
    # create data pipline
    wide_column, deep_column = build_feature_cols(train_file, test_file)
   
 
    is_saved_model = True    
    inputs = {}
    # I1-I13
    for x in range(1, 10):
        inputs['I'+str(x)] = tf.placeholder(tf.float32,
                                            [None], name='I'+str(x))
    inputs['I10'] = tf.placeholder(tf.int64, [None], name='I10')
    for x in range(11, 14):
        inputs['I'+str(x)] = tf.placeholder(tf.float32,
                                            [None], name='I'+str(x))
    # C1-C26
    for x in range(1, 27):
        inputs['C'+str(x)] = tf.placeholder(tf.string,
                                            [None], name='C'+str(x))
    real_input = inputs

    # create model
    model = WDL(wide_column=wide_column,
                deep_column=deep_column,
                linear_learning_rate=args.linear_learning_rate,
                deep_learning_rate=args.deep_learning_rate,
                bf16=args.bf16,
                inputs=real_input,
                input_layer_partitioner=None,
                dense_layer_partitioner=None,
                saved_model=True)

    with tf.Session() as sess1:

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
            inputs = model.feature,
            outputs = {"Sigmoid":model.predict}
        )
    

if __name__ == "__main__":
    parser = get_arg_parser()
    args = parser.parse_args()

    
    main()
   