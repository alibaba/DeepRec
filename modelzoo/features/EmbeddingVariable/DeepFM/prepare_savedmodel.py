import time
import argparse
import tensorflow as tf
import os
import sys
import math
import collections
from tensorflow.python.client import timeline
import json



CONTINUOUS_COLUMNS = ["I" + str(i) for i in range(1, 14)]  # 1-13 inclusive
CATEGORICAL_COLUMNS = ["C" + str(i) for i in range(1, 27)]  # 1-26 inclusive
LABEL_COLUMN = ["clicked"]
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


def add_layer_summary(value, tag):
    tf.summary.scalar('%s/fraction_of_zero_values' % tag,
                      tf.nn.zero_fraction(value))
    tf.summary.histogram('%s/activation' % tag, value)




def build_feature_cols():
    wide_column = []
    deep_column = []
    fm_column = []
    for column_name in FEATURE_COLUMNS:
        if column_name in CATEGORICAL_COLUMNS:
            categorical_column = tf.feature_column.categorical_column_with_embedding(
                column_name,
                dtype=tf.string)

            categorical_embedding_column = tf.feature_column.embedding_column(
                categorical_column, dimension=16, combiner='mean')

            wide_column.append(categorical_embedding_column)
            deep_column.append(categorical_embedding_column)
            fm_column.append(categorical_embedding_column)
        else:
            column = tf.feature_column.numeric_column(column_name, shape=(1, ))
            wide_column.append(column)
            deep_column.append(column)

    return wide_column, fm_column, deep_column


class DeepFM():
    def __init__(self,
                 wide_column=None,
                 fm_column=None,
                 deep_column=None,
                 dnn_hidden_units=[1024, 256, 32],
                 final_hidden_units=[128, 64],
                 optimizer_type='adam',
                 learning_rate=0.001,
                 inputs=None,
                 use_bn=True,
                 bf16=False,
                 input_layer_partitioner=None,
                 dense_layer_partitioner=None):
        if not inputs:
            raise ValueError('Dataset is not defined.')
        self.wide_column = wide_column
        self.deep_column = deep_column
        self.fm_column = fm_column
        if not wide_column or not fm_column or not deep_column:
            raise ValueError(
                'Wide column, FM column or Deep column is not defined.')
        self.dnn_hidden_units = dnn_hidden_units
        self.final_hidden_units = final_hidden_units
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        self.input_layer_partitioner = input_layer_partitioner
        self.dense_layer_partitioner = dense_layer_partitioner

        self.feature = inputs
        self.bf16 = bf16

       
        self.use_bn = use_bn

        self.predict = self.prediction()
       

    def dnn(self, dnn_input, dnn_hidden_units=None, layer_name=''):
        for layer_id, num_hidden_units in enumerate(dnn_hidden_units):
            with tf.variable_scope(layer_name + "_%d" % layer_id,
                                   partitioner=self.dense_layer_partitioner,
                                   reuse=tf.AUTO_REUSE) as dnn_layer_scope:
                dnn_input = tf.layers.dense(dnn_input,
                                            units=num_hidden_units,
                                            activation=tf.nn.relu,
                                            name=dnn_layer_scope)
                if self.use_bn:
                    dnn_input = tf.layers.batch_normalization(
                        dnn_input)
                add_layer_summary(dnn_input, dnn_layer_scope.name)

        return dnn_input

    def prediction(self):
        # input features
        with tf.variable_scope('input_layer',
                               partitioner=self.input_layer_partitioner,
                               reuse=tf.AUTO_REUSE):

            fm_cols = {}
            wide_input = tf.feature_column.input_layer(
                self.feature, self.wide_column, cols_to_output_tensors=fm_cols)
            fm_input = tf.stack([fm_cols[cols] for cols in self.fm_column], 1)
            dnn_input = tf.feature_column.input_layer(self.feature,
                                                      self.deep_column)

        if self.bf16:
            wide_input = tf.cast(wide_input, dtype=tf.bfloat16)
            fm_input = tf.cast(fm_input, dtype=tf.bfloat16)
            dnn_input = tf.cast(dnn_input, dtype=tf.bfloat16)

        # DNN part
        if self.bf16:
            with tf.variable_scope('dnn').keep_weights():
                dnn_output = self.dnn(dnn_input, self.dnn_hidden_units,
                                      'dnn_layer')
        else:
            with tf.variable_scope('dnn'):
                dnn_output = self.dnn(dnn_input, self.dnn_hidden_units,
                                      'dnn_layer')

        # linear / fisrt order part
        with tf.variable_scope('linear', reuse=tf.AUTO_REUSE) as linear:
            linear_output = tf.reduce_sum(wide_input, axis=1, keepdims=True)

        # FM second order part
        with tf.variable_scope('fm', reuse=tf.AUTO_REUSE) as fm:
            sum_square = tf.square(tf.reduce_sum(fm_input, axis=1))
            square_sum = tf.reduce_sum(tf.square(fm_input), axis=1)
            fm_output = 0.5 * tf.subtract(sum_square, square_sum)

        # Final dnn layer
        all_input = tf.concat([dnn_output, linear_output, fm_output], 1)
        if self.bf16:
            with tf.variable_scope('final_dnn').keep_weights():
                net = self.dnn(all_input, self.final_hidden_units, 'final_dnn')
            net = tf.cast(net, dtype=tf.float32)
        else:
            with tf.variable_scope('final_dnn'):
                net = self.dnn(all_input, self.final_hidden_units, 'final_dnn')

        net = tf.layers.dense(net, units=1)
        net = tf.math.sigmoid(net)
        self.output = net
        return net




def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_location',
                        help='Full path of train data',
                        required=False,
                        default='./data')
    parser.add_argument('--batch_size',
                        help='Batch size to train. Default is 512',
                        type=int,
                        default=512)
    parser.add_argument('--checkpoint',
                        help='Full path to checkpoints input/output directory',
                        required=False)
    parser.add_argument('--bf16',
                        help='enable DeepRec BF16 in deep model. Default FP32',
                        action='store_true')
    parser.add_argument("--optimizer",
                        type=str,
                        choices=["adam", "adagrad", "adamasync"],
                        default="adam")
    parser.add_argument('--learning_rate',
                        help='Learning rate for model',
                        type=float,
                        default=0.001)
    return parser


def main(tf_config=None, server=None):
    
    
    with tf.Session() as sess1:
        batch_size = args.batch_size
        
        # set fixed random seed
        tf.set_random_seed(2021)

    

        # create data pipline
        wide_column, fm_column, deep_column = build_feature_cols()
    
        final_input = {}


        for i in range(1,14):
            final_input["I"+str(i)] = tf.placeholder(tf.float32,[None], name='I'+str(i))   
        for j in range(1,27):
            final_input["C"+str(j)] = tf.placeholder(tf.string, [None], name='C'+str(j)) 


        # create model
        model = DeepFM(wide_column=wide_column,
                    fm_column=fm_column,
                    deep_column=deep_column,
                    optimizer_type=args.optimizer,
                    learning_rate=args.learning_rate,
                    bf16=args.bf16,
                    inputs=final_input,
                    input_layer_partitioner=None,
                    dense_layer_partitioner=None)

            
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
            outputs = {"Sigmoid":model.output}
        )


if __name__ == "__main__":
    parser = get_arg_parser()
    args = parser.parse_args()

  
    main()
   