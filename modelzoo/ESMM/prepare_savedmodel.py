import argparse
import collections
import json
import math
import numbers
import os
import tensorflow as tf
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.client import timeline
import time

# Set to INFO for tracking training, default is WARN. ERROR for least messages
tf.logging.set_verbosity(tf.logging.INFO)
print(f'Using TensorFlow version {tf.__version__}')

USER_COLUMN = [
    'user_id', 'cms_segid', 'cms_group_id', 'age_level', 'pvalue_level',
    'shopping_level', 'occupation', 'new_user_class_level'
]
ITEM_COLUMN = [
    'adgroup_id', 'cate_id', 'campaign_id', 'customer', 'brand', 'price'
]
COMBO_COLUMN = [
    'pid', 'tag_category_list', 'tag_brand_list'
]
LABEL_COLUMNS = ['clk', 'buy']
TAG_COLUMN = ['tag_category_list', 'tag_brand_list']

HASH_INPUTS = [
    'pid', 'adgroup_id', 'cate_id', 'campaign_id', 'customer', 'brand',
    'user_id', 'cms_segid', 'cms_group_id', 'final_gender_code', 'age_level',
    'pvalue_level', 'shopping_level', 'occupation', 'new_user_class_level',
    'tag_category_list', 'tag_brand_list', 'price'
]

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

defaults = [[0]] * len(LABEL_COLUMNS) + [[' ']] * len(HASH_INPUTS)
ALL_FEATURE_COLUMNS = HASH_INPUTS
headers = LABEL_COLUMNS + ALL_FEATURE_COLUMNS
class ESMM():
    def __init__(self,
                 input,
                 user_column,
                 item_column,
                 combo_column,
                 user_mlp=[256, 128, 96, 64],
                 item_mlp=[256, 128, 96, 64],
                 combo_mlp=[128, 96, 64, 32],
                 cvr_mlp=[256, 128, 96, 64],
                 ctr_mlp=[256, 128, 96, 64],
                 batch_size=None,
                 optimizer_type='adam',
                 bf16=False,
                 stock_tf=None,
                 adaptive_emb=False,
                 learning_rate=0.1,
                 l2_scale=1e-6,
                 input_layer_partitioner=None,
                 dense_layer_partitioner=None):
        if not input:
            raise ValueError('Dataset is not defined.')
        if not user_column or not item_column or not combo_column:
            raise ValueError('User column, item column or combo column is not defined.')
        self._user_column = user_column
        self._item_column = item_column
        self._combo_column = combo_column

        self._user_mlp = user_mlp
        self._item_mlp = item_mlp
        self._combo_mlp = combo_mlp
        self._cvr_mlp = cvr_mlp
        self._ctr_mlp = ctr_mlp
        self._batch_size = batch_size

        self._optimizer_type = optimizer_type
        self._learning_rate = learning_rate
        self._l2_regularization = self._l2_regularizer(l2_scale) if l2_scale else None
        self._tf = stock_tf
        self._adaptive_emb = adaptive_emb
        self._bf16 = False if self._tf else bf16

        self._input_layer_partitioner = input_layer_partitioner
        self._dense_layer_partitioner = dense_layer_partitioner

        self.feature = input
        self._create_model()



    # used to add summary in tensorboard
    def _add_layer_summary(self, value, tag):
        tf.summary.scalar('%s/fraction_of_zero_values' % tag,
                          tf.nn.zero_fraction(value))
        tf.summary.histogram('%s/activation' % tag, value)

    

    def _create_dense_layer(self, input, num_hidden_units, activation, layer_name):
        with tf.variable_scope(layer_name, reuse=tf.AUTO_REUSE) as mlp_layer_scope:
            dense_layer = tf.layers.dense(input,
                                       units=num_hidden_units,
                                       activation=activation,
                                       kernel_regularizer=self._l2_regularization,
                                       name=mlp_layer_scope)
            self._add_layer_summary(dense_layer, mlp_layer_scope.name)
        return dense_layer

    def _l2_regularizer(self, scale, scope=None):
        if isinstance(scale, numbers.Integral):
            raise ValueError(f'Scale cannot be an integer: {scale}')
        if isinstance(scale, numbers.Real):
            if scale < 0.:
                raise ValueError(f'Setting a scale less than 0 on a regularizer: {scale}.')
            if scale == 0.:
                return lambda _: None

        def l2(weights):
            with tf.name_scope(scope, 'l2_regularizer', [weights]) as name:
                my_scale = tf.convert_to_tensor(scale, dtype=weights.dtype.base_dtype, name='scale')
                return tf.math.multiply(my_scale, tf.nn.l2_loss(weights), name=name)

        return l2

    def _make_scope(self, name, bf16):
        if(bf16):
            return tf.variable_scope(name, reuse=tf.AUTO_REUSE).keep_weights(dtype=tf.float32)
        else:
            return tf.variable_scope(name, reuse=tf.AUTO_REUSE)

    # create model
    def _create_model(self):
        with tf.variable_scope('user_input_layer',
                               partitioner=self._input_layer_partitioner,
                               reuse=tf.AUTO_REUSE):
            if not self._tf and self._adaptive_emb:
                '''Adaptive Embedding Feature Part 1 of 2'''
                adaptive_mask_tensors = {}
                for col in USER_COLUMN:
                    adaptive_mask_tensors[col] = tf.ones([self._batch_size],
                                                         tf.int32)
                user_emb = tf.feature_column.input_layer(
                    self.feature,
                    self._user_column,
                    adaptive_mask_tensors=adaptive_mask_tensors)
            else:
                user_emb = tf.feature_column.input_layer(self.feature,
                                                         self._user_column)
        with tf.variable_scope('item_input_layer',
                               partitioner=self._input_layer_partitioner,
                               reuse=tf.AUTO_REUSE):
            if not self._tf and self._adaptive_emb:
                '''Adaptive Embedding Feature Part 1 of 2'''
                adaptive_mask_tensors = {}
                for col in ITEM_COLUMN:
                    adaptive_mask_tensors[col] = tf.ones([self._batch_size],
                                                         tf.int32)
                item_emb = tf.feature_column.input_layer(
                    self.feature,
                    self._item_column,
                    adaptive_mask_tensors=adaptive_mask_tensors)
            else:
                item_emb = tf.feature_column.input_layer(self.feature,
                                                         self._item_column)
        with tf.variable_scope('combo_input_layer',
                               partitioner=self._input_layer_partitioner,
                               reuse=tf.AUTO_REUSE):
            if not self._tf and self._adaptive_emb:
                '''Adaptive Embedding Feature Part 1 of 2'''
                adaptive_mask_tensors = {}
                for col in COMBO_COLUMN:
                    adaptive_mask_tensors[col] = tf.ones([self._batch_size],
                                                         tf.int32)
                combo_emb = tf.feature_column.input_layer(
                    self.feature,
                    self._combo_column,
                    adaptive_mask_tensors=adaptive_mask_tensors)
            else:
                for key in TAG_COLUMN:
                    self.feature[key] = tf.strings.split(self.feature[key], '|')
                combo_emb = tf.feature_column.input_layer(self.feature,
                                                          self._combo_column)

        with self._make_scope('ESMM', self._bf16):
            if self._bf16:
                user_emb = tf.cast(user_emb, dtype=tf.bfloat16)
                item_emb = tf.cast(item_emb, dtype=tf.bfloat16)
                combo_emb = tf.cast(combo_emb, dtype=tf.bfloat16)

            with tf.variable_scope('user_mlp_layer',
                                   partitioner=self._dense_layer_partitioner,
                                   reuse=tf.AUTO_REUSE):
                for layer_id, num_hidden_units in enumerate(self._user_mlp):
                    user_emb = self._create_dense_layer(user_emb,
                                                         num_hidden_units,
                                                         tf.nn.relu,
                                                         f'user_mlp_{layer_id}')

            with tf.variable_scope('item_mlp_layer',
                                   partitioner=self._dense_layer_partitioner,
                                   reuse=tf.AUTO_REUSE):
                for layer_id, num_hidden_units in enumerate(self._item_mlp):
                    item_emb = self._create_dense_layer(item_emb,
                                                         num_hidden_units,
                                                         tf.nn.relu,
                                                         f'item_mlp_{layer_id}')

            with tf.variable_scope('combo_mlp_layer',
                                   partitioner=self._dense_layer_partitioner,
                                   reuse=tf.AUTO_REUSE):
                for layer_id, num_hidden_units in enumerate(self._combo_mlp):
                    combo_emb = self._create_dense_layer(combo_emb,
                                                          num_hidden_units,
                                                          tf.nn.relu,
                                                          f'combo_mlp_{layer_id}')

            concat = tf.concat([user_emb, item_emb, combo_emb], axis=1)

            pCVR = self._build_cvr_model(concat)
            pCTR = self._build_ctr_model(concat)

            pCTCVR = tf.cast(tf.multiply(pCVR, pCTR), tf.float32)
            self.probability = pCTCVR
            self.output = tf.round(self.probability)

    def _build_cvr_model(self, net):
        with tf.variable_scope('cvr_mlp',
                               partitioner=self._dense_layer_partitioner,
                               reuse=tf.AUTO_REUSE):
            for layer_id, num_hidden_units in enumerate(self._cvr_mlp):
                net = self._create_dense_layer(net,
                                                num_hidden_units,
                                                tf.nn.relu,
                                                f'cvr_mlp_hiddenlayer_{layer_id}')
            net = self._create_dense_layer(net,
                                            1,
                                            tf.math.sigmoid,
                                            'cvr_mlp_hiddenlayer_last')
        return net

    def _build_ctr_model(self, net):
        with tf.variable_scope('ctr_mlp',
                               partitioner=self._dense_layer_partitioner,
                               reuse=tf.AUTO_REUSE):
            for layer_id, num_hidden_units in enumerate(self._ctr_mlp):
                net = self._create_dense_layer(net,
                                                num_hidden_units,
                                                tf.nn.relu,
                                                f'ctr_mlp_hiddenlayer_{layer_id}')
            net = self._create_dense_layer(net,
                                            1,
                                            tf.math.sigmoid,
                                            'ctr_mlp_hiddenlayer_last')
        return net

# generate dataset pipline
def build_model_input(filename, batch_size, num_epochs, seed, stock_tf, workqueue=None):
    def parse_csv(value):
        tf.logging.info('Parsing {}'.format(filename))
        columns = tf.io.decode_csv(value, record_defaults=defaults)
        all_columns = collections.OrderedDict(zip(headers, columns))
        labels = [all_columns.pop(LABEL_COLUMNS[0]), all_columns.pop(LABEL_COLUMNS[1])]
        label = tf.multiply(labels[0], labels[1])
        features = all_columns
        return features, label

    '''Work Queue Feature'''
    if not stock_tf and workqueue:
        from tensorflow.python.ops.work_queue import WorkQueue
        work_queue = WorkQueue([filename])
        files = work_queue.input_dataset()
    else:
        files = filename
    return (tf.data.TextLineDataset(files)
            .shuffle(buffer_size=10000, seed=seed)
            .repeat(num_epochs)
            .batch(batch_size)
            .map(parse_csv, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            .prefetch(2))

# generate feature columns
def build_feature_columns(stock_tf,
                          emb_fusion,
                          emb_variable=None,
                          emb_var_elimination=None,
                          emb_var_filter=None,
                          adaptive_emb=None,
                          dynamic_emb_var=None):
    user_column = []
    item_column = []
    combo_column = []
    for column_name in ALL_FEATURE_COLUMNS:
        column = tf.feature_column.categorical_column_with_hash_bucket(
            column_name,
            hash_bucket_size=HASH_BUCKET_SIZES[column_name],
            dtype=tf.string)

        if not stock_tf:
            '''Feature Elimination of EmbeddingVariable Feature'''
            if emb_var_elimination == 'gstep':
                # Feature elimination based on global steps
                evict_opt = tf.GlobalStepEvict(steps_to_live=4000)
            elif emb_var_elimination == 'l2':
                # Feature elimination based on l2 weight
                evict_opt = tf.L2WeightEvict(l2_weight_threshold=1.0)
            else:
                evict_opt = None

            '''Feature Filter of EmbeddingVariable Feature'''
            if emb_var_filter == 'cbf':
                # CBF-based feature filter
                filter_option = tf.CBFFilter(filter_freq=3,
                                             max_element_size=2**30,
                                             false_positive_probability=0.01,
                                             counter_type=tf.int64)
            elif emb_var_filter == 'counter':
                # Counter-based feature filter
                filter_option = tf.CounterFilter(filter_freq=3)
            else:
                filter_option = None

            ev_opt = tf.EmbeddingVariableOption(evict_option=evict_opt, filter_option=filter_option)

            if emb_variable:
                '''Embedding Variable Feature'''
                column = tf.feature_column.categorical_column_with_embedding(column_name,
                                                                             dtype=tf.string,
                                                                             ev_option=ev_opt)
            elif adaptive_emb:
                '''                 Adaptive Embedding Feature Part 2 of 2
                Except the following code, a dict, 'adaptive_mask_tensors', is needede as the input of
                'tf.feature_column.input_layer(adaptive_mask_tensors=adaptive_mask_tensors)'.
                For column 'COL_NAME',the value of adaptive_mask_tensors['$COL_NAME'] is an int32
                tensor with shape [batch_size].
                '''
                column = tf.feature_column.categorical_column_with_adaptive_embedding(
                    column_name,
                    hash_bucket_size=HASH_BUCKET_SIZES[column_name],
                    dtype=tf.string,
                    ev_option=ev_opt)
            elif dynamic_emb_var:
                '''Dynamic-dimension Embedding Variable'''
                raise ValueError('Dynamic-dimension Embedding Variable is not enabled in the model')

        if not stock_tf and emb_fusion:
            '''Embedding Fusion Feature'''
            embedding_column = tf.feature_column.embedding_column(column,
                                                                  dimension=16,
                                                                  combiner='mean',
                                                                  do_fusion=emb_fusion)
        else:
            embedding_column = tf.feature_column.embedding_column(column,
                                                       dimension=16,
                                                       combiner='mean')


        if column_name in USER_COLUMN:
            user_column.append(embedding_column)
        elif column_name in ITEM_COLUMN:
            item_column.append(embedding_column)
        elif column_name in COMBO_COLUMN:
            combo_column.append(embedding_column)

    return user_column, item_column, combo_column


def main(stock_tf, tf_config=None, server=None):
   
    # set batch size, eporch & steps
    batch_size = math.ceil(
                  args.batch_size / args.micro_batch
                  ) if args.micro_batch and not stock_tf else args.batch_size
    

    
    # set fixed random seed
    SEED = args.seed
    tf.set_random_seed(SEED)

    # create feature column
    user_column, item_column, combo_column = build_feature_columns(stock_tf,
                                                                   args.emb_fusion,
                                                                   args.ev,
                                                                   args.ev_elimination,
                                                                   args.ev_filter,
                                                                   args.adaptive_emb,
                                                                   args.dynamic_ev)

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
    for fea in HASH_INPUTS:
        final_input[fea] = tf.placeholder(tf.string,[None],name=fea)
    

    # create model
    model = ESMM(final_input,
                 user_column,
                 item_column,
                 combo_column,
                 batch_size=batch_size,
                 optimizer_type=args.optimizer,
                 bf16=args.bf16,
                 stock_tf=stock_tf,
                 adaptive_emb=args.adaptive_emb,
                 learning_rate=args.learning_rate,
                 l2_scale=args.l2_regularization,
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
            inputs = model.feature,
            outputs = {"predict":model.output}
        )


def boolean_string(string):
    low_string = string.lower()
    if low_string not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return low_string == 'true'

def get_arg_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', help='set random seed', type=int, default=2020)
    parser.add_argument('--data_location',
                        help='Full path of train data',
                        default='./data')
    parser.add_argument('--steps',
                        help='set the number of steps on train dataset',
                        type=int,
                        default=0)
    parser.add_argument('--batch_size',
                        help='Batch size to train',
                        type=int,
                        default=512)
    parser.add_argument('--checkpoint',
                        help='path to the checkpoint file (generated during training)',
                        default='./result')
    parser.add_argument('--output_dir',
                        help='Full path to logs & model output directory',
                        default='./result')
    parser.add_argument('--checkpoint_dir',
                        help='Full path to checkpoints output directory')
    parser.add_argument('--learning_rate',
                        help='Learning rate for model',
                        type=float,
                        default=0.1)
    parser.add_argument('--l2_regularization',
                        help='L2 regularization for the model',
                        type=float)
    parser.add_argument('--timeline',
                        help='number of steps on saving timeline',
                        type=int)
    parser.add_argument('--save_steps',
                        help='set the number of steps on saving checkpoints',
                        type=int)
    parser.add_argument('--keep_checkpoint_max',
                        help='Maximum number of recent checkpoint to keep',
                        type=int,
                        default=1)
    parser.add_argument('--bf16',
                        help='enable DeepRec BF16 in deep model',
                        action='store_true')
    parser.add_argument('--no_eval',
                        help='not evaluate trained model by eval dataset',
                        action='store_true')
    parser.add_argument('--protocol',
                        type=str,
                        choices=['grpc', 'grpc++', 'star_server'],
                        default='grpc')
    parser.add_argument('--inter',
                        help='set inter op parallelism threads',
                        type=int,
                        default=0)
    parser.add_argument('--intra',
                        help='set intra op parallelism threads',
                        type=int,
                        default=0)
    parser.add_argument('--input_layer_partitioner',
                        help='slice size of input layer partitioner. units MB',
                        type=int,
                        default=0)
    parser.add_argument('--dense_layer_partitioner',
                        help='slice size of dense layer partitioner. units KB',
                        type=int,
                        default=0)
    parser.add_argument('--tf',
                        help='Use TF 1.15.5 API and disable DeepRec feature to run a baseline.',
                        action='store_true')
    '''Enabled By Default DeepRec optimizations'''
    parser.add_argument('--smartstaged', \
                        help='Whether to enable SmartStage feature of DeepRec',
                        type=boolean_string,
                        default=True)
    parser.add_argument('--emb_fusion', \
                        help='Whether to enable embedding fusion',
                        type=boolean_string,
                        default=True)
    parser.add_argument('--op_fusion', \
                        help='Whether to enable Auto graph fusion feature',
                        type=boolean_string,
                        default=True)
    parser.add_argument('--micro_batch',
                        help='Set number for Auto Micro Batch',
                        type=int,
                        default=0)
    parser.add_argument('--optimizer', type=str,
                        choices=['adam', 'adamasync', 'adagraddecay',
                                 'adagrad', 'gradientdescent'],
                        default='adamasync')
    '''Optional DeepRec optimizations'''
    parser.add_argument('--ev', \
                        help='Whether to enable DeepRec EmbeddingVariable',
                        type=boolean_string,
                        default=False)
    parser.add_argument('--ev_elimination', \
                        help='Feature Elimination of EmbeddingVariable Feature',
                        type=str,
                        choices=[None, 'l2', 'gstep'],
                        default=None)
    parser.add_argument('--ev_filter', \
                        help='Feature Filter of EmbeddingVariable Feature',
                        type=str,
                        choices=[None, 'counter', 'cbf'],
                        default=None)
    parser.add_argument('--adaptive_emb', \
                        help='Whether to enable Adaptive Embedding',
                        type=boolean_string,
                        default=False)
    parser.add_argument('--dynamic_ev', \
                        help='Whether to enable Dynamic-dimension Embedding Variable',
                        type=boolean_string,
                        default=False)
    parser.add_argument('--incremental_ckpt', \
                        help='Set time[in seconds] of save Incremental Checkpoint',
                        type=int,
                        default=None)
    parser.add_argument('--workqueue', \
                        help='Whether to enable WorkQueue',
                        type=boolean_string,
                        default=False)
    return parser

# Parse distributed training configuration and generate cluster information
def generate_cluster_info(TF_CONFIG):
    print(f'Running distributed training with TF_CONFIG: {TF_CONFIG}')

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
        raise ValueError(f'[TF_CONFIG ERROR] Incorrect ps_hosts or incorrect worker_hosts')

    task_config = tf_config.get('task')
    task_type = task_config.get('type')
    task_index = task_config.get('index') + (1 if task_type == 'worker'
                                             and chief_hosts else 0)

    if task_type == 'chief':
        task_type = 'worker'

    is_chief = True if task_index == 0 else False
    cluster = tf.train.ClusterSpec({'ps': ps_hosts, 'worker': worker_hosts})

    if args.protocol in ['grpc++', 'star_server']:
        raise ValueError('Grpc++ and star_server protocols have not been enabled in the model yet')

    server = tf.distribute.Server(cluster,
                                  job_name=task_type,
                                  task_index=task_index,
                                  protocol=args.protocol)
    if task_type == 'ps':
        server.join()
    elif task_type == 'worker':
        tf_config = {'ps_hosts': ps_hosts,
                     'worker_hosts': worker_hosts,
                     'type': task_type,
                     'index': task_index,
                     'is_chief': is_chief}

        tf_device = tf.device(tf.train.replica_device_setter(
                              worker_device=f'/job:worker/task:{task_index}',
                              cluster=cluster))
        return tf_config, server, tf_device
    else:
        raise ValueError(f'[TF_CONFIG ERROR] Task type or index error.')

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
    DNNL_MAX_CPU_ISA: Specify the highest instruction set used by oneDNN (when the version is less than 2.5.0),
        it will be set to AVX512_CORE_AMX to enable Intel CPU's feature.
    '''
    os.environ['START_STATISTIC_STEP'] = '100'
    os.environ['STOP_STATISTIC_STEP'] = '110'
    os.environ['MALLOC_CONF'] = \
        'background_thread:true,metadata_thp:auto,dirty_decay_ms:20000,muzzy_decay_ms:20000'

def check_stock_tf():
    import pkg_resources
    detailed_version = pkg_resources.get_distribution('Tensorflow').version
    return not ('deeprec' in detailed_version)

def check_DeepRec_features():
    return args.smartstaged or args.emb_fusion or args.op_fusion or args.micro_batch or args.bf16 or \
           args.ev or args.adaptive_emb or args.dynamic_ev or (args.optimizer == 'adamasync') or \
           args.incremental_ckpt  or args.workqueue

if __name__ == '__main__':
    parser = get_arg_parser()
    args = parser.parse_args()

    stock_tf = args.tf
    if not stock_tf and check_stock_tf() and check_DeepRec_features():
        raise ValueError('Stock Tensorflow does not support DeepRec features. '
                         'For Stock Tensorflow run the script with `--tf` argument.')

    if not stock_tf:
        set_env_for_DeepRec()

    TF_CONFIG = os.getenv('TF_CONFIG')
    if not TF_CONFIG:
        print('Running stand-alone mode training')
        main(stock_tf)
    else:
        tf_config, server, tf_device = generate_cluster_info(TF_CONFIG)
        main(stock_tf, tf_config, server)
