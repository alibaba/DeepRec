# Embedding Variable
## 功能介绍
Tensorflow对Embedding的支持是通过Variable实现的。其中，用于存储Embedding的Variable大小为[vocabulary_size, embedding_dimension]，需要事先确定。在大规模稀疏特征的场景中，会有以下弊端：

1. vocabulary_size一般由id空间决定，在线学习场景中，新id不断加入导致vocabulary_size难估计；
1. id一般为string类型且规模庞大，进行Embedding之前需要先Hash到vocabulary_size范围内：
   - vocabulary_size过小，会导致Hash冲撞率增加，不同特征可能查到相同的Embedding，即特征减少；
   - vocabulary_size过大，会导致Variable内部存储了永远不会被到查到的Embedding，即内存冗余；
3. Embedding变量过大是模型变大的主要原因，即便通过正则手段使得某些特征的Embedding对整个模型效果影响不大，也无法把这些Emebdding从模型中去掉；



为解决上述问题，DeepRec新设计了一套支持动态Embedding语义的EmbeddingVariable，在特征无损训练的同时以最经济的方式使用内存资源，使得超大规模特征的模型更容易增量上线。
​

DeepRec的EmbeddingVariable经过了若干版本的迭代，在支持特征淘汰、特征准入、特征统计等基础功能的基础之上，进行了包括稀疏特征存储结构优化，无锁化hashmap，混合存储架构（gpu, mem, ssd），Embedding GPU算子支持、HashTable GPU存储等等的支持。当前TensorFlow recommenders-addons中提供了基础Embedding Variable功能的支持（[https://github.com/tensorflow/recommenders-addons/pull/16](https://github.com/tensorflow/recommenders-addons/pull/16)）。


## 用户接口
我们向用户提供两个层面的API，分别为embedding variable和feature_column
下面API是创建一个新的EmbeddingVariable变量
```python
def get_embedding_variable(name,
                           embedding_dim,
                           key_dtype=dtypes.int64,
                           value_dtype=None,
                           initializer=None,
                           regularizer=None,
                           trainable=True,
                           collections=None,
                           caching_device=None,
                           partitioner=None,
                           validate_shape=True,
                           custom_getter=None,
                           constraint=None,
                           init_data_source=None,
                           ev_option = tf.EmbeddingVariableOption()):
```

- `name`: EmbeddingVariable名称
- `embedding_dim`: embedding之后的维度, eg: 8, 64
- `key_dtype`: lookup时key的类型，默认值为int64，允许的值为int64和int32
- `value_dtype`: embedding vector的类型，目前仅限于float
- `initializer`: embedding vector初始化值，可以传入的参数为Initializer或list
- `trainable`: 是否被添加到GraphKeys.TRAINABLE_VARIABLES的collection
- `partitioner`: 分区函数
- `ev_opt`: 一些基于EV的功能参数配置

通过`tf.feature_column`使用Embedding Variable功能的API：
```python
def categorical_column_with_embedding(key,
                                  dtype=dtypes.string,
                                  partition_num=None,
                                  ev_option=tf.EmbeddingVariableOption()
                                  )
```
另外也可以通过`tf.contrib.feature_column`使用Embedding Variable功能
```python
def sparse_column_with_embedding(column_name,
                                 dtype=dtypes.string,
                                 partition_num=None,
                                 steps_to_live=None,
                                 init_data_source=None,
                                 ht_partition_num=1000,
                                 evconfig = variables.EmbeddingVariableOption()
```
## 功能使用示例
使用`get_embedding_variable`接口
```python
import tensorflow as tf

var = tf.get_embedding_variable("var_0",
                                embedding_dim=3,
                                initializer=tf.ones_initializer(tf.float32),
                                partitioner=tf.fixed_size_partitioner(num_shards=4))

shape = [var1.total_count() for var1 in var]

emb = tf.nn.embedding_lookup(var, tf.cast([0,1,2,5,6,7], tf.int64))
fun = tf.multiply(emb, 2.0, name='multiply')
loss = tf.reduce_sum(fun, name='reduce_sum')
opt = tf.train.AdagradOptimizer(0.1)

g_v = opt.compute_gradients(loss)
train_op = opt.apply_gradients(g_v)

init = tf.global_variables_initializer()

sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
with tf.Session(config=sess_config) as sess:
  sess.run([init])
  print(sess.run([emb, train_op, loss]))
  print(sess.run([emb, train_op, loss]))
  print(sess.run([emb, train_op, loss]))
  print(sess.run([shape]))
```
使用`categorical_column_with_embedding`接口：
```python
import tensorflow as tf
from tensorflow.python.framework import ops


columns = tf.feature_column.categorical_column_with_embedding("col_emb", dtype=tf.dtypes.int64)
W = tf.feature_column.embedding_column(categorical_column=columns,
            dimension=3,
            initializer=tf.ones_initializer(tf.dtypes.float32))

ids={}
ids["col_emb"] = tf.SparseTensor(indices=[[0,0],[1,1],[2,2],[3,3],[4,4]], values=tf.cast([1,2,3,4,5], tf.dtypes.int64), dense_shape=[5, 5])

emb = tf.feature_column.input_layer(ids, [W])
fun = tf.multiply(emb, 2.0, name='multiply')
loss = tf.reduce_sum(fun, name='reduce_sum')
opt = tf.train.FtrlOptimizer(0.1, l1_regularization_strength=2.0, l2_regularization_strength=0.00001)
g_v = opt.compute_gradients(loss)
train_op = opt.apply_gradients(g_v)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print("init global done")
    print(sess.run([emb, train_op,loss]))
    print(sess.run([emb, train_op,loss]))
    print(sess.run([emb, train_op,loss]))
```
使用`sparse_column_with_embedding`接口：
```python
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.contrib.layers.python.layers import feature_column_ops
from tensorflow.contrib.layers.python.layers import feature_column


columns = feature_column.sparse_column_with_embedding(column_name="col_emb", dtype=tf.dtypes.int64)
W = feature_column.embedding_column(sparse_id_column=columns,
            dimension=3,
            initializer=tf.ones_initializer(tf.dtypes.float32))

ids={}
ids["col_emb"] = tf.SparseTensor(indices=[[0,0],[1,1],[2,2],[3,3],[4,4]], values=tf.cast([1,2,3,4,5], tf.dtypes.int64), dense_shape=[5, 5])

emb = feature_column_ops.input_from_feature_columns(columns_to_tensors=ids, feature_columns=[W])
fun = tf.multiply(emb, 2.0, name='multiply')
loss = tf.reduce_sum(fun, name='reduce_sum')
opt = tf.train.FtrlOptimizer(0.1, l1_regularization_strength=2.0, l2_regularization_strength=0.00001)
g_v = opt.compute_gradients(loss)
train_op = opt.apply_gradients(g_v)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print("init global done")
    print(sess.run([emb, train_op,loss]))
    print(sess.run([emb, train_op,loss]))
    print(sess.run([emb, train_op,loss]))
```
使用`sequence_categorical_column_with_embedding`接口：
```python
import tensorflow as tf
from tensorflow.python.feature_column import sequence_feature_column


columns = sequence_feature_column.sequence_categorical_column_with_embedding(key="col_emb", dtype=tf.dtypes.int32)
W = tf.feature_column.embedding_column(categorical_column=columns,
            dimension=3,
            initializer=tf.ones_initializer(tf.dtypes.float32))

ids={}
ids["col_emb"] = tf.SparseTensor(indices=[[0,0],[0,1],[1,1],[2,2],[3,3],[4,4]], \
                                 values=tf.cast([1,3,2,3,4,5], tf.dtypes.int64), 
                                 dense_shape=[5, 5])

# 目前还不支持使用SequenceFeatures接口创建
emb, length = tf.contrib.feature_column.sequence_input_layer(ids, [W])
fun = tf.multiply(emb, 2.0, name='multiply')
loss = tf.reduce_sum(fun, name='reduce_sum')
opt = tf.train.FtrlOptimizer(0.1, l1_regularization_strength=2.0, l2_regularization_strength=0.00001)
g_v = opt.compute_gradients(loss)
train_op = opt.apply_gradients(g_v)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print("init global done")
    print(sess.run([emb, train_op,loss]))
    print(sess.run([emb, train_op,loss]))
    print(sess.run([emb, train_op,loss]))
```
使用`weighted_categorical_column`接口：
```python
import tensorflow as tf


categorical_column = tf.feature_column.categorical_column_with_embedding("col_emb", dtype=tf.dtypes.int64)

ids={}
ids["col_emb"] = tf.SparseTensor(indices=[[0,0],[0,1],[1,1],[2,2],[3,3],[4,3],[4,4]], \
                        values=tf.cast([1,3,2,3,4,5,3], tf.dtypes.int64), dense_shape=[5, 5])    
ids['weight'] = [[2.0],[5.0],[4.0],[8.0],[3.0],[1.0],[2.5]]

columns = tf.feature_column.weighted_categorical_column(categorical_column, 'weight')

W = tf.feature_column.embedding_column(categorical_column=columns,
            dimension=3,
            initializer=tf.ones_initializer(tf.dtypes.float32))
emb = tf.feature_column.input_layer(ids, [W])
fun = tf.multiply(emb, 2.0, name='multiply')
loss = tf.reduce_sum(fun, name='reduce_sum')
opt = tf.train.FtrlOptimizer(0.1, l1_regularization_strength=2.0, l2_regularization_strength=0.00001)
g_v = opt.compute_gradients(loss)
train_op = opt.apply_gradients(g_v)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print("init global done")
    print(sess.run([emb, train_op,loss]))
    print(sess.run([emb, train_op,loss]))
    print(sess.run([emb, train_op,loss]))
```

## EV Initializer 
在尝试使用EV训练WDL模型、DIN模型和DIEN模型时发现如果不使用glorot uniform initializer就会比较明显的影响模型训练的效果（例如使用ones_initializer会导致训练AUC下降以及AUC增长速度变慢，使用truncated intializer会导致训练无法收敛）。但是由于EV的shape是动态的，因此无法支持glorot uniform initializer等需要配置静态shape的initializer。动态的Embedding在PS-Worker架构中广泛存在，在别的框架中有以下几种解决方法：

- PAI-TF140 & 1120: 固定每次Gather的时候生成的default value的大小以支持静态的variable。这样的缺点是为了更大的default value matrix以获得更好的训练效果时会带来额外的generate default value的开销。
- XDL：XDL的实现是在initialize的时候生成一个固定大小的default value matrix，每有一个新的特征来就获取一个default value，当这个matrix被消耗完后，再重新生成一个matrix。这样的方法好处在于可以确保每一个特征都会有唯一的default value。缺点在于首先重新生成default value matrix的过程需要加锁，会影响性能；其次，他们的generate方法是在C++构造临时一个context然后调用initializer的Op，这会导致runtime缺少graph信息，无法设置随机数的seed，在大部分模型的训练中，是需要设置seed的。
- Abacus：给每一个feature单独生成一个default value的方式来获得静态shape，这种方法性能会比较好，但是由于生成的shape太小，可能不符合分布。同时当用户固定seed的时候，每个特征的default value都会是固定的。

综上所示，我们提供了EV initializer，EV Initializer会在Initialize的时候生成一个固定shape的default value matrix，之后所有特征会根据id mod default value dim来从matrix中获取一个default value。这样的方法首先避免了加锁以及多次生成对性能的影响，其次也可以使得default value符合用户想要的分布，最后还可以通过设置seed固定default value。
### 使用方法
用户可以通过下面的方法配置EV Initializer

```python
init_opt = tf.InitializerOption(initializer=tf.glorot_uniform_initializer,
                                default_value_dim = 10000)
ev_opt = tf.EmbeddingVariableOption(init_option=init)

#通过底层API设置
emb_var = tf.get_embedding_variable("var", embedding_dim = 16, ev_option=ev_opt)

通过feature column API设置
from tensorflow.contrib.layers.python.layers import feature_column
emb_var = feature_column.sparse_column_wth_embedding("var", ev_option=ev_opt)

emb_var = tf.feature_column.categorical_column_with_embedding("var", ev_option=ev_opt)
```
下面是EV Initializer的接口定义：
```python
@tf_export(v1=["InitializerOption"])
class InitializerOption(object):
  def __init__(self,
               initializer = None,
               default_value_dim = 4096):
    self.initializer = initializer
    self.default_value_dim  = default_value_dim
    if default_value_dim <=0:
      print("default value dim must larger than 1, the default value dim is set to default 4096.")
      default_value_dim = 4096
```
下面是参数的解释

- `initializer`：Embedding Variable使用的Initializer，如果不配置的话则会被设置EV默认设置为truncated normal initializer。
- `default value dim`：生成的default value的数量，设置可以参考hash bucket size或是特征的数量，默认是4096。



