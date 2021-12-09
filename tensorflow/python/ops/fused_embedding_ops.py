from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import gen_fused_embedding_ops
from tensorflow.python.ops.gen_fused_embedding_ops import fused_embedding_sparse_pre_look_up
from tensorflow.python.ops.gen_fused_embedding_ops import fused_embedding_sparse_post_look_up
from tensorflow.python.ops.gen_fused_embedding_ops import fused_embedding_sparse_post_look_up_grad
from tensorflow.python.util.tf_export import tf_export


@tf_export(v1=["nn.fused_embedding_lookup_sparse"])
def fused_embedding_lookup_sparse(params,
                                  sp_ids,
                                  partition_strategy=None,
                                  name=None,
                                  combiner=None,
                                  max_norm=None,
                                  blocknums=None):
  valid_partition_strategy = ['div']
  if partition_strategy not in valid_partition_strategy:
    raise ValueError("{} is not supported yet. Currently only support {}".format(
      partition_strategy, valid_partition_strategy))

  if blocknums is not None:
    raise ValueError("Using blocknums for DynamicEmbeddingVariable is not supported yet")

  partition_nums = len(params)
  partition_shapes = [w.shape for w in params]
  with ops.name_scope(name, "fused_embedding_lookup_sparse",
                      params + [sp_ids]) as name:

    partitioned_values, partitioned_indices = fused_embedding_sparse_pre_look_up(
      partition_shapes=partition_shapes,
      sp_values=sp_ids.values,
      sp_indices=sp_ids.indices,
    )
    emb_shards = []
    for i in range(partition_nums):
      param = params[i]
      sub_partition_values = partitioned_values[i]
      with ops.colocate_with(param):
        shard = array_ops.gather(param, sub_partition_values)
        emb_shards.append(shard)
    emb_vectors, _ = fused_embedding_sparse_post_look_up(
      emb_shards=emb_shards, partitioned_indices=partitioned_indices,
      sp_dense_shape=sp_ids.dense_shape,
      partitioned_values=partitioned_values,
      combiner=combiner, max_norm=max_norm
    )
  return emb_vectors

@ops.RegisterGradient("FusedEmbeddingSparsePostLookUp")
def fused_embedding_sparse_post_look_up_grad(op, top_grad_emb_vec, _):
  num_partitions = op.get_attr("num_partitions")
  grad_shards = gen_fused_embedding_ops.fused_embedding_sparse_post_look_up_grad(
    top_grad=top_grad_emb_vec, emb_shards=[op.inputs[i] for i in range(0, num_partitions)],
    partitioned_indices=[op.inputs[i] for i in range(num_partitions, 2 * num_partitions)],
    feature_nums=op.outputs[1], combiner=op.get_attr("combiner"),
    max_norm=op.get_attr("max_norm")
  )
  return grad_shards + [None for _ in range(0, 2 * num_partitions + 1)]
