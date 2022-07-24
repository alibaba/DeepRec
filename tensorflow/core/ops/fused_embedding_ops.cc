#include <stdio.h>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

REGISTER_OP("FusedEmbeddingSparsePreLookUp")
    .Attr("num_partitions: int >= 1 = 1")
    .Attr("partition_axis: int >= 0 = 0")  // for now only support = 0,
                                           // will consider support = 1
                                           // if necessary
    .Attr("fill_empty_row: bool = false")
    .Attr("prune_invalid_id: bool = false")
    .Attr("default_id: int = -1")
    .Attr("partition_strategy: {'div','mod'} = 'div'")
    .Input("partition_shapes: num_partitions * int64")
    .Input("sp_values: int64")
    .Input("sp_indices: int64")
    .Input("sp_dense_shape: int64")
    .Output("partitioned_values: num_partitions * int64")
    .Output("partitioned_indices: num_partitions * int64")
    .Output("row_empty_and_invalid_flags: int32")
    .SetShapeFn([](InferenceContext* ctx) {
      int num_partitions;
      TF_RETURN_IF_ERROR(ctx->GetAttr("num_partitions", &num_partitions));
      int partition_axis;
      TF_RETURN_IF_ERROR(ctx->GetAttr("partition_axis", &partition_axis));

      ShapeHandle unused;
      // sp_values
      TF_RETURN_IF_ERROR(ctx->WithRank(ctx->input(num_partitions), 1, &unused));
      // sp_indices
      TF_RETURN_IF_ERROR(
          ctx->WithRank(ctx->input(num_partitions + 1), 2, &unused));
      DimensionHandle unused_dim;
      TF_RETURN_IF_ERROR(ctx->WithValue(ctx->Dim(unused, 1), 2, &unused_dim));
      // sp_dense_shape
      TF_RETURN_IF_ERROR(
          ctx->WithRank(ctx->input(num_partitions + 2), 1, &unused));

      // partition_shapes
      for (int i = 0; i < num_partitions; i++) {
        ShapeHandle partition_shape;
        TF_RETURN_IF_ERROR(ctx->WithRank(ctx->input(i), 1, &partition_shape));
        TF_RETURN_IF_ERROR(
            ctx->WithValue(ctx->NumElements(partition_shape), 2, &unused_dim));

        ShapeHandle values_result_shape, indices_result_shape;
        if (int(partition_axis) == 0) {
          values_result_shape = ctx->MakeShape({ctx->UnknownDim()});
          indices_result_shape = ctx->MakeShape({ctx->UnknownDim(), 2});
        } else {
          return errors::InvalidArgument("partition_axis > 0 not implemented!");
        }
        ctx->set_output(i, values_result_shape);
        ctx->set_output(i + num_partitions, indices_result_shape);
      }
      ctx->set_output(2 * num_partitions, ctx->MakeShape({ctx->UnknownDim()}));

      return Status::OK();
    });
//     .Doc(R"doc(
// A fused embedding op, usually using for partitioned and distriuted embedding
// variables. FusedEmbeddingSparsePreLookUp, FusedEmbeddingSparsePostLookUp
// should be used together. This op will first read the partition pattern of
// embedding variables through partition_shapes, then sort, re-calculate and
// assign the embedding indices to the corresponding partition. Several Gather
// ops usually should be appended after this op to gather embedding shards from
// multiple partitioned embedding variables. This op has no gradient function.
//     )doc");

REGISTER_OP("FusedEmbeddingSparsePostLookUp")
    .Attr("T : {float32}")
    .Attr("num_partitions: int >= 1 = 1")
    .Attr("default_id: int = -1")
    .Attr("partition_axis: int >= 0 = 0")  // for now only support = 0,
                                           // will consider support = 1
                                           // if necessary
    .Attr("combiner: {'sqrtn', 'mean', 'sum'}")
    .Attr("max_norm: float = -1.0")
    .Input("emb_shards: num_partitions * T")
    .Input("partitioned_indices: num_partitions * int64")
    .Input("sp_dense_shape: int64")
    .Input("row_empty_and_invalid_flags: int32")
    .Input(
        "partitioned_values: num_partitions * int64")  // only for backward use.
                                                       // actually directly port
                                                       // to python grad op
                                                       // output
    .Output("emb_vectors: T")
    .Output("feature_nums: int32")
    .SetShapeFn([](InferenceContext* ctx) {
      int num_partitions;
      TF_RETURN_IF_ERROR(ctx->GetAttr("num_partitions", &num_partitions));

      ShapeHandle first_emb_shard_shape;
      TF_RETURN_IF_ERROR(
          ctx->WithRank(ctx->input(0), 2, &first_emb_shard_shape));

      ShapeHandle unused;
      for (int i = 0; i < num_partitions; i++) {
        // emb_shards
        TF_RETURN_IF_ERROR(ctx->WithRank(ctx->input(i), 2, &unused));
        // partitioned_indices
        TF_RETURN_IF_ERROR(
            ctx->WithRank(ctx->input(i + num_partitions), 2, &unused));
        DimensionHandle unused_dim;
        TF_RETURN_IF_ERROR(ctx->WithValue(ctx->Dim(unused, 1), 2, &unused_dim));
      }
      // sp_dense_shape
      TF_RETURN_IF_ERROR(
          ctx->WithRank(ctx->input(2 * num_partitions), 1, &unused));
      // row_empty_and_invalid_flags
      TF_RETURN_IF_ERROR(
          ctx->WithRank(ctx->input(2 * num_partitions + 1), 1, &unused));

      DimensionHandle emb_vec_size_dim = ctx->Dim(first_emb_shard_shape, 1);
      ctx->set_output(0, ctx->MakeShape({ctx->UnknownDim(), emb_vec_size_dim}));
      ctx->set_output(1, ctx->MakeShape({ctx->UnknownDim()}));
      return Status::OK();
    });

//     .Doc(R"doc(
// A fused embedding op, usually using for partitioned and distriuted embedding
// variables. FusedEmbeddingSparsePreLookUp, FusedEmbeddingSparsePostLookUp
// should be used together. There should be several Gather ops before this op.
// The Gather ops gather embedding shards from embedding variable and this op
// glue them together, then apply combiner and max_morm according to embedding
// indices.
//     )doc");

REGISTER_OP("FusedEmbeddingSparsePostLookUpGrad")
    .Attr("T : {float32}")
    .Attr("num_partitions: int >= 1 = 1")
    .Attr("partition_axis: int >= 0 = 0")  // for now only support = 0,
                                           // will consider support = 1
                                           // if necessary
    .Attr("default_id: int = -1")
    .Attr("combiner: {'sqrtn', 'mean', 'sum'}")
    .Attr("max_norm: float = -1.0")
    .Input("top_grad: T")
    .Input("emb_shards: num_partitions * T")
    .Input("partitioned_indices: num_partitions * int64")
    .Input("feature_nums: int32")
    .Input("row_empty_and_invalid_flags: int32")
    .Output("grad_shards: num_partitions * T")
    .SetShapeFn([](InferenceContext* ctx) {
      int num_partitions;
      TF_RETURN_IF_ERROR(ctx->GetAttr("num_partitions", &num_partitions));

      ShapeHandle unused;
      ShapeHandle top_grad_shape;

      // top_grad
      TF_RETURN_IF_ERROR(ctx->WithRank(ctx->input(0), 2, &top_grad_shape));
      // emb_shards
      for (int i = 1; i < num_partitions + 1; i++) {
        TF_RETURN_IF_ERROR(ctx->WithRank(ctx->input(i), 2, &unused));
      }
      // partitioned_indices
      for (int i = num_partitions + 1; i < 2 * num_partitions + 1; i++) {
        TF_RETURN_IF_ERROR(ctx->WithRank(ctx->input(i), 2, &unused));
        DimensionHandle unused_dim;
        TF_RETURN_IF_ERROR(ctx->WithValue(ctx->Dim(unused, 1), 2, &unused_dim));
      }
      // feature_nums
      TF_RETURN_IF_ERROR(
          ctx->WithRank(ctx->input(2 * num_partitions + 1), 1, &unused));
      // row_empty_and_invalid_flags
      TF_RETURN_IF_ERROR(
          ctx->WithRank(ctx->input(2 * num_partitions + 2), 1, &unused));

      DimensionHandle emb_vec_size_dim = ctx->Dim(top_grad_shape, 1);

      ShapeHandle output_shape =
          ctx->MakeShape({ctx->UnknownDim(), emb_vec_size_dim});
      for (int i = 0; i < num_partitions; i++) {
        ctx->set_output(i, output_shape);
      }
      return Status::OK();
    });

//     .Doc(R"doc(
// Calculate gradient of FusedEmbeddingSparsePostLookUp
//     )doc");

REGISTER_OP("FusedSafeEmbeddingLookupSparseLocal")
    .Input("weight: T_weight")
    .Input("id_input: T_id")
    .Input("dense_shape: T_shape")
    .Input("indice: T_shape")
    .Input("weight_input: T_id")
    .Output("embedded: T")
    .Attr("combiner: {'sqrtn', 'mean', 'sum'} = 'mean'")
    .Attr("prune: bool = true")
    .Attr("max_norm: float = -1.0")
    .Attr("default_id: int = -1")
    .Attr("partition_strategy: {'div','mod'} = 'div'")
    .Attr("T_id: {int64, int32}")
    .Attr("T_shape: {int64, int32}")
    .Attr("T_weight: {float, resource}")
    .Attr("T: {float} = DT_FLOAT")
    .SetShapeFn([](InferenceContext* ctx) {
      ShapeHandle temp;
      TF_RETURN_IF_ERROR(ctx->WithRank(ctx->input(1), 1, &temp));
      TF_RETURN_IF_ERROR(ctx->WithRank(ctx->input(3), 2, &temp));
      TF_RETURN_IF_ERROR(ctx->WithRank(ctx->input(2), 1, &temp));
      ShapeHandle emb_var_shape;
      TF_RETURN_IF_ERROR(ctx->WithRank(ctx->input(0), 2, &emb_var_shape));

      DimensionHandle emb_vec_size_dim = ctx->Dim(emb_var_shape, 1);
      DimensionHandle batch_dim = ctx->UnknownDim();

      ShapeHandle output_shape = ctx->MakeShape({batch_dim, emb_vec_size_dim});
      ctx->set_output(0, output_shape);

      return Status::OK();
    });

REGISTER_OP("FusedSafeEmbeddingLookupSparseLocalGrad")
    .Input("gradients: T")
    .Input("input: Tinput")
    .Input("indices: Tindices")
    .Input("dense_shape: Tdense_shape")
    .Output("output: T")
    .Output("unique_value: Tinput")
    .Attr("T: {float}")
    .Attr("Tinput: {int64}")
    .Attr("Tindices: {int64, int32}")
    .Attr("Tdense_shape: {int64, int32}")
    .Attr("combiner: {'sqrtn', 'mean', 'sum'} = 'mean'")
    .SetShapeFn([](InferenceContext* ctx) {
      ShapeHandle emb_var_shape;
      TF_RETURN_IF_ERROR(ctx->WithRank(ctx->input(0), 2, &emb_var_shape));

      DimensionHandle emb_vec_size_dim = ctx->Dim(emb_var_shape, 1);
      DimensionHandle unique_dim = ctx->UnknownDim();

      ShapeHandle output_shape = ctx->MakeShape({unique_dim, emb_vec_size_dim});
      ctx->set_output(0, output_shape);

      ShapeHandle unique_value_shape = ctx->MakeShape({unique_dim});
      ctx->set_output(1, unique_value_shape);

      return Status::OK();
    });

REGISTER_OP("PruneInvalidAndFillEmptyRows")
    .Attr("fill_empty_row: bool = false")
    .Attr("prune: bool = false")
    .Attr("default_id: int = -1")
    .Attr("use_sparse_weights: bool = false")
    .Attr("prune_sparse_weights: bool = false")
    .Attr("default_weight: float = 1.0")
    .Input("sp_values: int64")
    .Input("sp_indices: int64")
    .Input("sp_dense_shape: int64")
    .Input("sp_weights_values: float")
    .Output("sp_values_out: int64")
    .Output("sp_indices_out: int64")
    .Output("sp_weights_values_out: float")
    .Output("is_row_empty: bool")
    .SetShapeFn([](InferenceContext* ctx) {
      ShapeHandle unused;
      std::vector<ShapeHandle> unused_list;

      ctx->input("sp_values", &unused_list);
      TF_RETURN_IF_ERROR(ctx->WithRank(unused_list[0], 1, &unused));

      ctx->input("sp_indices", &unused_list);
      TF_RETURN_IF_ERROR(ctx->WithRank(unused_list[0], 2, &unused));
      DimensionHandle unused_dim;
      TF_RETURN_IF_ERROR(ctx->WithValue(ctx->Dim(unused, 1), 2, &unused_dim));

      ctx->input("sp_dense_shape", &unused_list);
      TF_RETURN_IF_ERROR(ctx->WithRank(unused_list[0], 1, &unused));

      ctx->input("sp_weights_values", &unused_list);
      TF_RETURN_IF_ERROR(ctx->WithRank(unused_list[0], 1, &unused));

      unused_list.clear();
      unused_list.resize(1);
      unused_list[0] = ctx->MakeShape({ctx->UnknownDim()});
      ctx->set_output("sp_values_out", unused_list);
      ctx->set_output("is_row_empty", unused_list);
      unused_list[0] = ctx->MakeShape({ctx->UnknownDim(), 2});
      ctx->set_output("sp_indices_out", unused_list);

      return Status::OK();
    });

REGISTER_OP("UniqueWithCountsV3")
    .Attr("KeyType: {int32, int64} = DT_INT64")
    .Attr("CounterType: {int32, int64} = DT_INT32")
    .Input("input: KeyType")
    .Output("unique_keys: KeyType")
    .Output("unique_idxs: CounterType")
    .Output("unique_counts: CounterType")

    .SetShapeFn([](InferenceContext* ctx) {
      ShapeHandle unused;
      std::vector<ShapeHandle> unused_list;

      ctx->input("input", &unused_list);
      TF_RETURN_IF_ERROR(ctx->WithRank(unused_list[0], 1, &unused));

      unused_list.clear();
      unused_list.resize(1);
      unused_list[0] = ctx->MakeShape({ctx->UnknownDim()});

      ctx->set_output("unique_keys", unused_list);
      ctx->set_output("unique_idxs", unused_list);
      ctx->set_output("unique_counts", unused_list);

      return Status::OK();
    });

REGISTER_OP("PartitionWithPermutation")
    .Attr("num_partitions: int >= 2 = 2")
    .Attr("partition_axis: int >= 0 = 0")
    .Attr("partition_strategy : {'div', 'mod', 'mod_ev'}")
    .Input("input: int64")
    .Input("partition_shapes: num_partitions * int64")
    .Output("partitioned_values: num_partitions * int64")
    .Output("partition_permutation: int32")
    .SetShapeFn([](InferenceContext* ctx) {
      ShapeHandle unused;
      std::vector<ShapeHandle> unused_list;
      DimensionHandle unused_dim;

      int num_partitions;
      TF_RETURN_IF_ERROR(ctx->GetAttr("num_partitions", &num_partitions));
      int partition_axis;
      TF_RETURN_IF_ERROR(ctx->GetAttr("partition_axis", &partition_axis));

      ctx->input("input", &unused_list);
      TF_RETURN_IF_ERROR(ctx->WithRank(unused_list[0], 1, &unused));

      ctx->input("partition_shapes", &unused_list);
      for (int i = 0; i < num_partitions; i++) {
        TF_RETURN_IF_ERROR(ctx->WithRank(unused_list[i], 1, &unused));
        TF_RETURN_IF_ERROR(
            ctx->WithValue(ctx->NumElements(unused), 2, &unused_dim));
      }

      unused_list.clear();
      unused_list.resize(num_partitions);
      for (int i = 0; i < num_partitions; i++) {
        unused_list[i] = ctx->MakeShape({ctx->UnknownDim()});
      }
      ctx->set_output("partitioned_values", unused_list);

      unused_list.clear();
      unused_list.resize(1);
      unused_list[0] = ctx->MakeShape({ctx->UnknownDim(), 2});
      ctx->set_output("partition_permutation", unused_list);

      return Status::OK();
    });

REGISTER_OP("FusedEmbeddingSparsePostLookUpV2")
    .Attr("T : {float32}")
    .Attr("num_partitions: int >= 1 = 1")
    .Attr("fill_empty_row: bool = false")
    .Attr("default_id: int = -1")
    .Attr("partition_axis: int >= 0 = 0")
    .Attr("combiner: {'sqrtn', 'mean', 'sum'}")
    .Attr("max_norm: float = -1.0")
    .Attr("use_sparse_weights: bool = false")
    .Input("emb_shards: num_partitions * T")
    .Input("partition_permutation: int32")
    .Input("sp_dense_shape: int64")
    .Input("indices_before_unique: int64")
    .Input("is_row_empty: bool")
    .Input("unique_idxs: int32")
    .Input("sp_weights_values: float")
    .Output("emb_vectors: T")
    .Output("feature_nums: int32")
    .Output("emb_shard_ptrs: uint64")
    .SetShapeFn([](InferenceContext* ctx) {
      int num_partitions;
      TF_RETURN_IF_ERROR(ctx->GetAttr("num_partitions", &num_partitions));

      std::vector<ShapeHandle> unused_list;
      ShapeHandle unused;
      DimensionHandle unused_dim;

      // emb_shards
      ctx->input("emb_shards", &unused_list);
      ShapeHandle first_emb_shard_shape;
      TF_RETURN_IF_ERROR(
          ctx->WithRank(unused_list[0], 2, &first_emb_shard_shape));
      DimensionHandle emb_vec_size_dim = ctx->Dim(first_emb_shard_shape, 1);
      int64 emb_vec_size = ctx->Value(emb_vec_size_dim);

      for (int i = 0; i < num_partitions; i++) {
        TF_RETURN_IF_ERROR(ctx->WithRank(unused_list[i], 2, &unused));
        TF_RETURN_IF_ERROR(
            ctx->WithValue(ctx->Dim(unused, 1), emb_vec_size, &unused_dim));
      }

      // partition_permutation
      ctx->input("partition_permutation", &unused_list);
      TF_RETURN_IF_ERROR(ctx->WithRank(unused_list[0], 2, &unused));

      // sp_dense_shape
      ctx->input("sp_dense_shape", &unused_list);
      TF_RETURN_IF_ERROR(ctx->WithRank(unused_list[0], 1, &unused));

      // indices_before_unique
      ctx->input("indices_before_unique", &unused_list);
      TF_RETURN_IF_ERROR(ctx->WithRank(unused_list[0], 2, &unused));

      // is_row_empty
      ctx->input("is_row_empty", &unused_list);
      TF_RETURN_IF_ERROR(ctx->WithRank(unused_list[0], 1, &unused));

      // unique_counts
      ctx->input("unique_idxs", &unused_list);
      TF_RETURN_IF_ERROR(ctx->WithRank(unused_list[0], 1, &unused));

      ctx->input("sp_weights_values", &unused_list);
      TF_RETURN_IF_ERROR(ctx->WithRank(unused_list[0], 1, &unused));

      // emb_vectors
      unused_list.clear();
      unused_list.resize(1);
      unused_list[0] = ctx->MakeShape({ctx->UnknownDim(), emb_vec_size_dim});
      ctx->set_output("emb_vectors", unused_list);

      // feature_nums
      unused_list[0] = ctx->MakeShape({ctx->UnknownDim()});
      ctx->set_output("feature_nums", unused_list);

      // emb_shard_ptrs
      unused_list[0] = ctx->MakeShape({num_partitions});
      ctx->set_output("emb_shard_ptrs", unused_list);
      return Status::OK();
    });

//     .Doc(R"doc(
// A fused embedding op, usually using for partitioned and distriuted embedding
// variables. FusedEmbeddingSparse`LookUp, FusedEmbeddingSparsePostLookUp
// should be used together. There should be several Gather ops before this op.
// The Gather ops gather embedding shards from embedding variable and this op
// glue them together, then apply combiner and max_morm according to embedding
// indices.
//     )doc");

REGISTER_OP("FusedEmbeddingSparsePostLookUpV2Grad")
    .Attr("T : {float32}")
    .Attr("num_partitions: int >= 1 = 1")
    .Attr("fill_empty_row: bool = false")
    .Attr("partition_axis: int >= 0 = 0")  // for now only support = 0,
                                           // will consider support = 1
                                           // if necessary
    .Attr("default_id: int = -1")
    .Attr("combiner: {'sqrtn', 'mean', 'sum'}")
    .Attr("max_norm: float = -1.0")
    .Attr("use_sparse_weights: bool = false")
    .Input("top_grad: T")
    .Input("emb_shards: num_partitions * T")
    .Input("emb_shard_ptrs: uint64")
    .Input("partition_permutation: int32")
    .Input("feature_nums: int32")
    .Input("indices_before_unique: int64")
    .Input("unique_idxs: int32")
    .Input("is_row_empty: bool")
    .Input("sp_weights_values: float")
    .Output("grad_shards: num_partitions * T")
    .SetShapeFn([](InferenceContext* ctx) {
      int num_partitions;
      TF_RETURN_IF_ERROR(ctx->GetAttr("num_partitions", &num_partitions));

      std::vector<ShapeHandle> unused_list;
      ShapeHandle unused;
      DimensionHandle unused_dim;

      ctx->input("top_grad", &unused_list);
      // top_grad
      TF_RETURN_IF_ERROR(ctx->WithRank(unused_list[0], 2, &unused));
      DimensionHandle emb_vec_size_dim = ctx->Dim(unused, 1);

      // emb_shards
      ctx->input("emb_shards", &unused_list);
      for (int i = 0; i < num_partitions; i++) {
        TF_RETURN_IF_ERROR(ctx->WithRank(unused_list[i], 2, &unused));
      }

      // emb_shard_ptrs
      ctx->input("emb_shard_ptrs", &unused_list);
      TF_RETURN_IF_ERROR(ctx->WithRank(unused_list[0], 1, &unused));
      TF_RETURN_IF_ERROR(
          ctx->WithValue(ctx->Dim(unused, 0), num_partitions, &unused_dim));

      // partition_permutation
      ctx->input("partition_permutation", &unused_list);
      TF_RETURN_IF_ERROR(ctx->WithRank(unused_list[0], 2, &unused));

      // feature_nums
      ctx->input("feature_nums", &unused_list);
      TF_RETURN_IF_ERROR(ctx->WithRank(unused_list[0], 1, &unused));

      // indices_before_unique
      ctx->input("indices_before_unique", &unused_list);
      TF_RETURN_IF_ERROR(ctx->WithRank(unused_list[0], 2, &unused));

      // unique_idxs
      ctx->input("unique_idxs", &unused_list);
      TF_RETURN_IF_ERROR(ctx->WithRank(unused_list[0], 1, &unused));

      // is_row_empty
      ctx->input("is_row_empty", &unused_list);
      TF_RETURN_IF_ERROR(ctx->WithRank(unused_list[0], 1, &unused));

      // grad_shards
      unused_list.clear();
      unused_list.resize(1);
      unused_list[0] = ctx->MakeShape({ctx->UnknownDim(), emb_vec_size_dim});
      ctx->set_output("grad_shards", unused_list);

      return Status::OK();
    });

//     .Doc(R"doc(
// Calculate gradient of FusedEmbeddingSparsePostLookUp
//     )doc");

}  // namespace tensorflow
