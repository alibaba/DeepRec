# DLRM with GPU Fused Embedding

The model structure, hyper params, dataset, etc, are all same to [DLRM](../../../DLRM/README.md). Please follow the instruction there to prepare, setup and run the model.

The only difference is that this model use GPU Fused Embedding to acclerate the lookup process. Only change is:

```python
categorical_embedding_column = tf.feature_column.embedding_column(
    categorical_column, dimension=16, combiner='mean',
    do_fusion='v2')
```

## Benchmark

On A100-80GB-PCIE GPU, with 8 cores AMD EPYC 7232P CPU @ 3.20GHz. Average of 5000 iterations.
Let tensorflow use private single thread for GPU kernels:

```bash
export TF_GPU_THREAD_MODE="gpu_private"
export TF_GPU_THREAD_COUNT=1
```

The perf boost:

|                              | Unfused | Fused   | Speedup |
| ---------------------------- | ------- |
| Step Time, Batch Size = 512  | 19.98ms | 14.81ms | 1.34x   |
| Step Time, Batch Size = 4096 | 37.82ms | 28.82ms | 1.31x   |