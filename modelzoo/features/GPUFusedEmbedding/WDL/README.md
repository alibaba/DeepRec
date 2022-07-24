# WDL with GPU Fused Embedding

The model structure, hyper params, dataset, etc, are all same to [WDL](../../../WDL/README.md). Please follow the instruction there to prepare, setup and run the model.

The only difference is that this model use GPU Fused Embedding to acclerate the lookup process. Only change is:

```python
deep_columns.append(tf.feature_column.embedding_column(
    categorical_column,
    dimension=EMBEDDING_DIMENSIONS[column_name],
    combiner='mean', do_fusion='v2'))
```

## Benchmark

On A100-80GB-PCIE GPU, with 8 cores AMD EPYC 7232P CPU @ 3.20GHz. Average of 5000 iterations. The perf boost:

Let tensorflow use private single thread for GPU kernels:

```bash
export TF_GPU_THREAD_MODE="gpu_private"
export TF_GPU_THREAD_COUNT=1
```

|                              | Unfused | Fused   | Speedup |
| ---------------------------- | ------- |
| Step Time, Batch Size = 512  | 41.3ms | 38.4ms | 1.07x   |
| Step Time, Batch Size = 4096 | 75.1ms | 66.5ms | 1.12x   |