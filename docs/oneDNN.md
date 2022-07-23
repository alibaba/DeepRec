# oneDNN

## 介绍

[oneDNN](https://github.com/oneapi-src/oneDNN) 是 Intel 开源的跨平台深度学习性能加速库，通过 [文档](https://oneapi-src.github.io/oneDNN/) 可以了解到被支持的原语，DeepRec 中已经加入了 oneDNN 的支持，只需要在 DeepRec 编译命令中加入关于 oneDNN 的编译选项：`--config=mkl_threadpool` 即可开启 oneDNN 加速算子计算。在支持 AVX512 指令集的机器（Sky Lake 及其之后的 CPU）上添加 `--config=opt` 选项，默认会打开 `--copt=-march=native` 的优化，可以进一步加速算子计算性能。

Tips: MKL-DNN 被重命名为 DNNL，之后又被重命名为 oneDNN；TensorFlow 初期采用的是 MKL 加速算子计算，在之后的版本迭代中，逐步使用 oneDNN 替换了 MKL，但宏定义还是仍然保留。

DeepRec 关于 oneDNN 的宏定义：

| 宏定义                           | 可设值（默认值加粗）                           | 解释                                                         |
| :------------------------------- | ---------------------------------------------- | ------------------------------------------------------------ |
| TF_MKL_PRIMITIVE_ONLY_FOR_RECO   | **1/true**, 0/false                            | 1: 仅替换推荐模型中oneDNN支持[算子](https://github.com/alibaba/DeepRec/blob/main/tensorflow/core/graph/mkl_layout_pass.cc#L824-L840)；0: 替换成所有oneDNN支持的的算子 |
| TF_MKL_OPTIMIZE_PRIMITIVE_MEMUSE | **1/true**,  0/false                           | 1: 通过释放原语来减少内存，再次使用会重建；0: 不释放原语     |
| TF_DISABLE_MKL                   | **0**, 1                                       | 0: Enable MKL; 1: Disable MKL                                |
| TF_MKL_NUM_INTRAOP               | 整数值，如14，**默认不设置**                   | 整数值：设置 oneDNN 使用的 intra 线程数；不设置：使用最多的 TF intra 线程数 |
| ONEDNN_VERBOSE                   | **0**/1/2                                      | 打印 oneDNN 原语输出的 log 的[等级](https://oneapi-src.github.io/oneDNN/dev_guide_verbose.html) |
| DNNL_MAX_CPU_ISA                 | **ALL**, AVX512_CORE_AMX,  AVX512_CORE_BF16, … | 指定 oneDNN(版本小于2.5.0时) 使用的[最高指令集](https://oneapi-src.github.io/oneDNN/v2.4/dev_guide_cpu_dispatcher_control.html#run-time-controls) |
| ONEDNN_MAX_CPU_ISA               | **ALL**, AVX512_CORE_AMX,  AVX512_CORE_BF16, … | 指定 oneDNN(版本大于等于2.5.0时) 使用的[最高指令集](https://oneapi-src.github.io/oneDNN/dev_guide_cpu_dispatcher_control.html) |

oneDNN 支持的原语：

| 原语                                   | 支持的类型                  | 支持的后向操作                    |
| :------------------------------------- | :-------------------------- | --------------------------------- |
| Matrix Multiplication                  | f32, bf16, f16, u8, s8      | Scale, Zero, Eltwise, Sum, Binary |
| Inner Product                          | f32, bf16, f16, u8, s8      | Scale, Eltwise, Sum, Binary       |
| Layer Normalization                    | f32, bf16, f16              | /                                 |
| Batch Normalization                    | f32, bf16, f16, s8          | Eltwise                           |
| Local Response Normalization (LRN)     | f32, bf16, f16              | /                                 |
| Binary (+, =, *, /, >, <, min, max...) | f32, bf16, f16, u8, s8      | Scale, Eltwise, Sum, Binary       |
| Eltwise (relu, gelu, tanh, linear...)  | f32, s32, bf16, f16, u8, s8 | Binary                            |
| PReLU                                  | f32, s32, bf16, s8, u8      | /                                 |
| Sum                                    | f32, s32, bf16, f16, u8, s8 | /                                 |
| Reduction                              | f32, bf16, u8, s8           | Eltwise, Sum, Binary              |
| Softmax                                | f32, bf16, f16              | /                                 |
| LogSoftmax                             | f32, bf16                   | /                                 |
| Reorder                                | f32, s32, bf16, f16, u8, s8 | Scale, Sum                        |
| Concat                                 | f32, s32, bf16, f16, u8, s8 | /                                 |
| Convolution                            | f32, bf16, f16, u8, s8      | Scale, Zero, Eltwise, Sum, Binary |
| Pooling                                | f32, s32, bf16, f16, u8, s8 | Binary                            |
| RNN (LSTM, GRU, Vanilla RNN...)        | f32, bf16, f16, u8, s8      | /                                 |
| Resampling                             | f32, s32, bf16, f16, s8, u8 | Eltwise, Sum, Binary              |
| Shuffle                                | f32, s32, bf16, s8, u8      | /                                 |
