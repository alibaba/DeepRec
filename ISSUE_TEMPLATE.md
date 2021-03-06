Please go to Stack Overflow for help and support:

https://stackoverflow.com/questions/tagged/deeprec

If you open a GitHub issue, here is our policy:

1. It must be a bug, a feature request, or a significant problem with documentation (for small docs fixes please send a PR instead).
2. The form below must be filled out.
3. It shouldn't be a TensorBoard issue.

**Here's why we have that policy**: DeepRec developers respond to issues. We want to focus on work that benefits the whole community, e.g., fixing bugs and adding features. Support only helps individuals. GitHub also notifies thousands of people when issues are filed. We want them to see you communicating an interesting problem, rather than being redirected to Stack Overflow.

------------------------

### System information
- **Have I written custom code (as opposed to using a stock example script provided in DeepRec)**:
- **OS Platform and Distribution (e.g., Linux Ubuntu 18.04)**:
- **Mobile device (e.g. iPhone 8, Pixel 2, Samsung Galaxy) if the issue happens on mobile device**:
- **DeepRec installed from (source or binary)**:
- **DeepRec version (use command below)**:
- **Python version**:
- **Bazel version (if compiling from source)**:
- **GCC/Compiler version (if compiling from source)**:
- **CUDA/cuDNN version**:
- **GPU model and memory**:
- **Exact command to reproduce**:

You can collect some of this information using our environment capture script:

https://github.com/alibaba/DeepRec/tree/main/tools/tf_env_collect.sh

You can obtain the DeepRec version with:

```bash
python -c "import tensorflow as tf; print(tf.version.GIT_VERSION, tf.version.VERSION)"
```

### Describe the problem
Describe the problem clearly here. Be sure to convey here why it's a bug in DeepRec or a feature request.

### Source code / logs
Include any logs or source code that would be helpful to diagnose the problem. If including tracebacks, please include the full traceback. Large logs and files should be attached. Try to provide a reproducible test case that is the bare minimum necessary to generate the problem.
