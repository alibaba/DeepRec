## How to use prepare_savedmodel.py to get savedmodel

- Current support model: \
BST, DBMTL, DeepFM, DIEN, DIN, DLRM, DSSM, ESMM, MMoE, SimpleMultiTask, WDL

- Usage: \
 For every model listed above, there is a prepare_savedmodel.py. To run this script please firstly ensure you have gotten the checkpoint file from training. To use prepare_savedmodel.py, please use:

  ```
    cd [modelfolder] 
    python prepare_savedmodel.py --checkpoint [ckpt path]
  ``` 
 - If you choose --bf16 during training stage, please also add it above. For example:

   ```
    cd [modelfolder] 
    python prepare_savedmodel.py --bf16 --checkpoint [ckpt path]
   ``` 


 - Example: \
  This is an example for BST model without bf16 feature
   ```
    cd modelzoo/BST
    python prepare_savedmodel.py  --checkpoint ./result/model_BST_1657777492
   ``` 

 - Output: \
  The savedmodel will be stored under ./savedmodels folder


## How to use start_serving.cc to retrieving serving result

- Functionality: \
start_serving.cc provides functionality such that you can get serving result after getting the savedmodel.

- Parameter: 
  1. At the start of main(), there is a file_path variable. Please substitude it to your own evaluation file path (the format should be the same as training one)
  2. Please edit the savedmodel and checkpoint path in the model_config which is at the very begining.

- Usage: 
  1. Please edit BUILD file to add the start_serving
  2. Please make sure you have installed serving part properly
  3. Then go back to the Deeprec folder and do as the follows, assuming the cc_binary name in BUILD is "wdl_demo":
   ``` 
   bazel build //serving/processor/tests:wdl_demo
   bazel-bin/serving/processor/tests/wdl_demo
   ``` 
  



