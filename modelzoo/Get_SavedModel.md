## How to use prepare_savedmodel.py to get savedmodel

- Current support model: \
BST, DBMTL, DeepFM, DIEN, DIN, DLRM, DSSM, ESMM, MMoE, SimpleMultiTask, WDL

- Usage: \
 For every model listed above, there is a prepare_savedmodel.py. To run this script please firstly ensure you have gotten the checkpoint file from training. To use prepare_savedmodel.py, please use:

 ```
    cd [modelfolder] 
    python prepare_savedmodel.py --tf --checkpoint [ckpt path]
 ``` 

 - Example: \
  This is an example for BST model
  ```
    cd modelzoo/BST
    python prepare_savedmodel.py --tf --checkpoint ./result/model_BST_1657777492
  ``` 

 - Output: \
  The savedmodel will be stored under ./savedmodels folder
