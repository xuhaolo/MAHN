# MAHN
This repository implement the proposed model MAHN in the paper of "Multiple Adaptive Hawkes Network for Popularity Prediction".

# DataSet
We use the Sina Weibo Dataset collected by the paper of DeepHawkes (Qi Cao, Huawei Shen, Keting Cen, Wentao Ouyang, Xueqi Cheng. 2017. DeepHawkes: Bridging the Gap between Prediction and Understanding of Information Cascades. In Proceedings of CIKM'17, Singapore., November 
6-10, 2017, 11 pages.).

Download link: https://pan.baidu.com/s/1c2rnvJq

password: ijp6

# Run MAHN model

### 1. Preprocessing
    cd preprocssing
    python generate_cascade
    #you can get the preprocessed dataset an the dirctory data
### 2. Train MAHN
    python train
    #you can get the learned parameters int the file *.pkl and the prediction results in the file *.txt.
