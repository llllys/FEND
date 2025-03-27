# Unofficial Implementation of FEND: A Future Enhanced Distribution-Aware Contrastive Learning Framework for Long-Tail Trajectory Prediction

This repository is an **unofficial** implementation of the  [CVPR 2023](https://cvpr2023.thecvf.com/) paper: [FEND: A Future Enhanced Distribution-Aware Contrastive Learning Framework for Long-Tail Trajectory Prediction](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_FEND_A_Future_Enhanced_Distribution-Aware_Contrastive_Learning_Framework_for_Long-Tail_CVPR_2023_paper.pdf).

What we have implemented:
- Future Enhanced Trajectory Clustering
- PCL loss
- Experiments on Argoverse 2 Motion Forecasting Dataset with QCNet + FEND

What we have not implemented:
- Distribution-Aware Hyper Predictor
- Experiments on ETH-UCY and Nuscenes as described in the paper

## Getting Started
Please follow the instructions in [QCNet](https://github.com/ZikangZhou/QCNet) to set up the environment and prepare the dataset.

## Offline Clustering
Run the ```offline_cluster.ipynb``` step by step to generate the offline clustering results. A 2-level Kmeans cluster label for training set can be downloaded [here](https://drive.google.com/file/d/1VRwTx0iYxK_CcWu26UbiyAHwZkOIpsG8/view?usp=drive_link).

## Train QCNet+FEND
Please follow the instructions in [QCNet](https://github.com/ZikangZhou/QCNet) to train the model. We have made the following main changes in ```QCNet/``` to support FEND:
- ```datasets/argoverse_v2_dataset.py```: incorperate the offline clustering results
- ```losses/pcl_loss.py```: implement the PCL loss
- ```predictors/qcnet.py```: incorporate the PCL loss in the training process

## Results
We train a small QCNet and QCNet+FEND with 4 RTX 4090 GPUs for 5 epochs. The results are as follows:
| Model | Dataset | Split | Checkpoint | minFDE (K=6) | minADE (K=6) | MR (K=6) 
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| QCNet | AV2 | Val | [QCNet](https://drive.google.com/file/d/1kJ4OUUeHmbtVfiy8K4sO-wEIER4FY3zV/view?usp=drive_link) | 2.71 | 1.27 | 0.38 | 
| QCNet+FEND | AV2 | Val | [QCNet+FEND](https://drive.google.com/file/d/1Zg58PLp5E8ed2a-Hs3MpmD6rXsCvlL9U/view?usp=drive_link) | 3.29 | 1.50 | 0.48|

In our experiment, the performance of QCNet+FEND is worse than that of QCNet alone. Several potential reasons may explain this:
-  In the QCNet+FEND implementation, contrastive learning is applied using only the agent encoder features ```scene_enc['x_a'][:, -1, :]``` from the agent encoder. This excludes the map-related features, potentially weakening the contrastive loss’s ability to capture the full scene context. This partial feature utilization might disrupt the balance of the original model, contributing to the observed performance drop.
- Hyperparameters are not optimized


Readers are encouraged to implement the full FEND framework and experiment on ETH-UCY and Nuscenes to compare the results with the paper.

## Acknowledgement
We appreciate [QCNet](https://github.com/ZikangZhou/QCNet) for their valuable code base.
