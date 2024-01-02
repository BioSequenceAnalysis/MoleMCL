This is a Pytorch implementation (stay tuned) of the MoleMCL paper: 

## Installation
We used the following Python packages for core development. We tested on `Python 3.7`.
```
pytorch                   1.13.1                   
torch-cluster             1.6.1
torch-geometric           2.3.1          
torch-scatter             2.1.1
torch-sparse              0.6.17
torch-spline-conv         1.2.2
torchvision               0.14.1
rdkit                     2023.3.2
tqdm                      4.26.0
tensorboardx              2.6.1
```

## Dataset download
All the necessary data files can be downloaded from the following links.

For the chemistry dataset, download from [chem data](http://snap.stanford.edu/gnn-pretrain/data/chem_dataset.zip) (2.5GB), unzip it, and put it under `dataset/`.


## Pre-training and fine-tuning
#### 1. Self-supervised pre-training
```
python MoleMCL.py --output_model_file OUTPUT_MODEL_PATH
```
This will save the resulting pre-trained model to `OUTPUT_MODEL_PATH`.

#### 2. Fine-tuning
```
python finetune.py --input_model_file INPUT_MODEL_PATH --dataset DOWNSTREAM_DATASET --filename OUTPUT_FILE_PATH
```
This will finetune pre-trained model specified in `INPUT_MODEL_PATH` using dataset `DOWNSTREAM_DATASET.` The result of fine-tuning will be saved to `OUTPUT_FILE_PATH.`



## Acknowledgement
[1] Strategies for Pre-training Graph Neural Networks (Hu et al., ICLR 2020)
[2] Mole-BERT: Rethinking Pre-training Graph Neural Networks for Molecules (Xia et al., ICLR 2023)

