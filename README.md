# MGCLforMole

## Tokenizer Training
```
python vqvae.py --output_model_file OUTPUT_MODEL_PATH
```
This will save the resulting tokenizer to `OUTPUT_MODEL_PATH`.

## Pre-training and fine-tuning
#### 1. Self-supervised pre-training
```
python pretrain.py --output_model_file OUTPUT_MODEL_PATH
```
This will save the resulting pre-trained model to `OUTPUT_MODEL_PATH`.

#### 2. Fine-tuning
```
python finetune.py --input_model_file INPUT_MODEL_PATH --dataset DOWNSTREAM_DATASET --filename OUTPUT_FILE_PATH
```
This will finetune pre-trained model specified in `INPUT_MODEL_PATH` using dataset `DOWNSTREAM_DATASET.` The result of fine-tuning will be saved to `OUTPUT_FILE_PATH.`
