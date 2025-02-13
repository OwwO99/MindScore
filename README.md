# MindScore

**MindScore**: Quantifying Human Preference for Text-to-Image Generation through Multi-view Lens

## Requirements

To set up the environment for MindScore, follow these steps:

1. Create a new Conda environment with Python 3.9:

```bash
conda create -n mindscore python=3.9
conda activate mindscore
```

2. Install the required Python packages using pip:

```bash
pip install torch torchvision timm lavis transformers pandas numpy pyarrow Pillow tqdm
```

## Datasets

- **Pick-a-Pic**: The dataset can be downloaded from [Pick-a-Pic dataset](https://huggingface.co/datasets/yuvalkirstain/pickapic_v1/tree/main/data).

- **ImagePrefer**: Our dataset will be released soon.

## Training and Evaluation

### Training with Pick-a-Pic

To train the model using the Pick-a-Pic dataset, run the following command:

```bash
python train.py --is_train --is_eval --epochs 1 --batch 128 --dataset Pick-a-pic --lr1 5e-6 --lr2 5e-5 --tau 0.2 --is_save --check steps --model_type blip --layers
```

### Training with ImagePrefer

To train the model using the ImagePrefer dataset, use this command:

```bash
python train.py --is_train --is_eval --epochs 5 --batch 128 --dataset ImagePrefer --lr1 1e-5 --lr2 5e-5 --tau 0.2 --is_save --check epochs --model_type blip --layers
```

### Evaluation

To evaluate the model, execute the following:

```bash
python test.py --model_id --batch 128 --dataset Pick-a-pic --model_type blip
```



