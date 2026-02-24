# DeepSeek-OCR Fine-tuning Guide

This guide explains how to fine-tune DeepSeek-OCR on your custom dataset of handwritten document images with Excel ground truth.

## Overview

The fine-tuning pipeline consists of three main scripts:

1. **`prepare_dataset.py`** - Converts Excel + image pairs to training format
2. **`finetune_qlora.py`** - Fine-tunes the model using QLoRA (4-bit quantization)
3. **`inference_finetuned.py`** - Runs inference with the fine-tuned model

## Requirements

### Hardware
- **Minimum**: NVIDIA GPU with 8GB VRAM (RTX 3070 Ti, RTX 3080, etc.)
- **Recommended**: NVIDIA GPU with 24GB+ VRAM (RTX 4090, A100)

### Software
```bash
pip install -r requirements.txt
```

## Step 1: Prepare Your Data

### Directory Structure

Organize your data like this:

```
training_data/
├── images/
│   ├── doc001_A.jpg       # Image for sheet A of doc001
│   ├── doc001_A1.jpg      # Image for sheet A1 of doc001
│   ├── doc001_B1.jpg
│   ├── doc002_A.jpg
│   └── ...
└── excel/
    ├── doc001.xlsx        # Contains sheets: A, A1, B1, B2...
    ├── doc002.xlsx
    └── ...
```

### Naming Convention

The script matches Excel sheets to images by name. For example:
- Excel file `doc001.xlsx` with sheet `A` → looks for `doc001_A.jpg`
- Excel file `February 2023.xlsx` with sheet `B1` → looks for `February 2023_B1.jpg`

### Run Data Preparation

```bash
# Full dataset
python prepare_dataset.py \
    --data_dir ./training_data \
    --output ./training_data/dataset.jsonl \
    --format html \
    --train_split 0.9

# Test with a single file first
python prepare_dataset.py \
    --test_single \
    --test_image ./input/February\ 2023\ page\ 002.jpeg \
    --test_excel ./output/img_1/img_1.xlsx \
    --test_sheet Sheet1
```

This creates:
- `dataset_train.jsonl` - Training examples (90%)
- `dataset_val.jsonl` - Validation examples (10%)

## Step 2: Fine-tune the Model

### Basic Training (8GB GPU)

```bash
python finetune_qlora.py \
    --train_data ./training_data/dataset_train.jsonl \
    --val_data ./training_data/dataset_val.jsonl \
    --output_dir ./finetuned_model \
    --epochs 3 \
    --batch_size 1 \
    --grad_accum 8 \
    --lr 2e-5 \
    --lora_r 16 \
    --lora_alpha 32
```

### If You Run Out of Memory

Try these options:
```bash
python finetune_qlora.py \
    --train_data ./training_data/dataset_train.jsonl \
    --output_dir ./finetuned_model \
    --batch_size 1 \
    --grad_accum 16 \
    --lora_r 8 \
    --max_length 1024
```

### Training Parameters Explained

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 3 | Number of training passes |
| `--batch_size` | 1 | Per-GPU batch size |
| `--grad_accum` | 8 | Gradient accumulation (effective batch = batch_size × grad_accum) |
| `--lr` | 2e-5 | Learning rate |
| `--lora_r` | 16 | LoRA rank (lower = less memory, 8-64 typical) |
| `--lora_alpha` | 32 | LoRA alpha (typically 2× rank) |
| `--max_length` | 2048 | Max sequence length |

### Monitor Training

```bash
# View training logs
tensorboard --logdir ./finetuned_model/logs
```

## Step 3: Run Inference

### Single Image

```bash
# With fine-tuned model
python inference_finetuned.py \
    --image ./test_image.jpg \
    --adapter_path ./finetuned_model

# With base model (for comparison)
python inference_finetuned.py \
    --image ./test_image.jpg
```

### Batch Processing

```bash
python inference_finetuned.py \
    --image_dir ./test_images \
    --adapter_path ./finetuned_model \
    --output_dir ./results
```

### Compare Base vs Fine-tuned

```bash
python inference_finetuned.py \
    --image ./test_image.jpg \
    --adapter_path ./finetuned_model \
    --compare \
    --output_file comparison.json
```

## Troubleshooting

### Out of Memory (OOM)

1. Reduce `--lora_r` to 8
2. Reduce `--max_length` to 1024
3. Increase `--grad_accum` to 16 or 32
4. Ensure no other GPU processes are running

### Slow Training

1. Ensure you're using GPU: `nvidia-smi` should show utilization
2. Install flash-attention: `pip install flash-attn`
3. Use fewer workers: `--dataloader_num_workers 1`

### Poor Results

1. Check your data - run `prepare_dataset.py --test_single` first
2. Train longer: increase `--epochs` to 5-10
3. Try higher LoRA rank: `--lora_r 32 --lora_alpha 64`
4. Add more training data

## Cloud Training Option

If your local GPU is insufficient, use cloud GPUs:

### Google Colab Pro ($10/month)
- Provides T4 (16GB) or A100 (40GB)
- Upload your dataset to Google Drive
- Run the training notebook

### RunPod / Lambda Labs (~$1-2/hour)
- Rent A100 40GB/80GB
- SSH in and run the scripts directly

### Example Cloud Setup
```bash
# On cloud instance
git clone https://github.com/your-repo/DeepSeek-OCR.git
cd DeepSeek-OCR/finetune
pip install -r requirements.txt

# Upload your data
# Run training
python finetune_qlora.py --train_data ./data/train.jsonl ...

# Download results
# scp -r ./finetuned_model user@local:/path/to/local/
```

## File Reference

```
finetune/
├── prepare_dataset.py      # Data preparation
├── finetune_qlora.py       # Training script
├── inference_finetuned.py  # Inference script
├── requirements.txt        # Python dependencies
└── README.md              # This file
```
