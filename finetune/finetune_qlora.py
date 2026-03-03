#!/usr/bin/env python3
"""
QLoRA Fine-tuning Script for DeepSeek-OCR

Optimized for low VRAM (8GB) with:
- 4-bit quantization (QLoRA)
- Gradient checkpointing
- Small batch size with gradient accumulation
- Memory-efficient attention

Usage:
    python finetune_qlora.py --train_data ./training_data/dataset_train.jsonl \
                             --val_data ./training_data/dataset_val.jsonl \
                             --output_dir ./finetuned_model

Requirements:
    pip install torch transformers peft accelerate bitsandbytes datasets pillow
"""

import os
import json
import math
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field

import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageOps

# DeepSeek-OCR image constants (no-crop mode for simplicity)
IMAGE_TOKEN = "<image>"
IMAGE_TOKEN_ID = 128815
IMAGE_SIZE = 640       # image patch size
BASE_SIZE = 640        # global view size (same as IMAGE_SIZE in no-crop)
PATCH_SIZE = 16
DOWNSAMPLE_RATIO = 4
IMAGE_MEAN = IMAGE_STD = (0.5, 0.5, 0.5)  # model's own normalization

from transformers import (
    AutoTokenizer,
    AutoProcessor,
    AutoModelForCausalLM,
    AutoModel,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class ModelConfig:
    """Model configuration"""
    model_name: str = "deepseek-ai/DeepSeek-OCR"
    trust_remote_code: bool = True
    use_flash_attention: bool = True
    max_length: int = 2048


@dataclass
class LoRAConfig:
    """LoRA configuration optimized for vision-language models"""
    r: int = 16  # LoRA rank (lower = less memory, 8-64 typical)
    lora_alpha: int = 32  # LoRA alpha (typically 2x rank)
    lora_dropout: float = 0.05
    bias: str = "none"
    # Target modules for DeepSeek-VL2 architecture
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
        "gate_proj", "up_proj", "down_proj",      # MLP
    ])


@dataclass
class TrainConfig:
    """Training configuration optimized for 8GB VRAM"""
    # Batch size settings
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8  # Effective batch size = 8

    # Training settings
    num_train_epochs: int = 3
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1

    # Memory optimization
    gradient_checkpointing: bool = True
    fp16: bool = False  # Use bf16 instead if available
    bf16: bool = True

    # Logging and saving
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    save_total_limit: int = 3

    # Other
    dataloader_num_workers: int = 2
    seed: int = 42


# ============================================================================
# Dataset
# ============================================================================

class OCRFineTuneDataset(Dataset):
    """Dataset for OCR fine-tuning with image-text pairs."""

    def __init__(
        self,
        data_path: str,
        tokenizer,
        processor,
        max_length: int = 2048,
        image_size: int = 1024,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_length = max_length
        self.image_size = image_size

        # Load data
        self.examples = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.examples.append(json.loads(line))

        logger.info(f"Loaded {len(self.examples)} examples from {data_path}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx) -> Dict:
        example = self.examples[idx]

        # Load image
        image_path = example["image"]
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.warning(f"Error loading image {image_path}: {e}")
            image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (128, 128, 128))

        # Preprocess image using model's normalization (no-crop mode)
        transform = T.Compose([T.ToTensor(), T.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD)])
        pad_color = tuple(int(x * 255) for x in IMAGE_MEAN)
        padded = ImageOps.pad(image, (IMAGE_SIZE, IMAGE_SIZE), color=pad_color)
        images_ori = transform(padded)                              # [3, H, W]
        images_crop = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE)    # no crops
        images_spatial_crop = torch.tensor([1, 1], dtype=torch.long)

        # Build image token sequence (no-crop mode, fixed size)
        num_q = math.ceil((IMAGE_SIZE // PATCH_SIZE) / DOWNSAMPLE_RATIO)
        image_token_ids = ([IMAGE_TOKEN_ID] * num_q + [IMAGE_TOKEN_ID]) * num_q + [IMAGE_TOKEN_ID]

        # Tokenize conversation parts
        conversations = example["conversations"]
        user_content = conversations[0]["content"]   # contains "<image>"
        assistant_content = conversations[1]["content"]

        text_parts = user_content.split(IMAGE_TOKEN)
        prefix_ids = self.tokenizer.encode(f"<|User|>{text_parts[0]}", add_special_tokens=False)
        suffix_ids  = self.tokenizer.encode(text_parts[1] if len(text_parts) > 1 else "", add_special_tokens=False)
        asst_prefix = self.tokenizer.encode("<|Assistant|>", add_special_tokens=False)
        response_ids = self.tokenizer.encode(assistant_content, add_special_tokens=False)
        eos_id = self.tokenizer.eos_token_id or 2
        bos_id = self.tokenizer.bos_token_id or 0

        full_ids = [bos_id] + prefix_ids + image_token_ids + suffix_ids + asst_prefix + response_ids + [eos_id]

        # images_seq_mask: True only for image token positions
        seq_mask = (
            [False] +
            [False] * len(prefix_ids) +
            [True]  * len(image_token_ids) +
            [False] * len(suffix_ids) +
            [False] * len(asst_prefix) +
            [False] * len(response_ids) +
            [False]
        )

        # Labels: -100 everywhere except assistant response + EOS
        resp_start = len(full_ids) - len(response_ids) - 1
        labels = [-100] * resp_start + response_ids + [eos_id]

        # Truncate or pad to max_length
        pad_id = self.tokenizer.pad_token_id or 0
        if len(full_ids) > self.max_length:
            full_ids  = full_ids[:self.max_length]
            seq_mask  = seq_mask[:self.max_length]
            labels    = labels[:self.max_length]
        else:
            pad_len   = self.max_length - len(full_ids)
            full_ids  = full_ids  + [pad_id] * pad_len
            seq_mask  = seq_mask  + [False]  * pad_len
            labels    = labels    + [-100]   * pad_len

        input_ids        = torch.tensor(full_ids, dtype=torch.long)
        attention_mask   = (input_ids != pad_id).long()
        images_seq_mask  = torch.tensor(seq_mask, dtype=torch.bool)
        labels_tensor    = torch.tensor(labels,   dtype=torch.long)

        return {
            "input_ids":          input_ids,
            "attention_mask":     attention_mask,
            "images_ori":         images_ori,
            "images_crop":        images_crop,
            "images_spatial_crop": images_spatial_crop,
            "images_seq_mask":    images_seq_mask,
            "labels":             labels_tensor,
        }


# ============================================================================
# Data Collator
# ============================================================================

class OCRDataCollator:
    """Custom data collator for OCR training."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch: List[Dict]) -> Dict:
        input_ids        = torch.stack([x["input_ids"]          for x in batch])
        attention_mask   = torch.stack([x["attention_mask"]     for x in batch])
        labels           = torch.stack([x["labels"]             for x in batch])
        images_seq_mask  = torch.stack([x["images_seq_mask"]    for x in batch])
        images_spatial_crop = torch.stack([x["images_spatial_crop"] for x in batch])
        images_crop      = torch.stack([x["images_crop"]        for x in batch])
        images_ori       = torch.stack([x["images_ori"]         for x in batch])

        return {
            "input_ids":           input_ids,
            "attention_mask":      attention_mask,
            "labels":              labels,
            "images":              [(images_crop, images_ori)],
            "images_seq_mask":     images_seq_mask,
            "images_spatial_crop": images_spatial_crop,
        }


# ============================================================================
# Model Loading
# ============================================================================

def load_model_for_training(
    model_config: ModelConfig,
    lora_config: LoRAConfig,
    device_map: str = "auto",
):
    """Load model with QLoRA configuration."""

    logger.info(f"Loading model: {model_config.model_name}")

    # Quantization config for 4-bit QLoRA
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,  # Nested quantization for more memory savings
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name,
        trust_remote_code=model_config.trust_remote_code,
    )

    # Ensure pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Try to load processor (for image preprocessing)
    try:
        processor = AutoProcessor.from_pretrained(
            model_config.model_name,
            trust_remote_code=model_config.trust_remote_code,
        )
        # Check if it actually handles images (not just a tokenizer)
        if not hasattr(processor, 'image_processor') and not hasattr(processor, 'feature_extractor'):
            logger.warning("Processor does not support images, using torchvision fallback.")
            processor = None
    except Exception as e:
        logger.warning(f"Could not load processor: {e}. Using basic preprocessing.")
        processor = None

    # Load model with quantization
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_config.model_name,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=model_config.trust_remote_code,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2" if model_config.use_flash_attention else "eager",
        )
    except ValueError:
        logger.warning("AutoModelForCausalLM failed, falling back to AutoModel with trust_remote_code")
        model = AutoModel.from_pretrained(
            model_config.model_name,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=model_config.trust_remote_code,
            torch_dtype=torch.bfloat16,
        )

    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )

    # Configure LoRA
    peft_config = LoraConfig(
        r=lora_config.r,
        lora_alpha=lora_config.lora_alpha,
        lora_dropout=lora_config.lora_dropout,
        bias=lora_config.bias,
        target_modules=lora_config.target_modules,
        task_type=TaskType.CAUSAL_LM,
    )

    # Apply LoRA
    model = get_peft_model(model, peft_config)

    # Print trainable parameters
    model.print_trainable_parameters()

    return model, tokenizer, processor


# ============================================================================
# Training
# ============================================================================

def train(
    train_data_path: str,
    val_data_path: Optional[str],
    output_dir: str,
    model_config: ModelConfig,
    lora_config: LoRAConfig,
    train_config: TrainConfig,
):
    """Main training function."""

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load model
    model, tokenizer, processor = load_model_for_training(
        model_config, lora_config
    )

    # Create datasets
    train_dataset = OCRFineTuneDataset(
        train_data_path,
        tokenizer,
        processor,
        max_length=model_config.max_length,
    )

    eval_dataset = None
    if val_data_path and Path(val_data_path).exists():
        eval_dataset = OCRFineTuneDataset(
            val_data_path,
            tokenizer,
            processor,
            max_length=model_config.max_length,
        )

    # Data collator
    data_collator = OCRDataCollator(tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,

        # Batch size
        per_device_train_batch_size=train_config.per_device_train_batch_size,
        per_device_eval_batch_size=train_config.per_device_eval_batch_size,
        gradient_accumulation_steps=train_config.gradient_accumulation_steps,

        # Training
        num_train_epochs=train_config.num_train_epochs,
        learning_rate=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
        warmup_ratio=train_config.warmup_ratio,
        lr_scheduler_type="cosine",

        # Memory optimization
        gradient_checkpointing=train_config.gradient_checkpointing,
        fp16=train_config.fp16,
        bf16=train_config.bf16,
        optim="paged_adamw_8bit",  # Memory-efficient optimizer

        # Logging
        logging_steps=train_config.logging_steps,
        logging_dir=f"{output_dir}/logs",
        report_to=["tensorboard"],

        # Saving
        save_steps=train_config.save_steps,
        save_total_limit=train_config.save_total_limit,

        # Evaluation
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=train_config.eval_steps if eval_dataset else None,

        # Other
        dataloader_num_workers=train_config.dataloader_num_workers,
        seed=train_config.seed,
        remove_unused_columns=False,

        # For low VRAM
        dataloader_pin_memory=False,
        max_grad_norm=1.0,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # Train
    logger.info("Starting training...")
    trainer.train()

    # Save final model
    logger.info(f"Saving model to {output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    # Save LoRA config
    model.save_pretrained(output_dir)

    logger.info("Training complete!")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Fine-tune DeepSeek-OCR with QLoRA")

    # Data paths
    parser.add_argument("--train_data", type=str, required=True,
                        help="Path to training JSONL file")
    parser.add_argument("--val_data", type=str, default=None,
                        help="Path to validation JSONL file")
    parser.add_argument("--output_dir", type=str, default="./finetuned_model",
                        help="Output directory for fine-tuned model")

    # Model settings
    parser.add_argument("--model_name", type=str, default="deepseek-ai/DeepSeek-OCR",
                        help="Base model name/path")
    parser.add_argument("--max_length", type=int, default=2048,
                        help="Maximum sequence length")

    # LoRA settings
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout")

    # Training settings
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Per-device batch size")
    parser.add_argument("--grad_accum", type=int, default=8,
                        help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Learning rate")

    # Memory settings
    parser.add_argument("--no_gradient_checkpointing", action="store_true",
                        help="Disable gradient checkpointing")
    parser.add_argument("--no_flash_attention", action="store_true",
                        help="Disable flash attention")

    args = parser.parse_args()

    # Create configs
    model_config = ModelConfig(
        model_name=args.model_name,
        max_length=args.max_length,
        use_flash_attention=not args.no_flash_attention,
    )

    lora_config = LoRAConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    train_config = TrainConfig(
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        gradient_checkpointing=not args.no_gradient_checkpointing,
    )

    # Start training
    train(
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        output_dir=args.output_dir,
        model_config=model_config,
        lora_config=lora_config,
        train_config=train_config,
    )


if __name__ == "__main__":
    main()
