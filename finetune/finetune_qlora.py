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
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

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

        # Load and process image
        image_path = example["image"]
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.warning(f"Error loading image {image_path}: {e}")
            # Return a blank image as fallback
            image = Image.new("RGB", (self.image_size, self.image_size), "white")

        # Get conversation
        conversations = example["conversations"]
        user_content = conversations[0]["content"]  # Includes <image> token
        assistant_content = conversations[1]["content"]  # Ground truth

        # Format as chat
        # DeepSeek format: <|User|>content<|Assistant|>response<|end▁of▁sentence|>
        full_text = f"<|User|>{user_content}<|Assistant|>{assistant_content}<|end▁of▁sentence|>"

        # Tokenize
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        # Process image
        if self.processor is not None:
            pixel_values = self.processor(images=image, return_tensors="pt")["pixel_values"]
        else:
            # Fallback: basic image preprocessing
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            pixel_values = transform(image).unsqueeze(0)

        # Create labels (mask user input, only train on assistant response)
        input_ids = encoding["input_ids"].squeeze()
        labels = input_ids.clone()

        # Find where assistant response starts
        assistant_token = self.tokenizer.encode("<|Assistant|>", add_special_tokens=False)
        assistant_start = None
        for i in range(len(input_ids) - len(assistant_token)):
            if input_ids[i:i+len(assistant_token)].tolist() == assistant_token:
                assistant_start = i + len(assistant_token)
                break

        # Mask everything before assistant response
        if assistant_start:
            labels[:assistant_start] = -100

        # Also mask padding
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": encoding["attention_mask"].squeeze(),
            "pixel_values": pixel_values.squeeze(),
            "labels": labels,
        }


# ============================================================================
# Data Collator
# ============================================================================

class OCRDataCollator:
    """Custom data collator for OCR training."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch: List[Dict]) -> Dict:
        input_ids = torch.stack([x["input_ids"] for x in batch])
        attention_mask = torch.stack([x["attention_mask"] for x in batch])
        labels = torch.stack([x["labels"] for x in batch])
        pixel_values = torch.stack([x["pixel_values"] for x in batch])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values,
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
