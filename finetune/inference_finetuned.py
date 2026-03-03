#!/usr/bin/env python3
"""
Inference Script for Fine-tuned DeepSeek-OCR

Loads the base model with LoRA adapters and runs OCR on images.

Usage:
    python inference_finetuned.py --image ./test_image.jpg \
                                  --adapter_path ./finetuned_model

    # Batch processing
    python inference_finetuned.py --image_dir ./test_images \
                                  --adapter_path ./finetuned_model \
                                  --output_dir ./results
"""

import os
import re
import json
import argparse
from pathlib import Path
from typing import Optional, List

import torch
from PIL import Image


# Phrases written on blank form pages (case-insensitive, accent-tolerant).
_RAS_PATTERN = re.compile(
    r'\b(rien\s+[aà]\s+signaler|pien\s+[aà]\s+signaler|n[eé]ant|r\.?\s*a\.?\s*s\.?)\b',
    re.IGNORECASE,
)


def is_blank_page(ocr_output: str) -> bool:
    """
    Return True if the model output indicates a blank page
    (contains only a 'rien à signaler' phrase and no table data).
    """
    text = ocr_output.strip()
    # If it contains a table tag, it's not blank
    if "<table" in text.lower():
        return False
    return bool(_RAS_PATTERN.search(text))

from transformers import AutoTokenizer, AutoModel
from peft import PeftModel


def load_finetuned_model(
    base_model_name: str = "deepseek-ai/DeepSeek-OCR",
    adapter_path: Optional[str] = None,
    device: str = "cuda",
    load_in_4bit: bool = True,
):
    print(f"Loading base model: {base_model_name}")

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)

    model = AutoModel.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model = model.eval().cuda()

    if adapter_path and Path(adapter_path).exists():
        print(f"Loading LoRA adapters from: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)

    return model, tokenizer


def run_ocr(
    model,
    tokenizer,
    image_path: str,
    prompt: str = "<image>\n<|grounding|>Convert the document to markdown.",
    max_new_tokens: int = 2048,
    temperature: float = 0.1,
    do_sample: bool = False,
) -> str:
    """
    Run OCR on a single image.

    Args:
        model: The loaded model
        tokenizer: The tokenizer
        image_path: Path to the image file
        prompt: The prompt to use
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        do_sample: Whether to use sampling

    Returns:
        OCR result as string
    """
    # Load image
    image = Image.open(image_path).convert("RGB")

    # Use model's built-in infer() method
    # Settings must match training: IMAGE_SIZE=640, BASE_SIZE=640, no crop
    if hasattr(model, 'infer'):
        result = model.infer(
            tokenizer,
            prompt=prompt,
            image_file=image_path,
            output_path="/tmp/ocr_output",
            base_size=640,
            image_size=640,
            crop_mode=False,
            save_results=False,
            eval_mode=True,
        )
        return result

    # Otherwise, use standard generation
    # Format prompt
    full_prompt = f"<|User|>{prompt}<|Assistant|>"

    # Tokenize
    inputs = tokenizer(full_prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Process image
    # Note: This is simplified - actual implementation depends on model architecture
    # DeepSeek-OCR may have specific image processing requirements

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else None,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract assistant response
    if "<|Assistant|>" in generated_text:
        result = generated_text.split("<|Assistant|>")[-1].strip()
    else:
        result = generated_text

    return result


def run_ocr_vllm(
    image_path: str,
    adapter_path: Optional[str] = None,
    prompt: str = "<image>\n<|grounding|>Convert the document to markdown.",
) -> str:
    """
    Run OCR using vLLM for faster inference.

    Note: vLLM LoRA support may require specific setup.
    """
    from vllm import LLM, SamplingParams

    # Initialize vLLM
    llm = LLM(
        model="deepseek-ai/DeepSeek-OCR",
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        enable_lora=adapter_path is not None,
    )

    # Load image
    image = Image.open(image_path).convert("RGB")

    # Sampling parameters
    sampling_params = SamplingParams(
        temperature=0.1,
        max_tokens=2048,
    )

    # Generate
    # Note: vLLM multimodal interface may differ
    outputs = llm.generate(
        [{"prompt": prompt, "multi_modal_data": {"image": image}}],
        sampling_params,
        lora_request=adapter_path,
    )

    return outputs[0].outputs[0].text


def batch_process(
    model,
    tokenizer,
    image_dir: str,
    output_dir: str,
    extensions: List[str] = [".jpg", ".jpeg", ".png", ".tiff"],
):
    """
    Process all images in a directory.

    Args:
        model: The loaded model
        tokenizer: The tokenizer
        image_dir: Directory containing images
        output_dir: Directory to save results
    """
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all images
    image_files = []
    for ext in extensions:
        image_files.extend(image_dir.glob(f"*{ext}"))
        image_files.extend(image_dir.glob(f"*{ext.upper()}"))

    print(f"Found {len(image_files)} images to process")

    results = {}
    for i, image_path in enumerate(image_files):
        print(f"Processing [{i+1}/{len(image_files)}]: {image_path.name}")

        try:
            result = run_ocr(model, tokenizer, str(image_path))

            if is_blank_page(result):
                print(f"  Skipped (blank page: 'rien à signaler')")
                results[image_path.name] = {"status": "skipped_blank"}
                continue

            # Save individual result
            output_file = output_dir / f"{image_path.stem}.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(result)

            results[image_path.name] = {
                "status": "success",
                "output_file": str(output_file),
            }
        except Exception as e:
            print(f"  Error: {e}")
            results[image_path.name] = {
                "status": "error",
                "error": str(e),
            }

    # Save summary
    summary_file = output_dir / "processing_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_dir}")
    print(f"Summary: {summary_file}")

    return results


def compare_models(
    image_path: str,
    adapter_path: str,
    output_file: Optional[str] = None,
):
    """
    Compare base model vs fine-tuned model on the same image.
    """
    print("=" * 60)
    print("Comparing Base Model vs Fine-tuned Model")
    print("=" * 60)

    # Load base model
    print("\n[1] Loading base model...")
    base_model, tokenizer = load_finetuned_model(adapter_path=None)

    print("\n[2] Running OCR with base model...")
    base_result = run_ocr(base_model, tokenizer, image_path)

    # Free memory
    del base_model
    torch.cuda.empty_cache()

    # Load fine-tuned model
    print("\n[3] Loading fine-tuned model...")
    ft_model, tokenizer = load_finetuned_model(adapter_path=adapter_path)

    print("\n[4] Running OCR with fine-tuned model...")
    ft_result = run_ocr(ft_model, tokenizer, image_path)

    # Display results
    print("\n" + "=" * 60)
    print("BASE MODEL OUTPUT:")
    print("=" * 60)
    print(base_result[:1000] + "..." if len(base_result) > 1000 else base_result)

    print("\n" + "=" * 60)
    print("FINE-TUNED MODEL OUTPUT:")
    print("=" * 60)
    print(ft_result[:1000] + "..." if len(ft_result) > 1000 else ft_result)

    # Save comparison
    if output_file:
        comparison = {
            "image": image_path,
            "base_model_output": base_result,
            "finetuned_model_output": ft_result,
        }
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(comparison, f, indent=2, ensure_ascii=False)
        print(f"\nComparison saved to: {output_file}")

    return base_result, ft_result


def main():
    parser = argparse.ArgumentParser(description="Run inference with fine-tuned DeepSeek-OCR")

    # Input
    parser.add_argument("--image", type=str, help="Path to single image")
    parser.add_argument("--image_dir", type=str, help="Directory of images for batch processing")

    # Model
    parser.add_argument("--adapter_path", type=str, default=None,
                        help="Path to LoRA adapter (None for base model)")
    parser.add_argument("--base_model", type=str, default="deepseek-ai/DeepSeek-OCR",
                        help="Base model name/path")

    # Output
    parser.add_argument("--output_dir", type=str, default="./ocr_results",
                        help="Output directory for batch processing")
    parser.add_argument("--output_file", type=str, help="Output file for single image")

    # Generation settings
    parser.add_argument("--max_tokens", type=int, default=2048,
                        help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Generation temperature")

    # Options
    parser.add_argument("--compare", action="store_true",
                        help="Compare base vs fine-tuned model")
    parser.add_argument("--no_4bit", action="store_true",
                        help="Disable 4-bit quantization")
    parser.add_argument("--use_vllm", action="store_true",
                        help="Use vLLM for inference")

    args = parser.parse_args()

    # Validate inputs
    if not args.image and not args.image_dir:
        parser.error("Either --image or --image_dir is required")

    # Comparison mode
    if args.compare:
        if not args.image:
            parser.error("--compare requires --image")
        if not args.adapter_path:
            parser.error("--compare requires --adapter_path")

        compare_models(
            args.image,
            args.adapter_path,
            args.output_file,
        )
        return

    # Load model
    model, tokenizer = load_finetuned_model(
        base_model_name=args.base_model,
        adapter_path=args.adapter_path,
        load_in_4bit=not args.no_4bit,
    )

    # Single image
    if args.image:
        print(f"Processing: {args.image}")
        result = run_ocr(
            model, tokenizer, args.image,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
        )

        if is_blank_page(result):
            print("\nBlank page detected ('rien à signaler') — skipped.")
            return

        print("\nOCR Result:")
        print("-" * 40)
        print(result)

        if args.output_file:
            with open(args.output_file, 'w', encoding='utf-8') as f:
                f.write(result)
            print(f"\nSaved to: {args.output_file}")

    # Batch processing
    elif args.image_dir:
        batch_process(
            model, tokenizer,
            args.image_dir,
            args.output_dir,
        )


if __name__ == "__main__":
    main()
