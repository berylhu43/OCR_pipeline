#!/usr/bin/env python3
"""
Preview model outputs vs ground truth for the validation set.

Prints raw model output and ground truth side by side so you can
inspect format differences before writing the comparison logic.

Usage:
    python evaluation/preview_outputs.py \
        --val_data ./training_data/dataset_val.jsonl \
        --adapter_path ./finetuned_model_v3 \
        --n 5
"""

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "finetune"))
from inference_finetuned import load_finetuned_model, run_ocr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_data",     required=True)
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument("--base_model",   default="deepseek-ai/DeepSeek-OCR")
    parser.add_argument("--n",            type=int, default=5, help="Number of examples to show")
    parser.add_argument("--save",         type=str, default=None, help="Save outputs to JSON file")
    args = parser.parse_args()

    examples = []
    with open(args.val_data, encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line.strip()))
    examples = examples[:args.n]

    print(f"Loading model...")
    model, tokenizer = load_finetuned_model(
        base_model_name=args.base_model,
        adapter_path=args.adapter_path,
    )

    saved = []
    for i, ex in enumerate(examples):
        image_path = ex["image"]
        gt = ex["conversations"][1]["content"]
        prompt = ex["conversations"][0]["content"]

        print(f"\n{'='*70}")
        print(f"[{i+1}/{len(examples)}] {image_path}")
        print(f"PROMPT: {prompt[:120]}")
        print(f"{'='*70}")

        try:
            pred = run_ocr(model, tokenizer, image_path, prompt=prompt)
        except Exception as e:
            pred = f"ERROR: {e}"

        print(f"\n--- GROUND TRUTH ---\n{gt}\n")
        print(f"--- MODEL OUTPUT ---\n{pred}\n")

        saved.append({"image": image_path, "ground_truth": gt, "model_output": pred})

    if args.save:
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(saved, f, indent=2, ensure_ascii=False)
        print(f"\nSaved to {args.save}")


if __name__ == "__main__":
    main()
