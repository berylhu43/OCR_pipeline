# DeepSeek-OCR Fine-tuning Pipeline for Structured Administrative Forms

> **Work in Progress** — This project is actively being developed. The data pipeline is complete and tested; QLoRA fine-tuning is next.

## Overview

This project fine-tunes [DeepSeek-OCR](https://huggingface.co/deepseek-ai/DeepSeek-OCR) on a private corpus of scanned administrative forms, teaching the model to accurately extract handwritten and typed data into structured HTML tables.

The core challenge: standard OCR models handle printed text well, but struggle with domain-specific forms that have a fixed layout, mixed handwriting styles, and organization-specific terminology. Fine-tuning on matched image–ground-truth pairs addresses this directly.

## What It Does

```
Scanned form images  +  Excel ground truth
         │                      │
         └──────────┬───────────┘
                    ▼
         prepare_dataset.py
         (image ↔ row matching via page_num)
                    │
                    ▼
         JSONL training pairs
         (image path + HTML table)
                    │
                    ▼
         finetune_qlora.py
         (QLoRA, 4-bit, 8GB GPU)
                    │
                    ▼
         Fine-tuned adapter
                    │
                    ▼
         inference_finetuned.py
```

## Technical Highlights

- **Exact image–row matching** via a `page_num` column in each Excel file — no approximate distribution, every training example corresponds to the precise page the data was entered from.
- **Two image naming conventions handled automatically**: coded style (`SECTION_003.jpeg`) for some months, paged style (`Month Year page N.jpeg`) for others.
- **Multi-section Excel files**: some Excel files span multiple image sections (e.g., `L_front.xlsx` covers pages 111–112); rows are distributed across the correct pages.
- **Robust month normalization**: handles French (Janvier, Fevrier), English (January, February), and common typos (Feburary, Fevier).
- **Secondary header detection**: files with sub-header rows are filtered automatically so they don't pollute training data.
- **QLoRA fine-tuning** with 4-bit quantization, gradient checkpointing, and gradient accumulation — designed to run on a single 8GB consumer GPU.

## Project Structure

```
DeepSeek-OCR/
├── finetune/
│   ├── prepare_dataset.py      # Converts Excel + images → JSONL training pairs
│   ├── finetune_qlora.py       # QLoRA fine-tuning (optimized for 8GB VRAM)
│   ├── inference_finetuned.py  # Run inference with the fine-tuned adapter
│   └── requirements.txt
├── DeepSeek-OCR-master/        # Upstream DeepSeek-OCR inference code (vLLM / HF)
├── training_data/              # Gitignored — private data via Dropbox symlink
│   ├── images/                 # Scanned form images, organized by month
│   ├── excel/                  # Ground truth Excel files, one per form section
│   ├── dataset_train.jsonl     # Generated training set
│   └── dataset_val.jsonl       # Generated validation set
└── .gitignore
```

## Data Pipeline

### Input structure

```
training_data/
├── images/
│   ├── 94. January 2023/       # "paged" style: "January 2023 page N.jpeg"
│   └── 95. February 2023/      # "coded" style: "SECTION_NNN.jpeg"
└── excel/
    ├── A.xlsx                  # Section A: Effectifs summary
    ├── A1.xlsx                 # Section A1: Recrutement (pages 3–14)
    ├── B1.xlsx                 # Section B1 (pages 24–35)
    ├── D_beer.xlsx             # Section D, beverage sub-category
    ├── L_front.xlsx            # Section L1+L2 (pages 111–112, multi-page)
    └── ...                     # 25 Excel files total covering ~142 pages
```

Each Excel file includes a `page_num` column mapping each data row to the exact scanned page it appears on.

### Output format

Each training example pairs one image with its ground-truth HTML table:

```json
{
  "image": "training_data/images/95. February 2023/A1_003.jpeg",
  "conversations": [
    {
      "role": "user",
      "content": "<image>\n<|grounding|>Convert the document to markdown."
    },
    {
      "role": "assistant",
      "content": "<table><tr><td>annee</td><td>mois</td><td>grade</td>...</tr><tr><td>2023</td><td>Fevrier</td>...</tr></table>"
    }
  ]
}
```

**Current dataset size:** 158 training examples across 2 months of data (January and February 2023).

## Usage

### 1. Install dependencies

```bash
pip install -r finetune/requirements.txt
```

### 2. Prepare dataset

```bash
python finetune/prepare_dataset.py \
    --data_dir ./training_data \
    --output ./training_data/dataset.jsonl
```

Outputs `dataset_train.jsonl` (90%) and `dataset_val.jsonl` (10%).

### 3. Fine-tune

```bash
python finetune/finetune_qlora.py \
    --train_data ./training_data/dataset_train.jsonl \
    --val_data   ./training_data/dataset_val.jsonl \
    --output_dir ./finetuned_model
```

### 4. Run inference

```bash
python finetune/inference_finetuned.py \
    --image        ./test.jpg \
    --adapter_path ./finetuned_model
```

## Roadmap

- [x] Data pipeline: Excel + images → JSONL training pairs
- [x] Exact page-level image–row matching via `page_num`
- [x] Two-month dataset (January & February 2023, 158 examples)
- [ ] QLoRA fine-tuning run and evaluation
- [ ] Expand dataset to remaining months of 2023
- [ ] Benchmark: pre-fine-tune vs. post-fine-tune extraction accuracy

## Upstream: DeepSeek-OCR

This project builds on [DeepSeek-OCR](https://huggingface.co/deepseek-ai/DeepSeek-OCR) by deepseek-ai — a vision-language model optimized for document OCR with context-aware optical compression.

- [Paper (arXiv)](https://arxiv.org/abs/2510.18234)
- [Hugging Face model](https://huggingface.co/deepseek-ai/DeepSeek-OCR)
- [vLLM inference docs](https://docs.vllm.ai/projects/recipes/en/latest/DeepSeek/DeepSeek-OCR.html)

### Quick inference with upstream model (vLLM)

```python
from vllm import LLM, SamplingParams
from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor
from PIL import Image

llm = LLM(
    model="deepseek-ai/DeepSeek-OCR",
    enable_prefix_caching=False,
    mm_processor_cache_gb=0,
    logits_processors=[NGramPerReqLogitsProcessor]
)

image = Image.open("your_form.jpg").convert("RGB")
prompt = "<image>\n<|grounding|>Convert the document to markdown."

outputs = llm.generate(
    [{"prompt": prompt, "multi_modal_data": {"image": image}}],
    SamplingParams(
        temperature=0.0, max_tokens=8192,
        extra_args=dict(ngram_size=30, window_size=90,
                        whitelist_token_ids={128821, 128822}),
        skip_special_tokens=False
    )
)
print(outputs[0].outputs[0].text)
```
