# DeepSeek-OCR Fine-tuning Pipeline for Structured Administrative Forms

## Overview

This project fine-tunes [DeepSeek-OCR](https://huggingface.co/deepseek-ai/DeepSeek-OCR) on a private corpus of scanned administrative forms, teaching the model to accurately extract handwritten and typed data into structured HTML tables.

The core challenge: standard OCR models handle printed text well, but struggle with domain-specific forms that have a fixed layout, mixed handwriting styles, and organization-specific terminology (Congolese French). Fine-tuning on matched image–ground-truth pairs addresses this directly.

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
         (QLoRA, 4-bit, 24GB GPU)
                    │
                    ▼
         Fine-tuned adapter
                    │
                    ▼
         inference_finetuned.py / evaluate.py
```

## Technical Highlights

- **Exact image–row matching** via a `page_num` column in each Excel file — no approximate distribution, every training example corresponds to the precise page the data was entered from.
- **Three image naming conventions handled automatically**: coded style (`SECTION_003.jpeg`), paged style (`Month Year page N.jpeg`), and numbered style (`1.jpg`, `2.jpg`).
- **PDF support**: months where scans are stored as PDFs are handled alongside JPEGs/PNGs.
- **Multi-section Excel files**: some Excel files span multiple image sections (e.g., `L_front.xlsx` covers pages 111–112); rows are distributed across the correct pages.
- **Robust month normalization**: handles French (Janvier, Fevrier), English (January, February), and common typos (Feburary, Fevier, Auot).
- **Per-section prompt instructions**: section-specific instructions in the training prompt guide the model on domain-specific formatting rules (e.g. unit designation stripping for B1/L_front/L_combat, total extraction for I_autorisation).
- **Date reformatting**: model is instructed to convert handwritten day/month/year to MM/DD/YYYY output format.
- **Blank page detection**: pages with "rien à signaler", "néant", or "r.a.s." are automatically skipped during inference.
- **QLoRA fine-tuning** with 4-bit quantization, gradient checkpointing, and gradient accumulation — runs on a single RTX 3090 (24GB) or A40 (48GB).
- **Selective section augmentation**: `--sections` and `--append` flags allow adding extra months for data-starved sections without duplicating well-represented ones.

## Project Structure

```
DeepSeek-OCR/
├── finetune/
│   ├── prepare_dataset.py      # Converts Excel + images → JSONL training pairs
│   ├── finetune_qlora.py       # QLoRA fine-tuning
│   ├── inference_finetuned.py  # Run inference with the fine-tuned adapter
│   └── requirements.txt
├── evaluation/
│   ├── evaluate.py             # Per-column accuracy evaluation against ground truth
│   └── preview_outputs.py      # Side-by-side model output vs ground truth viewer
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
│   ├── 95. February 2023/      # "coded" style: "SECTION_NNN.jpeg"
│   ├── 96. March 2023/         # "coded" style
│   ├── 97. April 2023/         # "numbered" style: "1.jpg", "2.jpg", ...
│   └── 98. May 2023/           # "numbered" style
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
      "content": "<image>\n<|grounding|>This is section A1. The document is in Congolese French. Convert the document to markdown. Dates are handwritten as day/month/year — reformat them to month/day/year (MM/DD/YYYY) in your output."
    },
    {
      "role": "assistant",
      "content": "<table><tr><td>annee</td><td>mois</td><td>grade</td>...</tr><tr><td>2023</td><td>Fevrier</td>...</tr></table>"
    }
  ]
}
```

**Current dataset size:** ~600+ training examples across 8 months (January–August 2023).

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

To add extra months for data-starved sections only (without duplicating existing data):

```bash
python finetune/prepare_dataset.py \
    --data_dir ./training_data \
    --output ./training_data/dataset.jsonl \
    --sections C_beer C_cig I_autorisation J K L_front L_combat \
    --append
```

### 3. Fine-tune

```bash
nohup python finetune/finetune_qlora.py \
    --train_data ./training_data/dataset_train.jsonl \
    --val_data   ./training_data/dataset_val.jsonl \
    --output_dir ./finetuned_model_v4b \
    --epochs 10 --lora_r 64 --lora_alpha 128 \
    --batch_size 1 --grad_accum 8 \
    > training.log 2>&1 &
```

**Note:** `--batch_size` must be 1 — batch size 2 causes a 5D tensor error in the SAM encoder.

To resume from a checkpoint after interruption:

```bash
python finetune/finetune_qlora.py \
    ... \
    --resume_from_checkpoint ./finetuned_model_v4/checkpoint-300
```

### 4. Run inference

```bash
python finetune/inference_finetuned.py \
    --image        ./test.jpg \
    --adapter_path ./finetuned_model_v4b
```

### 5. Evaluate

```bash
# Preview raw outputs vs ground truth, save to JSON
python evaluation/preview_outputs.py \
    --val_data     ./training_data/dataset_val.jsonl \
    --adapter_path ./finetuned_model_v4b \
    --n 20 \
    --save ./evaluation/results.json

# Convert saved JSON to browsable HTML pages (no GPU needed)
python evaluation/view_results.py \
    --input ./evaluation/results.json \
    --output_dir ./evaluation/results/
# Open evaluation/results/index.html in your browser

# Score from saved JSON (fast, no GPU needed)
python evaluation/evaluate.py \
    --from_json ./evaluation/results.json \
    --output_dir ./evaluation/eval_results

# Or run inference + score in one step
python evaluation/evaluate.py \
    --val_data     ./training_data/dataset_val.jsonl \
    --adapter_path ./finetuned_model_v4b \
    --output_dir   ./evaluation/eval_results
```

## RunPod Setup

Recommended GPU: **RTX 3090 (24GB)** — if unavailable, **A40 (48GB)**.
Use a **Network Volume** for storage (survives pod termination, ~$0.07/GB/mo).

```bash
# Copy rclone config from local
ssh -p <PORT> root@<IP> "mkdir -p ~/.config/rclone"
scp -P <PORT> ~/.config/rclone/rclone.conf root@<IP>:~/.config/rclone/rclone.conf

# Clone repo and install
git clone https://github.com/<your-repo>/DeepSeek-OCR.git OCR_pipeline
cd OCR_pipeline
pip install -r finetune/requirements.txt

# Sync Excel data from Dropbox
rclone sync "dropbox_OCR:18. Administrative data NDC/04. Data entry/2023" \
    ./training_data/excel/ --progress

# Sync images per month
rclone sync "dropbox_OCR:.../94. January 2023" "./training_data/images/94. January 2023/" --progress
```

## Evaluation Results (finetuned_model_v4b)

Evaluated on 37 validation examples across all sections (January–August 2023).

| Metric | Value |
|---|---|
| Examples evaluated | 32 / 37 |
| **Avg cell accuracy** | **38.5%** |

**Strong columns (≥80%):**

| Column | Accuracy |
|---|---|
| Prix | 100% |
| nombre_de_payeurs_de_la_taxe | 100% |
| l_prix_de_vente_a_l_unitefc | 100% |
| currency | 100% |
| l_quantite_vendue | 89.7% |
| l_remain_last_mth | 89.7% |
| l_date_de_vente | 87.2% |
| l_quantite_distribuee | 84.6% |
| l_overall_total | 82.1% |

**Weak areas:** Observations column (transcription style mismatch), incident-record section (M3), and D_cig dates.

The evaluation script strips DB-only metadata columns (`annee`, `mois`, sequential row counters) from both GT and model output before scoring, to only compare what is actually visible in the scanned image.

## Roadmap

- [x] Data pipeline: Excel + images → JSONL training pairs
- [x] Exact page-level image–row matching via `page_num`
- [x] Three image naming styles (coded, paged, numbered) + PDF support
- [x] 8-month dataset (January–August 2023, ~600 examples)
- [x] QLoRA fine-tuning (v2: Jan–Feb; v3: Jan–Apr; v4b: Jan–Aug, 38.5% cell accuracy)
- [x] Evaluation pipeline with per-column accuracy reporting and HTML result viewer
- [ ] Improve weak sections (M3 incidents, Observations text normalization)
- [ ] Expand to remaining months of 2023

## Upstream: DeepSeek-OCR

This project builds on [DeepSeek-OCR](https://huggingface.co/deepseek-ai/DeepSeek-OCR) by deepseek-ai — a vision-language model optimized for document OCR with context-aware optical compression.

- [Paper (arXiv)](https://arxiv.org/abs/2510.18234)
- [Hugging Face model](https://huggingface.co/deepseek-ai/DeepSeek-OCR)
- [vLLM inference docs](https://docs.vllm.ai/projects/recipes/en/latest/DeepSeek/DeepSeek-OCR.html)
