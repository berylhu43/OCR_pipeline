#!/usr/bin/env python3
"""
Evaluation Script for Fine-tuned DeepSeek-OCR

Runs inference on the validation set and compares against ground truth.
Reports per-column accuracy to identify systematic errors (e.g. dates, names).

Usage:
    # Full validation set:
    python evaluation/evaluate.py \
        --val_data ./training_data/dataset_val.jsonl \
        --adapter_path ./finetuned_model_v3 \
        --output_dir ./evaluation/results

    # Quick test on first N examples:
    python evaluation/evaluate.py \
        --val_data ./training_data/dataset_val.jsonl \
        --adapter_path ./finetuned_model_v3 \
        --limit 10
"""

import sys
import json
import re
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from html.parser import HTMLParser

# Add finetune/ to path so we can reuse load_finetuned_model / run_ocr
sys.path.insert(0, str(Path(__file__).parent.parent / "finetune"))
from inference_finetuned import load_finetuned_model, run_ocr


# ---------------------------------------------------------------------------
# HTML / markdown table parsing
# ---------------------------------------------------------------------------

class _TableParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows: List[List[str]] = []
        self._row: Optional[List[str]] = None
        self._cell: Optional[str] = None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self._row = []
        elif tag == "td" and self._row is not None:
            self._cell = ""

    def handle_endtag(self, tag):
        if tag == "tr" and self._row is not None:
            self.rows.append(self._row)
            self._row = None
        elif tag == "td" and self._cell is not None:
            self._row.append(self._cell.strip())
            self._cell = None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell += data


def _parse_html(html: str) -> Tuple[Optional[List[str]], List[List[str]]]:
    m = re.search(r'<table.*?>(.*?)</table>', html, re.DOTALL | re.IGNORECASE)
    if not m:
        return None, []
    p = _TableParser()
    p.feed(m.group(0))
    if not p.rows:
        return None, []
    return p.rows[0], p.rows[1:]


def _parse_markdown(text: str) -> Tuple[Optional[List[str]], List[List[str]]]:
    lines = [l for l in text.strip().splitlines() if l.strip().startswith("|")]
    if len(lines) < 2:
        return None, []

    def split(line):
        return [c.strip() for c in line.strip().strip("|").split("|")]

    headers = split(lines[0])
    data_start = 2 if len(lines) > 1 and re.match(r"^\|[\s\-|:]+\|$", lines[1]) else 1
    return headers, [split(l) for l in lines[data_start:]]


def parse_table(text: str) -> Tuple[Optional[List[str]], List[List[str]]]:
    """Parse HTML or markdown table; return (headers, data_rows)."""
    if "<table" in text.lower():
        return _parse_html(text)
    if "|" in text:
        return _parse_markdown(text)
    return None, []


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def _norm_date(v: str) -> str:
    """
    Normalize date strings to MM/DD/YYYY for comparison.
    Handles:
      - MM/DD/YYYY or M/D/YYYY  → already target format
      - DD/MM/YYYY              → swap day and month (handwritten convention)
      - YYYY-MM-DD              → reformat to MM/DD/YYYY
      - MM/YYYY or YYYY-MM      → normalize to MM/YYYY
    Returns original string lowercased if no pattern matches.

    Ambiguity note: D/M/YYYY and M/D/YYYY are indistinguishable when both
    values are ≤ 12. We assume the GT is already in MM/DD/YYYY and the model
    output is in DD/MM/YYYY (handwritten), so we always swap group(1)/group(2).
    The GT dates coming from Excel are stored as YYYY-MM-DD and converted here.
    """
    v = v.strip()
    # YYYY-MM-DD (from Excel GT)
    m = re.match(r'^(\d{4})-(\d{2})-(\d{2})$', v)
    if m:
        return f"{m.group(2)}/{m.group(3)}/{m.group(1)}"
    # YYYY-MM (partial date from Excel GT)
    m = re.match(r'^(\d{4})-(\d{2})$', v)
    if m:
        return f"{m.group(2)}/{m.group(1)}"
    # DD/MM/YYYY or D/M/YYYY (handwritten, model output before retraining)
    # Target format is MM/DD/YYYY so swap day and month
    m = re.match(r'^(\d{1,2})/(\d{1,2})/(\d{4})$', v)
    if m:
        day, month, year = m.group(1), m.group(2), m.group(3)
        return f"{month.zfill(2)}/{day.zfill(2)}/{year}"
    # MM/YYYY (month/year only)
    m = re.match(r'^(\d{1,2})/(\d{4})$', v)
    if m:
        return f"{m.group(1).zfill(2)}/{m.group(2)}"
    return v.lower()


def _norm_number(v: str) -> str:
    """Normalize numeric values: strip currency suffixes, spaces, dots used as thousands sep."""
    v = v.strip()
    # Remove currency suffix (FC, Fc, etc.) and trailing spaces
    v = re.sub(r'\s*[Ff][Cc]$', '', v).strip()
    # Remove dots/spaces used as thousands separators (e.g. 147.000 or 147 000)
    # Only do this if remaining string is purely numeric after removal
    candidate = re.sub(r'[.\s]', '', v)
    if candidate.isdigit():
        return candidate
    # Handle comma as decimal separator (0,14 → 0.14)
    v = v.replace(',', '.')
    return v.lower()


def _norm(v: str) -> str:
    """General normalization: try date, then number, then lowercase."""
    if v is None:
        return ""
    v = str(v).strip()
    if not v:
        return ""
    # Try date pattern
    normed = _norm_date(v)
    if normed != v.lower():
        return normed
    # Try number normalization
    return _norm_number(v)


def _is_empty_row(row: List[str]) -> bool:
    """Return True if all cells in the row are blank."""
    return all(c.strip() in ("", "/", "-") for c in row)


def cells_match(pred: str, gt: str) -> bool:
    return _norm(pred) == _norm(gt)


# ---------------------------------------------------------------------------
# Per-example evaluation
# ---------------------------------------------------------------------------

def evaluate_example(pred_text: str, gt_text: str) -> Dict:
    """
    Compare predicted output to ground truth table.

    Matching strategy:
    - Columns matched by position (model uses verbose French headers, GT uses short names)
    - Skip header rows in model output that contain no numeric data (multi-row headers)
    - Skip empty rows in model output (blank form lines)
    - Normalize dates (DD/MM/YYYY → YYYY-MM-DD) and currency (147.000FC → 147000)

    Returns a dict with:
      - overall_cell_accuracy: fraction of cells that match exactly
      - column_accuracy: per-column breakdown (keyed by GT column name)
      - row_count_match: whether data row counts match after filtering
      - pred_parsed / gt_parsed: whether tables could be parsed
    """
    # Strip bounding-box preamble emitted by DeepSeek-OCR grounding tokens
    pred_text = re.sub(r'<\|ref\|>.*?<\|/det\|>', '', pred_text, flags=re.DOTALL).strip()

    pred_headers, pred_rows = parse_table(pred_text)
    gt_headers, gt_rows = parse_table(gt_text)

    # Remove empty rows from model output (blank form lines)
    pred_rows = [r for r in pred_rows if not _is_empty_row(r)]

    # Model sometimes emits extra header rows (e.g. merged headers with no numbers).
    # Drop leading pred rows that are all non-numeric and look like sub-headers.
    while pred_rows and all(not re.search(r'\d', c) for c in pred_rows[0]):
        pred_rows = pred_rows[1:]

    result = {
        "pred_parsed": pred_headers is not None,
        "gt_parsed": gt_headers is not None,
        "pred_rows": len(pred_rows),
        "gt_rows": len(gt_rows),
        "row_count_match": len(pred_rows) == len(gt_rows),
        "overall_cell_accuracy": None,
        "column_accuracy": {},
    }

    if pred_headers is None or gt_headers is None:
        return result

    # Determine column offset: model often prepends a row-number column not in GT.
    # Match columns by position (GT index → pred index + offset).
    # Detect by checking if pred has one more column than GT and first col is numeric.
    col_offset = 0
    if len(pred_headers) == len(gt_headers) + 1:
        # Check first data row
        for r in pred_rows[:3]:
            if r and re.match(r'^\d+$', r[0].strip()):
                col_offset = 1
                break

    col_stats: Dict[str, Dict] = defaultdict(lambda: {"correct": 0, "total": 0, "errors": []})
    total_correct = 0
    total_cells = 0

    n_rows = min(len(pred_rows), len(gt_rows))
    for row_idx in range(n_rows):
        pred_row = pred_rows[row_idx]
        gt_row = gt_rows[row_idx]

        # Match by position: gt_col_idx → pred_col_idx + col_offset
        for gt_idx, gt_col_name in enumerate(gt_headers):
            pred_idx = gt_idx + col_offset
            gt_val = gt_row[gt_idx] if gt_idx < len(gt_row) else ""
            pred_val = pred_row[pred_idx] if pred_idx < len(pred_row) else ""

            match = cells_match(pred_val, gt_val)
            col_stats[gt_col_name]["total"] += 1
            total_cells += 1
            if match:
                col_stats[gt_col_name]["correct"] += 1
                total_correct += 1
            else:
                if len(col_stats[gt_col_name]["errors"]) < 5:
                    col_stats[gt_col_name]["errors"].append(
                        {"row": row_idx, "pred": pred_val, "gt": gt_val}
                    )

    result["overall_cell_accuracy"] = total_correct / total_cells if total_cells > 0 else 0
    result["column_accuracy"] = {
        col: {
            "accuracy": s["correct"] / s["total"] if s["total"] > 0 else 0,
            "correct": s["correct"],
            "total": s["total"],
            "errors": s["errors"],
        }
        for col, s in col_stats.items()
    }
    return result


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(
    val_data_path: str,
    adapter_path: Optional[str],
    base_model: str,
    output_dir: str,
    limit: Optional[int],
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load validation examples
    examples = []
    with open(val_data_path, encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line.strip()))
    if limit:
        examples = examples[:limit]
    print(f"Evaluating {len(examples)} examples from {val_data_path}")

    # Load model once
    model, tokenizer = load_finetuned_model(
        base_model_name=base_model,
        adapter_path=adapter_path,
    )

    results = []
    agg: Dict[str, Dict] = defaultdict(lambda: {"correct": 0, "total": 0, "errors": []})

    for i, ex in enumerate(examples):
        image_path = ex["image"]
        gt_text = ex["conversations"][1]["content"]
        prompt = ex["conversations"][0]["content"]

        print(f"[{i+1}/{len(examples)}] {Path(image_path).name}", end=" ... ", flush=True)

        try:
            pred_text = run_ocr(model, tokenizer, image_path, prompt=prompt)
            ev = evaluate_example(pred_text, gt_text)

            results.append({
                "image": image_path,
                "pred": pred_text,
                "gt": gt_text,
                "eval": ev,
            })

            # Aggregate per-column stats
            for col, stats in ev["column_accuracy"].items():
                agg[col]["correct"] += stats["correct"]
                agg[col]["total"] += stats["total"]
                if len(agg[col]["errors"]) < 10:
                    agg[col]["errors"].extend(stats["errors"])

            acc = ev.get("overall_cell_accuracy")
            print(f"cell_acc={acc:.1%}" if acc is not None else "parse_failed")

        except Exception as e:
            print(f"ERROR: {e}")
            results.append({"image": image_path, "error": str(e)})

    # Build summary
    valid = [r for r in results if "eval" in r and r["eval"].get("overall_cell_accuracy") is not None]
    avg_acc = sum(r["eval"]["overall_cell_accuracy"] for r in valid) / len(valid) if valid else 0

    # Sort columns by error rate (worst first)
    def error_rate(stats):
        return 1 - (stats["correct"] / stats["total"]) if stats["total"] > 0 else 0

    per_col = {
        col: {
            "accuracy": s["correct"] / s["total"] if s["total"] > 0 else 0,
            "correct": s["correct"],
            "total": s["total"],
            "sample_errors": s["errors"][:5],
        }
        for col, s in sorted(agg.items(), key=lambda x: error_rate(x[1]), reverse=True)
    }

    summary = {
        "n_examples": len(examples),
        "n_valid": len(valid),
        "n_errors": len(examples) - len(valid),
        "avg_cell_accuracy": avg_acc,
        "per_column_accuracy": per_col,
    }

    # Save
    results_file = output_path / "results.json"
    summary_file = output_path / "summary.json"

    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Print summary table
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Examples evaluated : {len(valid)} / {len(examples)}")
    print(f"Avg cell accuracy  : {avg_acc:.1%}")
    print()
    print(f"{'Column':<30} {'Accuracy':>10} {'Correct':>8} {'Total':>8}")
    print("-" * 60)
    for col, stats in per_col.items():
        acc = stats["accuracy"]
        flag = "  <<<" if acc < 0.7 else ""
        print(f"{col:<30} {acc:>9.1%} {stats['correct']:>8} {stats['total']:>8}{flag}")
        # Show a sample error for bad columns
        if acc < 0.9 and stats["sample_errors"]:
            e = stats["sample_errors"][0]
            print(f"  e.g. pred={e['pred']!r:20}  gt={e['gt']!r}")

    print(f"\nFull results : {results_file}")
    print(f"Summary      : {summary_file}")
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned DeepSeek-OCR")
    parser.add_argument("--val_data",     required=True,  help="Validation JSONL path")
    parser.add_argument("--adapter_path", default=None,   help="LoRA adapter path (None = base model)")
    parser.add_argument("--base_model",   default="deepseek-ai/DeepSeek-OCR")
    parser.add_argument("--output_dir",   default="./evaluation/results")
    parser.add_argument("--limit",        type=int, default=None, help="Only evaluate first N examples")
    args = parser.parse_args()

    run_evaluation(
        val_data_path=args.val_data,
        adapter_path=args.adapter_path,
        base_model=args.base_model,
        output_dir=args.output_dir,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
