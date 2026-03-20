#!/usr/bin/env python3
"""
Evaluation Script for Fine-tuned DeepSeek-OCR

Two modes:
  1. Run inference + evaluate (slow):
     python evaluation/evaluate.py \
         --val_data ./training_data/dataset_val.jsonl \
         --adapter_path ./finetuned_model_v4b

  2. Evaluate from existing results JSON (fast, no GPU needed):
     python evaluation/evaluate.py \
         --from_json ./evaluation/results.json
"""

import sys
import json
import re
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from html.parser import HTMLParser


# ---------------------------------------------------------------------------
# HTML table parsing
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
        elif tag in ("td", "th") and self._row is not None:
            self._cell = ""

    def handle_endtag(self, tag):
        if tag == "tr" and self._row is not None:
            if self._row:
                self.rows.append(self._row)
            self._row = None
        elif tag in ("td", "th") and self._cell is not None:
            self._row.append(self._cell.strip())
            self._cell = None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell += data


def parse_html_table(html: str) -> Tuple[Optional[List[str]], List[List[str]]]:
    """Extract the last <table> in html. Returns (headers, data_rows)."""
    tables = list(re.finditer(r'<table[^>]*>.*?</table>', html, re.DOTALL | re.IGNORECASE))
    if not tables:
        return None, []
    p = _TableParser()
    p.feed(tables[-1].group(0))
    if not p.rows:
        return None, []
    return p.rows[0], p.rows[1:]


def clean_model_output(text: str) -> str:
    """Strip grounding/detection tags emitted by base model."""
    text = re.sub(r'<\|ref\|>.*?<\|/ref\|>', '', text, flags=re.DOTALL)
    text = re.sub(r'<\|det\|>.*?<\|/det\|>', '', text, flags=re.DOTALL)
    return text.strip()


# ---------------------------------------------------------------------------
# Column alignment helpers
# ---------------------------------------------------------------------------

_DATE_META_NAMES = {"annee", "year", "année", "mois", "month", "an"}
# Legacy alias used elsewhere
_DATE_COL_NAMES = _DATE_META_NAMES


def _is_date_meta_col(name: str) -> bool:
    """Return True for annee/mois/year columns added by data entry, not visible in image."""
    return name.strip().lower() in _DATE_META_NAMES


def _is_row_number_col(col_values: List[str]) -> bool:
    """Return True if a column looks like a sequential row-number column."""
    nums = [v.strip() for v in col_values if v.strip()]
    if not nums:
        return False
    return sum(1 for v in nums if re.match(r'^\d+$', v)) / len(nums) > 0.8


def _is_sequential_int_col(col_values: List[str]) -> bool:
    """Return True if column values are consecutive integers (e.g. 16,17,18 or 1,2,3).
    Used to detect DB row-counter columns not visible in the handwritten image."""
    nums = [v.strip() for v in col_values if v.strip()]
    if len(nums) < 2:
        return False
    try:
        ints = [int(v) for v in nums]
    except ValueError:
        return False
    return all(ints[i] == ints[i - 1] + 1 for i in range(1, len(ints)))


def align_columns(
    gt_headers: List[str],
    gt_rows: List[List[str]],
    pred_headers: List[str],
    pred_rows: List[List[str]],
) -> Tuple[List[str], List[List[str]], List[List[str]]]:
    """
    Align GT and pred columns for fair comparison:
    1. Strip date-meta (annee/mois) and sequential row-counter columns from GT
       (these are DB artifacts not visible in the handwritten image)
    2. Strip date-meta columns from pred by header name
    3. Strip leading row-number column from pred if pred still has more cols than GT
    Returns (aligned_gt_headers, aligned_gt_rows, aligned_pred_rows)
    """
    # Step 1: strip date-meta AND sequential row-counter columns from GT
    gt_keep = []
    for i, h in enumerate(gt_headers):
        if _is_date_meta_col(h):
            continue
        col_vals = [r[i] if i < len(r) else "" for r in gt_rows]
        if _is_sequential_int_col(col_vals):
            continue  # e.g. total_ou_detail_vente_l = 16,17,18... (row ID, not in image)
        gt_keep.append(i)
    aligned_gt_headers = [gt_headers[i] for i in gt_keep]
    aligned_gt_rows = [[r[i] if i < len(r) else "" for i in gt_keep] for r in gt_rows]

    # Step 2: strip date-meta columns from pred by header name
    pred_keep = [i for i, h in enumerate(pred_headers) if not _is_date_meta_col(h)]
    trimmed_pred_headers = [pred_headers[i] for i in pred_keep]
    trimmed_pred_rows = [[r[i] if i < len(r) else "" for i in pred_keep] for r in pred_rows]

    # Step 3: strip leading row-number column from pred if pred has more cols than GT
    pred_offset = 0
    if trimmed_pred_rows and len(trimmed_pred_headers) > len(aligned_gt_headers):
        first_col_vals = [r[0] if r else "" for r in trimmed_pred_rows]
        if _is_row_number_col(first_col_vals):
            pred_offset = 1

    n_cols = len(aligned_gt_headers)
    aligned_pred_rows = [
        [r[pred_offset + i] if (pred_offset + i) < len(r) else "" for i in range(n_cols)]
        for r in trimmed_pred_rows
    ]

    return aligned_gt_headers, aligned_gt_rows, aligned_pred_rows


# ---------------------------------------------------------------------------
# Value normalization
# ---------------------------------------------------------------------------

def _norm_date(v: str) -> str:
    v = v.strip()
    # YYYY-MM-DD
    m = re.match(r'^(\d{4})-(\d{2})-(\d{2})$', v)
    if m:
        return f"{m.group(2)}/{m.group(3)}/{m.group(1)}"
    # YYYY-MM
    m = re.match(r'^(\d{4})-(\d{2})$', v)
    if m:
        return f"{m.group(2)}/{m.group(1)}"
    # D/M/YYYY or M/D/YYYY — normalize order (assume day first for handwritten)
    m = re.match(r'^(\d{1,2})/(\d{1,2})/(\d{4})$', v)
    if m:
        return f"{m.group(2).zfill(2)}/{m.group(1).zfill(2)}/{m.group(3)}"
    # MM/YYYY
    m = re.match(r'^(\d{1,2})/(\d{4})$', v)
    if m:
        return f"{m.group(1).zfill(2)}/{m.group(2)}"
    return v.lower()


def _norm_number(v: str) -> str:
    v = re.sub(r'\s*[Ff][Cc]$', '', v).strip()
    candidate = re.sub(r'[.\s]', '', v)
    if candidate.isdigit():
        return candidate
    return v.replace(',', '.').lower()


def normalize(v: str) -> str:
    if not v:
        return ""
    v = str(v).strip()
    normed = _norm_date(v)
    if normed != v.lower():
        return normed
    return _norm_number(v)


def cells_match(pred: str, gt: str) -> bool:
    return normalize(pred) == normalize(gt)


def _is_empty_row(row: List[str]) -> bool:
    return all(c.strip() in ("", "/", "-", "—") for c in row)


# ---------------------------------------------------------------------------
# Per-example evaluation
# ---------------------------------------------------------------------------

def evaluate_example(pred_text: str, gt_text: str) -> Dict:
    pred_text = clean_model_output(pred_text)
    pred_headers, pred_rows = parse_html_table(pred_text)
    gt_headers, gt_rows = parse_html_table(gt_text)

    result = {
        "pred_parsed": pred_headers is not None,
        "gt_parsed": gt_headers is not None,
        "pred_rows": 0,
        "gt_rows": len(gt_rows) if gt_rows else 0,
        "row_count_match": False,
        "overall_cell_accuracy": None,
        "column_accuracy": {},
    }

    if pred_headers is None or gt_headers is None:
        return result

    # Remove empty rows and extra sub-header rows from pred
    pred_rows = [r for r in pred_rows if not _is_empty_row(r)]
    while pred_rows and all(not re.search(r'\d', c) for c in pred_rows[0]):
        pred_rows = pred_rows[1:]

    result["pred_rows"] = len(pred_rows)
    result["row_count_match"] = len(pred_rows) == len(gt_rows)

    # Align columns: strip annee/mois from GT, row-num from pred
    aligned_gt_headers, aligned_gt_rows, aligned_pred_rows = align_columns(
        gt_headers, gt_rows, pred_headers, pred_rows
    )

    col_stats: Dict[str, Dict] = defaultdict(lambda: {"correct": 0, "total": 0, "errors": []})
    total_correct = 0
    total_cells = 0

    n_rows = min(len(aligned_pred_rows), len(aligned_gt_rows))
    for row_idx in range(n_rows):
        pred_row = aligned_pred_rows[row_idx]
        gt_row = aligned_gt_rows[row_idx]
        for col_idx, col_name in enumerate(aligned_gt_headers):
            gt_val = gt_row[col_idx] if col_idx < len(gt_row) else ""
            pred_val = pred_row[col_idx] if col_idx < len(pred_row) else ""
            match = cells_match(pred_val, gt_val)
            col_stats[col_name]["total"] += 1
            total_cells += 1
            if match:
                col_stats[col_name]["correct"] += 1
                total_correct += 1
            elif len(col_stats[col_name]["errors"]) < 3:
                col_stats[col_name]["errors"].append(
                    {"row": row_idx, "pred": pred_val, "gt": gt_val}
                )

    result["overall_cell_accuracy"] = total_correct / total_cells if total_cells > 0 else 0.0
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
# Aggregate & print
# ---------------------------------------------------------------------------

def aggregate_and_print(results: List[Dict]) -> Dict:
    valid = [r for r in results if "eval" in r and r["eval"].get("overall_cell_accuracy") is not None]
    agg: Dict[str, Dict] = defaultdict(lambda: {"correct": 0, "total": 0, "errors": []})

    for r in valid:
        for col, stats in r["eval"]["column_accuracy"].items():
            agg[col]["correct"] += stats["correct"]
            agg[col]["total"] += stats["total"]
            if len(agg[col]["errors"]) < 5:
                agg[col]["errors"].extend(stats["errors"])

    avg_acc = sum(r["eval"]["overall_cell_accuracy"] for r in valid) / len(valid) if valid else 0

    def error_rate(s):
        return 1 - s["correct"] / s["total"] if s["total"] > 0 else 0

    per_col = {
        col: {
            "accuracy": s["correct"] / s["total"] if s["total"] > 0 else 0,
            "correct": s["correct"],
            "total": s["total"],
            "sample_errors": s["errors"][:3],
        }
        for col, s in sorted(agg.items(), key=lambda x: error_rate(x[1]), reverse=True)
    }

    summary = {
        "n_examples": len(results),
        "n_evaluated": len(valid),
        "n_errors": len(results) - len(valid),
        "avg_cell_accuracy": avg_acc,
        "per_column_accuracy": per_col,
    }

    print("\n" + "=" * 65)
    print("EVALUATION SUMMARY")
    print("=" * 65)
    print(f"Examples evaluated : {len(valid)} / {len(results)}")
    print(f"Avg cell accuracy  : {avg_acc:.1%}")
    print()
    print(f"{'Column':<35} {'Accuracy':>10} {'Correct':>8} {'Total':>8}")
    print("-" * 65)
    for col, stats in per_col.items():
        acc = stats["accuracy"]
        flag = "  <<<" if acc < 0.7 else ""
        print(f"{col:<35} {acc:>9.1%} {stats['correct']:>8} {stats['total']:>8}{flag}")
        if acc < 0.9 and stats["sample_errors"]:
            e = stats["sample_errors"][0]
            print(f"    e.g. pred={e['pred']!r:25}  gt={e['gt']!r}")

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned DeepSeek-OCR")
    parser.add_argument("--from_json",    default=None, help="Evaluate from existing results JSON (no inference)")
    parser.add_argument("--val_data",     default=None, help="Validation JSONL (needed if not using --from_json)")
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument("--base_model",   default="deepseek-ai/DeepSeek-OCR")
    parser.add_argument("--output_dir",   default="./evaluation/eval_results")
    parser.add_argument("--limit",        type=int, default=None)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.from_json:
        # Fast mode: load existing results and re-evaluate
        print(f"Loading results from {args.from_json}")
        with open(args.from_json, encoding="utf-8") as f:
            raw = json.load(f)
        results = []
        for item in raw:
            pred = item.get("model_output", item.get("pred", ""))
            gt   = item.get("ground_truth", item.get("gt", ""))
            ev   = evaluate_example(pred, gt)
            results.append({"image": item.get("image", ""), "eval": ev})
            acc = ev.get("overall_cell_accuracy")
            print(f"  {item.get('image',''):<50} {f'{acc:.1%}' if acc is not None else 'parse_failed'}")
    else:
        if not args.val_data:
            parser.error("--val_data required when not using --from_json")

        sys.path.insert(0, str(Path(__file__).parent.parent / "finetune"))
        from inference_finetuned import load_finetuned_model, run_ocr

        examples = []
        with open(args.val_data, encoding="utf-8") as f:
            for line in f:
                examples.append(json.loads(line.strip()))
        if args.limit:
            examples = examples[:args.limit]

        model, tokenizer = load_finetuned_model(
            base_model_name=args.base_model,
            adapter_path=args.adapter_path,
        )

        results = []
        for i, ex in enumerate(examples):
            image_path = ex["image"]
            gt_text    = ex["conversations"][1]["content"]
            prompt     = ex["conversations"][0]["content"]
            print(f"[{i+1}/{len(examples)}] {Path(image_path).name}", end=" ... ", flush=True)
            try:
                pred_text = run_ocr(model, tokenizer, image_path, prompt=prompt)
                ev = evaluate_example(pred_text, gt_text)
                results.append({"image": image_path, "pred": pred_text, "gt": gt_text, "eval": ev})
                acc = ev.get("overall_cell_accuracy")
                print(f"{acc:.1%}" if acc is not None else "parse_failed")
            except Exception as e:
                print(f"ERROR: {e}")
                results.append({"image": image_path, "error": str(e)})

        with open(out_dir / "results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    summary = aggregate_and_print(results)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSummary saved → {out_dir}/summary.json")


if __name__ == "__main__":
    main()
