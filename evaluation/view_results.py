#!/usr/bin/env python3
"""
Convert preview_outputs.py JSON results to a viewable HTML comparison page.

Usage:
    # First save results:
    python evaluation/preview_outputs.py \
        --val_data ./training_data/dataset_val.jsonl \
        --adapter_path ./finetuned_model_v4b \
        --n 20 --save ./evaluation/results.json

    # Then view:
    python evaluation/view_results.py --input ./evaluation/results.json --output_dir ./evaluation/results/
    # Open evaluation/results/index.html in your browser
"""

import json
import argparse
import re
from pathlib import Path


def strip_grounding_tags(text: str) -> str:
    """Remove <|ref|>...<|/ref|> and <|det|>...<|/det|> tags from model output."""
    text = re.sub(r'<\|ref\|>.*?<\|/ref\|>', '', text, flags=re.DOTALL)
    text = re.sub(r'<\|det\|>.*?<\|/det\|>', '', text, flags=re.DOTALL)
    return text.strip()


def extract_table(text: str) -> str:
    """Extract the last <table>...</table> block from text."""
    matches = list(re.finditer(r'<table>.*?</table>', text, flags=re.DOTALL))
    if matches:
        return matches[-1].group(0)
    return text


STYLE = """
<style>
  body { font-family: sans-serif; font-size: 13px; margin: 20px; }
  h2 { color: #333; }
  h4 { margin: 0 0 8px 0; color: #555; border-bottom: 1px solid #eee; padding-bottom: 4px; }
  .image-path { font-size: 12px; color: #888; margin-bottom: 16px; }
  .section { margin-bottom: 24px; }
  table { border-collapse: collapse; width: 100%; font-size: 12px; margin-bottom: 8px; }
  td, th { border: 1px solid #ddd; padding: 5px 10px; text-align: left; }
  tr:nth-child(even) { background: #f9f9f9; }
  thead tr { background: #e8f0fe; }
  pre { background: #f5f5f5; padding: 10px; font-size: 11px; white-space: pre-wrap; word-break: break-word; border-radius: 4px; }
  .nav { margin-bottom: 20px; font-size: 13px; }
  .nav a { margin-right: 12px; color: #1a73e8; text-decoration: none; }
  .nav a:hover { text-decoration: underline; }
</style>
"""


def make_page(i: int, total: int, ex: dict, out_dir: Path) -> str:
    image = ex.get("image", "")
    gt = ex.get("ground_truth", "")
    raw_output = ex.get("model_output", "")

    cleaned = strip_grounding_tags(raw_output)
    table_only = extract_table(cleaned)

    prev_link = f'<a href="{i-1:03d}.html">← Prev</a>' if i > 1 else ''
    next_link = f'<a href="{i+1:03d}.html">Next →</a>' if i < total else ''
    index_link = '<a href="index.html">Index</a>'

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Example {i}/{total}</title>
{STYLE}
</head>
<body>
<div class="nav">{index_link} {prev_link} {next_link} <span style="color:#999">{i}/{total}</span></div>
<h2>Example {i}</h2>
<div class="image-path">{image}</div>

<div class="section">
  <h4>Ground Truth</h4>
  {gt}
</div>

<div class="section">
  <h4>Model Output (cleaned)</h4>
  {table_only}
</div>

<div class="section">
  <h4>Raw Output</h4>
  <pre>{raw_output}</pre>
</div>

<div class="nav">{index_link} {prev_link} {next_link}</div>
</body>
</html>"""


def make_index(examples: list) -> str:
    rows = []
    for i, ex in enumerate(examples, 1):
        image = ex.get("image", "")
        section = image.split("/")[-2] if "/" in image else ""
        filename = image.split("/")[-1]
        rows.append(f'<tr><td><a href="{i:03d}.html">{i}</a></td><td>{section}</td><td>{filename}</td></tr>')

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>OCR Results Index</title>
{STYLE}
</head>
<body>
<h2>OCR Results — {len(examples)} examples</h2>
<table>
<thead><tr><th>#</th><th>Month/Folder</th><th>Image</th></tr></thead>
{''.join(rows)}
</table>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True, help="JSON file from preview_outputs.py --save")
    parser.add_argument("--output_dir", default="evaluation/results/")
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        examples = json.load(f)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, ex in enumerate(examples, 1):
        page = make_page(i, len(examples), ex, out_dir)
        (out_dir / f"{i:03d}.html").write_text(page, encoding="utf-8")

    (out_dir / "index.html").write_text(make_index(examples), encoding="utf-8")

    print(f"Saved {len(examples)} pages → {out_dir}/index.html")


if __name__ == "__main__":
    main()
