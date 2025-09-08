#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_html_from_csv.py
Reads esm_design_compare_table.csv and writes an embedded, searchable/sortable HTML table.
Output path: /sci/labs/orzuk/meirab/stability_algo/full_pipeline_outputs/esm_design_compare_table_embedded.html
"""
import pandas as pd
from pathlib import Path
import datetime

ROOT = Path("/sci/labs/orzuk/meirab/stability_algo/full_pipeline_outputs")
CSV = ROOT / "esm_design_compare_table.csv"
OUT = ROOT / "esm_design_compare_table_embedded.html"

def df_to_html_embedded(df: pd.DataFrame, title="ESM design compare table"):
    table_html = df.to_html(index=False, classes="display", table_id="esm_table", na_rep="")
    now = datetime.datetime.utcnow().isoformat() + "Z"
    template = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <link rel="stylesheet" href="https://cdn.datatables.net/1.13.6/css/jquery.dataTables.min.css">
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial; padding: 18px; }}
    h1 {{ font-size: 1.2rem; margin-bottom: 8px; }}
    .meta {{ color: #666; font-size: 0.9rem; margin-bottom: 12px; }}
    table.dataTable tbody tr td {{ white-space: nowrap; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <div class="meta">Generated: {now} - Source CSV: {CSV.name}</div>

  {table_html}

  <script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
  <script src="https://cdn.datatables.net/1.13.6/js/jquery.dataTables.min.js"></script>
  <script>
    $(document).ready(function() {{
      $('#esm_table').DataTable({{
        "pageLength": 25,
        "order": [[0, "asc"]]
      }});
    }});
  </script>
</body>
</html>"""
    return template

def main():
    if not CSV.exists():
        print("CSV not found:", CSV)
        return
    df = pd.read_csv(CSV)
    wanted = ["pdb id","DG esm WT","DDG esm WT","DG esm Design","DDG esm Design"]
    for c in wanted:
        if c not in df.columns:
            df[c] = ""
    df = df[wanted]
    html = df_to_html_embedded(df, title="ESM design compare table")
    OUT.write_text(html)
    print("Wrote:", OUT)

if __name__ == "__main__":
    main()
