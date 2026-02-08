#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import re
import pandas as pd
import json

# === PATHS ===
STRUCTURE_METRICS_CSV = Path(
    "/sci/labs/orzuk/meirab/stability_algo/structure_results/structure_metrics_only.csv"
)

EVAL_ROOT = Path("/sci/labs/orzuk/meirab/stability_algo/best_results/evaluation")
CSV_OUT = Path("/sci/labs/orzuk/meirab/stability_algo/full_pipeline_outputs/weight05_05_summary_table.csv")
HTML_OUT =  Path("/sci/labs/orzuk/meirab/stability_algo/full_pipeline_outputs/weight05_05_summary_table.html")
PER_PDB_JSON_ROOT = Path(
    "/sci/labs/orzuk/meirab/stability_algo/full_pipeline_outputs/per_pdb"
)



def get_protein_length_from_json(pdb_id):
    """
    Read protein sequence length from per-PDB JSON file.
    """
    json_path = PER_PDB_JSON_ROOT / f"{pdb_id}_weight05_05_dg0.5_ddg0.5.json"
    if not json_path.exists():
        return None

    try:
        with open(json_path) as f:
            data = json.load(f)
        return int(data.get("Sequence length"))
    except Exception:
        return None

# === REGEX HELPERS ===
def extract_value(text, key):
    """Extract numeric value (float) from a line like: 'wt_dg: 19.893447875976562'"""
    m = re.search(rf"{key}:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", text)
    return round(float(m.group(1)), 3) if m else None

def extract_pdb_id(text):
    m = re.search(r"pdb_id:\s*([A-Za-z0-9]+)", text)
    return m.group(1) if m else None

def parse_summary_file(path: Path):
    """Parse evaluation_summary.txt for one protein."""
    try:
        text = path.read_text()
    except Exception:
        return None

    pdb_id = extract_pdb_id(text)
    protein_length = get_protein_length_from_json(pdb_id)

    if not pdb_id:
        pdb_id = path.parent.name.split("_")[0]  # fallback

    # === Extract mutation list (letters and numbers, e.g. 'V75E') ===
    m = re.search(r"mutated_positions:\s*\[([^\]]*)\]", text)
    mutations = []
    if m:
        inside = m.group(1)
        # extract quoted items: 'A34T', "V75E"
        mutations = re.findall(r"'([^']+)'|\"([^\"]+)\"", inside)
        # flatten tuples
        mutations = [a or b for a, b in mutations]

    record = {
        "pdb id": pdb_id,
        "Protein Length": protein_length,
        "# Mutations (count)": int(len(mutations)),
        "Mutations list": ", ".join(mutations) if mutations else "",
        "WT ΔG": extract_value(text, "wt_dg"),
        "Design ΔG": extract_value(text, "design_dg"),
        "WT ΔΔG": extract_value(text, "wt_overall_ddg"),
        "Design ΔΔG": extract_value(text, "design_overall_ddg"),
        "WT Loss": extract_value(text, "wt_loss"),
        "Design Loss": extract_value(text, "design_loss"),
    }
    return record


def main():
    # find all weight05_05 folders
    folders = sorted(EVAL_ROOT.glob("*_weight05_05"))
    print(f"Found {len(folders)} weight05_05 experiment folders.")

    records = []
    for folder in folders:
        summary_path = folder / "evaluation_summary.txt"
        if summary_path.exists():
            rec = parse_summary_file(summary_path)
            if rec:
                records.append(rec)
        else:
            print(f"[WARN] Missing summary: {summary_path}")

    if not records:
        print("[WARNING] No valid summaries found.")
        return

    # === Build DataFrame ===
    df = pd.DataFrame.from_records(records)
    df["# Mutations (count)"] = df["# Mutations (count)"].astype(int)
    df["Protein Length"] = df["Protein Length"].astype("Int64")

    df["Improvement"] = df["WT Loss"] - df["Design Loss"]
    # === Merge AlphaFold structural metrics (TM-score, RMSD) ===
    if STRUCTURE_METRICS_CSV.exists():
        df_struct = pd.read_csv(STRUCTURE_METRICS_CSV)

        # Normalize PDB IDs for safe merge
        df_struct["pdb id"] = df_struct["pdb_id"].str.upper()
        df["pdb id"] = df["pdb id"].str.upper()

        df = df.merge(
            df_struct[["pdb id", "tm_score", "rmsd_ca"]],
            on="pdb id",
            how="left"
        )

        df.rename(
            columns={
                "tm_score": "TM-score",
                "rmsd_ca": "RMSD (Å)"
            },
            inplace=True
        )
    else:
        print(f"[WARN] Structure metrics CSV not found: {STRUCTURE_METRICS_CSV}")
	

    df = df[
        ["pdb id","Protein Length", "# Mutations (count)", "Mutations list",
         "WT ΔG", "Design ΔG",
         "WT ΔΔG", "Design ΔΔG",
         "WT Loss", "Design Loss","Improvement","TM-score", "RMSD (Å)"]
    ]
    df = df.round(3)

    # === Save CSV ===
    df.to_csv(CSV_OUT, index=False, float_format="%.3f")
    print(f"[OK] Saved CSV → {CSV_OUT}")

    # === HTML version ===
    float_cols = df.select_dtypes(include=["float"]).columns
    styled = df.style.set_properties(**{"text-align": "right"}).format(
        {col: "{:.3f}".format for col in float_cols}
    )

    # Highlight whichever (WT or Design) has lower values for each metric
    def highlight_better(row):
        """
        Highlight the ENTIRE ROW only if:
        Design DG   < WT DG
        Design DDG  < WT DDG
        Design Loss < WT Loss
        """
        styles = [''] * len(row)

        # Extract values
        wt_dg, des_dg = row["WT ΔG"], row["Design ΔG"]
        wt_ddg, des_ddg = row["WT ΔΔG"], row["Design ΔΔG"]
        wt_loss, des_loss = row["WT Loss"], row["Design Loss"]

        # Check that all are valid numbers
        if not pd.isna(wt_dg) and not pd.isna(des_dg) and \
           not pd.isna(wt_ddg) and not pd.isna(des_ddg) and \
           not pd.isna(wt_loss) and not pd.isna(des_loss):

            # Condition: design better in *all three* metrics
            if (des_dg < wt_dg) and (des_ddg < wt_ddg) and (des_loss < wt_loss):
                styles = ["background-color: #c6f6c6; font-weight: bold;"] * len(row)

        return styles

    styled = styled.apply(highlight_better, axis=1)

    html = styled.to_html(index=False)
    html_full = f"""<!doctype html>
<html lang="en">
 <head>
 <meta charset="utf-8"> 
<title>Weight 0.5 / 0.5 Summary Table</title> 
<style> body{{font-family:Arial,Helvetica,sans-serif;padding:16px}} table{{border-collapse:collapse; width:100%;}} th,td{{border:1px solid #ddd;padding:6px 10px;text-align:right;vertical-align:top}} th{{background:#f3f3f3;text-align:center}} td:first-child, th:first-child{{text-align:left}} </style> </head>
<body> {html} </body>
</html>"""  
    HTML_OUT.write_text(html_full)

    print(f"[OK] Saved HTML → {HTML_OUT}")


if __name__ == "__main__":
    main()
