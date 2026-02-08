import subprocess
from pathlib import Path
import pandas as pd
import re

# -------------------------
# Paths
# -------------------------
BASE = Path("/sci/labs/orzuk/meirab/stability_algo")

WT_DIR = BASE / "data/full_pdbs"
AF_DIR = BASE / "alphafold_runs/colabfold_out"
OVERLAY_SCRIPT = BASE / "weight_exp/comarison_runs/overlay_af_vs_wt.py"

OUT_DIR = BASE / "structure_results"
PNG_DIR = OUT_DIR / "overlays"

TM_EXEC = "TMscore"  # or full path if needed

OUT_DIR.mkdir(exist_ok=True)
PNG_DIR.mkdir(exist_ok=True)

# -------------------------
# Helpers
# -------------------------
def find_rank1_pdb(af_dir: Path) -> Path:
    pdbs = list(af_dir.glob("*rank_001*.pdb"))
    if len(pdbs) != 1:
        raise RuntimeError(f"Expected 1 rank_001 pdb in {af_dir}, found {len(pdbs)}")
    return pdbs[0]


def run_tmscore(design_pdb: Path, wt_pdb: Path):
    proc = subprocess.run(
        [TM_EXEC, str(design_pdb), str(wt_pdb)],
        capture_output=True,
        text=True,
        check=True
    )

    out = proc.stdout

    tm = float(re.search(r"TM-score\s*=\s*([0-9.]+)", out).group(1))
    rmsd = float(re.search(r"RMSD\s*=\s*([0-9.]+)", out).group(1))

    return tm, rmsd


def run_overlay(wt_pdb, design_pdb, tm_score, out_png):
    subprocess.run(
        [
            "pymol",
            "-cq",
            str(OVERLAY_SCRIPT),
            str(wt_pdb),
            str(design_pdb),
            f"{tm_score:.3f}",
            str(out_png),
        ],
        check=True
    )


# -------------------------
# Main loop
# -------------------------
records = []

for af_run in sorted(AF_DIR.glob("*_beam_weight05_05.result")):
    pdb_id = af_run.name.split("_")[0]
    pdb_id_upper = pdb_id.upper()
    pdb_id_lower = pdb_id.lower()


    wt_pdb = WT_DIR / f"{pdb_id_lower}.pdb"
    if not wt_pdb.exists():
        print(f"[SKIP] Missing WT PDB for {pdb_id}")
        continue

    design_pdb = find_rank1_pdb(af_run)

    print(f"[RUN] {pdb_id}")

    tm, rmsd = run_tmscore(design_pdb, wt_pdb)

    png_path = PNG_DIR / f"{pdb_id}.png"
    run_overlay(wt_pdb, design_pdb, tm, png_path)

    records.append(
        dict(
            pdb_id=pdb_id_upper,   # keep table clean
            tm_score=tm,
            rmsd_ca=rmsd,
          )
     )

# -------------------------
# Save tables
# -------------------------
df = pd.DataFrame(records).sort_values("pdb_id")

csv_path = OUT_DIR / "structure_metrics.csv"
html_path = OUT_DIR / "structure_metrics.html"

df.to_csv(csv_path, index=False)

# HTML with image links
df_html = df.copy()
df_html["overlay_png"] = df_html["overlay_png"].apply(
    lambda p: f'<a href="{p}">view</a>'
)

df_html.to_html(html_path, escape=False, index=False)

print("\n✓ Done")
print(f"CSV : {csv_path}")
print(f"HTML: {html_path}")
