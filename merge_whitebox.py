"""
merge_whitebox_results.py
Recombine les sorties des 3 runners séparés (fgsm/pgd/cw) en un seul
CSV + JSON, dans le MÊME FORMAT que whitebox_multirun.py original.

À lancer une fois que whitebox_fgsm_multirun.py, whitebox_pgd_multirun.py
et whitebox_cw_multirun.py ont terminé pour le (dataset, eps) voulu.

Usage :
  python merge_whitebox_results.py --dataset swat    --eps 0.1
  python merge_whitebox_results.py --dataset swat    --eps 0.3
  python merge_whitebox_results.py --dataset batadal --eps 0.1
  python merge_whitebox_results.py --dataset batadal --eps 0.3

Sorties :
  ~/<dataset>/results/whitebox_multirun_<dataset>_eps<eps>.csv   (identique au script original)
  ~/<dataset>/results/whitebox_multirun_<dataset>_eps<eps>.json  (identique au script original)
"""

import argparse
import json
from pathlib import Path
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="swat", choices=["swat", "batadal"])
parser.add_argument("--eps", default=0.1, type=float)
args = parser.parse_args()

DATASET = args.dataset
EPS = args.eps
RESULTS_DIR = Path(f"~/{DATASET}/results").expanduser()
TAG = f"{DATASET}_eps{EPS}"

ATTACK_FILES = {
    "FGSM": RESULTS_DIR / f"whitebox_fgsm_{TAG}.csv",
    "PGD":  RESULTS_DIR / f"whitebox_pgd_{TAG}.csv",
    "C&W":  RESULTS_DIR / f"whitebox_cw_{TAG}.csv",
}

PERSAMPLE_FILES = {
    "FGSM": RESULTS_DIR / f"whitebox_persample_fgsm_{TAG}.csv",
    "PGD":  RESULTS_DIR / f"whitebox_persample_pgd_{TAG}.csv",
    "C&W":  RESULTS_DIR / f"whitebox_persample_cw_{TAG}.csv",
}


def merge_persample():
    """Fusionne les 3 CSV par-échantillon (timestamps) s'ils existent."""
    dfs = []
    for attack_name, path in PERSAMPLE_FILES.items():
        if path.exists():
            dfs.append(pd.read_csv(path))
            print(f"✓ persample {attack_name:<6} chargé depuis {path}  ({len(dfs[-1])} lignes)")

    if not dfs:
        print("⚠ Aucun CSV par-échantillon trouvé (timestamps absents ou runners "
              "pas encore lancés avec timestamps_test.npy présent) — étape ignorée.")
        return None

    df_persample = pd.concat(dfs, ignore_index=True)
    df_persample = df_persample.sort_values(["seed", "model", "attack", "timestamp"]).reset_index(drop=True)

    persample_path = RESULTS_DIR / f"whitebox_persample_{TAG}.csv"
    df_persample.to_csv(persample_path, index=False)
    print(f"✓ CSV par-échantillon fusionné → {persample_path}  ({len(df_persample)} lignes)")
    return df_persample


def run():
    dfs = []
    missing = []
    for attack_name, path in ATTACK_FILES.items():
        if path.exists():
            dfs.append(pd.read_csv(path))
            print(f"✓ {attack_name:<6} chargé depuis {path}  ({len(dfs[-1])} lignes)")
        else:
            missing.append((attack_name, path))

    if missing:
        print("\n⚠ Fichiers manquants (le runner correspondant n'a peut-être pas fini) :")
        for attack_name, path in missing:
            print(f"    {attack_name:<6} → {path}")
        if not dfs:
            print("\n✗ Aucun fichier trouvé, rien à fusionner.")
            return None

    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values(["seed", "model", "attack"]).reset_index(drop=True)

    csv_path = RESULTS_DIR / f"whitebox_multirun_{TAG}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✓ CSV fusionné → {csv_path}  ({len(df)} lignes)")

    summary = (df.groupby(["attack", "model"])["asr"]
                 .agg(["median", "std", "min", "max"]).round(4))
    print(f"\n{'═'*55}")
    print(f"  RÉSUMÉ FUSIONNÉ — {DATASET.upper()}  eps={EPS}")
    print(f"{'═'*55}")
    print(summary.to_string())

    out = {}
    for model_name in df["model"].unique():
        out[model_name] = {}
        sub = df[df["model"] == model_name]
        for attack_name in sub["attack"].unique():
            vals = sub[sub["attack"] == attack_name]["asr"]
            out[model_name][attack_name] = {
                "evasion_rate_median": round(float(vals.median()) * 100, 2),
                "evasion_rate_std":    round(float(vals.std()) * 100, 2),
                "evasion_rate_min":    round(float(vals.min()) * 100, 2),
                "evasion_rate_max":    round(float(vals.max()) * 100, 2),
            }

    json_path = RESULTS_DIR / f"whitebox_multirun_{TAG}.json"
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"✓ JSON fusionné → {json_path}")

    merge_persample()

    return df


if __name__ == "__main__":
    run()