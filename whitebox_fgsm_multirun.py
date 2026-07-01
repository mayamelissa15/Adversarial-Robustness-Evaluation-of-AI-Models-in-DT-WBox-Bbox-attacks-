"""
whitebox_fgsm_multirun.py
Runner FGSM seul — extrait de whitebox_multirun.py pour permettre de
lancer FGSM / PGD / C&W en parallèle sur des process séparés.

Usage :
  python whitebox_fgsm_multirun.py --dataset swat    --eps 0.1
  python whitebox_fgsm_multirun.py --dataset swat    --eps 0.3
  python whitebox_fgsm_multirun.py --dataset batadal --eps 0.1
  python whitebox_fgsm_multirun.py --dataset batadal --eps 0.3

Sorties :
  ~/<dataset>/results/whitebox_fgsm_<dataset>_eps<eps>.csv
  ~/<dataset>/results/whitebox_fgsm_<dataset>_eps<eps>.json
  ~/<dataset>/results/whitebox_fgsm_<dataset>_eps<eps>_tmp.csv  (checkpoint)

Ensuite : python merge_whitebox_results.py --dataset ... --eps ...
"""

import warnings
import pandas as pd
import json
warnings.filterwarnings("ignore")

from common_whitebox import (
    build_arg_parser, setup_paths, get_device, eval_sizes,
    load_victims, build_per_model_eval, set_all_seeds, VICTIMS_SPEC,
)
from models import eval_attack
from whitebox import fgsm_mlp, fgsm_logreg, fgsm_xgb

ATTACK_NAME = "FGSM"

args = build_arg_parser("Whitebox multirun — FGSM uniquement").parse_args()
DATASET, EPS, N_RUNS, FAST = args.dataset, args.eps, args.n_runs, args.fast

SAVE_DIR, RESULTS_DIR, TAG = setup_paths(DATASET, EPS)
DEVICE = get_device()
SEEDS = list(range(N_RUNS))
EVAL_ATK_SIZE, EVAL_NRM_SIZE = eval_sizes(DATASET)

print(f"\n{'═'*55}")
print(f"  Attaque  : {ATTACK_NAME}")
print(f"  Dataset  : {DATASET.upper()}")
print(f"  Epsilon  : {EPS}")
print(f"  N_RUNS   : {N_RUNS}")
print(f"  Device   : {DEVICE}")
print(f"  Sorties  : {RESULTS_DIR}")
print(f"{'═'*55}")


def run():
    X_test, y_test, mlp_w, logreg_w, xgb_w = load_victims(SAVE_DIR, DEVICE)
    victims = {"MLP": mlp_w, "LogReg": logreg_w, "XGBoost": xgb_w}

    all_results = []

    for seed in SEEDS:
        print(f"\n{'═'*55}")
        print(f"  SEED {seed+1}/{N_RUNS}  —  {DATASET.upper()}  eps={EPS}  [{ATTACK_NAME}]")
        print(f"{'═'*55}")

        set_all_seeds(seed)

        for vic_name, is_lr, is_xgb in VICTIMS_SPEC:
            vic_w = victims[vic_name]
            print(f"\n  ── {vic_name} {'─'*(45-len(vic_name))}")

            X_eval, y_eval, X_atk, y_atk, idx_ev = build_per_model_eval(
                X_test, y_test, vic_w, seed, EVAL_ATK_SIZE, EVAL_NRM_SIZE, DATASET)

            print(f"  Eval set : {len(X_eval)} exemples "
                  f"({(y_eval==1).sum()} attaques, {(y_eval==0).sum()} normaux)")

            print(f"  [{ATTACK_NAME}]")
            if is_lr:
                X_adv = fgsm_logreg(vic_w, X_atk, y_atk, EPS)
            elif is_xgb:
                X_adv = fgsm_xgb(vic_w, X_atk, y_atk, EPS)
            else:
                X_adv = fgsm_mlp(vic_w, X_atk, y_atk, EPS)

            r = eval_attack(vic_w, X_eval, y_eval, X_adv, ATTACK_NAME, vic_name, threshold=0.45)
            r.update({"seed": seed, "family": "Whitebox",
                      "eps": EPS, "dataset": DATASET,
                      "n_atk": int((y_eval == 1).sum())})
            all_results.append(r)
            print(f"    ASR={r['asr']*100:.1f}%  "
                  f"F1 {r['f1_clean']:.3f}→{r['f1_adv']:.3f}  "
                  f"margin_adv_mean={r.get('margin_adv_mean', float('nan')):.4f}  "
                  f"L∞mean={r.get('linf_mean', float('nan')):.4f}")

        pd.DataFrame(all_results).to_csv(
            RESULTS_DIR / f"whitebox_fgsm_{TAG}_tmp.csv", index=False)
        print(f"\n  ✓ Checkpoint seed {seed} sauvegardé")

    df = pd.DataFrame(all_results)
    csv_path = RESULTS_DIR / f"whitebox_fgsm_{TAG}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✓ CSV → {csv_path}")

    summary = (df.groupby(["attack", "model"])["asr"]
                 .agg(["median", "std", "min", "max"]).round(4))
    print(f"\n{'═'*55}")
    print(f"  RÉSUMÉ {ATTACK_NAME} — {DATASET.upper()}  eps={EPS}")
    print(f"{'═'*55}")
    print(summary.to_string())

    out = {}
    for model_name in df["model"].unique():
        sub = df[df["model"] == model_name]
        vals = sub["asr"]
        out[model_name] = {ATTACK_NAME: {
            "evasion_rate_median": round(float(vals.median()) * 100, 2),
            "evasion_rate_std":    round(float(vals.std()) * 100, 2),
            "evasion_rate_min":    round(float(vals.min()) * 100, 2),
            "evasion_rate_max":    round(float(vals.max()) * 100, 2),
        }}

    json_path = RESULTS_DIR / f"whitebox_fgsm_{TAG}.json"
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"✓ JSON → {json_path}")

    return df


if __name__ == "__main__":
    run()