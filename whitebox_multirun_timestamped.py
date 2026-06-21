# whitebox_multirun_timestamped.py
"""
Copie de whitebox_multirun.py + export par-échantillon avec timestamps.

N

En plus du résumé agrégé habituel, ce script produit un CSV détaillé
sous-échantillonné (max 50 samples par seed × modèle × attaque) avec :
timestamp, succès individuel, marge logit, L∞ — pour analyse temporelle.

Pré-requis : avoir lancé 00_extract_timestamps.py --dataset <dataset>
avant, pour générer timestamps_test.npy dans artifacts/. Si ce fichier
n'existe pas, le script tourne normalement mais SKIP l'export par-échantillon
(avec un warning), sans planter.

Usage :
  python whitebox_multirun_timestamped.py --dataset swat   --eps 0.1
  python whitebox_multirun_timestamped.py --dataset swat   --eps 0.3
  python whitebox_multirun_timestamped.py --dataset batadal --eps 0.1
  python whitebox_multirun_timestamped.py --dataset batadal --eps 0.3

Sorties :
  ~/<dataset>/results/whitebox_multirun_<dataset>_eps<eps>.csv   (identique à l'existant)
  ~/<dataset>/results/whitebox_multirun_<dataset>_eps<eps>.json  (identique à l'existant)
  ~/<dataset>/results/whitebox_persample_<dataset>_eps<eps>.csv  (NOUVEAU — détail temporel)
"""

import argparse
import numpy as np
import torch
import joblib
import pandas as pd
import json
import warnings
from pathlib import Path
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

import sys
sys.path.append(str(Path(__file__).parent))

from models import (MLP, MLPWrapper, LogRegWrapper, XGBoostWrapper,
                    build_eval_set, eval_attack, eval_attack_persample)
from whitebox import (
    fgsm_mlp, fgsm_logreg, fgsm_xgb,
    pgd_mlp,  pgd_logreg,  pgd_xgb,
    cw_mlp,   cw_logreg,   cw_xgb,
    THRESHOLD_LOGIT, EPS_FD, CW_LR_XGB, CW_LR_LR,
    PGD_ALPHA_K,
)

# ══════════════════════════════════════════════════════════════
# ARGUMENTS
# ══════════════════════════════════════════════════════════════

parser = argparse.ArgumentParser()
parser.add_argument("--dataset",  default="swat",
                    choices=["swat", "batadal"],
                    help="Dataset cible")
parser.add_argument("--eps",      default=0.1,  type=float,
                    help="Epsilon L∞ (0.1 ou 0.3)")
parser.add_argument("--n_runs",   default=10,   type=int,
                    help="Nombre de seeds")
parser.add_argument("--fast",     action="store_true",
                    help="FAST_MODE : PGD 50×3 au lieu de 200×10")
parser.add_argument("--persample_n", default=50, type=int,
                    help="Nombre max de samples gardés par (seed, modèle, attaque) "
                         "dans le CSV par-échantillon")
args = parser.parse_args()

DATASET      = args.dataset
EPS          = args.eps
N_RUNS       = args.n_runs
FAST         = args.fast
PERSAMPLE_N  = args.persample_n

# ── Chemins selon dataset ──────────────────────────────────────
SAVE_DIR    = Path(f"~/{DATASET}/artifacts").expanduser()
RESULTS_DIR = Path(f"~/{DATASET}/results").expanduser()
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TAG = f"{DATASET}_eps{EPS}"   # utilisé dans tous les noms de fichiers

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS  = list(range(N_RUNS))

# ── Hyperparamètres PGD / C&W selon FAST_MODE ─────────────────
PGD_ITERS    = 50  if FAST else 200
PGD_RESTARTS = 3   if FAST else 10
CW_ITERS     = 150 if FAST else 500

# ── Taille eval set selon dataset ─────────────────────────────
EVAL_ATK_SIZE = 200 if DATASET == "batadal" else 500
EVAL_NRM_SIZE = 500

# ── Chargement timestamps (optionnel — ne plante pas si absent) ──
TIMESTAMPS_PATH = SAVE_DIR / "timestamps_test.npy"
if TIMESTAMPS_PATH.exists():
    TIMESTAMPS_TEST = np.load(TIMESTAMPS_PATH, allow_pickle=True)
    HAS_TIMESTAMPS  = True
else:
    TIMESTAMPS_TEST = None
    HAS_TIMESTAMPS  = False
    print(f"\n⚠ {TIMESTAMPS_PATH} introuvable — export par-échantillon désactivé "
          f"pour ce run (lance 00_extract_timestamps.py --dataset {DATASET} d'abord).")

print(f"\n{'═'*55}")
print(f"  Dataset  : {DATASET.upper()}")
print(f"  Epsilon  : {EPS}")
print(f"  N_RUNS   : {N_RUNS}")
print(f"  Device   : {DEVICE}")
print(f"  FAST     : {FAST}")
print(f"  PGD      : {PGD_ITERS} iters × {PGD_RESTARTS} restarts")
print(f"  C&W      : {CW_ITERS} iters")
print(f"  Eval atk : {EVAL_ATK_SIZE} exemples max")
print(f"  Timestamps : {'OK' if HAS_TIMESTAMPS else 'absents (export persample skip)'}")
print(f"  Sorties  : {RESULTS_DIR}")
print(f"{'═'*55}")


# ══════════════════════════════════════════════════════════════
# CHARGEMENT DES VICTIMES
# ══════════════════════════════════════════════════════════════

def load_victims():
    X_test = np.load(SAVE_DIR / "X_test.npy")
    y_test = np.load(SAVE_DIR / "y_test.npy")

    mlp_model = MLP(input_size=X_test.shape[1]).to(DEVICE)
    mlp_model.load_state_dict(
        torch.load(SAVE_DIR / "best_mlp.pt", map_location=DEVICE))
    mlp_model.eval()
    mlp_w = MLPWrapper(mlp_model, DEVICE)

    logreg_w = LogRegWrapper(joblib.load(SAVE_DIR / "logreg.pkl"))

    xgb_model = XGBClassifier()
    xgb_model.load_model(str(SAVE_DIR / "xgb.json"))
    xgb_w = XGBoostWrapper(xgb_model)

    print(f"\n✓ Modèles chargés depuis {SAVE_DIR}")
    print(f"  X_test : {X_test.shape} — attaques : {y_test.sum()} / {len(y_test)}")

    return X_test, y_test, mlp_w, logreg_w, xgb_w


# ══════════════════════════════════════════════════════════════
# EVAL SET PAR MODÈLE  (identique à whitebox_multirun.py + idx_ev en plus)
# ══════════════════════════════════════════════════════════════

def build_per_model_eval(X_test, y_test, victim_w, seed):
    """
    Identique à la version de whitebox_multirun.py, SAUF qu'on retourne
    aussi idx_ev (les positions dans X_test/timestamps_test utilisées),
    pour pouvoir récupérer les timestamps correspondants.
    """
    rng        = np.random.default_rng(seed)
    idx_normal = np.where(y_test == 0)[0]
    idx_attack = np.where(y_test == 1)[0]

    preds_vic     = victim_w.predict(X_test[idx_attack], threshold=0.45)
    idx_attack_ok = idx_attack[preds_vic == 1]

    n_atk = min(EVAL_ATK_SIZE, len(idx_attack_ok))
    n_nrm = min(EVAL_NRM_SIZE, len(idx_normal))

    if n_atk == 0:
        raise ValueError(
            f"Aucun TP pour ce modèle sur {DATASET} — vérifier le F1 baseline.")

    sel_n  = rng.choice(idx_normal,    size=n_nrm, replace=False)
    sel_a  = rng.choice(idx_attack_ok, size=n_atk, replace=n_atk > len(idx_attack_ok))
    idx_ev = np.concatenate([sel_n, sel_a])
    rng.shuffle(idx_ev)

    X_eval = X_test[idx_ev]
    y_eval = y_test[idx_ev]
    mask   = (y_eval == 1)
    X_atk  = X_eval[mask].astype(np.float32)
    y_atk  = y_eval[mask]

    return X_eval, y_eval, X_atk, y_atk, idx_ev


def set_all_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ══════════════════════════════════════════════════════════════
# BOUCLE PRINCIPALE
# ══════════════════════════════════════════════════════════════

def run_one_attack(attack_label, X_adv, vic_w, X_eval, y_eval, X_atk, y_atk,
                    vic_name, seed, idx_ev, all_results, persample_results):
    """
    Calcule l'agrégat (eval_attack, INCHANGÉ) + le détail par-échantillon
    (eval_attack_persample, NOUVEAU) pour une attaque donnée.
    """
    r = eval_attack(vic_w, X_eval, y_eval, X_adv, attack_label, vic_name, threshold=0.45)
    r.update({"seed": seed, "family": "Whitebox",
              "eps": EPS, "dataset": DATASET,
              "n_atk": int((y_eval == 1).sum())})
    all_results.append(r)
    print(f"    ASR={r['asr']*100:.1f}%  "
          f"F1 {r['f1_clean']:.3f}→{r['f1_adv']:.3f}  "
          f"margin_adv_mean={r.get('margin_adv_mean', float('nan')):.4f}  "
          f"L∞mean={r.get('linf_mean', float('nan')):.4f}")

    if HAS_TIMESTAMPS:
        timestamps_eval = TIMESTAMPS_TEST[idx_ev]
        records = eval_attack_persample(
            vic_w, X_eval, y_eval, X_adv, attack_label, vic_name,
            timestamps_full=timestamps_eval, threshold=0.45,
            max_samples=PERSAMPLE_N, seed=seed,
        )
        for rec in records:
            rec.update({"eps": EPS, "dataset": DATASET})
        persample_results.extend(records)

    return r


def run():
    X_test, y_test, mlp_w, logreg_w, xgb_w = load_victims()

    victims = [
        ("MLP",     mlp_w,    False, False),
        ("LogReg",  logreg_w, True,  False),
        ("XGBoost", xgb_w,    False, True),
    ]

    all_results       = []
    persample_results = []

    for seed in SEEDS:
        print(f"\n{'═'*55}")
        print(f"  SEED {seed+1}/{N_RUNS}  —  {DATASET.upper()}  eps={EPS}")
        print(f"{'═'*55}")

        set_all_seeds(seed)

        for vic_name, vic_w, is_lr, is_xgb in victims:
            print(f"\n  ── {vic_name} {'─'*(45-len(vic_name))}")

            X_eval, y_eval, X_atk, y_atk, idx_ev = build_per_model_eval(
                X_test, y_test, vic_w, seed=seed)

            print(f"  Eval set : {len(X_eval)} exemples "
                  f"({(y_eval==1).sum()} attaques, {(y_eval==0).sum()} normaux)")

            # ── FGSM ──────────────────────────────────────────
            print("  [FGSM]")
            if is_lr:
                X_adv = fgsm_logreg(vic_w, X_atk, y_atk, EPS)
            elif is_xgb:
                X_adv = fgsm_xgb(vic_w, X_atk, y_atk, EPS)
            else:
                X_adv = fgsm_mlp(vic_w, X_atk, y_atk, EPS)
            run_one_attack("FGSM", X_adv, vic_w, X_eval, y_eval, X_atk, y_atk,
                           vic_name, seed, idx_ev, all_results, persample_results)

            # ── PGD ───────────────────────────────────────────
            print("  [PGD]")
            alpha = EPS / PGD_ALPHA_K
            if is_lr:
                X_adv = pgd_logreg(vic_w, X_atk, y_atk, EPS,
                                   iters=PGD_ITERS, restarts=PGD_RESTARTS,
                                   alpha=alpha)
            elif is_xgb:
                X_adv = pgd_xgb(vic_w, X_atk, y_atk, EPS,
                                iters=PGD_ITERS, restarts=PGD_RESTARTS,
                                alpha=alpha)
            else:
                X_adv = pgd_mlp(vic_w, X_atk, y_atk, EPS,
                                iters=PGD_ITERS, restarts=PGD_RESTARTS,
                                alpha=alpha)
            run_one_attack("PGD", X_adv, vic_w, X_eval, y_eval, X_atk, y_atk,
                           vic_name, seed, idx_ev, all_results, persample_results)

            # ── C&W ───────────────────────────────────────────
            print("  [C&W]")
            if is_lr:
                X_adv = cw_logreg(vic_w, X_atk, y_atk, EPS, iters=CW_ITERS)
            elif is_xgb:
                X_adv = cw_xgb(vic_w, X_atk, y_atk, EPS, iters=CW_ITERS)
            else:
                X_adv = cw_mlp(vic_w, X_atk, y_atk, EPS, iters=CW_ITERS)
            run_one_attack("C&W", X_adv, vic_w, X_eval, y_eval, X_atk, y_atk,
                           vic_name, seed, idx_ev, all_results, persample_results)

        # ── Checkpoint après chaque seed ──────────────────────
        pd.DataFrame(all_results).to_csv(
            RESULTS_DIR / f"whitebox_multirun_{TAG}_tmp.csv", index=False)
        if HAS_TIMESTAMPS:
            pd.DataFrame(persample_results).to_csv(
                RESULTS_DIR / f"whitebox_persample_{TAG}_tmp.csv", index=False)
        print(f"\n  ✓ Checkpoint seed {seed} sauvegardé")

    # ══════════════════════════════════════════════════════════
    # SAUVEGARDE FINALE — agrégat (identique à whitebox_multirun.py)
    # ══════════════════════════════════════════════════════════

    df = pd.DataFrame(all_results)
    csv_path = RESULTS_DIR / f"whitebox_multirun_{TAG}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✓ CSV agrégat → {csv_path}")

    summary = (df.groupby(["attack", "model"])["asr"]
                 .agg(["median", "std", "min", "max"])
                 .round(4))
    print(f"\n{'═'*55}")
    print(f"  RÉSUMÉ — {DATASET.upper()}  eps={EPS}")
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
                "evasion_rate_std":    round(float(vals.std())    * 100, 2),
                "evasion_rate_min":    round(float(vals.min())    * 100, 2),
                "evasion_rate_max":    round(float(vals.max())    * 100, 2),
            }

    json_path = RESULTS_DIR / f"whitebox_multirun_{TAG}.json"
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"✓ JSON agrégat → {json_path}")

    # ══════════════════════════════════════════════════════════
    # SAUVEGARDE FINALE — détail par-échantillon (NOUVEAU)
    # ══════════════════════════════════════════════════════════

    if HAS_TIMESTAMPS and persample_results:
        df_persample = pd.DataFrame(persample_results)
        persample_path = RESULTS_DIR / f"whitebox_persample_{TAG}.csv"
        df_persample.to_csv(persample_path, index=False)
        print(f"✓ CSV par-échantillon (timestamps) → {persample_path}")
        print(f"  {len(df_persample)} lignes "
              f"(max {PERSAMPLE_N} samples × seed × modèle × attaque)")
    else:
        print("⚠ Pas de CSV par-échantillon produit (timestamps absents).")

    return df

if __name__ == "__main__":
    run()