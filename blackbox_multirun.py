# blackbox_multirun.py
"""
Blackbox multi-run — SWaT et BATADAL, eps 0.1 (et optionnellement 0.3).

Regroupe en un seul script :
  - Score-based   : Square, NES
  - Transfer      : MI-FGSM, VMI-FGSM, Ensemble-MI (3 substituts)
  - Decision-based: HSJA, RayS  (longs → crash-resume automatique)

Usage :
  python blackbox_multirun.py --dataset swat    --eps 0.1
  python blackbox_multirun.py --dataset batadal  --eps 0.1
  python blackbox_multirun.py --dataset swat    --eps 0.1 --skip_transfer
  python blackbox_multirun.py --dataset swat    --eps 0.1 --only decision

Sorties :
  ~/<dataset>/results/blackbox_multirun_<dataset>_eps<eps>.csv
  ~/<dataset>/results/blackbox_multirun_<dataset>_eps<eps>.json
  ~/<dataset>/results/blackbox_multirun_<dataset>_eps<eps>_tmp.csv   (checkpoint)
"""

import argparse
import json
import time
import warnings
from datetime import timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

import sys
sys.path.append(str(Path(__file__).parent))

from models import (MLP, SmallMLP, DeepMLP,
                    MLPWrapper, LogRegWrapper, XGBoostWrapper,
                    build_eval_set, eval_attack)
from blackbox import square_attack, nes_attack, hsja, rays
from transfer import (mi_fgsm, vmi_fgsm, ensemble_mi_fgsm,
                      train_substitute, eval_transfer)

# ══════════════════════════════════════════════════════════════
# ARGUMENTS
# ══════════════════════════════════════════════════════════════

parser = argparse.ArgumentParser()
parser.add_argument("--dataset",  default="swat",
                    choices=["swat", "batadal"],
                    help="Dataset cible")
parser.add_argument("--eps",      default=0.1, type=float,
                    help="Epsilon L∞ (0.1 ou 0.3)")
parser.add_argument("--n_runs",   default=10,  type=int,
                    help="Nombre de seeds")
parser.add_argument("--only",     default=None,
                    choices=["score", "transfer", "decision"],
                    help="Exécuter seulement une famille d'attaques")
parser.add_argument("--skip_transfer", action="store_true",
                    help="Sauter la famille Transfer (lente à cause des substituts)")
args = parser.parse_args()

DATASET = args.dataset
EPS     = args.eps
N_RUNS  = args.n_runs
SEEDS   = list(range(N_RUNS))

RUN_SCORE    = args.only in (None, "score")
RUN_TRANSFER = args.only in (None, "transfer") and not args.skip_transfer
RUN_DECISION = args.only in (None, "decision")

# ── Chemins selon dataset ──────────────────────────────────────
SAVE_DIR    = Path(f"~/{DATASET}/artifacts").expanduser()
RESULTS_DIR = Path(f"~/{DATASET}/results").expanduser()
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TAG = f"{DATASET}_eps{EPS}"

OUT_CSV  = RESULTS_DIR / f"blackbox_multirun_{TAG}.csv"
TMP_CSV  = RESULTS_DIR / f"blackbox_multirun_{TAG}_tmp.csv"
OUT_JSON = RESULTS_DIR / f"blackbox_multirun_{TAG}.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Taille eval set selon dataset ─────────────────────────────
EVAL_ATK_SIZE = 200 if DATASET == "batadal" else 500
EVAL_NRM_SIZE = 500

# ── Hyperparamètres decision-based ────────────────────────────
MAX_DECISION_BOUNDARY = 300
HSJA_ITERS  = 40
HSJA_N_EST  = 100
RAYS_ITERS  = 60
RAYS_SEARCH = 15

# ── Timer global ──────────────────────────────────────────────
_T0 = time.time()

def elapsed():
    return str(timedelta(seconds=int(time.time() - _T0)))

def elapsed_s():
    return int(time.time() - _T0)

def banner(msg, level=1):
    if level == 1:
        print(f"\n{'═'*65}")
        print(f"  {msg}  [{elapsed()}]")
        print(f"{'═'*65}")
    elif level == 2:
        print(f"\n{'─'*55}")
        print(f"  {msg}  [{elapsed()}]")
        print(f"{'─'*55}")
    else:
        print(f"  >> {msg}  [{elapsed()}]")


# ══════════════════════════════════════════════════════════════
# CALCUL DU NOMBRE TOTAL DE TÂCHES (pour la barre de progression)
# ══════════════════════════════════════════════════════════════

VICTIMS    = ["MLP", "LogReg", "XGBoost"]
N_VICTIMS  = len(VICTIMS)

# Nombre d'attaques par famille × victime
N_SCORE_PER_SEED    = 2 * N_VICTIMS  if RUN_SCORE    else 0  # Square, NES × 3
N_TRANSFER_PER_SEED = (3 * 2 * N_VICTIMS + N_VICTIMS) if RUN_TRANSFER else 0
#   3 substituts × (MI-FGSM + VMI-FGSM) × 3 victimes  +  Ensemble × 3 victimes
N_DECISION_PER_SEED = 2 * N_VICTIMS  if RUN_DECISION else 0  # HSJA, RayS × 3

N_TASKS_PER_SEED = N_SCORE_PER_SEED + N_TRANSFER_PER_SEED + N_DECISION_PER_SEED
N_TASKS_TOTAL    = N_TASKS_PER_SEED * N_RUNS

_task_counter = {"done": 0, "skipped": 0}

def progress_bar(done, total, width=35):
    pct   = done / total if total > 0 else 0
    filled = int(width * pct)
    bar    = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {pct*100:5.1f}%  ({done}/{total})"

def tick(skipped=False):
    """Incrémente le compteur global et affiche la progression."""
    _task_counter["done"] += 1
    if skipped:
        _task_counter["skipped"] += 1
    done  = _task_counter["done"]
    total = N_TASKS_TOTAL
    pct   = done / total if total > 0 else 0

    # ETA basé sur le temps écoulé / tâches faites (hors skips)
    real_done = done - _task_counter["skipped"]
    if real_done > 0:
        avg_s = elapsed_s() / real_done
        remaining = total - done
        eta_s = int(avg_s * remaining)
        eta_str = str(timedelta(seconds=eta_s))
    else:
        eta_str = "?"

    bar = progress_bar(done, total)
    print(f"  {bar}  elapsed={elapsed()}  ETA≈{eta_str}")


print(f"\n{'═'*65}")
print(f"  Dataset  : {DATASET.upper()}")
print(f"  Epsilon  : {EPS}")
print(f"  N_RUNS   : {N_RUNS}")
print(f"  Device   : {DEVICE}")
print(f"  Familles : Score={RUN_SCORE} | Transfer={RUN_TRANSFER} | Decision={RUN_DECISION}")
print(f"  MAX_DB   : {MAX_DECISION_BOUNDARY}  |  HSJA {HSJA_ITERS}×{HSJA_N_EST}  |  RayS {RAYS_ITERS}×{RAYS_SEARCH}")
print(f"  Eval atk : {EVAL_ATK_SIZE} max  |  Eval nrm : {EVAL_NRM_SIZE} max")
print(f"  Tâches   : {N_TASKS_TOTAL} au total  ({N_TASKS_PER_SEED}/seed × {N_RUNS} seeds)")
print(f"  Sorties  : {RESULTS_DIR}")
print(f"{'═'*65}")


# ══════════════════════════════════════════════════════════════
# CHARGEMENT
# ══════════════════════════════════════════════════════════════

def load_victims():
    X_train = np.load(SAVE_DIR / "X_train.npy")
    y_train = np.load(SAVE_DIR / "y_train.npy")
    X_test  = np.load(SAVE_DIR / "X_test.npy")
    y_test  = np.load(SAVE_DIR / "y_test.npy")

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

    return X_train, y_train, X_test, y_test, mlp_w, logreg_w, xgb_w


# ══════════════════════════════════════════════════════════════
# EVAL SET PAR MODÈLE
# ══════════════════════════════════════════════════════════════

def build_per_model_eval(X_test, y_test, victim_w, seed):
    """
    Eval set propre à UN modèle :
    - normaux : tirage aléatoire
    - attaques : uniquement les TP de CE modèle

    Cohérent avec whitebox_multirun.py (pas d'intersection des 3 modèles).
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
    sel_a  = rng.choice(idx_attack_ok, size=n_atk,
                        replace=(n_atk > len(idx_attack_ok)))
    idx_ev = np.concatenate([sel_n, sel_a])
    rng.shuffle(idx_ev)

    X_eval = X_test[idx_ev]
    y_eval = y_test[idx_ev]
    mask   = (y_eval == 1)
    X_atk  = X_eval[mask].astype(np.float32)
    y_atk  = y_eval[mask]

    return X_eval, y_eval, X_atk, y_atk


# ══════════════════════════════════════════════════════════════
# HELPERS DECISION-BASED
# ══════════════════════════════════════════════════════════════

def subsample_bb(X_atk, y_atk, seed):
    """Sous-échantillonnage pour les attaques decision-based (coûteuses)."""
    if len(X_atk) <= MAX_DECISION_BOUNDARY:
        return X_atk, y_atk
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_atk), MAX_DECISION_BOUNDARY, replace=False)
    return X_atk[idx], y_atk[idx]


def build_bb_eval(X_ev, y_ev, X_atk_bb):
    """
    Reconstruit un eval set limité aux exemples bb + normaux.
    Évite la dilution de l'ASR.
    """
    mask_normal = (y_ev == 0)
    X_normal    = X_ev[mask_normal]
    y_normal    = y_ev[mask_normal]

    X_ev_bb  = np.concatenate([X_normal, X_atk_bb], axis=0)
    y_ev_bb  = np.concatenate([y_normal,
                                np.ones(len(X_atk_bb), dtype=y_ev.dtype)], axis=0)

    return X_ev_bb, y_ev_bb


def already_done(existing_df, seed, attack, model):
    if existing_df is None or existing_df.empty:
        return False
    return not existing_df[
        (existing_df["seed"]   == seed)   &
        (existing_df["attack"] == attack) &
        (existing_df["model"]  == model)
    ].empty


# ══════════════════════════════════════════════════════════════
# SEEDS
# ══════════════════════════════════════════════════════════════

def set_all_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ══════════════════════════════════════════════════════════════
# BOUCLE PRINCIPALE
# ══════════════════════════════════════════════════════════════

def run():
    X_train, y_train, X_test, y_test, mlp_w, logreg_w, xgb_w = load_victims()

    victims = [
        ("MLP",     mlp_w),
        ("LogReg",  logreg_w),
        ("XGBoost", xgb_w),
    ]

    # ── Chargement du checkpoint ───────────────────────────────
    if TMP_CSV.exists():
        all_results = pd.read_csv(TMP_CSV).to_dict("records")
        existing_df = pd.DataFrame(all_results)
        # Ajuster le compteur si on reprend depuis un checkpoint
        _task_counter["done"]    = len(all_results)
        _task_counter["skipped"] = len(all_results)  # tâches déjà faites = skippées
        banner(f"Reprise depuis {TMP_CSV} — {len(all_results)} résultats déjà présents",
               level=2)
    else:
        all_results = []
        existing_df = None

    # ══════════════════════════════════════════════════════════
    # BOUCLE SEEDS
    # ══════════════════════════════════════════════════════════

    for seed in SEEDS:
        banner(f"SEED {seed+1}/{N_RUNS}  —  {DATASET.upper()}  eps={EPS}", level=1)
        set_all_seeds(seed)
        t_seed = time.time()

        # ── SCORE-BASED : Square + NES ─────────────────────────
        if RUN_SCORE:
            banner(f"[Score-based] seed={seed}", level=2)
            for vic_name, vic_w in victims:
                X_eval, y_eval, X_atk, y_atk = build_per_model_eval(
                    X_test, y_test, vic_w, seed=seed)
                print(f"  {vic_name} — eval={len(X_eval)}  atk={len(X_atk)}")

                for atk_name, atk_fn in [("Square", square_attack),
                                          ("NES",    nes_attack)]:
                    if already_done(existing_df, seed, atk_name, vic_name):
                        print(f"    [{atk_name}] déjà fait — skip.")
                        tick(skipped=True)
                        continue

                    set_all_seeds(seed)
                    X_adv = atk_fn(vic_w, X_atk, y_atk, EPS)
                    r = eval_attack(vic_w, X_eval, y_eval, X_adv,
                                    atk_name, vic_name, threshold=0.45)
                    r.update({"seed": seed, "family": "Score-based",
                               "eps": EPS, "dataset": DATASET,
                               "n_atk": int((y_eval==1).sum())})
                    all_results.append(r)
                    existing_df = pd.DataFrame(all_results)
                    print(f"    [{atk_name}] ASR={r['asr']*100:.1f}%  "
                          f"F1 {r['f1_clean']:.3f}→{r['f1_adv']:.3f}")
                    tick()

            pd.DataFrame(all_results).to_csv(TMP_CSV, index=False)

        # ── TRANSFER : MI-FGSM, VMI-FGSM, Ensemble ────────────
        if RUN_TRANSFER:
            banner(f"[Transfer] seed={seed}", level=2)
            set_all_seeds(seed)

            sub1 = train_substitute(MLP,      X_train, y_train, DEVICE,
                                    f"Sub1-seed{seed}", noise_std=0.02)
            sub2 = train_substitute(SmallMLP, X_train, y_train, DEVICE,
                                    f"Sub2-seed{seed}", noise_std=0.02)
            sub3 = train_substitute(DeepMLP,  X_train, y_train, DEVICE,
                                    f"Sub3-seed{seed}", noise_std=0.02)

            substitutes = [
                ("Sub1-MLP",      sub1),
                ("Sub2-SmallMLP", sub2),
                ("Sub3-DeepMLP",  sub3),
            ]

            # ✅ FIX : un eval set par victime, les adv sont générés
            #          sur X_atk de la victime courante (pas toujours MLP).
            eval_sets_vic = {}
            for vic_name, vic_w in victims:
                X_eval, y_eval, X_atk, y_atk = build_per_model_eval(
                    X_test, y_test, vic_w, seed=seed)
                eval_sets_vic[vic_name] = (X_eval, y_eval, X_atk, y_atk, vic_w)

            for sub_name, sub_w in substitutes:
                for vic_name, vic_w in victims:
                    X_eval, y_eval, X_atk_vic, y_atk_vic, _ = eval_sets_vic[vic_name]

                    # Adv générés sur X_atk propre à cette victime
                    set_all_seeds(seed)
                    X_adv_mi  = mi_fgsm(sub_w,  X_atk_vic, y_atk_vic, eps=EPS)
                    X_adv_vmi = vmi_fgsm(sub_w, X_atk_vic, y_atk_vic, eps=EPS)

                    for atk_name, X_adv in [("MI-FGSM",  X_adv_mi),
                                             ("VMI-FGSM", X_adv_vmi)]:
                        full_atk = f"{atk_name}_{sub_name}"
                        if already_done(existing_df, seed, full_atk, vic_name):
                            print(f"    [{full_atk} → {vic_name}] skip.")
                            tick(skipped=True)
                            continue

                        r = eval_transfer(X_eval, y_eval, X_adv,
                                          vic_w, sub_name, vic_name, atk_name)
                        r["seed"]    = seed
                        r["family"]  = "Transfer"
                        r["model"]   = r.pop("victim")
                        r["attack"]  = full_atk
                        r["eps"]     = EPS
                        r["dataset"] = DATASET
                        r["n_atk"]   = int((y_eval==1).sum())
                        all_results.append(r)
                        existing_df = pd.DataFrame(all_results)
                        print(f"    [{full_atk} → {vic_name}] ASR={r['asr']*100:.1f}%")
                        tick()

            # Ensemble — ✅ FIX : un X_adv_ens par victime
            for vic_name, vic_w in victims:
                X_eval, y_eval, X_atk_vic, y_atk_vic, _ = eval_sets_vic[vic_name]

                atk_name = "Ensemble-MI"
                if already_done(existing_df, seed, atk_name, vic_name):
                    print(f"    [{atk_name} → {vic_name}] skip.")
                    tick(skipped=True)
                    continue

                set_all_seeds(seed)
                X_adv_ens = ensemble_mi_fgsm(
                    [sub1, sub2, sub3], X_atk_vic, y_atk_vic, eps=EPS,
                    weights=[1/3, 1/3, 1/3]
                )
                r = eval_transfer(X_eval, y_eval, X_adv_ens,
                                  vic_w, "Ensemble(S1+S2+S3)", vic_name, atk_name)
                r["seed"]    = seed
                r["family"]  = "Transfer"
                r["model"]   = r.pop("victim")
                r["eps"]     = EPS
                r["dataset"] = DATASET
                r["n_atk"]   = int((y_eval==1).sum())
                all_results.append(r)
                existing_df = pd.DataFrame(all_results)
                print(f"    [{atk_name} → {vic_name}] ASR={r['asr']*100:.1f}%")
                tick()

            pd.DataFrame(all_results).to_csv(TMP_CSV, index=False)

        # ── DECISION-BASED : HSJA + RayS ──────────────────────
        if RUN_DECISION:
            banner(f"[Decision-based] seed={seed}", level=2)

            for vic_name, vic_w in victims:
                X_eval, y_eval, X_atk, y_atk = build_per_model_eval(
                    X_test, y_test, vic_w, seed=seed)

                X_atk_bb, y_atk_bb = subsample_bb(X_atk, y_atk, seed)

                print(f"\n  {vic_name} — eval={len(X_eval)}  "
                      f"atk={len(X_atk)} → bb={len(X_atk_bb)}")

                for atk_name, atk_fn, atk_kw in [
                    ("HSJA", hsja,
                     {"iters": HSJA_ITERS, "n_est": HSJA_N_EST}),
                    ("RayS", rays,
                     {"iters": RAYS_ITERS, "search_steps": RAYS_SEARCH}),
                ]:
                    if already_done(existing_df, seed, atk_name, vic_name):
                        sub = existing_df[
                            (existing_df["seed"]   == seed)   &
                            (existing_df["attack"] == atk_name) &
                            (existing_df["model"]  == vic_name)
                        ].iloc[0]
                        print(f"    [{atk_name}] déjà fait — "
                              f"ASR={sub['asr']:.1%}  skip.")
                        tick(skipped=True)
                        continue

                    print(f"    [{atk_name}] démarrage...")
                    t0 = time.time()
                    set_all_seeds(seed)

                    X_adv_bb = atk_fn(vic_w, X_atk_bb, y_atk_bb, EPS, **atk_kw)

                    X_ev_bb, y_ev_bb = build_bb_eval(
                         X_ev=X_eval, y_ev=y_eval, X_atk_bb=X_atk_bb)

                    r = eval_attack(vic_w, X_ev_bb, y_ev_bb, X_adv_bb,atk_name, vic_name, threshold=0.45)
                    
                    r.update({"seed": seed, "family": "Decision-based",
                               "eps": EPS, "dataset": DATASET,
                               "n_atk": len(X_atk_bb)})
                    all_results.append(r)
                    existing_df = pd.DataFrame(all_results)
                    pd.DataFrame(all_results).to_csv(TMP_CSV, index=False)

                    dt = timedelta(seconds=int(time.time() - t0))
                    print(f"    [{atk_name}] ✓  ASR={r['asr']:.1%}  "
                          f"F1adv={r['f1_adv']:.4f}  durée={dt}  "
                          f"[total : {elapsed()}]")
                    tick()

        # ── Checkpoint global après chaque seed ───────────────
        pd.DataFrame(all_results).to_csv(TMP_CSV, index=False)
        t_seed_dt  = timedelta(seconds=int(time.time() - t_seed))
        seeds_left = N_RUNS - seed - 1
        eta        = timedelta(seconds=int((time.time() - t_seed) * seeds_left))
        print(f"\n  Seed {seed} terminée en {t_seed_dt}  |  "
              f"seeds restantes : {seeds_left}  |  ETA ≈ {eta}")

    # ══════════════════════════════════════════════════════════
    # SAUVEGARDE FINALE
    # ══════════════════════════════════════════════════════════

    df = pd.DataFrame(all_results)
    df.to_csv(OUT_CSV, index=False)
    print(f"\n✓ CSV → {OUT_CSV}")

    if TMP_CSV.exists():
        TMP_CSV.unlink()

    banner("RÉSULTATS FINAUX", level=1)
    summary = (df.groupby(["family", "attack", "model"])["asr"]
                 .agg(["median", "std", "min", "max"])
                 .round(4))
    print(summary.to_string())

    out = {}
    for model_name in df["model"].unique():
        out[model_name] = {}
        sub_m = df[df["model"] == model_name]
        for attack_name in sub_m["attack"].unique():
            vals = sub_m[sub_m["attack"] == attack_name]["asr"]
            out[model_name][attack_name] = {
                "evasion_rate_median": round(float(vals.median()) * 100, 2),
                "evasion_rate_std":    round(float(vals.std())    * 100, 2),
                "evasion_rate_min":    round(float(vals.min())    * 100, 2),
                "evasion_rate_max":    round(float(vals.max())    * 100, 2),
            }

    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"✓ JSON → {OUT_JSON}")
    print(f"  Durée totale : {elapsed()}")

    return df


if __name__ == "__main__":
    run()