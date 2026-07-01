"""
common_whitebox.py
Fonctions partagées entre les runners whitebox_fgsm_multirun.py,
whitebox_pgd_multirun.py, whitebox_cw_multirun.py.

IMPORTANT : garantit que les 3 runners construisent EXACTEMENT le même
eval set pour un (dataset, seed, modèle) donné, car build_per_model_eval()
utilise np.random.default_rng(seed) — un RNG local, indépendant de tout
ce qui tourne ailleurs dans le process (donc peu importe l'ordre / le
parallélisme entre les 3 scripts, les résultats restent comparables).

Ce fichier doit être dans le même dossier que models.py et whitebox.py.
"""

import argparse
import numpy as np
import torch
import joblib
from pathlib import Path
from xgboost import XGBClassifier

import sys
sys.path.append(str(Path(__file__).parent))

from models import MLP, MLPWrapper, LogRegWrapper, XGBoostWrapper


# ══════════════════════════════════════════════════════════════
# ARGUMENTS
# ══════════════════════════════════════════════════════════════

def build_arg_parser(description=""):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--dataset", default="swat", choices=["swat", "batadal"],
                         help="Dataset cible")
    parser.add_argument("--eps", default=0.1, type=float,
                         help="Epsilon L∞ (0.1 ou 0.3)")
    parser.add_argument("--n_runs", default=10, type=int,
                         help="Nombre de seeds")
    parser.add_argument("--fast", action="store_true",
                         help="FAST_MODE : PGD 50x3 au lieu de 200x10, C&W 150 iters")
    parser.add_argument("--persample_n", default=50, type=int,
                         help="Nb max de samples gardés par (seed, modèle) dans "
                              "le CSV par-échantillon (timestamps)")
    return parser


# ══════════════════════════════════════════════════════════════
# CHEMINS
# ══════════════════════════════════════════════════════════════

def setup_paths(dataset, eps):
    save_dir = Path(f"~/{dataset}/artifacts").expanduser()
    results_dir = Path(f"~/{dataset}/results").expanduser()
    results_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{dataset}_eps{eps}"
    return save_dir, results_dir, tag


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def eval_sizes(dataset):
    """SWaT : beaucoup d'attaques -> 500. BATADAL : seulement 219 en test -> 200 max."""
    eval_atk_size = 200 if dataset == "batadal" else 500
    eval_nrm_size = 500
    return eval_atk_size, eval_nrm_size


# ══════════════════════════════════════════════════════════════
# CHARGEMENT DES VICTIMES
# ══════════════════════════════════════════════════════════════

def load_victims(save_dir, device):
    X_test = np.load(save_dir / "X_test.npy")
    y_test = np.load(save_dir / "y_test.npy")

    mlp_model = MLP(input_size=X_test.shape[1]).to(device)
    mlp_model.load_state_dict(
        torch.load(save_dir / "best_mlp.pt", map_location=device))
    mlp_model.eval()
    mlp_w = MLPWrapper(mlp_model, device)

    logreg_w = LogRegWrapper(joblib.load(save_dir / "logreg.pkl"))

    xgb_model = XGBClassifier()
    xgb_model.load_model(str(save_dir / "xgb.json"))
    xgb_w = XGBoostWrapper(xgb_model)

    print(f"\n✓ Modèles chargés depuis {save_dir}")
    print(f"  X_test : {X_test.shape} — attaques : {y_test.sum()} / {len(y_test)}")

    return X_test, y_test, mlp_w, logreg_w, xgb_w


# ══════════════════════════════════════════════════════════════
# EVAL SET PAR MODÈLE — NE PAS MODIFIER LA LOGIQUE DE TIRAGE
# (sinon les 3 runners ne partagent plus le même eval set et les
#  ASR FGSM / PGD / C&W ne sont plus comparables entre eux)
# ══════════════════════════════════════════════════════════════

def build_per_model_eval(X_test, y_test, victim_w, seed,
                          eval_atk_size, eval_nrm_size, dataset):
    rng = np.random.default_rng(seed)
    idx_normal = np.where(y_test == 0)[0]
    idx_attack = np.where(y_test == 1)[0]

    preds_vic = victim_w.predict(X_test[idx_attack], threshold=0.45)
    idx_attack_ok = idx_attack[preds_vic == 1]

    n_atk = min(eval_atk_size, len(idx_attack_ok))
    n_nrm = min(eval_nrm_size, len(idx_normal))

    if n_atk == 0:
        raise ValueError(
            f"Aucun TP pour ce modèle sur {dataset} — vérifier le F1 baseline.")

    sel_n = rng.choice(idx_normal, size=n_nrm, replace=False)
    sel_a = rng.choice(idx_attack_ok, size=n_atk, replace=n_atk > len(idx_attack_ok))
    idx_ev = np.concatenate([sel_n, sel_a])
    rng.shuffle(idx_ev)

    X_eval = X_test[idx_ev]
    y_eval = y_test[idx_ev]
    mask = (y_eval == 1)
    X_atk = X_eval[mask].astype(np.float32)
    y_atk = y_eval[mask]

    return X_eval, y_eval, X_atk, y_atk, idx_ev


def load_timestamps(save_dir):
    """
    Charge timestamps_test.npy si présent (généré par 00_extract_timestamps.py).
    Ne plante jamais si absent — retourne juste has_timestamps=False, et les
    runners doivent alors sauter l'export par-échantillon.
    """
    path = save_dir / "timestamps_test.npy"
    if path.exists():
        return np.load(path, allow_pickle=True), True
    print(f"\n⚠ {path} introuvable — export par-échantillon désactivé "
          f"(lance 00_extract_timestamps.py --dataset <dataset> d'abord).")
    return None, False


def set_all_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


VICTIMS_SPEC = [
    # (nom, is_logreg, is_xgb)
    ("MLP", False, False),
    ("LogReg", True, False),
    ("XGBoost", False, True),
]