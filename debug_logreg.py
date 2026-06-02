# ~/swat/debug_logreg2.py
"""
42.2% résiste à tout changement de gradient → le plafond n'est pas
un problème de gradient mais de géométrie de la frontière de décision.
"""

import numpy as np
import torch
import joblib
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import sys
sys.path.append(str(Path(__file__).parent))

from models import MLP, MLPWrapper, LogRegWrapper, XGBoostWrapper

SAVE_DIR = Path("~/swat/artifacts").expanduser()
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
EPS      = 0.1

X_test = np.load(SAVE_DIR / "X_test.npy")
y_test = np.load(SAVE_DIR / "y_test.npy")

mlp_model = MLP(input_size=X_test.shape[1]).to(DEVICE)
mlp_model.load_state_dict(torch.load(SAVE_DIR / "best_mlp.pt", map_location=DEVICE))
mlp_model.eval()
mlp_w    = MLPWrapper(mlp_model, DEVICE)
logreg_w = LogRegWrapper(joblib.load(SAVE_DIR / "logreg.pkl"))
xgb_w    = XGBoostWrapper(__import__('xgboost').XGBClassifier())
xgb_w.model.load_model(str(SAVE_DIR / "xgb.json"))

def build_shared_eval(seed=0):
    rng           = np.random.default_rng(seed)
    idx_normal    = np.where(y_test == 0)[0]
    idx_attack    = np.where(y_test == 1)[0]
    ok_mask       = ((mlp_w.predict(X_test[idx_attack]) == 1) &
                     (logreg_w.predict(X_test[idx_attack]) == 1) &
                     (xgb_w.predict(X_test[idx_attack]) == 1))
    idx_attack_ok = idx_attack[ok_mask]
    sel_n  = rng.choice(idx_normal,    size=500, replace=False)
    sel_a  = rng.choice(idx_attack_ok, size=min(500, len(idx_attack_ok)), replace=False)
    idx_ev = np.concatenate([sel_n, sel_a])
    rng.shuffle(idx_ev)
    X_eval = X_test[idx_ev]; y_eval = y_test[idx_ev]
    mask   = (y_eval == 1)
    return X_eval, y_eval, X_eval[mask].astype(np.float32), y_eval[mask]

np.random.seed(0); torch.manual_seed(0)
X_eval, y_eval, X_atk, y_atk = build_shared_eval(seed=0)

# ══════════════════════════════════════════════════════════════
# CHECK A : distribution des logits clean sur X_atk
# ══════════════════════════════════════════════════════════════
print("═"*55)
print("CHECK A — Distribution des logits clean")
print("═"*55)

logits_clean = logreg_w.logits_np(X_atk)
from whitebox import THRESHOLD_LOGIT
print(f"THRESHOLD_LOGIT : {THRESHOLD_LOGIT:.4f}")
print(f"Logits — min: {logits_clean.min():.2f}  max: {logits_clean.max():.2f}  "
      f"mean: {logits_clean.mean():.2f}  median: {np.median(logits_clean):.2f}")

# Distance au seuil pour chaque exemple
dist = logits_clean - THRESHOLD_LOGIT
print(f"\nDistance au seuil (logit - threshold) :")
print(f"  min    : {dist.min():.4f}")
print(f"  médiane: {np.median(dist):.4f}")
print(f"  max    : {dist.max():.4f}")

# Combien peuvent être évadés avec eps=0.1 ?
# Un exemple est évadable si : logit - |w|_1 * eps <= threshold
# Pour L∞ : la réduction max du logit = eps * ||w||_1
w     = logreg_w.model.coef_[0]
max_reduction = EPS * np.abs(w).sum()
print(f"\n||w||_1 = {np.abs(w).sum():.4f}")
print(f"Réduction max du logit avec eps={EPS} (L∞) : {max_reduction:.4f}")
print(f"  → Exemples évadables théoriquement "
      f"(dist <= {max_reduction:.2f}) : "
      f"{(dist <= max_reduction).sum()} / {len(X_atk)}")
print(f"  → Exemples évadables théoriquement (%) : "
      f"{(dist <= max_reduction).mean()*100:.1f}%")

# ══════════════════════════════════════════════════════════════
# CHECK B : attaque optimale analytique (pas d'itération)
# ══════════════════════════════════════════════════════════════
print("\n" + "═"*55)
print("CHECK B — Attaque optimale analytique L∞")
print("═"*55)
print("Pour LogReg linéaire, l'optimum L∞ est :")
print("  x_adv = x - eps * sign(w)  (en une seule ligne)")

X_opt = X_atk - EPS * np.sign(w)[None, :]   # broadcast : même signe pour tous
logits_opt = logreg_w.logits_np(X_opt)

evaded_opt = (logits_opt < THRESHOLD_LOGIT).sum()
print(f"\nAttaque optimale → évadés : {evaded_opt} / {len(X_atk)} "
      f"({evaded_opt/len(X_atk)*100:.1f}%)")
print(f"Logits après attaque optimale — "
      f"min: {logits_opt.min():.4f}  mean: {logits_opt.mean():.4f}")

# Vérifie que la réduction est bien max_reduction pour tous
reductions = logits_clean - logits_opt
print(f"Réduction logit — "
      f"min: {reductions.min():.4f}  max: {reductions.max():.4f}  "
      f"(attendu: {max_reduction:.4f} pour tous)")

# ══════════════════════════════════════════════════════════════
# CHECK C : les features sont-elles bornées / clippées ?
# ══════════════════════════════════════════════════════════════
print("\n" + "═"*55)
print("CHECK C — Les features ont-elles des bornes naturelles ?")
print("═"*55)
print("Si les features sont dans [0,1] ou ont des bornes serrées,")
print("clip(x - eps*sign(w), x-eps, x+eps) peut ne pas atteindre x-eps*sign(w)")

# Valeurs min/max des features dans X_atk
feat_min = X_atk.min(axis=0)
feat_max = X_atk.max(axis=0)
feat_range = feat_max - feat_min

print(f"\nRange des features (max - min) :")
print(f"  min range : {feat_range.min():.4f}  (feature la plus étroite)")
print(f"  max range : {feat_range.max():.4f}")
print(f"  median    : {np.median(feat_range):.4f}")
print(f"  < 2*eps ({2*EPS}) : {(feat_range < 2*EPS).sum()} features sur {len(feat_range)}")

# Features où sign(w) * eps dépasse la borne naturelle
w_abs_top10 = np.argsort(np.abs(w))[-10:][::-1]
print(f"\nTop 10 features par |coef_| :")
print(f"{'feat':>5}  {'coef':>8}  {'range':>8}  {'eps*|sign|':>10}  {'bloqué?':>8}")
for j in w_abs_top10:
    bloque = feat_range[j] < EPS
    print(f"  {j:3d}  {w[j]:8.3f}  {feat_range[j]:8.4f}  {EPS:10.4f}  "
          f"{'OUI ⚠' if bloque else 'non':>8}")

# ══════════════════════════════════════════════════════════════
# CHECK D : eps beaucoup plus grand
# ══════════════════════════════════════════════════════════════
print("\n" + "═"*55)
print("CHECK D — ASR avec eps × 10 (eps=1.0) pour tester le plafond dur")
print("═"*55)

for test_eps in [0.1, 0.3, 0.5, 1.0, 2.0, 5.0]:
    X_test_adv   = X_atk - test_eps * np.sign(w)[None, :]
    logits_test  = logreg_w.logits_np(X_test_adv)
    evaded       = (logits_test < THRESHOLD_LOGIT).sum()
    print(f"  eps={test_eps:.1f}  → évadés {evaded:3d}/500 ({evaded/5:.1f}%)")

# ══════════════════════════════════════════════════════════════
# CHECK E : normalisation des données
# ══════════════════════════════════════════════════════════════
print("\n" + "═"*55)
print("CHECK E — Normalisation des données X_test")
print("═"*55)
print(f"X_test global — min: {X_test.min():.4f}  max: {X_test.max():.4f}  "
      f"mean: {X_test.mean():.4f}  std: {X_test.std():.4f}")
print(f"X_atk         — min: {X_atk.min():.4f}  max: {X_atk.max():.4f}  "
      f"mean: {X_atk.mean():.4f}  std: {X_atk.std():.4f}")

# Distribution par feature
per_feat_std = X_test.std(axis=0)
print(f"\nStd par feature — min: {per_feat_std.min():.4f}  "
      f"max: {per_feat_std.max():.4f}  median: {np.median(per_feat_std):.4f}")
print(f"Features avec std < 0.01 : {(per_feat_std < 0.01).sum()}")
print(f"Features avec std > 1.0  : {(per_feat_std > 1.0).sum()}")