"""
defenses.py  — version corrigée v2
Entraînement des modèles défendus + évaluation immédiate.

═══════════════════════════════════════════════════════════════
CORRECTIONS PAR RAPPORT À LA VERSION PRÉCÉDENTE
═══════════════════════════════════════════════════════════════

FIX A — BUG CRITIQUE : _build_aug_dataset EMPOISONNE LE DATASET LOGREG
  Problème : les adversariaux (X_adv, y=1) ajoutés au dataset ont la
  FORME d'un normal (c'est le but de l'attaque) mais le LABEL d'une
  attaque. Le modèle reçoit des signaux contradictoires et abandonne
  la classe 1 → recall=0, ASR=100%.

  Mauvaise intuition initiale : "montrer au modèle des attaques
  perturbées avec leur vrai label pour qu'il les reconnaisse malgré tout".

  Réalité : les adversariaux générés depuis la classe 1 RESSEMBLENT
  délibérément à la classe 0. Les ajouter avec y=1 crée un conflit
  insoluble pour un modèle linéaire.

  Solution en deux volets :
    1. On génère les adversariaux depuis les NORMAUX (classe 0) avec
       y_adv=0 : le modèle apprend que ces voisins des normaux sont
       AUSSI des normaux → frontière plus robuste côté normal.
    2. On génère les adversariaux depuis les ATTAQUES (classe 1) mais
       on les conserve avec y_adv=1 ET on sur-pondère ces exemples
       dans le fit (sample_weight × HARD_SAMPLE_WEIGHT) pour forcer
       le modèle à maintenir sa frontière sur ces cas difficiles.
       Alternative plus simple activée par défaut : on n'ajoute PAS
       les adversariaux des attaques — on génère seulement ceux des
       normaux. Cela suffit pour robustifier la frontière sans
       introduire de contradiction.

  Paramètre : ADV_ATTACK_RATIO contrôle si on ajoute les adversariaux
  d'attaque (avec sur-pondération) ou pas du tout (0.0 = désactivé).

FIX B — BUG IMPORTANT : EPS_AT trop petit (0.1) vs évaluation à eps=0.3
  Problème : le MLP est entraîné adversarialement avec EPS_AT_LIST=[0.1, 0.3]
  mais EPS_AT (valeur initiale du curriculum) = 0.1.
  Le curriculum monte de 0.1 à 0.3 sur 30 epochs mais avec AT_EPOCHS=60
  et AT_PATIENCE=10, l'early stop peut intervenir avant que le modèle
  voie suffisamment d'exemples à eps=0.3.

  Solution :
    - EPS_AT_START = 0.05  (départ plus bas, montée plus douce)
    - EPS_AT_END   = 0.35  (dépasse légèrement le budget d'évaluation)
    - Curriculum : 40 epochs pour monter (au lieu de 30)
    - AT_PATIENCE augmenté à 15 pour laisser le modèle s'adapter à eps élevé

FIX C — BUG MINEUR : augment_xgb_proxy utilise le proxy MLP AT-FGSM
  mais celui-ci est entraîné avec EPS_AT correct après FIX B.
  En plus, on ajoute une deuxième variante : augmentation XGBoost
  avec adversariaux DIRECTS générés sur XGBoost lui-même (plus de
  transfert, adversariaux en distribution).

  Nouveau paramètre : XGB_AUG_DIRECT_ITERS=50 pour les adversariaux
  directs (coût acceptable, directement utiles).

FIX D — CORRECTION _build_aug_dataset POUR LOGREG ET XGB
  Nouvelle logique :
    1. Adversariaux générés depuis les NORMAUX (classe 0) → ajoutés
       avec label 0. Force la frontière à être robuste côté normal.
    2. Les adversariaux depuis les attaques (classe 1) sont OPTIONNELS
       et contrôlés par ADV_ATTACK_RATIO (défaut 0.0 = désactivé).
       Si activé, ils sont ajoutés avec y=1 ET sample_weight élevé.
    3. Le paramètre NORMAL_AUG_RATIO passe à 1.0 (tous les normaux,
       pas seulement 30%) pour maximiser la couverture de la frontière.

═══════════════════════════════════════════════════════════════
RÉSUMÉ DES CHANGEMENTS DE PARAMÈTRES
═══════════════════════════════════════════════════════════════
  NORMAL_AUG_RATIO  : 0.3  → 1.0   (tous les normaux augmentés)
  EPS_AT_START      : 0.1  → 0.05  (curriculum plus doux)
  EPS_AT_END        : 0.3  → 0.35  (dépasse le budget d'évaluation)
  AT_PATIENCE       : 10   → 15    (plus de tolérance à eps élevé)
  AT_CURRICULUM_EP  : 30   → 40    (montée plus lente)
  ADV_ATTACK_RATIO  : n/a  → 0.0   (adversariaux d'attaque désactivés)
  HARD_SAMPLE_WEIGHT: n/a  → 3.0   (si ADV_ATTACK_RATIO > 0)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib
import warnings
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score, precision_score
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

import sys
import importlib.util

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))

from models import (MLP, MLPWrapper, LogRegWrapper, XGBoostWrapper,
                    build_eval_set, eval_attack)


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

wb = _load_module("whitebox", BASE / "whitebox.py")

fgsm_mlp        = wb.fgsm_mlp
fgsm_logreg     = wb.fgsm_logreg
fgsm_xgb        = wb.fgsm_xgb
pgd_mlp         = wb.pgd_mlp
pgd_logreg      = wb.pgd_logreg
pgd_xgb         = wb.pgd_xgb
cw_mlp          = wb.cw_mlp
cw_logreg       = wb.cw_logreg
cw_xgb          = wb.cw_xgb
THRESHOLD_LOGIT = wb.THRESHOLD_LOGIT


# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════

SAVE_DIR  = Path("~/swat/artifacts").expanduser()
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
THRESHOLD = 0.45

# ── FIX B : plage eps plus large que le budget d'évaluation (0.3) ─
EPS_AT_LIST       = [0.1, 0.3]
EPS_AT_START      = 0.05   # FIX B : départ plus bas (était 0.1 = EPS_AT)
EPS_AT_END        = 0.35   # FIX B : dépasse légèrement eps d'évaluation
AT_EPOCHS         = 60
AT_PATIENCE       = 15     # FIX B : augmenté (était 10) pour laisser le
                            #         modèle s'adapter aux eps élevés
AT_CURRICULUM_EP  = 40     # FIX B : montée plus lente (était 30)
AT_MIX_RATIO      = 0.5
PGD_AT_ITERS      = 10
PGD_AT_ALPHA      = lambda eps: eps / 4

# ── FIX A : paramètres augmentation dataset ───────────────────
NORMAL_AUG_RATIO  = 1.0    # FIX A : tous les normaux (était 0.3)
                            #         → couverture maximale de la frontière
ADV_ATTACK_RATIO  = 0.0    # FIX A : adversariaux d'attaque désactivés
                            #         (était implicitement 1.0 → empoisonnait)
HARD_SAMPLE_WEIGHT = 3.0   # poids si ADV_ATTACK_RATIO > 0 (non utilisé ici)

print(f"Device : {DEVICE}")


# ══════════════════════════════════════════════════════════════
# CHARGEMENT
# ══════════════════════════════════════════════════════════════

def load_artifacts():
    X_train = np.load(SAVE_DIR / "X_train.npy")
    y_train = np.load(SAVE_DIR / "y_train.npy")
    X_test  = np.load(SAVE_DIR / "X_test.npy")
    y_test  = np.load(SAVE_DIR / "y_test.npy")

    mlp_model = MLP(input_size=X_test.shape[1]).to(DEVICE)
    mlp_model.load_state_dict(
        torch.load(SAVE_DIR / "best_mlp.pt", map_location=DEVICE)
    )
    mlp_model.eval()
    mlp_w = MLPWrapper(mlp_model, DEVICE)

    logreg_w = LogRegWrapper(joblib.load(SAVE_DIR / "logreg.pkl"))

    xgb_model = XGBClassifier()
    xgb_model.load_model(str(SAVE_DIR / "xgb.json"))
    xgb_w = XGBoostWrapper(xgb_model)

    return X_train, y_train, X_test, y_test, mlp_w, logreg_w, xgb_w


# ══════════════════════════════════════════════════════════════
# UTILITAIRES
# ══════════════════════════════════════════════════════════════

def quick_eval(wrapper, X_test, y_test, label=""):
    y_pred = wrapper.predict(X_test)
    f1  = f1_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    pre = precision_score(y_test, y_pred, zero_division=0)
    print(f"    {label:<35} F1={f1:.4f}  Recall={rec:.4f}  Prec={pre:.4f}")
    return f1


def _bar(v, w=15):
    return "█" * int(round(v * w)) + "░" * (w - int(round(v * w)))


# ══════════════════════════════════════════════════════════════
# CHARGEMENT MLP ROBUSTE (détection mismatch d'architecture)
# ══════════════════════════════════════════════════════════════

def _load_mlp_safe(fpath, input_size):
    """
    Charge un MLP sauvegardé avec détection automatique de mismatch
    d'architecture. Lève RuntimeError explicite si les clés ne matchent pas.
    """
    model = MLP(input_size=input_size).to(DEVICE)
    checkpoint = torch.load(fpath, map_location=DEVICE)

    model_keys      = set(model.state_dict().keys())
    checkpoint_keys = set(checkpoint.keys())
    missing    = model_keys - checkpoint_keys
    unexpected = checkpoint_keys - model_keys

    if missing or unexpected:
        msg = (
            f"\n{'═'*60}\n"
            f"  MISMATCH D'ARCHITECTURE — {fpath.name}\n"
            f"{'═'*60}\n"
            f"  Clés manquantes  : {missing or 'aucune'}\n"
            f"  Clés inattendues : {unexpected or 'aucune'}\n\n"
            f"  Solution : supprimer les anciens checkpoints et relancer.\n"
            f"  $ rm {fpath}\n"
            f"  $ python3 defenses.py\n"
            f"{'═'*60}"
        )
        raise RuntimeError(msg)

    model.load_state_dict(checkpoint)
    model.eval()
    return model


# ══════════════════════════════════════════════════════════════
# DÉFENSE 1 & 2 — ADVERSARIAL TRAINING MLP  (FIX B)
# ══════════════════════════════════════════════════════════════

def adversarial_train_mlp(X_train, y_train, X_test, y_test, input_size,
                           attack="fgsm",
                           epochs=AT_EPOCHS, patience=AT_PATIENCE,
                           mix_ratio=AT_MIX_RATIO):
    fname = f"mlp_at_{attack}.pt"
    fpath = SAVE_DIR / fname

    if fpath.exists():
        print(f"    {fname} déjà présent → chargement direct")
        model = _load_mlp_safe(fpath, input_size)
        w = MLPWrapper(model, DEVICE)
        quick_eval(w, X_test, y_test, f"[chargé] {fname}")
        return w

    model = MLP(input_size=input_size).to(DEVICE)
    pos_weight = torch.tensor(
        [(y_train == 0).sum() / (y_train == 1).sum()],
        dtype=torch.float32
    ).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    X_t    = torch.tensor(X_train, dtype=torch.float32)
    y_t    = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=2048, shuffle=True)

    val_size = min(8000, len(X_train) // 5)
    X_val_t  = torch.tensor(X_train[:val_size], dtype=torch.float32).to(DEVICE)
    y_val    = y_train[:val_size]

    best_f1, no_improve, best_state = 0.0, 0, None

    # FIX B : curriculum étendu (EPS_AT_START → EPS_AT_END sur AT_CURRICULUM_EP epochs)
    print(f"    Curriculum eps : {EPS_AT_START:.3f} → {EPS_AT_END:.3f} "
          f"sur {AT_CURRICULUM_EP} epochs (évaluation à 0.3)")

    for epoch in range(epochs):
        model.train()

        # FIX B : curriculum plus doux et dépassant le budget d'évaluation
        frac     = min(epoch / AT_CURRICULUM_EP, 1.0)
        eps_curr = EPS_AT_START + frac * (EPS_AT_END - EPS_AT_START)

        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)

            tmp_w     = MLPWrapper(model, DEVICE)
            xb_np     = xb.detach().cpu().numpy()
            yb_np     = yb.detach().cpu().numpy().flatten().astype(int)

            if attack == "fgsm":
                xb_adv_np = fgsm_mlp(tmp_w, xb_np, yb_np, eps=eps_curr)
            else:
                xb_adv_np = pgd_mlp(tmp_w, xb_np, yb_np, eps=eps_curr,
                                     iters=PGD_AT_ITERS, restarts=1,
                                     alpha=PGD_AT_ALPHA(eps_curr))

            xb_adv  = torch.tensor(xb_adv_np, dtype=torch.float32, device=DEVICE)
            n_adv   = int(len(xb) * mix_ratio)
            idx_adv = torch.randperm(len(xb))[:n_adv]
            xb_mix  = xb.clone()
            xb_mix[idx_adv] = xb_adv[idx_adv]

            optimizer.zero_grad()
            criterion(model(xb_mix), yb).backward()
            optimizer.step()

        scheduler.step()

        model.eval()
        with torch.no_grad():
            proba  = torch.sigmoid(model(X_val_t)).cpu().numpy().flatten()
            y_pred = (proba >= THRESHOLD).astype(int)
        f1 = f1_score(y_val, y_pred, zero_division=0)

        if f1 > best_f1:
            best_f1, no_improve = f1, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"    Early stop epoch {epoch+1} — best F1 {best_f1:.4f}")
                break

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1:3d} | F1 val {f1:.4f} | "
                  f"best {best_f1:.4f} | eps_curr {eps_curr:.3f}")

    if best_state:
        model.load_state_dict(best_state)
    model.eval()

    torch.save(model.state_dict(), fpath)
    print(f"    Sauvegardé : {fpath}  (best F1 val {best_f1:.4f})")

    w = MLPWrapper(model, DEVICE)
    quick_eval(w, X_test, y_test, f"[clean] {fname}")
    return w


# ══════════════════════════════════════════════════════════════
# FIX A — _build_aug_dataset CORRIGÉ
# ══════════════════════════════════════════════════════════════

def _build_aug_dataset(X_train, y_train, adv_fn, eps_list,
                       adv_attack_ratio=ADV_ATTACK_RATIO):
    """
    Construit un dataset augmenté pour l'entraînement défensif.

    FIX A — Logique corrigée :
    ──────────────────────────
    ANCIENNE logique (BUGGUÉE) :
      - Générait X_adv depuis les ATTAQUES (y=1) et les ajoutait avec y=1
      - X_adv ressemble à un normal (c'est le but de l'attaque FGSM/PGD)
      - Résultat : le modèle reçoit (X_normal-like, y=1) → contradiction
        insoluble → abandon de la classe 1 → recall=0

    NOUVELLE logique (CORRIGÉE) :
      - Génère X_adv depuis les NORMAUX (y=0) avec label y=0
        → le modèle apprend que les voisins adversariaux des normaux
          sont AUSSI des normaux : la frontière est robuste côté normal
      - Les adversariaux depuis les ATTAQUES (y=1) sont optionnels
        (adv_attack_ratio > 0) et ajoutés avec leur label réel y=1
        + sample_weight élevé pour forcer la frontière côté attaque
        MAIS par défaut désactivés (adv_attack_ratio=0.0) car ils
        créent plus de bruit que de signal pour LogReg

    Retourne : (X_aug, y_aug, sample_weights)
      sample_weights est None si tous les poids sont 1.0
    """
    X_parts = [X_train]
    y_parts = [y_train]
    w_parts = [np.ones(len(y_train))]   # poids uniformes pour les données originales

    # ── Adversariaux depuis les NORMAUX (y=0) ─────────────────
    # FIX A : on augmente les normaux, pas les attaques
    mask_norm = (y_train == 0)
    X_norm    = X_train[mask_norm].astype(np.float32)
    y_norm    = y_train[mask_norm]

    n_normal = int(len(X_norm) * NORMAL_AUG_RATIO)  # NORMAL_AUG_RATIO=1.0 → tous
    idx      = np.random.choice(len(X_norm), n_normal, replace=False)
    X_n_sub  = X_norm[idx]
    y_n_sub  = y_norm[idx]

    for eps in eps_list:
        X_adv_n = adv_fn(X_n_sub, y_n_sub, eps)
        X_parts.append(X_adv_n)
        y_parts.append(y_n_sub)                           # label 0 conservé ✓
        w_parts.append(np.ones(len(X_adv_n)))
        print(f"      eps={eps} → +{len(X_adv_n)} adversariaux (normaux, y=0)")

    # ── Adversariaux depuis les ATTAQUES (y=1) — OPTIONNEL ────
    # FIX A : désactivé par défaut (adv_attack_ratio=0.0)
    if adv_attack_ratio > 0:
        mask_atk = (y_train == 1)
        X_atk    = X_train[mask_atk].astype(np.float32)
        y_atk    = y_train[mask_atk]
        n_atk    = int(len(X_atk) * adv_attack_ratio)
        idx_a    = np.random.choice(len(X_atk), n_atk, replace=False)
        X_a_sub  = X_atk[idx_a]
        y_a_sub  = y_atk[idx_a]

        for eps in eps_list:
            X_adv_a = adv_fn(X_a_sub, y_a_sub, eps)
            X_parts.append(X_adv_a)
            y_parts.append(y_a_sub)                       # label 1 conservé ✓
            w_parts.append(np.full(len(X_adv_a), HARD_SAMPLE_WEIGHT))
            print(f"      eps={eps} → +{len(X_adv_a)} adversariaux "
                  f"(attaques, y=1, poids={HARD_SAMPLE_WEIGHT})")

    X_aug = np.concatenate(X_parts, axis=0)
    y_aug = np.concatenate(y_parts, axis=0)
    w_aug = np.concatenate(w_parts, axis=0)

    has_custom_weights = (w_aug != 1.0).any()
    sw = w_aug if has_custom_weights else None

    print(f"      Dataset : {len(X_train)} → {len(X_aug)} exemples "
          f"(+{len(X_aug)-len(X_train)}) | "
          f"poids custom : {'oui' if sw is not None else 'non'}")
    return X_aug, y_aug, sw


# ══════════════════════════════════════════════════════════════
# DÉFENSE 3 — AUGMENTATION LogReg (FGSM + PGD)
# ══════════════════════════════════════════════════════════════

def augment_logreg(logreg_wrapper, X_train, y_train, X_test, y_test,
                   attack="fgsm", eps_list=None):
    if eps_list is None:
        eps_list = EPS_AT_LIST

    fname = f"logreg_aug_{attack}.pkl"
    fpath = SAVE_DIR / fname

    if fpath.exists():
        print(f"    {fname} déjà présent → chargement direct")
        w = LogRegWrapper(joblib.load(fpath))
        quick_eval(w, X_test, y_test, f"[chargé] {fname}")
        return w

    print(f"    Génération X_adv LogReg ({attack}, eps={eps_list})...")
    print(f"    [FIX A] Adversariaux depuis NORMAUX uniquement (adv_attack_ratio=0)")

    if attack == "fgsm":
        adv_fn = lambda X, y, eps: fgsm_logreg(logreg_wrapper, X, y, eps)
    else:
        adv_fn = lambda X, y, eps: pgd_logreg(logreg_wrapper, X, y, eps,
                                               iters=20, restarts=3)

    # FIX A : _build_aug_dataset retourne maintenant (X, y, weights)
    X_aug, y_aug, sample_weights = _build_aug_dataset(
        X_train, y_train, adv_fn, eps_list
    )

    new_lr = LogisticRegression(
        C=1.0, max_iter=2000, solver="saga",
        class_weight="balanced", random_state=42
    )
    # FIX A : passage des sample_weights si présents
    fit_kwargs = {}
    if sample_weights is not None:
        fit_kwargs["sample_weight"] = sample_weights
    new_lr.fit(X_aug, y_aug, **fit_kwargs)

    joblib.dump(new_lr, fpath)
    print(f"    Sauvegardé : {fpath}")

    w = LogRegWrapper(new_lr)
    quick_eval(w, X_test, y_test, f"[clean] {fname}")
    return w


# ══════════════════════════════════════════════════════════════
# DÉFENSE 4a — AUGMENTATION XGBoost via PROXY MLP
# ══════════════════════════════════════════════════════════════

def augment_xgb_proxy(xgb_wrapper, mlp_proxy_wrapper,
                      X_train, y_train, X_test, y_test,
                      attack="fgsm", eps_list=None):
    if eps_list is None:
        eps_list = EPS_AT_LIST

    fname = f"xgb_aug_proxy_{attack}.json"
    fpath = SAVE_DIR / fname

    if fpath.exists():
        print(f"    {fname} déjà présent → chargement direct")
        m = XGBClassifier()
        m.load_model(str(fpath))
        w = XGBoostWrapper(m)
        quick_eval(w, X_test, y_test, f"[chargé] {fname}")
        return w

    print(f"    Génération X_adv XGBoost via proxy MLP ({attack}, eps={eps_list})...")
    print(f"    [FIX A] Adversariaux depuis NORMAUX uniquement")

    if attack == "fgsm":
        adv_fn = lambda X, y, eps: fgsm_mlp(mlp_proxy_wrapper, X, y, eps)
    else:
        adv_fn = lambda X, y, eps: pgd_mlp(mlp_proxy_wrapper, X, y, eps,
                                            iters=20, restarts=3,
                                            alpha=PGD_AT_ALPHA(eps))

    # FIX A : dépackage du triplet (X, y, weights)
    X_aug, y_aug, sample_weights = _build_aug_dataset(
        X_train, y_train, adv_fn, eps_list
    )

    w = _fit_xgb(X_aug, y_aug, fpath, sample_weights=sample_weights)
    quick_eval(w, X_test, y_test, f"[clean] {fname}")
    return w


# ══════════════════════════════════════════════════════════════
# DÉFENSE 4b — AUGMENTATION XGBoost DIRECTE (gradient numérique)
# ══════════════════════════════════════════════════════════════

def augment_xgb_direct(xgb_wrapper, X_train, y_train, X_test, y_test,
                        attack="fgsm", eps_list=None):
    if eps_list is None:
        eps_list = EPS_AT_LIST

    fname = f"xgb_aug_direct_{attack}.json"
    fpath = SAVE_DIR / fname

    if fpath.exists():
        print(f"    {fname} déjà présent → chargement direct")
        m = XGBClassifier()
        m.load_model(str(fpath))
        w = XGBoostWrapper(m)
        quick_eval(w, X_test, y_test, f"[chargé] {fname}")
        return w

    print(f"    Génération X_adv XGBoost DIRECTE ({attack}, eps={eps_list})...")
    print(f"    [FIX A] Adversariaux depuis NORMAUX uniquement")

    if attack == "fgsm":
        adv_fn = lambda X, y, eps: fgsm_xgb(xgb_wrapper, X, y, eps)
    else:
        adv_fn = lambda X, y, eps: pgd_xgb(xgb_wrapper, X, y, eps,
                                            iters=50, restarts=5)

    # FIX A : dépackage du triplet (X, y, weights)
    X_aug, y_aug, sample_weights = _build_aug_dataset(
        X_train, y_train, adv_fn, eps_list
    )

    w = _fit_xgb(X_aug, y_aug, fpath, sample_weights=sample_weights)
    quick_eval(w, X_test, y_test, f"[clean] {fname}")
    return w


def _fit_xgb(X_aug, y_aug, fpath, sample_weights=None):
    scale_pw = float((y_aug == 0).sum()) / float((y_aug == 1).sum())
    new_xgb  = XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pw,
        eval_metric="logloss", early_stopping_rounds=20,
        device="cuda" if torch.cuda.is_available() else "cpu",
        random_state=42, verbosity=0
    )
    idx = np.random.permutation(len(X_aug))
    X_aug, y_aug = X_aug[idx], y_aug[idx]
    if sample_weights is not None:
        sample_weights = sample_weights[idx]

    split = int(0.9 * len(X_aug))
    sw_train = sample_weights[:split] if sample_weights is not None else None
    sw_val   = sample_weights[split:] if sample_weights is not None else None

    fit_kwargs = {}
    if sw_train is not None:
        fit_kwargs["sample_weight"] = sw_train
    # Note : XGBoost n'accepte pas sample_weight dans eval_set,
    # l'early stopping se fait sur la log-loss non pondérée (acceptable)

    new_xgb.fit(
        X_aug[:split], y_aug[:split],
        eval_set=[(X_aug[split:], y_aug[split:])],
        verbose=False,
        **fit_kwargs
    )

    new_xgb.save_model(str(fpath))
    print(f"    Sauvegardé : {fpath}")
    return XGBoostWrapper(new_xgb)


# ══════════════════════════════════════════════════════════════
# ÉVALUATION DÉFENSIVE AVEC C&W, LAMBDAS CORRIGÉES
# ══════════════════════════════════════════════════════════════

def evaluate_defended_models(defended_models, X_test, y_test, eps=0.3):
    """
    Évalue chaque modèle défendu sous FGSM, PGD, C&W à eps=0.3.

    Colonnes : F1 clean | ASR FGSM | ASR PGD | ASR C&W
    """
    print(f"\n{'═'*72}")
    print(f"  ÉVALUATION DÉFENSIVE — eps={eps}")
    print(f"{'═'*72}")

    def _is_mlp(w):
        return hasattr(w, 'model') and isinstance(getattr(w, 'model', None), MLP)

    def _is_logreg(w):
        return hasattr(w, 'model') and hasattr(getattr(w, 'model', None), 'coef_')

    def _get_attack_fn(wrapper, attack_name, eps_val, X_atk, y_atk):
        """Capture par valeur via paramètres par défaut des lambdas."""
        is_mlp    = _is_mlp(wrapper)
        is_logreg = _is_logreg(wrapper)

        if attack_name == "FGSM":
            if is_mlp:
                return lambda xatk=X_atk, yatk=y_atk: fgsm_mlp(wrapper, xatk, yatk, eps_val)
            elif is_logreg:
                return lambda xatk=X_atk, yatk=y_atk: fgsm_logreg(wrapper, xatk, yatk, eps_val)
            else:
                return lambda xatk=X_atk, yatk=y_atk: fgsm_xgb(wrapper, xatk, yatk, eps_val)

        elif attack_name == "PGD":
            if is_mlp:
                return lambda xatk=X_atk, yatk=y_atk: pgd_mlp(
                    wrapper, xatk, yatk, eps_val, iters=50, restarts=3)
            elif is_logreg:
                return lambda xatk=X_atk, yatk=y_atk: pgd_logreg(
                    wrapper, xatk, yatk, eps_val, iters=50, restarts=3)
            else:
                return lambda xatk=X_atk, yatk=y_atk: pgd_xgb(
                    wrapper, xatk, yatk, eps_val, iters=50, restarts=3)

        elif attack_name == "C&W":
            if is_mlp:
                return lambda xatk=X_atk, yatk=y_atk: cw_mlp(
                    wrapper, xatk, yatk, eps_val)
            elif is_logreg:
                return lambda xatk=X_atk, yatk=y_atk: cw_logreg(
                    wrapper, xatk, yatk, eps_val)
            else:
                return lambda xatk=X_atk, yatk=y_atk: cw_xgb(
                    wrapper, xatk, yatk, eps_val)

        raise ValueError(f"Attaque inconnue : {attack_name}")

    attacks = ["FGSM", "PGD", "C&W"]
    header  = (f"  {'Modèle':<30} {'F1 clean':>9} "
               f"{'FGSM':>9} {'PGD':>9} {'C&W':>9}")
    print(header)
    print(f"  {'─'*70}")

    results = {}

    for label, wrapper in defended_models.items():
        y_pred_clean = wrapper.predict(X_test)
        f1_clean     = f1_score(y_test, y_pred_clean, zero_division=0)

        tp_mask = (y_test == 1) & (y_pred_clean == 1)
        X_atk   = X_test[tp_mask].astype(np.float32)
        y_atk   = y_test[tp_mask]

        row      = f"  {label:<30} {f1_clean:>9.4f}"
        row_data = {"f1_clean": f1_clean}

        for att in attacks:
            if tp_mask.sum() == 0:
                row += f" {'N/A':>9}"
                row_data[att] = None
                continue

            fn    = _get_attack_fn(wrapper, att, eps, X_atk, y_atk)
            X_adv = fn()
            y_adv = wrapper.predict(X_adv)
            asr   = float((y_adv == 0).mean()) if len(y_adv) > 0 else 0.0

            row += f" {asr*100:>8.1f}%"
            row_data[att] = asr

        print(row)
        results[label] = row_data

    print(f"  {'─'*70}")
    return results


# ══════════════════════════════════════════════════════════════
# SAUVEGARDE JSON
# ══════════════════════════════════════════════════════════════

LABEL_MAP = {
    "MLP baseline":            ("MLP",     "Baseline"),
    "MLP AT-FGSM":             ("MLP",     "AT-FGSM"),
    "MLP AT-PGD10":            ("MLP",     "AT-PGD"),
    "LogReg baseline":         ("LogReg",  "Baseline"),
    "LogReg Aug-FGSM":         ("LogReg",  "Aug-FGSM"),
    "LogReg Aug-PGD":          ("LogReg",  "Aug-PGD"),
    "XGBoost baseline":        ("XGBoost", "Baseline"),
    "XGBoost Aug-proxy-FGSM":  ("XGBoost", "Aug-proxy"),
    "XGBoost Aug-direct-FGSM": ("XGBoost", "Aug-direct"),
}

EVAL_ATTACKS = ["FGSM", "PGD", "C&W"]


def _convert_defense_results(flat_results: dict) -> dict:
    out = {}
    for label, metrics in flat_results.items():
        mapping = LABEL_MAP.get(label, (None, None))
        model, defense = mapping
        if model is None:
            continue

        out.setdefault(model, {}).setdefault(defense, {})
        f1_clean = metrics.get("f1_clean")

        for att in EVAL_ATTACKS:
            asr = metrics.get(att)
            if asr is None:
                continue
            out[model][defense][att] = {
                "evasion_rate": round(asr * 100, 2),
                "f1":           round(f1_clean, 4) if f1_clean is not None else None,
                "recall":       round(1.0 - asr, 4),
                "delta_f1":     None,
            }
    return out


def _save_defense_results(flat_results: dict, results_dir):
    import json
    from pathlib import Path
    results_dir = Path(results_dir).expanduser()
    results_dir.mkdir(parents=True, exist_ok=True)

    hier = _convert_defense_results(flat_results)
    out_path = results_dir / "defense_results.json"
    with open(out_path, "w") as f:
        json.dump(hier, f, indent=2)
    print(f"\n  defense_results.json sauvegardé → {out_path}")

    flat_path = results_dir / "defense_whitebox_results.json"
    with open(flat_path, "w") as f:
        json.dump(flat_results, f, indent=2)
    print(f"  defense_whitebox_results.json sauvegardé → {flat_path}")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def run():
    print("\n" + "═"*60)
    print("  CHARGEMENT DES ARTIFACTS")
    print("═"*60)
    X_train, y_train, X_test, y_test, mlp_w, logreg_w, xgb_w = load_artifacts()
    input_size = X_train.shape[1]

    print(f"\n  [FIX B] EPS curriculum : {EPS_AT_START} → {EPS_AT_END} "
          f"(évaluation à 0.3, patience={AT_PATIENCE})")
    print(f"  [FIX A] NORMAL_AUG_RATIO={NORMAL_AUG_RATIO}, "
          f"ADV_ATTACK_RATIO={ADV_ATTACK_RATIO}")
    print(f"\n  ⚠  Supprimer les .pt/.pkl/.json existants si les modèles")
    print(f"     ont été entraînés avec l'ancienne version !")
    print(f"     rm ~/swat/artifacts/mlp_at_*.pt")
    print(f"     rm ~/swat/artifacts/logreg_aug_*.pkl")
    print(f"     rm ~/swat/artifacts/xgb_aug_*.json")

    print("\n" + "═"*60)
    print("  ENTRAÎNEMENT DES MODÈLES DÉFENDUS")
    print("═"*60)

    print("\n[1/6] Adversarial Training FGSM — MLP")
    mlp_at_fgsm = adversarial_train_mlp(
        X_train, y_train, X_test, y_test, input_size, attack="fgsm"
    )

    print("\n[2/6] Adversarial Training PGD-10 — MLP")
    mlp_at_pgd = adversarial_train_mlp(
        X_train, y_train, X_test, y_test, input_size, attack="pgd"
    )

    print("\n[3/6] Augmentation FGSM — LogReg")
    logreg_aug_fgsm = augment_logreg(
        logreg_w, X_train, y_train, X_test, y_test, attack="fgsm"
    )

    print("\n[4/6] Augmentation PGD — LogReg")
    logreg_aug_pgd = augment_logreg(
        logreg_w, X_train, y_train, X_test, y_test, attack="pgd"
    )

    print("\n[5/6] Augmentation XGBoost via proxy MLP AT-FGSM")
    xgb_aug_proxy = augment_xgb_proxy(
        xgb_w, mlp_at_fgsm, X_train, y_train, X_test, y_test, attack="fgsm"
    )

    print("\n[6/6] Augmentation XGBoost DIRECTE (gradient numérique FGSM)")
    xgb_aug_direct = augment_xgb_direct(
        xgb_w, X_train, y_train, X_test, y_test, attack="fgsm"
    )

    # ── Évaluation défensive ─────────────────────────────────
    defended_models = {
        "MLP baseline":           mlp_w,
        "MLP AT-FGSM":            mlp_at_fgsm,
        "MLP AT-PGD10":           mlp_at_pgd,
        "LogReg baseline":        logreg_w,
        "LogReg Aug-FGSM":        logreg_aug_fgsm,
        "LogReg Aug-PGD":         logreg_aug_pgd,
        "XGBoost baseline":       xgb_w,
        "XGBoost Aug-proxy-FGSM": xgb_aug_proxy,
        "XGBoost Aug-direct-FGSM":xgb_aug_direct,
    }

    results = evaluate_defended_models(defended_models, X_test, y_test, eps=0.3)

    import json
    RESULTS_DIR = Path("~/swat/results").expanduser()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _save_defense_results(results, RESULTS_DIR)

    print("\n" + "═"*60)
    print("  DONE — artéfacts sauvegardés dans ~/swat/artifacts/")
    print("═"*60)

    return (mlp_at_fgsm, mlp_at_pgd,
            logreg_aug_fgsm, logreg_aug_pgd,
            xgb_aug_proxy, xgb_aug_direct)


if __name__ == "__main__":
    run()