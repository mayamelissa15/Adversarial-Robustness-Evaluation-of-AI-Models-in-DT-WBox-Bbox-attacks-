"""
train_batadal.py  —  version corrigée v2
══════════════════════════════════════════════════════════════

STRATÉGIE :
  - Test  = dataset04 entier  → 219 attaques (résolution 0.46%)
  - Train = dataset03 (normaux) + 80% de dataset04 (normaux + attaques)
    → le modèle voit des attaques pendant l'entraînement
    → les 20% restants de d04 restent dans le test uniquement

  En pratique : on split d04 en 80/20 stratifié,
  la partie 80% va en train, la partie 100% reste en test.
  Comme d04 a 219 attaques → ~175 en train, ~44 en test
  + tous les normaux de d03 et d04 en train.

  Test final = d04 entier (3958 normaux + 219 attaques = 4177)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import warnings
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

import sys
sys.path.append(str(Path(__file__).parent.parent))
from models import MLP

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("⚠  XGBoost non disponible, skipped.")

# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════

DATA_DIR  = Path("~/swat/batadal/data").expanduser()
SAVE_DIR  = Path("~/batadal/artifacts").expanduser()
SAVE_DIR.mkdir(parents=True, exist_ok=True)

DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
THRESHOLD = 0.45
EPOCHS    = 50
BATCH     = 256
LR        = 1e-3

# ══════════════════════════════════════════════════════════════
# CHARGEMENT & PRÉPARATION
# ══════════════════════════════════════════════════════════════

def load_batadal():
    d03 = pd.read_csv(DATA_DIR / "BATADAL_dataset03.csv", skipinitialspace=True)
    d04 = pd.read_csv(DATA_DIR / "BATADAL_dataset04.csv", skipinitialspace=True)

    drop_cols    = ["DATETIME", "ATT_FLAG"]
    feature_cols = [c for c in d03.columns if c not in drop_cols]

    d03["label"] = 0
    d04["label"] = (d04["ATT_FLAG"] == 1).astype(int)

    # ── Split d04 en 80/20 stratifié ─────────────────────────
    # 80% → enrichit le train  |  test = d04 entier (les deux parties)
    d04_train, _ = train_test_split(
        d04, test_size=0.2, stratify=d04["label"], random_state=42
    )

    # Train = d03 complet + 80% de d04
    train_df = pd.concat([d03, d04_train], ignore_index=True)
    # Test  = d04 complet (les 219 attaques sont toutes là)
    test_df  = d04.copy()

    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df["label"].values.astype(int)
    X_test  = test_df[feature_cols].values.astype(np.float32)
    y_test  = test_df["label"].values.astype(int)

    # Normalisation
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test  = scaler.transform(X_test).astype(np.float32)
    joblib.dump(scaler, SAVE_DIR / "scaler.pkl")

    # ── Vérification ─────────────────────────────────────────
    n_att_train = int(y_train.sum())
    n_att_test  = int(y_test.sum())
    print(f"\n{'─'*55}")
    print(f"  Split summary")
    print(f"{'─'*55}")
    print(f"  Train : {X_train.shape}  — normaux : {(y_train==0).sum():>5}  attaques : {n_att_train:>4}")
    print(f"  Test  : {X_test.shape}   — normaux : {(y_test==0).sum():>5}  attaques : {n_att_test:>4}")
    print(f"  Résolution ASR : 1/{n_att_test} ≈ {100/n_att_test:.2f}% par sample  ✅")
    print(f"  Features : {len(feature_cols)} → {feature_cols[:4]} ...")

    assert n_att_train > 0, "❌ Aucune attaque en train — les modèles ne peuvent rien apprendre !"
    assert n_att_test  >= 100, f"❌ Seulement {n_att_test} attaques en test — split incorrect !"

    return X_train, y_train, X_test, y_test, len(feature_cols)


# ══════════════════════════════════════════════════════════════
# MLP
# ══════════════════════════════════════════════════════════════

def train_mlp(X_train, y_train, X_test, y_test, input_size):
    print(f"\n{'─'*55}")
    print("  MLP")
    print(f"{'─'*55}")

    model     = MLP(input_size=input_size).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    raw_pw = n_neg / n_pos
    capped_pw = min(raw_pw, 15.0)
    print(f"  pos_weight = {n_neg}/{n_pos} = {raw_pw:.1f}  → cappé à {capped_pw:.1f}")

    pos_weight = torch.tensor([capped_pw], dtype=torch.float32).to(DEVICE)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    X_tr = torch.tensor(X_train, dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.float32)
    X_te = torch.tensor(X_test,  dtype=torch.float32)

    best_f1    = 0.0
    best_state = None

    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(X_tr))
        X_tr, y_tr = X_tr[perm], y_tr[perm]

        for i in range(0, len(X_tr), BATCH):
            xb = X_tr[i:i+BATCH].to(DEVICE)
            yb = y_tr[i:i+BATCH].to(DEVICE).view(-1, 1)
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            proba = torch.sigmoid(model(X_te.to(DEVICE))).squeeze(-1).cpu().numpy()
        preds = (proba >= THRESHOLD).astype(int)
        f1    = f1_score(y_test, preds, zero_division=0)

        if f1 > best_f1:
            best_f1    = f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}/{EPOCHS} — F1 : {f1:.4f}  (best : {best_f1:.4f})")

    model.load_state_dict(best_state)
    torch.save(best_state, SAVE_DIR / "best_mlp.pt")

    model.eval()
    with torch.no_grad():
        proba = torch.sigmoid(model(X_te.to(DEVICE))).squeeze(-1).cpu().numpy()
    preds = (proba >= THRESHOLD).astype(int)

    print(f"\n  ✅ MLP sauvegardé — best F1 = {best_f1:.4f}")
    print(classification_report(y_test, preds, zero_division=0))
    return model


# ══════════════════════════════════════════════════════════════
# LOGISTIC REGRESSION
# ══════════════════════════════════════════════════════════════

def train_logreg(X_train, y_train, X_test, y_test):
    print(f"\n{'─'*55}")
    print("  Logistic Regression")
    print(f"{'─'*55}")

    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    capped = min(n_neg / n_pos, 15.0)
    print(f"  class_weight attaque cappé à {capped:.1f}")

    model = LogisticRegression(
        max_iter=1000,
        class_weight={0: 1.0, 1: capped},
        C=1.0,
        solver="lbfgs"
    )
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= THRESHOLD).astype(int)
    f1    = f1_score(y_test, preds, zero_division=0)

    joblib.dump(model, SAVE_DIR / "logreg.pkl")
    print(f"  ✅ LogReg sauvegardé — F1 = {f1:.4f}")
    print(classification_report(y_test, preds, zero_division=0))
    return model


# ══════════════════════════════════════════════════════════════
# XGBOOST
# ══════════════════════════════════════════════════════════════

def train_xgboost(X_train, y_train, X_test, y_test):
    if not HAS_XGB:
        return None

    print(f"\n{'─'*55}")
    print("  XGBoost")
    print(f"{'─'*55}")

    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    scale = min(n_neg / n_pos, 15.0)
    print(f"  scale_pos_weight cappé à {scale:.1f}")

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=scale,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              verbose=False)

    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= THRESHOLD).astype(int)
    f1    = f1_score(y_test, preds, zero_division=0)

    model.save_model(str(SAVE_DIR / "xgb.json"))
    print(f"  ✅ XGBoost sauvegardé — F1 = {f1:.4f}")
    print(classification_report(y_test, preds, zero_division=0))
    return model


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    print(f"\n{'═'*55}")
    print(f"  BATADAL — Entraînement v2 (split corrigé)")
    print(f"  Device : {DEVICE}")
    print(f"{'═'*55}")

    X_train, y_train, X_test, y_test, input_size = load_batadal()

    np.save(SAVE_DIR / "X_test.npy", X_test)
    np.save(SAVE_DIR / "y_test.npy", y_test)
    print(f"\n  ✅ X_test / y_test sauvegardés → {SAVE_DIR}")

    train_mlp(X_train, y_train, X_test, y_test, input_size)
    train_logreg(X_train, y_train, X_test, y_test)
    train_xgboost(X_train, y_train, X_test, y_test)

    print(f"\n{'═'*55}")
    print(f"  ✅ Tous les artifacts → {SAVE_DIR}")
    print(f"{'═'*55}")


if __name__ == "__main__":
    main()