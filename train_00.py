# ~/swat/00_train.py

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score
from xgboost import XGBClassifier
import joblib
import warnings
from pathlib import Path
from tqdm import tqdm
import json, os
from pathlib import Path

warnings.filterwarnings('ignore')

import sys
sys.path.append(str(Path(__file__).parent))
from models import MLP

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

DATA_PATH  = Path("~/swat/merged.csv").expanduser()
SAVE_DIR   = Path("~/swat/artifacts").expanduser()
SAVE_DIR.mkdir(parents=True, exist_ok=True)

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 2048
LR         = 0.001
MAX_EPOCHS = 50
PATIENCE   = 5
THRESHOLD  = 0.45

print(f"Device : {DEVICE}")


# ─────────────────────────────────────────────
# CHARGEMENT ET PREPROCESSING
# ─────────────────────────────────────────────

print("\n=== Chargement des données ===")
data = pd.read_csv(DATA_PATH)
print(f"Shape : {data.shape}")
print(f"Labels :\n{data['Normal/Attack'].value_counts()}")

y = data["Normal/Attack"].map({"Normal": 0, "Attack": 1})
X = data.drop([" Timestamp", "Normal/Attack"], axis=1)
X = X.fillna(X.mean())

print(f"\nFeatures : {X.shape[1]}")
print(f"NaN restants : {X.isna().sum().sum()}")
print(f"Normal : {(y==0).sum()} | Attack : {(y==1).sum()}")


# ─────────────────────────────────────────────
# SPLIT
# ─────────────────────────────────────────────

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test  = scaler.transform(X_test_raw)

y_train_np = y_train.values
y_test_np  = y_test.values

print(f"\nX_train : {X_train.shape} | X_test : {X_test.shape}")
print(f"y_train — Normal : {(y_train_np==0).sum()} | Attack : {(y_train_np==1).sum()}")
print(f"y_test  — Normal : {(y_test_np==0).sum()}  | Attack : {(y_test_np==1).sum()}")

np.save(SAVE_DIR / "X_train.npy", X_train)
np.save(SAVE_DIR / "X_test.npy",  X_test)
np.save(SAVE_DIR / "y_train.npy", y_train_np)
np.save(SAVE_DIR / "y_test.npy",  y_test_np)
joblib.dump(scaler, SAVE_DIR / "scaler.pkl")
print("Splits + scaler sauvegardés ✓")


# ─────────────────────────────────────────────
# MLP
# ─────────────────────────────────────────────

print("\n=== Entraînement MLP ===")

INPUT_SIZE = X_train.shape[1]
model      = MLP(INPUT_SIZE).to(DEVICE)
print(f"Paramètres : {sum(p.numel() for p in model.parameters()):,}")

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train_np, dtype=torch.float32).view(-1, 1)
X_test_t  = torch.tensor(X_test, dtype=torch.float32)

train_loader = DataLoader(
    TensorDataset(X_train_t, y_train_t),
    batch_size=BATCH_SIZE, shuffle=True
)

pos_weight = torch.tensor(
    [(y_train_np == 0).sum() / (y_train_np == 1).sum()],
    dtype=torch.float32
).to(DEVICE)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=LR)

best_f1    = 0
no_improve = 0

for epoch in tqdm(range(MAX_EPOCHS)):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        criterion(model(xb), yb).backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        proba  = torch.sigmoid(model(X_test_t.to(DEVICE))).cpu().numpy().flatten()
        y_pred = (proba >= THRESHOLD).astype(int)

    f1 = f1_score(y_test_np, y_pred, zero_division=0)

    if f1 > best_f1:
        best_f1    = f1
        no_improve = 0
        torch.save(model.state_dict(), SAVE_DIR / "best_mlp.pt")
    else:
        no_improve += 1
        if no_improve >= PATIENCE:
            print(f"Early stopping epoch {epoch+1}")
            break

model.load_state_dict(torch.load(SAVE_DIR / "best_mlp.pt"))
model.eval()
print(f"Best F1 MLP : {best_f1:.4f}")

with torch.no_grad():
    proba  = torch.sigmoid(model(X_test_t.to(DEVICE))).cpu().numpy().flatten()
    y_pred = (proba >= THRESHOLD).astype(int)

print("\n=== Résultats MLP ===")
print(classification_report(y_test_np, y_pred, target_names=["Normal", "Attack"]))


# ─────────────────────────────────────────────
# LOGISTIC REGRESSION
# ─────────────────────────────────────────────

print("\n=== Entraînement LogReg ===")

logreg = LogisticRegression(
    C=1.0,
    max_iter=1000,
    solver="saga",
    class_weight="balanced",
    random_state=42,
)
logreg.fit(X_train, y_train_np)

y_pred_lr = logreg.predict(X_test)
print("\n=== Résultats LogReg ===")
print(classification_report(y_test_np, y_pred_lr, target_names=["Normal", "Attack"]))

joblib.dump(logreg, SAVE_DIR / "logreg.pkl")
print("LogReg sauvegardé ✓")


# ─────────────────────────────────────────────
# XGBOOST
# ─────────────────────────────────────────────

print("\n=== Entraînement XGBoost ===")

scale_pw = float((y_train_np == 0).sum()) / float((y_train_np == 1).sum())
print(f"scale_pos_weight : {scale_pw:.1f}")

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train_np, test_size=0.1, stratify=y_train_np, random_state=42
)

xgb = XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pw,
    eval_metric="logloss",
    early_stopping_rounds=20,
    device="cuda" if torch.cuda.is_available() else "cpu",
    random_state=42,
    verbosity=0,
)

xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=50)

y_pred_xgb = xgb.predict(X_test)
print("\n=== Résultats XGBoost ===")
print(classification_report(y_test_np, y_pred_xgb, target_names=["Normal", "Attack"]))

xgb.save_model(str(SAVE_DIR / "xgb.json"))
print("XGBoost sauvegardé ✓")

print("\n=== Tous les artefacts sauvegardés dans", SAVE_DIR, "===")

# ─────────────────────────────────────────────
# SAUVEGARDE RÉSULTATS JSON (pour le dashboard)
# ─────────────────────────────────────────────

from sklearn.metrics import precision_score, recall_score

RESULTS_DIR = Path("~/swat/results").expanduser()
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# MLP — y_pred déjà calculé juste au-dessus
mlp_f1   = f1_score(y_test_np, y_pred, zero_division=0)
mlp_prec = precision_score(y_test_np, y_pred, zero_division=0)
mlp_rec  = recall_score(y_test_np, y_pred, zero_division=0)
mlp_acc  = float((y_pred == y_test_np).mean()) * 100

# LogReg — y_pred_lr déjà calculé
lr_f1   = f1_score(y_test_np, y_pred_lr, zero_division=0)
lr_prec = precision_score(y_test_np, y_pred_lr, zero_division=0)
lr_rec  = recall_score(y_test_np, y_pred_lr, zero_division=0)
lr_acc  = float((y_pred_lr == y_test_np).mean()) * 100

# XGBoost — y_pred_xgb déjà calculé
xgb_f1   = f1_score(y_test_np, y_pred_xgb, zero_division=0)
xgb_prec = precision_score(y_test_np, y_pred_xgb, zero_division=0)
xgb_rec  = recall_score(y_test_np, y_pred_xgb, zero_division=0)
xgb_acc  = float((y_pred_xgb == y_test_np).mean()) * 100

train_results = {
    "MLP": {
        "clean_accuracy": round(mlp_acc,  2),
        "f1":             round(mlp_f1,   4),
        "precision":      round(mlp_prec, 4),
        "recall":         round(mlp_rec,  4),
    },
    "LogReg": {
        "clean_accuracy": round(lr_acc,  2),
        "f1":             round(lr_f1,   4),
        "precision":      round(lr_prec, 4),
        "recall":         round(lr_rec,  4),
    },
    "XGBoost": {
        "clean_accuracy": round(xgb_acc,  2),
        "f1":             round(xgb_f1,   4),
        "precision":      round(xgb_prec, 4),
        "recall":         round(xgb_rec,  4),
    },
}

with open(RESULTS_DIR / "train_results.json", "w") as f:
    json.dump(train_results, f, indent=2)

print(f"\nJSON dashboard sauvegardé → {RESULTS_DIR / 'train_results.json'}")