# 00_extract_timestamps.py
"""
Extraction des timestamps alignés avec X_test.npy / X_train.npy ( JP remarque )


  1. Recharge merged.csv
  2. Refait EXACTEMENT le même split que 00_train.py (même random_state,
     mêmes colonnes droppées)
  3. Vérifie que ce split recalculé donne bien le même y_test que celui
     déjà sauvegardé dans artifacts/ (sécurité anti-désalignement)
  4. Sauvegarde timestamps_train.npy et timestamps_test.npy, alignés
     POSITION PAR POSITION avec X_train.npy/y_train.npy et X_test.npy/y_test.npy

Usage :
  python 00_extract_timestamps.py --dataset swat

"""

import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

# ══════════════════════════════════════════════════════════════
# CONFIG PAR DATASET
# ══════════════════════════════════════════════════════════════
# Pour ajouter batadal plus tard : juste compléter cette entrée
# avec le bon nom de colonne timestamp / label / mapping.

DATASET_CONFIG = {
    "swat": {
        "data_path":     "~/swat/merged.csv",
        "save_dir":      "~/swat/artifacts",
        "timestamp_col": " Timestamp",
        "label_col":     "Normal/Attack",
        "label_map":     {"Normal": 0, "Attack": 1},
        "test_size":     0.2,
        "random_state":  42,
    },
    # "batadal": {...}  # à compléter quand on s'y attaque
}

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="swat", choices=list(DATASET_CONFIG.keys()),
                     help="Dataset cible")
args = parser.parse_args()

cfg = DATASET_CONFIG[args.dataset]

DATA_PATH = Path(cfg["data_path"]).expanduser()
SAVE_DIR  = Path(cfg["save_dir"]).expanduser()
SAVE_DIR.mkdir(parents=True, exist_ok=True)

print(f"\n{'═'*55}")
print(f"  Extraction timestamps — {args.dataset.upper()}")
print(f"{'═'*55}")
print(f"  Source      : {DATA_PATH}")
print(f"  Artifacts   : {SAVE_DIR}")

# ══════════════════════════════════════════════════════════════
# RECHARGEMENT + SPLIT IDENTIQUE À 00_train.py
# ══════════════════════════════════════════════════════════════

data = pd.read_csv(DATA_PATH)

y = data[cfg["label_col"]].map(cfg["label_map"])
X = data.drop([cfg["timestamp_col"], cfg["label_col"]], axis=1)
X = X.fillna(X.mean())

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y,
    test_size=cfg["test_size"],
    random_state=cfg["random_state"],
)

# ══════════════════════════════════════════════════════════════
# VÉRIFICATION D'ALIGNEMENT (sécurité avant de sauvegarder quoi que ce soit)
# ══════════════════════════════════════════════════════════════

y_test_saved = np.load(SAVE_DIR / "y_test.npy")

assert len(y_test_saved) == len(y_test), (
    f"Tailles différentes : y_test.npy existant = {len(y_test_saved)}, "
    f"split recalculé = {len(y_test)}. "
    "Vérifie test_size / random_state / colonnes droppées dans DATASET_CONFIG."
)

assert np.array_equal(y_test_saved, y_test.values), (
    "Le split recalculé ne correspond PAS à y_test.npy existant ! "
    "Les timestamps seraient mal alignés — on s'arrête avant de sauvegarder n'importe quoi. "
    "Vérifie que 00_train.py n'a pas changé depuis (colonnes, random_state, version sklearn...)."
)

print("\n✓ Split recalculé identique à y_test.npy existant — alignement garanti.")

# ══════════════════════════════════════════════════════════════
# EXTRACTION DES TIMESTAMPS VIA LES INDEX PANDAS CONSERVÉS
# ══════════════════════════════════════════════════════════════
# X_train_raw / X_test_raw sont des slices pandas issues du split :
# elles gardent l'index original du CSV, ce qui permet de retrouver
# le timestamp de chaque ligne sans recalculer quoi que ce soit d'autre.

timestamps_train = data.loc[X_train_raw.index, cfg["timestamp_col"]].values
timestamps_test  = data.loc[X_test_raw.index,  cfg["timestamp_col"]].values

np.save(SAVE_DIR / "timestamps_train.npy", timestamps_train)
np.save(SAVE_DIR / "timestamps_test.npy",  timestamps_test)

print(f"\n✓ timestamps_train.npy  ({len(timestamps_train)} lignes) → {SAVE_DIR}")
print(f"✓ timestamps_test.npy   ({len(timestamps_test)} lignes)  → {SAVE_DIR}")
print(f"\nAperçu timestamps_test[:5] :\n{timestamps_test[:5]}")
print(f"\n{'═'*55}")
print("  Terminé — rien d'autre n'a été modifié.")
print(f"{'═'*55}\n")