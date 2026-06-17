import numpy as np
from train_batadal import load_batadal, SAVE_DIR

X_train, y_train, X_test, y_test, input_size = load_batadal()

np.save(SAVE_DIR / "X_train.npy", X_train)
np.save(SAVE_DIR / "y_train.npy", y_train)
print(f"✅ X_train/y_train sauvegardés → {SAVE_DIR}")