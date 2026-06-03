# ~/swat/02_blackbox.py

import numpy as np
import torch
import joblib
import pandas as pd
import warnings
from pathlib import Path
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

import sys
sys.path.append(str(Path(__file__).parent))
from models import MLP, MLPWrapper, LogRegWrapper, XGBoostWrapper, build_eval_set, eval_attack

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

SAVE_DIR  = Path("~/swat/artifacts").expanduser()
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
THRESHOLD = 0.45
EPS_LIST  = [0.1, 0.3, 0.5]

MAX_DECISION_BOUNDARY = 300

print(f"Device : {DEVICE}")


# ─────────────────────────────────────────────
# CHARGEMENT
# ─────────────────────────────────────────────

def load_artifacts():
    X_test    = np.load(SAVE_DIR / "X_test.npy")
    y_test    = np.load(SAVE_DIR / "y_test.npy")

    mlp_model = MLP(input_size=X_test.shape[1]).to(DEVICE)
    mlp_model.load_state_dict(torch.load(SAVE_DIR / "best_mlp.pt", map_location=DEVICE))
    mlp_model.eval()
    mlp_w = MLPWrapper(mlp_model, DEVICE)

    logreg   = joblib.load(SAVE_DIR / "logreg.pkl")
    logreg_w = LogRegWrapper(logreg)

    xgb_model = XGBClassifier()
    xgb_model.load_model(str(SAVE_DIR / "xgb.json"))
    xgb_w = XGBoostWrapper(xgb_model)

    return X_test, y_test, mlp_w, logreg_w, xgb_w


# ─────────────────────────────────────────────
# SQUARE ATTACK
# FIX : schedule racine (moins agressif), early-stop par exemple,
#       freeze des exemples déjà évasifs
# ─────────────────────────────────────────────

def square_attack(wrapper, X_np, y_np, eps, max_queries=2000, p_init=0.3):
    N, D   = X_np.shape
    X_orig = X_np.copy()
    X_adv  = X_np.copy()

    def bce_loss(X_batch, y_batch):
        probs = wrapper.predict_proba(X_batch).flatten()
        probs = np.clip(probs, 1e-7, 1 - 1e-7)
        return -(y_batch * np.log(probs) + (1 - y_batch) * np.log(1 - probs))

    # FIX : schedule racine au lieu de quadratique → perturbations plus grandes plus longtemps
    def p_schedule(q):
        return max(p_init * (1 - q / max_queries) ** 0.5, 0.05)

    curr_loss = bce_loss(X_adv, y_np)

    for q in range(max_queries):
        # FIX : freeze les exemples déjà évasifs
        evaded = (wrapper.predict(X_adv) != y_np)
        if evaded.all():
            print(f"    Square : tous évasifs à la requête {q}, arrêt anticipé")
            break

        p           = p_schedule(q)
        square_size = max(int(p * D), 1)
        X_cand      = X_adv.copy()

        for i in range(N):
            if evaded[i]:
                continue  # FIX : ne perturbe plus les exemples déjà évasifs
            idx   = np.random.choice(D, square_size, replace=False)
            delta = np.random.choice([-eps, eps], size=square_size)
            X_cand[i, idx] = np.clip(
                X_adv[i, idx] + delta,
                X_orig[i, idx] - eps,
                X_orig[i, idx] + eps
            )

        cand_loss = bce_loss(X_cand, y_np)

        # FIX : n'accepte l'update que pour les exemples non encore évasifs
        improved = (cand_loss > curr_loss) & ~evaded
        X_adv[improved]     = X_cand[improved]
        curr_loss[improved] = cand_loss[improved]

    return X_adv


# ─────────────────────────────────────────────
# NES ATTACK
# FIX : freeze des exemples évasifs, plus de samples et d'itérations
# ─────────────────────────────────────────────

def nes_attack(wrapper, X_np, y_np, eps,
               sigma=0.01, lr=0.01, n_samples=50, iters=100):
    N, D   = X_np.shape
    X_orig = X_np.copy()
    X_adv  = X_np.copy()

    def neg_bce(X_batch, y_batch):
        probs = wrapper.predict_proba(X_batch).flatten()
        probs = np.clip(probs, 1e-7, 1 - 1e-7)
        return -(y_batch * np.log(probs) + (1 - y_batch) * np.log(1 - probs))

    for it in range(iters):
        # FIX : calcule le masque des exemples pas encore évasifs
        evaded = (wrapper.predict(X_adv) != y_np)
        if evaded.all():
            print(f"    NES : tous évasifs à l'itération {it}, arrêt anticipé")
            break

        active = ~evaded
        grad_est = np.zeros_like(X_adv)

        for _ in range(n_samples // 2):
            noise    = np.random.randn(N, D)
            X_pos    = np.clip(X_adv + sigma * noise,  X_orig - eps, X_orig + eps)
            X_neg    = np.clip(X_adv - sigma * noise,  X_orig - eps, X_orig + eps)
            loss_pos = neg_bce(X_pos, y_np)
            loss_neg = neg_bce(X_neg, y_np)
            grad_est += ((loss_pos - loss_neg)[:, None] * noise) / (2 * sigma)

        grad_est /= (n_samples // 2)

        # FIX : n'applique la mise à jour qu'aux exemples encore actifs
        X_adv[active] = X_adv[active] + lr * np.sign(grad_est[active])
        X_adv = np.clip(X_adv, X_orig - eps, X_orig + eps)

    return X_adv


# ─────────────────────────────────────────────
# HSJA
# FIX : direction de mise à jour corrigée (+ au lieu de -),
#       vérification que le résultat de binary_search est bien adverse
# ─────────────────────────────────────────────

def hsja(wrapper, X_np, y_np, eps, iters=20, n_est=30, stepsize_init=0.1):
    N, D   = X_np.shape
    X_orig = X_np.copy()
    y_flat = y_np.flatten().astype(int)
    X_adv  = X_np.copy()

    def is_adverse(X_batch, y_batch):
        return wrapper.predict(X_batch).flatten() != y_batch

    def binary_search(x_orig, x_adv, y_i, n_steps=15):
        """Trouve le point sur la frontière entre x_orig (non-adverse) et x_adv (adverse)."""
        lo, hi = x_orig.copy(), x_adv.copy()
        for _ in range(n_steps):
            mid  = (lo + hi) / 2
            pred = wrapper.predict(mid[None])[0]
            if pred != y_i:
                hi = mid   # mid est adverse → on peut rapprocher encore
            else:
                lo = mid   # mid est clean → on recule
        # hi est le point adverse le plus proche de x_orig
        return hi

    print(f"    HSJA init : recherche point adverse pour {N} exemples...")
    n_init_ok = 0
    for i in range(N):
        for _ in range(200):
            noise     = np.random.uniform(-eps, eps, D)
            candidate = np.clip(X_orig[i] + noise, X_orig[i] - eps, X_orig[i] + eps)
            if wrapper.predict(candidate[None])[0] != y_flat[i]:
                # FIX : binary search dès l'init pour partir d'un point propre
                X_adv[i] = binary_search(X_orig[i], candidate, y_flat[i])
                n_init_ok += 1
                break
    print(f"    HSJA init : {n_init_ok}/{N} exemples initialisés avec succès")

    for it in range(iters):
        stepsize = stepsize_init / np.sqrt(it + 1)
        n_evaded = sum(
            wrapper.predict(X_adv[i][None])[0] != y_flat[i]
            for i in range(N)
        )
        print(f"    HSJA iter {it+1:03d}/{iters} — "
              f"évadés : {n_evaded}/{N} ({100*n_evaded/N:.1f}%) | "
              f"stepsize={stepsize:.5f}")

        for i in range(N):
            if not is_adverse(X_adv[i][None], y_flat[i:i+1])[0]:
                continue

            grads = np.zeros(D)
            for _ in range(n_est):
                u    = np.random.randn(D)
                u   /= np.linalg.norm(u) + 1e-12
                x_q  = np.clip(X_adv[i] + 0.01 * u, X_orig[i] - eps, X_orig[i] + eps)
                pred = wrapper.predict(x_q[None])[0]
                sign = 1 if pred != y_flat[i] else -1
                grads += sign * u

            grads /= (n_est + 1e-12)

            # FIX : + au lieu de - → on avance vers la région adverse
            x_new = X_adv[i] + stepsize * np.sign(grads)
            x_new = np.clip(x_new, X_orig[i] - eps, X_orig[i] + eps)

            if wrapper.predict(x_new[None])[0] != y_flat[i]:
                # x_new est adverse → binary search pour minimiser la perturbation
                candidate = binary_search(X_orig[i], x_new, y_flat[i])
                # FIX : vérifie que le résultat est bien adverse avant d'accepter
                if wrapper.predict(candidate[None])[0] != y_flat[i]:
                    # garde le meilleur des deux (perturbation L-inf minimale)
                    if np.abs(candidate - X_orig[i]).max() < np.abs(X_adv[i] - X_orig[i]).max():
                        X_adv[i] = candidate

    return X_adv


# ─────────────────────────────────────────────
# RAYS
# FIX : mémorise la meilleure direction (exploration par mutation),
#       au lieu de retirer une direction aléatoire à chaque itération
# ─────────────────────────────────────────────

def rays(wrapper, X_np, y_np, eps, iters=30, search_steps=10):
    N, D   = X_np.shape
    X_orig = X_np.copy()
    y_flat = y_np.flatten().astype(int)
    X_adv  = X_np.copy()

    # FIX : initialise et mémorise une direction par exemple
    best_dirs = np.random.choice([-1, 1], size=(N, D)).astype(float)

    print(f"    RayS init : recherche point adverse pour {N} exemples...")
    n_init_ok = 0
    for i in range(N):
        for _ in range(200):
            candidate = np.clip(X_orig[i] + eps * best_dirs[i],
                                X_orig[i] - eps, X_orig[i] + eps)
            if wrapper.predict(candidate[None])[0] != y_flat[i]:
                X_adv[i] = candidate
                n_init_ok += 1
                break
            # si ça rate, retente avec une direction aléatoire
            best_dirs[i] = np.random.choice([-1, 1], size=D).astype(float)
    print(f"    RayS init : {n_init_ok}/{N} exemples initialisés avec succès")

    def binary_search_amplitude(x_orig, direction, y_i, n_steps=15):
        """Cherche l'amplitude minimale dans la direction donnée."""
        lo, hi = 0.0, 1.0
        best   = x_orig + hi * eps * direction
        for _ in range(n_steps):
            mid   = (lo + hi) / 2
            x_try = np.clip(x_orig + mid * eps * direction,
                            x_orig - eps, x_orig + eps)
            if wrapper.predict(x_try[None])[0] != y_i:
                best = x_try.copy()
                hi   = mid
            else:
                lo   = mid
        return best

    for it in range(iters):
        n_evaded = sum(
            wrapper.predict(X_adv[i][None])[0] != y_flat[i]
            for i in range(N)
        )
        print(f"    RayS iter {it+1:03d}/{iters} — "
              f"évadés : {n_evaded}/{N} ({100*n_evaded/N:.1f}%)")

        for i in range(N):
            if wrapper.predict(X_adv[i][None])[0] == y_flat[i]:
                continue

            # FIX : mutation d'un seul bit de la direction courante
            j           = np.random.randint(D)
            new_dir     = best_dirs[i].copy()
            new_dir[j] *= -1

            x_try = np.clip(X_orig[i] + eps * new_dir,
                            X_orig[i] - eps, X_orig[i] + eps)

            if wrapper.predict(x_try[None])[0] != y_flat[i]:
                # la nouvelle direction est adverse → binary search pour réduire l'amplitude
                candidate = binary_search_amplitude(X_orig[i], new_dir, y_flat[i], search_steps)
                curr_dist = np.abs(X_adv[i]   - X_orig[i]).max()
                cand_dist = np.abs(candidate   - X_orig[i]).max()
                if cand_dist < curr_dist:
                    # FIX : mémorise la meilleure direction et le meilleur point
                    best_dirs[i] = new_dir
                    X_adv[i]     = candidate
            else:
                # la mutation rate → essaie directement de réduire l'amplitude sur la direction actuelle
                candidate = binary_search_amplitude(X_orig[i], best_dirs[i], y_flat[i], search_steps)
                curr_dist = np.abs(X_adv[i]  - X_orig[i]).max()
                cand_dist = np.abs(candidate  - X_orig[i]).max()
                if cand_dist < curr_dist:
                    X_adv[i] = candidate

    return X_adv


# ─────────────────────────────────────────────
# AFFICHAGE PROPRE
# ─────────────────────────────────────────────

def print_results(df):
    models  = list(df["model"].unique())
    attacks = sorted(df["attack"].str.split("_eps").str[0].unique())

    col_w   = 28
    label_w = 12

    header = f"{'Attaque':<{label_w}}" + "".join(f"  {m:^{col_w}}" for m in models)
    subhdr = " " * label_w + "".join(
        f"  {'ASR':>6} {'F1adv':>6} {'Rec':>5} {'Linf':>6}  ".center(col_w + 2)
        for _ in models
    )
    sep = "═" * len(header)

    print(f"\n{sep}")
    print("RÉSUMÉ BLACKBOX")
    print(sep)
    print(header)
    print(subhdr)

    for eps in sorted(df["attack"].str.extract(r"eps(\d+\.\d+)")[0].astype(float).unique()):
        print(f"  {'─'*4}  ε = {eps}  {'─'*4}")
        for atk in attacks:
            attack_key = f"{atk}_eps{eps}"
            row_str    = f"  {atk:<{label_w - 2}}"
            for m in models:
                sub = df[(df["model"] == m) & (df["attack"] == attack_key)]
                if sub.empty:
                    row_str += "  " + " " * col_w
                else:
                    r = sub.iloc[0]
                    row_str += (
                        f"  {r['asr']:>6.1%} {r['f1_adv']:>6.4f} "
                        f"{r['rec_adv']:>5.3f} {r['linf']:>6.4f}  "
                    )
            print(row_str)
        print()

    print(sep)


# ─────────────────────────────────────────────
# BOUCLE PRINCIPALE
# ─────────────────────────────────────────────

def run():
    X_test, y_test, mlp_w, logreg_w, xgb_w = load_artifacts()

    print("\n=== Build eval sets ===")
    eval_sets = {
        "MLP":     (build_eval_set(X_test, y_test, mlp_w),    mlp_w),
        "LogReg":  (build_eval_set(X_test, y_test, logreg_w), logreg_w),
        "XGBoost": (build_eval_set(X_test, y_test, xgb_w),    xgb_w),
    }

    results = []

    for eps in EPS_LIST:
        print(f"\n{'═'*60}")
        print(f"eps = {eps}")
        print(f"{'═'*60}")

        for model_name, ((X_ev, y_ev), victim_w) in eval_sets.items():
            mask  = (y_ev == 1)
            X_atk = X_ev[mask].astype(np.float32)
            y_atk = y_ev[mask]

            if len(X_atk) > MAX_DECISION_BOUNDARY:
                idx_bb   = np.random.choice(len(X_atk), MAX_DECISION_BOUNDARY, replace=False)
                X_atk_bb = X_atk[idx_bb]
                y_atk_bb = y_atk[idx_bb]
                print(f"\n  [info] HSJA/RayS : sous-échantillonnage "
                      f"{len(X_atk)} → {MAX_DECISION_BOUNDARY} exemples")
            else:
                X_atk_bb, y_atk_bb = X_atk, y_atk
                idx_bb = None

            print(f"\n--- {model_name} | X_atk={X_atk.shape} | X_atk_bb={X_atk_bb.shape} ---")

            # ── Square
            print(f"  [Square]")
            X_adv = square_attack(victim_w, X_atk, y_atk, eps)
            results.append(eval_attack(victim_w, X_ev, y_ev, X_adv,
                                       f"Square_eps{eps}", model_name))
            np.save(SAVE_DIR / f"adv_square_{model_name}_eps{eps}.npy", X_adv)

            # ── NES
            print(f"  [NES]")
            X_adv = nes_attack(victim_w, X_atk, y_atk, eps)
            results.append(eval_attack(victim_w, X_ev, y_ev, X_adv,
                                       f"NES_eps{eps}", model_name))
            np.save(SAVE_DIR / f"adv_nes_{model_name}_eps{eps}.npy", X_adv)

            # ── HSJA
            print(f"  [HSJA] N={len(X_atk_bb)} iters=40 n_est=100 "
                  f"→ ~{len(X_atk_bb)*40*100} appels predict")
            X_adv_bb = hsja(victim_w, X_atk_bb, y_atk_bb, eps)
            X_adv_full = X_atk.copy()
            if idx_bb is not None:
                X_adv_full[idx_bb] = X_adv_bb
            else:
                X_adv_full = X_adv_bb
            results.append(eval_attack(victim_w, X_ev, y_ev, X_adv_full,
                                       f"HSJA_eps{eps}", model_name))
            np.save(SAVE_DIR / f"adv_hsja_{model_name}_eps{eps}.npy", X_adv_full)

            # ── RayS
            print(f"  [RayS] N={len(X_atk_bb)} iters=60 search_steps=15 "
                  f"→ ~{len(X_atk_bb)*60*15} appels predict")
            X_adv_bb = rays(victim_w, X_atk_bb, y_atk_bb, eps)
            X_adv_full = X_atk.copy()
            if idx_bb is not None:
                X_adv_full[idx_bb] = X_adv_bb
            else:
                X_adv_full = X_adv_bb
            results.append(eval_attack(victim_w, X_ev, y_ev, X_adv_full,
                                       f"RayS_eps{eps}", model_name))
            np.save(SAVE_DIR / f"adv_rays_{model_name}_eps{eps}.npy", X_adv_full)

    df = pd.DataFrame(results)
    df.to_csv(SAVE_DIR / "blackbox_results.csv", index=False)

    print_results(df)

    return df


if __name__ == "__main__":
    df = run()

    import json

    RESULTS_DIR = Path("~/swat/results").expanduser()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    out = {}
    for model_name in df["model"].unique():
        out[model_name] = {}
        sub = df[df["model"] == model_name]
        for _, row in sub.iterrows():
            attack_clean = row["attack"].split("_eps")[0]
            if attack_clean not in out[model_name] or \
               row["asr"] > out[model_name][attack_clean]["evasion_rate"] / 100:
                out[model_name][attack_clean] = {
                    "evasion_rate": round(row["asr"] * 100, 2),
                    "precision":    round(row.get("prec_adv", 0), 4),
                    "recall":       round(row["rec_adv"], 4),
                    "f1":           round(row["f1_adv"], 4),
                    "n_queries":    int(row.get("n_queries", 0)),
                }

    with open(RESULTS_DIR / "blackbox_results.json", "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nJSON sauvegardé → {RESULTS_DIR / 'blackbox_results.json'}")