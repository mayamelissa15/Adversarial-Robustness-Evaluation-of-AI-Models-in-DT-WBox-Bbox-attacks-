"""
blackbox.py

Square, NES, HSJA, RayS — utilisés par blackbox_multirun.py.

FIX MAJEUR (par rapport à 02_blackbox.py) : HSJA et RayS appelaient
wrapper.predict() sur UN SEUL exemple à la fois, à l'intérieur de boucles
imbriquées (for i in range(N): for _ in range(n_est): predict(x_i)).
Résultat : ~1 200 000 appels predict() individuels pour un seul (seed, modèle)
en HSJA (N=300, n_est=100, iters=40).

Ici, tous les appels predict() sont BATCHÉS : on construit un tableau
(M, D) regroupant tous les exemples encore actifs à une étape donnée, et on
appelle predict() UNE FOIS dessus. Le nombre d'appels au modèle tombe à
quelques milliers (batchés), pour la même quantité de "travail" au sens
algorithmique.

⚠ Cette version n'est PAS reproductible bit-à-bit par rapport à l'ancienne :
l'ordre des tirages aléatoires change (on tire le bruit pour tous les
exemples actifs d'un coup au lieu d'un par un). La méthode (mêmes formules,
mêmes critères d'acceptation/de binary search) est strictement la même —
seul l'ordre d'exécution change. Attends-toi à des ASR très proches de tes
anciens runs, pas garantis identiques à la décimale près.
"""

import numpy as np


# ══════════════════════════════════════════════════════════════
# SQUARE ATTACK — déjà batché, inchangé
# ══════════════════════════════════════════════════════════════

def square_attack(wrapper, X_np, y_np, eps, max_queries=2000, p_init=0.3):
    N, D   = X_np.shape
    X_orig = X_np.copy()
    X_adv  = X_np.copy()

    def bce_loss(X_batch, y_batch):
        probs = wrapper.predict_proba(X_batch).flatten()
        probs = np.clip(probs, 1e-7, 1 - 1e-7)
        return -(y_batch * np.log(probs) + (1 - y_batch) * np.log(1 - probs))

    def p_schedule(q):
        return max(p_init * (1 - q / max_queries) ** 0.5, 0.05)

    curr_loss = bce_loss(X_adv, y_np)

    for q in range(max_queries):
        evaded = (wrapper.predict(X_adv) != y_np)
        if evaded.all():
            print(f"    Square : tous évasifs à la requête {q}, arrêt anticipé")
            break

        p           = p_schedule(q)
        square_size = max(int(p * D), 1)
        X_cand      = X_adv.copy()

        for i in range(N):
            if evaded[i]:
                continue
            idx   = np.random.choice(D, square_size, replace=False)
            delta = np.random.choice([-eps, eps], size=square_size)
            X_cand[i, idx] = np.clip(
                X_adv[i, idx] + delta,
                X_orig[i, idx] - eps,
                X_orig[i, idx] + eps
            )

        cand_loss = bce_loss(X_cand, y_np)
        improved  = (cand_loss > curr_loss) & ~evaded
        X_adv[improved]     = X_cand[improved]
        curr_loss[improved] = cand_loss[improved]

    return X_adv


# ══════════════════════════════════════════════════════════════
# NES ATTACK — déjà batché, inchangé
# ══════════════════════════════════════════════════════════════

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
        evaded = (wrapper.predict(X_adv) != y_np)
        if evaded.all():
            print(f"    NES : tous évasifs à l'itération {it}, arrêt anticipé")
            break

        active   = ~evaded
        grad_est = np.zeros_like(X_adv)

        for _ in range(n_samples // 2):
            noise    = np.random.randn(N, D)
            X_pos    = np.clip(X_adv + sigma * noise,  X_orig - eps, X_orig + eps)
            X_neg    = np.clip(X_adv - sigma * noise,  X_orig - eps, X_orig + eps)
            loss_pos = neg_bce(X_pos, y_np)
            loss_neg = neg_bce(X_neg, y_np)
            grad_est += ((loss_pos - loss_neg)[:, None] * noise) / (2 * sigma)

        grad_est /= (n_samples // 2)

        X_adv[active] = X_adv[active] + lr * np.sign(grad_est[active])
        X_adv = np.clip(X_adv, X_orig - eps, X_orig + eps)

    return X_adv


# ══════════════════════════════════════════════════════════════
# HSJA — BATCHÉ (fix majeur de perf)
# ══════════════════════════════════════════════════════════════

def hsja(wrapper, X_np, y_np, eps, iters=20, n_est=30, stepsize_init=0.1,
         n_init_tries=200, n_steps_bsearch=15):
    N, D   = X_np.shape
    X_orig = X_np.copy()
    y_flat = y_np.flatten().astype(int)
    X_adv  = X_np.copy()

    def predict_batch(Xb):
        return wrapper.predict(Xb).flatten()

    def binary_search_batch(x_orig_b, x_adv_b, y_b, n_steps=n_steps_bsearch):
        """Version batchée : (M, D) -> (M, D). Même logique de bissection
        que l'original, appliquée en parallèle sur M exemples à la fois."""
        lo = x_orig_b.copy()
        hi = x_adv_b.copy()
        for _ in range(n_steps):
            mid    = (lo + hi) / 2
            pred   = predict_batch(mid)
            is_adv = (pred != y_b)
            hi = np.where(is_adv[:, None], mid, hi)
            lo = np.where(is_adv[:, None], lo, mid)
        return hi

    # ── Initialisation batchée ─────────────────────────────────
    print(f"    HSJA init : recherche point adverse pour {N} exemples...")
    initialized = np.zeros(N, dtype=bool)
    for _ in range(n_init_tries):
        pending = ~initialized
        if not pending.any():
            break
        idx        = np.where(pending)[0]
        noise      = np.random.uniform(-eps, eps, (len(idx), D))
        candidates = np.clip(X_orig[idx] + noise, X_orig[idx] - eps, X_orig[idx] + eps)
        preds      = predict_batch(candidates)
        success    = preds != y_flat[idx]

        succ_idx = idx[success]
        if len(succ_idx) > 0:
            refined = binary_search_batch(X_orig[succ_idx], candidates[success], y_flat[succ_idx])
            X_adv[succ_idx] = refined
            initialized[succ_idx] = True

    n_init_ok = int(initialized.sum())
    print(f"    HSJA init : {n_init_ok}/{N} exemples initialisés avec succès")

    # ── Boucle principale batchée ──────────────────────────────
    for it in range(iters):
        stepsize = stepsize_init / np.sqrt(it + 1)

        preds_now    = predict_batch(X_adv)
        adverse_mask = (preds_now != y_flat)
        n_evaded     = int(adverse_mask.sum())
        print(f"    HSJA iter {it+1:03d}/{iters} — "
              f"évadés : {n_evaded}/{N} ({100*n_evaded/N:.1f}%) | "
              f"stepsize={stepsize:.5f}")

        active_idx = np.where(adverse_mask)[0]
        if len(active_idx) == 0:
            continue

        Xa = X_adv[active_idx]
        Xo = X_orig[active_idx]
        ya = y_flat[active_idx]
        M  = len(active_idx)

        # Estimation du gradient par échantillonnage aléatoire, batchée
        grads = np.zeros((M, D))
        for _ in range(n_est):
            u  = np.random.randn(M, D)
            u /= (np.linalg.norm(u, axis=1, keepdims=True) + 1e-12)
            x_q   = np.clip(Xa + 0.01 * u, Xo - eps, Xo + eps)
            preds = predict_batch(x_q)
            sign  = np.where(preds != ya, 1.0, -1.0)
            grads += sign[:, None] * u
        grads /= (n_est + 1e-12)

        x_new = np.clip(Xa + stepsize * np.sign(grads), Xo - eps, Xo + eps)

        preds_new = predict_batch(x_new)
        still_adv = preds_new != ya

        if still_adv.any():
            sub        = np.where(still_adv)[0]
            candidates = binary_search_batch(Xo[sub], x_new[sub], ya[sub])
            preds_cand = predict_batch(candidates)
            valid      = preds_cand != ya[sub]

            curr_dist = np.abs(Xa[sub] - Xo[sub]).max(axis=1)
            cand_dist = np.abs(candidates - Xo[sub]).max(axis=1)
            improve   = valid & (cand_dist < curr_dist)

            gidx = active_idx[sub[improve]]
            X_adv[gidx] = candidates[improve]

    return X_adv


# ══════════════════════════════════════════════════════════════
# RAYS — BATCHÉ (fix majeur de perf)
# ══════════════════════════════════════════════════════════════

def rays(wrapper, X_np, y_np, eps, iters=30, search_steps=10, n_init_tries=200):
    N, D   = X_np.shape
    X_orig = X_np.copy()
    y_flat = y_np.flatten().astype(int)
    X_adv  = X_np.copy()

    def predict_batch(Xb):
        return wrapper.predict(Xb).flatten()

    def binary_search_amplitude_batch(x_orig_b, direction_b, y_b, n_steps=search_steps):
        """Version batchée de la recherche d'amplitude minimale."""
        M    = len(x_orig_b)
        lo   = np.zeros(M)
        hi   = np.ones(M)
        best = np.clip(x_orig_b + hi[:, None] * eps * direction_b,
                       x_orig_b - eps, x_orig_b + eps)
        for _ in range(n_steps):
            mid   = (lo + hi) / 2
            x_try = np.clip(x_orig_b + mid[:, None] * eps * direction_b,
                            x_orig_b - eps, x_orig_b + eps)
            preds  = predict_batch(x_try)
            is_adv = preds != y_b
            best = np.where(is_adv[:, None], x_try, best)
            hi   = np.where(is_adv, mid, hi)
            lo   = np.where(is_adv, lo, mid)
        return best

    best_dirs = np.random.choice([-1.0, 1.0], size=(N, D))

    # ── Initialisation batchée ─────────────────────────────────
    print(f"    RayS init : recherche point adverse pour {N} exemples...")
    initialized = np.zeros(N, dtype=bool)
    for _ in range(n_init_tries):
        pending = ~initialized
        if not pending.any():
            break
        idx        = np.where(pending)[0]
        candidates = np.clip(X_orig[idx] + eps * best_dirs[idx],
                             X_orig[idx] - eps, X_orig[idx] + eps)
        preds   = predict_batch(candidates)
        success = preds != y_flat[idx]

        succ_idx = idx[success]
        X_adv[succ_idx] = candidates[success]
        initialized[succ_idx] = True

        fail_idx = idx[~success]
        if len(fail_idx) > 0:
            best_dirs[fail_idx] = np.random.choice([-1.0, 1.0], size=(len(fail_idx), D))

    n_init_ok = int(initialized.sum())
    print(f"    RayS init : {n_init_ok}/{N} exemples initialisés avec succès")

    # ── Boucle principale batchée ──────────────────────────────
    for it in range(iters):
        preds_now    = predict_batch(X_adv)
        adverse_mask = preds_now != y_flat
        n_evaded     = int(adverse_mask.sum())
        print(f"    RayS iter {it+1:03d}/{iters} — "
              f"évadés : {n_evaded}/{N} ({100*n_evaded/N:.1f}%)")

        active_idx = np.where(adverse_mask)[0]
        if len(active_idx) == 0:
            continue

        M        = len(active_idx)
        Xo       = X_orig[active_idx]
        ya       = y_flat[active_idx]
        cur_dirs = best_dirs[active_idx].copy()

        # Mutation d'un bit aléatoire par exemple (vectorisée)
        new_dirs = cur_dirs.copy()
        j_idx    = np.random.randint(0, D, size=M)
        new_dirs[np.arange(M), j_idx] *= -1

        x_try      = np.clip(Xo + eps * new_dirs, Xo - eps, Xo + eps)
        preds_try  = predict_batch(x_try)
        mutation_ok = preds_try != ya

        # Cas 1 : la mutation reste adverse -> réduire l'amplitude dessus
        if mutation_ok.any():
            sub        = np.where(mutation_ok)[0]
            candidates = binary_search_amplitude_batch(Xo[sub], new_dirs[sub], ya[sub])
            curr_dist  = np.abs(X_adv[active_idx[sub]] - Xo[sub]).max(axis=1)
            cand_dist  = np.abs(candidates - Xo[sub]).max(axis=1)
            improve    = cand_dist < curr_dist

            gidx = active_idx[sub[improve]]
            X_adv[gidx]     = candidates[improve]
            best_dirs[gidx] = new_dirs[sub[improve]]

        # Cas 2 : la mutation échoue -> retente sur l'ancienne direction
        if (~mutation_ok).any():
            sub        = np.where(~mutation_ok)[0]
            candidates = binary_search_amplitude_batch(Xo[sub], cur_dirs[sub], ya[sub])
            curr_dist  = np.abs(X_adv[active_idx[sub]] - Xo[sub]).max(axis=1)
            cand_dist  = np.abs(candidates - Xo[sub]).max(axis=1)
            improve    = cand_dist < curr_dist

            gidx = active_idx[sub[improve]]
            X_adv[gidx] = candidates[improve]

    return X_adv