# ~/swat/plot_boxplots.py

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

RESULTS_DIR = Path("~/swat/results").expanduser()

# Charge le CSV (tmp ou final)
csv_path = RESULTS_DIR / "multi_run_results_tmp.csv"
if not csv_path.exists():
    csv_path = RESULTS_DIR / "multi_run_results.csv"

df = pd.read_csv(csv_path)
print(f"Seeds disponibles : {sorted(df['seed'].unique())}")
print(f"Shape : {df.shape}")
print(df.head())

# ── Config ────────────────────────────────────────────────────
MODELS  = ["MLP", "LogReg", "XGBoost"]
COLORS  = {"MLP": "#534AB7", "LogReg": "#0F6E56", "XGBoost": "#993C1D"}
METRIC  = "asr"   # ou "f1_adv", "rec_adv"

# ── Plot 1 : Blackbox ─────────────────────────────────────────
bb = df[df["family"].isin(["Score-based", "Decision-based"])]
attacks_bb = sorted(bb["attack"].unique())

fig, axes = plt.subplots(1, len(attacks_bb), figsize=(14, 5), sharey=True)
fig.suptitle("Blackbox attacks — ASR par seed (box plot)", fontsize=13)

for ax, atk in zip(axes, attacks_bb):
    data   = [bb[(bb["attack"] == atk) & (bb["model"] == m)][METRIC].values
               for m in MODELS]
    bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                    medianprops=dict(color="white", linewidth=2))
    for patch, m in zip(bp["boxes"], MODELS):
        patch.set_facecolor(COLORS[m])
        patch.set_alpha(0.85)
    ax.set_title(atk, fontsize=11)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(MODELS, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.grid(axis="y", alpha=0.3)

axes[0].set_ylabel("Attack Success Rate")
legend_patches = [mpatches.Patch(color=COLORS[m], label=m) for m in MODELS]
fig.legend(handles=legend_patches, loc="lower center", ncol=3, fontsize=10)
plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig(RESULTS_DIR / "boxplot_blackbox.png", dpi=150, bbox_inches="tight")
plt.show()
print("✓ boxplot_blackbox.png sauvegardé")

# ── Plot 2 : Transfer ─────────────────────────────────────────
tr = df[df["family"] == "Transfer"]

# Pour le transfer, on garde le meilleur substitut par (seed, attack, model)
# car tu veux comparer les attaques, pas les substituts
tr_best = (tr.groupby(["seed", "attack", "model"])[METRIC]
             .max()
             .reset_index())

attacks_tr = sorted(tr_best["attack"].unique())

fig, axes = plt.subplots(1, len(attacks_tr), figsize=(12, 5), sharey=True)
if len(attacks_tr) == 1:
    axes = [axes]
fig.suptitle("Transfer attacks — ASR par seed (box plot)", fontsize=13)

for ax, atk in zip(axes, attacks_tr):
    data = [tr_best[(tr_best["attack"] == atk) & (tr_best["model"] == m)][METRIC].values
             for m in MODELS]
    bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                    medianprops=dict(color="white", linewidth=2))
    for patch, m in zip(bp["boxes"], MODELS):
        patch.set_facecolor(COLORS[m])
        patch.set_alpha(0.85)
    ax.set_title(atk, fontsize=11)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(MODELS, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.grid(axis="y", alpha=0.3)

axes[0].set_ylabel("Attack Success Rate")
fig.legend(handles=legend_patches, loc="lower center", ncol=3, fontsize=10)
plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig(RESULTS_DIR / "boxplot_transfer.png", dpi=150, bbox_inches="tight")
plt.show()
print("✓ boxplot_transfer.png sauvegardé")

# ── Stats résumé ─────────────────────────────────────────────
print("\n=== Résumé statistique (mean ± std) ===")
summary = (df.groupby(["family", "attack", "model"])[METRIC]
             .agg(["mean", "std", "count"])
             .round(4))
print(summary.to_string())