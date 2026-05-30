# ~/swat/plot_article.py

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
import json

RESULTS_DIR = Path("~/swat/results").expanduser()

df = pd.read_csv(RESULTS_DIR / "multi_run_results_tmp.csv")
with open(RESULTS_DIR / "blackbox_results.json") as f:
    bb_json = json.load(f)

MODELS = ["MLP", "LogReg", "XGBoost"]

# Couleurs douces cohérentes avec ton schéma LaTeX
COLORS = {
    "MLP":     "#DDEEFF",   # bleu clair ~ blue!10
    "LogReg":  "#E8E8E8",   # gris clair ~ gray!12
    "XGBoost": "#FFE8CC",   # orange clair ~ orange!15
}
EDGE = {
    "MLP":     "#5588BB",
    "LogReg":  "#888888",
    "XGBoost": "#CC7722",
}

def make_barchart(attacks, values_dict, title, filename,
                  note=None, figwidth=7):
    """
    values_dict : {model: [val_atk1, val_atk2, ...]}
    """
    x     = np.arange(len(attacks))
    width = 0.22
    offsets = [-width, 0, width]

    fig, ax = plt.subplots(figsize=(figwidth, 4.5))

    for i, model in enumerate(MODELS):
        vals = values_dict[model]
        bars = ax.bar(x + offsets[i], vals, width,
                      label=model,
                      color=COLORS[model],
                      edgecolor=EDGE[model],
                      linewidth=0.8)
        # Valeur au dessus de chaque barre
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.008,
                    f"{v:.0%}", ha="center", va="bottom",
                    fontsize=7.5, color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels(attacks, fontsize=10)
    ax.set_ylabel("Attack Success Rate (ASR)", fontsize=10)
    ax.set_ylim(0, 1.10)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend(fontsize=9, framealpha=0.5)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.set_title(title, fontsize=11, pad=10)
    if note:
        fig.text(0.5, -0.04, note, ha="center",
                 fontsize=8, color="gray", style="italic")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / filename,
                dpi=200, bbox_inches="tight")
    plt.show()
    print(f"✓ {filename} sauvegardé")


# ══════════════════════════════════════════════════════════════
# CHART 1 : Transfer attacks
# ══════════════════════════════════════════════════════════════

tr = df[df["family"] == "Transfer"]
tr_best = (tr.groupby(["seed", "attack", "model"])["asr"]
             .max().reset_index()
             .groupby(["attack", "model"])["asr"]
             .median().reset_index())

attacks_tr = ["MI-FGSM", "VMI-FGSM", "Ensemble-MI"]
vals_tr = {m: [float(tr_best[(tr_best["attack"] == a) &
                              (tr_best["model"] == m)]["asr"].values[0])
               for a in attacks_tr]
           for m in MODELS}

make_barchart(
    attacks_tr, vals_tr,
    title="Transfer attacks — Median ASR over 10 runs (ε = 0.1)",
    filename="chart_transfer.png",
    figwidth=7
)

# ══════════════════════════════════════════════════════════════
# CHART 2 : Score-based attacks (Square + NES)
# ══════════════════════════════════════════════════════════════

bb = df[df["family"] == "Score-based"]
bb_med = (bb.groupby(["attack", "model"])["asr"]
            .median().reset_index())

attacks_sc = ["Square", "NES"]
vals_sc = {m: [float(bb_med[(bb_med["attack"] == a) &
                             (bb_med["model"] == m)]["asr"].values[0])
               for a in attacks_sc]
           for m in MODELS}

make_barchart(
    attacks_sc, vals_sc,
    title="Score-based attacks — Median ASR over 10 runs (ε = 0.1)",
    filename="chart_score.png",
    figwidth=5
)

# ══════════════════════════════════════════════════════════════
# CHART 3 : Decision-based attacks (HSJA + RayS) single run
# ══════════════════════════════════════════════════════════════

attacks_db = ["HSJA", "RayS"]
vals_db = {m: [bb_json[m][a]["evasion_rate"] / 100
               for a in attacks_db]
           for m in MODELS}

make_barchart(
    attacks_db, vals_db,
    title="Decision-based attacks — ASR (ε = 0.1)",
    filename="chart_decision.png",
    note="Single run — excluded from multi-run analysis "
         "due to computational cost (~1.2M queries/run)",
    figwidth=5
)

# ══════════════════════════════════════════════════════════════
# RÉSUMÉ VALEURS ARTICLE
# ══════════════════════════════════════════════════════════════

print("\n=== VALEURS MÉDIANES POUR L'ARTICLE ===\n")

print("── Transfer (médiane ± std, 10 runs)")
for atk in attacks_tr:
    line = f"  {atk:<14}"
    for m in MODELS:
        v = tr_best[(tr_best["attack"]==atk) & (tr_best["model"]==m)]["asr"]
        s = (tr.groupby(["seed","attack","model"])["asr"]
               .max().reset_index())
        s = s[(s["attack"]==atk) & (s["model"]==m)]["asr"]
        line += f"  {m}: {float(v.values[0]):.1%}±{s.std():.1%}"
    print(line)

print("\n── Score-based (médiane ± std, 10 runs)")
for atk in attacks_sc:
    line = f"  {atk:<14}"
    for m in MODELS:
        v = bb[(bb["attack"]==atk) & (bb["model"]==m)]["asr"]
        line += f"  {m}: {v.median():.1%}±{v.std():.1%}"
    print(line)

print("\n── Decision-based (single run)")
for atk in attacks_db:
    line = f"  {atk:<14}"
    for m in MODELS:
        line += f"  {m}: {bb_json[m][atk]['evasion_rate']:.1f}%"
    print(line)