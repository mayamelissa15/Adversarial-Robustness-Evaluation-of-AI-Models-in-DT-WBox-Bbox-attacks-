"""
Génère les figures whitebox pour l'article à partir de
whitebox_multirun_results.csv produit par run_whitebox_multirun.py.

Figures produites :
  chart_whitebox_asr_fgsm.png  — barplot ASR FGSM par modèle
  chart_whitebox_asr_pgd.png   — barplot ASR PGD par modèle
  chart_whitebox_asr_cw.png    — barplot ASR C&W par modèle
  chart_whitebox_f1.png        — évolution F1 clean → adv (barres groupées)
  chart_whitebox_recall.png    — recall adversarial par attaque et modèle
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

RESULTS_DIR = Path("~/swat/results").expanduser()

df = pd.read_csv(RESULTS_DIR / "whitebox_multirun_results.csv")

MODELS  = ["MLP", "LogReg", "XGBoost"]
ATTACKS = ["FGSM", "PGD", "C&W"]

COLORS = {
    "MLP":     "#DDEEFF",
    "LogReg":  "#E8E8E8",
    "XGBoost": "#FFE8CC",
}
EDGE = {
    "MLP":     "#5588BB",
    "LogReg":  "#888888",
    "XGBoost": "#CC7722",
}
MEDIAN_COLOR = {
    "MLP":     "#2255AA",
    "LogReg":  "#555555",
    "XGBoost": "#AA5500",
}


def plot_asr_barplot(df, filename_prefix="chart_whitebox_asr"):
    """
    Une figure par attaque, barplot ASR médian avec barre d'erreur (std).
    3 barres sur l'axe X : MLP / LogReg / XGBoost.
    """
    for atk in ATTACKS:
        sub_atk = df[df["attack"] == atk]

        meds = [sub_atk[sub_atk["model"] == m]["asr"].median() * 100 for m in MODELS]
        stds = [sub_atk[sub_atk["model"] == m]["asr"].std()    * 100 for m in MODELS]

        fig, ax = plt.subplots(figsize=(6, 5))

        bars = ax.bar(
            np.arange(len(MODELS)),
            meds,
            width=0.5,
            yerr=stds,
            capsize=5,
            color=[COLORS[m] for m in MODELS],
            edgecolor=[EDGE[m] for m in MODELS],
            linewidth=0.9,
            error_kw=dict(elinewidth=1.0, ecolor="gray"),
        )

        for bar, med, model in zip(bars, meds, MODELS):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 2.5,
                f"{med:.1f}%",
                ha="center", va="bottom", fontsize=10,
                color=MEDIAN_COLOR[model], fontweight="bold"
            )

        ax.set_xticks(np.arange(len(MODELS)))
        ax.set_xticklabels(MODELS, fontsize=11)
        ax.set_ylabel("Attack Success Rate (%)", fontsize=10)
        ax.set_ylim(0, 115)
        ax.set_title(f"{atk} — ASR médian ± std sur {df['seed'].nunique()} runs (ε = 0.1)",
                     fontsize=11, pad=10)
        ax.grid(axis="y", alpha=0.25, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        filename = f"{filename_prefix}_{atk.lower()}.png"
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / filename, dpi=200, bbox_inches="tight")
        plt.show()
        print(f"✓ {filename} sauvegardé")


def plot_f1_drop(df, filename="chart_whitebox_f1.png"):
    x       = np.arange(len(ATTACKS))
    width   = 0.22
    offsets = np.linspace(-(len(MODELS) - 1) * width / 2,
                           (len(MODELS) - 1) * width / 2,
                           len(MODELS))

    agg = df.groupby(["attack", "model"])[["f1_clean", "f1_adv"]].median()

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, model in enumerate(MODELS):
        f1_clean = [float(agg.loc[(atk, model), "f1_clean"]) for atk in ATTACKS]
        f1_adv   = [float(agg.loc[(atk, model), "f1_adv"])   for atk in ATTACKS]
        drop     = [c - a for c, a in zip(f1_clean, f1_adv)]

        ax.bar(x + offsets[i], f1_adv, width,
               color=COLORS[model], edgecolor=EDGE[model], linewidth=0.8,
               label=f"{model} (adv)")
        ax.bar(x + offsets[i], drop, width, bottom=f1_adv,
               color=EDGE[model], alpha=0.35, edgecolor=EDGE[model],
               linewidth=0.6, hatch="///")

        for pos, adv, cl in zip(x + offsets[i], f1_adv, f1_clean):
            ax.text(pos, cl + 0.008, f"{adv:.2f}",
                    ha="center", va="bottom", fontsize=7,
                    color=MEDIAN_COLOR[model])

    ax.set_xticks(x)
    ax.set_xticklabels(ATTACKS, fontsize=11)
    ax.set_ylabel("F1 score (médiane)", fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_title("F1 clean → adversarial — Whitebox attacks (ε = 0.1)",
                 fontsize=11, pad=10)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_patches = [mpatches.Patch(facecolor=COLORS[m], edgecolor=EDGE[m],
                                     label=m) for m in MODELS]
    hatch_patch = mpatches.Patch(facecolor="gray", alpha=0.35,
                                 hatch="///", edgecolor="gray",
                                 label="F1 drop (attaque)")
    ax.legend(handles=legend_patches + [hatch_patch], fontsize=9, framealpha=0.5)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / filename, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"✓ {filename} sauvegardé")


def plot_recall_adv(df, filename="chart_whitebox_recall.png"):
    x       = np.arange(len(ATTACKS))
    width   = 0.22
    offsets = np.linspace(-(len(MODELS) - 1) * width / 2,
                           (len(MODELS) - 1) * width / 2,
                           len(MODELS))

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, model in enumerate(MODELS):
        sub  = df[df["model"] == model]
        meds = [sub[sub["attack"] == atk]["rec_adv"].median() for atk in ATTACKS]
        errs = [sub[sub["attack"] == atk]["rec_adv"].std()    for atk in ATTACKS]

        bars = ax.bar(x + offsets[i], meds, width,
                      yerr=errs, capsize=3,
                      color=COLORS[model], edgecolor=EDGE[model],
                      linewidth=0.8, label=model,
                      error_kw=dict(elinewidth=0.8, ecolor=EDGE[model]))

        for bar, v in zip(bars, meds):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.018,
                    f"{v:.2f}",
                    ha="center", va="bottom", fontsize=7,
                    color=MEDIAN_COLOR[model])

    ax.set_xticks(x)
    ax.set_xticklabels(ATTACKS, fontsize=11)
    ax.set_ylabel("Recall adversarial (médiane ± std)", fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_title("Recall post-attaque — Whitebox attacks (ε = 0.1)",
                 fontsize=11, pad=10)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=9, framealpha=0.5)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / filename, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"✓ {filename} sauvegardé")


def print_article_values(df):
    print("\n=== VALEURS POUR L'ARTICLE (whitebox multi-run) ===\n")
    print(f"{'Attaque':<8} {'Modèle':<10} "
          f"{'ASR med':>8} {'ASR std':>8} "
          f"{'F1 adv':>8} {'Recall':>8}")
    print("─" * 60)

    for atk in ATTACKS:
        for model in MODELS:
            sub = df[(df["attack"] == atk) & (df["model"] == model)]
            if sub.empty:
                continue
            print(f"{atk:<8} {model:<10} "
                  f"{sub['asr'].median()*100:>7.1f}% "
                  f"{sub['asr'].std()*100:>7.1f}% "
                  f"{sub['f1_adv'].median():>8.3f} "
                  f"{sub['rec_adv'].median():>8.3f}")
        print()


if __name__ == "__main__":
    print("Génération des figures whitebox multi-run...")

    plot_asr_barplot(df)
    plot_f1_drop(df)
    plot_recall_adv(df)
    print_article_values(df)

    print(f"\n✓ Figures sauvegardées dans {RESULTS_DIR}")