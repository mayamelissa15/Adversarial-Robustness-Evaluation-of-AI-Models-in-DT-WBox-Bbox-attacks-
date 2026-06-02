"""
Generate whitebox figures from the multi-run outputs.

Expected inputs, produced by whitebox_multirun.py:
  ~/<dataset>/results/whitebox_multirun_<dataset>_eps<eps>.csv

Usage:
  python whitebox_plot_multirun.py --dataset swat --eps 0.1
  python whitebox_plot_multirun.py --dataset swat --eps 0.3
  python whitebox_plot_multirun.py --dataset batadal --eps 0.1
  python whitebox_plot_multirun.py --dataset batadal --eps 0.3

Input:
  ~/<dataset>/results/whitebox_multirun_<dataset>_eps<eps>.csv

Figures produced:
  chart_whitebox_asr_<dataset>_eps<eps>.png
  chart_whitebox_f1_<dataset>_eps<eps>.png
  chart_whitebox_recall_<dataset>_eps<eps>.png
"""

import argparse
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATASETS = ["swat", "batadal"]
EPSILONS = [0.1, 0.3]
MODELS = ["MLP", "LogReg", "XGBoost"]
ATTACKS = ["FGSM", "PGD", "C&W"]

COLORS = {
    "MLP": "#DDEEFF",
    "LogReg": "#E8E8E8",
    "XGBoost": "#FFE8CC",
}
EDGE = {
    "MLP": "#5588BB",
    "LogReg": "#888888",
    "XGBoost": "#CC7722",
}
MEDIAN_COLOR = {
    "MLP": "#2255AA",
    "LogReg": "#555555",
    "XGBoost": "#AA5500",
}


def eps_tag(eps):
    return f"{eps:g}"


def default_csv_path(dataset, eps):
    dataset = dataset.lower()
    return (
        Path(f"~/{dataset}/results").expanduser()
        / f"whitebox_multirun_{dataset}_eps{eps_tag(eps)}.csv"
    )


def figure_suffix():
    if len(DATASETS) == 1 and len(EPSILONS) == 1:
        return f"{DATASETS[0]}_eps{eps_tag(EPSILONS[0])}"
    datasets = "-".join(DATASETS)
    epsilons = "-".join(eps_tag(eps) for eps in EPSILONS)
    return f"{datasets}_eps{epsilons}"


def load_results(datasets, epsilons):
    frames = []
    missing = []

    for dataset in datasets:
        for eps in epsilons:
            path = default_csv_path(dataset, eps)
            if not path.exists():
                missing.append(path)
                continue

            df = pd.read_csv(path)
            if "dataset" in df.columns:
                df["dataset"] = df["dataset"].astype(str).str.lower()
            else:
                df["dataset"] = dataset.lower()

            if "eps" in df.columns:
                df["eps"] = pd.to_numeric(df["eps"])
            else:
                df["eps"] = eps
            frames.append(df)

    if missing:
        msg = "\n".join(f"  - {p}" for p in missing)
        raise FileNotFoundError(f"CSV multi-run introuvable(s):\n{msg}")

    if not frames:
        raise FileNotFoundError("Aucun CSV whitebox multi-run trouve.")

    df = pd.concat(frames, ignore_index=True)
    validate_schema(df)
    return df


def validate_schema(df):
    required = {
        "dataset",
        "eps",
        "seed",
        "attack",
        "model",
        "asr",
        "f1_clean",
        "f1_adv",
        "rec_adv",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError("Colonnes manquantes dans les CSV: " + ", ".join(missing))


def panel_title(dataset, eps, df):
    sub = df[(df["dataset"] == dataset) & (np.isclose(df["eps"], eps))]
    n_runs = sub["seed"].nunique()
    return f"{dataset.upper()} - eps={eps_tag(eps)} - {n_runs} runs"


def add_value_labels(ax, bars, values, color):
    for bar, value in zip(bars, values):
        if pd.isna(value):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.8,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=7,
            color=color,
            fontweight="bold",
        )


def plot_asr_grid(df, output_dir):
    panel_configs = [(dataset, eps) for dataset in DATASETS for eps in EPSILONS]
    ncols = len(panel_configs)
    fig, axes = plt.subplots(
        nrows=len(ATTACKS),
        ncols=ncols,
        figsize=(max(5, 3.8 * ncols), 9),
        sharey=True,
    )
    axes = np.asarray(axes).reshape(len(ATTACKS), ncols)

    x = np.arange(len(MODELS))

    for row, attack in enumerate(ATTACKS):
        for col, (dataset, eps) in enumerate(panel_configs):
            ax = axes[row, col]
            sub = df[
                (df["dataset"] == dataset)
                & (np.isclose(df["eps"], eps))
                & (df["attack"] == attack)
            ]

            meds = []
            errs = []
            for model in MODELS:
                vals = sub[sub["model"] == model]["asr"] * 100
                meds.append(vals.median())
                errs.append(vals.std())

            bars = ax.bar(
                x,
                meds,
                width=0.58,
                yerr=errs,
                capsize=3,
                color=[COLORS[m] for m in MODELS],
                edgecolor=[EDGE[m] for m in MODELS],
                linewidth=0.9,
                error_kw=dict(elinewidth=0.9, ecolor="gray"),
            )

            for bar, med, model in zip(bars, meds, MODELS):
                add_value_labels(ax, [bar], [med], MEDIAN_COLOR[model])

            ax.set_ylim(0, 112)
            ax.set_xticks(x)
            ax.set_xticklabels(MODELS, rotation=25, ha="right", fontsize=8)
            ax.grid(axis="y", alpha=0.25, linestyle="--")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            if row == 0:
                ax.set_title(panel_title(dataset, eps, df), fontsize=10, pad=8)
            if col == 0:
                ax.set_ylabel(f"{attack}\nASR median +/- std (%)", fontsize=9)

    fig.suptitle("Whitebox multi-run - Attack Success Rate", fontsize=14, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = output_dir / f"chart_whitebox_asr_{figure_suffix()}.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.show()
    print(f"Saved: {out}")


def plot_f1_grid(df, output_dir):
    fig, axes = plt.subplots(
        nrows=len(DATASETS),
        ncols=len(EPSILONS),
        figsize=(max(6, 5 * len(EPSILONS)), max(4.8, 3.5 * len(DATASETS))),
        sharey=True,
    )
    axes = np.asarray(axes).reshape(len(DATASETS), len(EPSILONS))

    x = np.arange(len(ATTACKS))
    width = 0.22
    offsets = np.linspace(
        -(len(MODELS) - 1) * width / 2,
        (len(MODELS) - 1) * width / 2,
        len(MODELS),
    )

    for row, dataset in enumerate(DATASETS):
        for col, eps in enumerate(EPSILONS):
            ax = axes[row, col]
            sub = df[(df["dataset"] == dataset) & (np.isclose(df["eps"], eps))]
            agg = sub.groupby(["attack", "model"])[["f1_clean", "f1_adv"]].median()

            for i, model in enumerate(MODELS):
                f1_clean = [
                    float(agg.loc[(attack, model), "f1_clean"])
                    if (attack, model) in agg.index
                    else np.nan
                    for attack in ATTACKS
                ]
                f1_adv = [
                    float(agg.loc[(attack, model), "f1_adv"])
                    if (attack, model) in agg.index
                    else np.nan
                    for attack in ATTACKS
                ]
                drop = [clean - adv for clean, adv in zip(f1_clean, f1_adv)]

                pos = x + offsets[i]
                ax.bar(
                    pos,
                    f1_adv,
                    width,
                    color=COLORS[model],
                    edgecolor=EDGE[model],
                    linewidth=0.8,
                )
                ax.bar(
                    pos,
                    drop,
                    width,
                    bottom=f1_adv,
                    color=EDGE[model],
                    alpha=0.32,
                    edgecolor=EDGE[model],
                    linewidth=0.6,
                    hatch="///",
                )

                for px, adv, clean in zip(pos, f1_adv, f1_clean):
                    if not pd.isna(adv):
                        ax.text(
                            px,
                            min(clean + 0.025, 1.11),
                            f"{adv:.2f}",
                            ha="center",
                            va="bottom",
                            fontsize=7,
                            color=MEDIAN_COLOR[model],
                        )

            ax.set_title(panel_title(dataset, eps, df), fontsize=10, pad=8)
            ax.set_xticks(x)
            ax.set_xticklabels(ATTACKS, fontsize=9)
            ax.set_ylim(0, 1.15)
            ax.grid(axis="y", alpha=0.25, linestyle="--")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            if col == 0:
                ax.set_ylabel("F1 median", fontsize=9)

    legend_patches = [
        mpatches.Patch(facecolor=COLORS[m], edgecolor=EDGE[m], label=m)
        for m in MODELS
    ]
    drop_patch = mpatches.Patch(
        facecolor="gray",
        alpha=0.32,
        hatch="///",
        edgecolor="gray",
        label="Clean-to-adv drop",
    )
    fig.legend(
        handles=legend_patches + [drop_patch],
        loc="lower center",
        ncol=4,
        fontsize=9,
        framealpha=0.6,
    )
    fig.suptitle("Whitebox multi-run - F1 clean to adversarial", fontsize=14, y=0.99)
    fig.tight_layout(rect=[0, 0.07, 1, 0.96])
    out = output_dir / f"chart_whitebox_f1_{figure_suffix()}.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.show()
    print(f"Saved: {out}")


def plot_recall_grid(df, output_dir):
    fig, axes = plt.subplots(
        nrows=len(DATASETS),
        ncols=len(EPSILONS),
        figsize=(max(6, 5 * len(EPSILONS)), max(4.8, 3.5 * len(DATASETS))),
        sharey=True,
    )
    axes = np.asarray(axes).reshape(len(DATASETS), len(EPSILONS))

    x = np.arange(len(ATTACKS))
    width = 0.22
    offsets = np.linspace(
        -(len(MODELS) - 1) * width / 2,
        (len(MODELS) - 1) * width / 2,
        len(MODELS),
    )

    for row, dataset in enumerate(DATASETS):
        for col, eps in enumerate(EPSILONS):
            ax = axes[row, col]
            sub = df[(df["dataset"] == dataset) & (np.isclose(df["eps"], eps))]

            for i, model in enumerate(MODELS):
                sub_model = sub[sub["model"] == model]
                meds = [
                    sub_model[sub_model["attack"] == attack]["rec_adv"].median()
                    for attack in ATTACKS
                ]
                errs = [
                    sub_model[sub_model["attack"] == attack]["rec_adv"].std()
                    for attack in ATTACKS
                ]

                bars = ax.bar(
                    x + offsets[i],
                    meds,
                    width,
                    yerr=errs,
                    capsize=3,
                    color=COLORS[model],
                    edgecolor=EDGE[model],
                    linewidth=0.8,
                    label=model,
                    error_kw=dict(elinewidth=0.8, ecolor=EDGE[model]),
                )

                for bar, value in zip(bars, meds):
                    if not pd.isna(value):
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.025,
                            f"{value:.2f}",
                            ha="center",
                            va="bottom",
                            fontsize=7,
                            color=MEDIAN_COLOR[model],
                        )

            ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
            ax.set_title(panel_title(dataset, eps, df), fontsize=10, pad=8)
            ax.set_xticks(x)
            ax.set_xticklabels(ATTACKS, fontsize=9)
            ax.set_ylim(0, 1.15)
            ax.grid(axis="y", alpha=0.25, linestyle="--")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            if col == 0:
                ax.set_ylabel("Recall adv median +/- std", fontsize=9)

    handles = [
        mpatches.Patch(facecolor=COLORS[m], edgecolor=EDGE[m], label=m)
        for m in MODELS
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9, framealpha=0.6)
    fig.suptitle("Whitebox multi-run - Adversarial recall", fontsize=14, y=0.99)
    fig.tight_layout(rect=[0, 0.07, 1, 0.96])
    out = output_dir / f"chart_whitebox_recall_{figure_suffix()}.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.show()
    print(f"Saved: {out}")


def print_article_values(df):
    print("\n=== VALUES FOR ARTICLE - WHITEBOX MULTI-RUN ===\n")
    cols = ["dataset", "eps", "attack", "model"]
    summary = (
        df.groupby(cols)
        .agg(
            runs=("seed", "nunique"),
            asr_med=("asr", "median"),
            asr_std=("asr", "std"),
            f1_clean_med=("f1_clean", "median"),
            f1_adv_med=("f1_adv", "median"),
            rec_adv_med=("rec_adv", "median"),
        )
        .reset_index()
    )
    summary["asr_med"] *= 100
    summary["asr_std"] *= 100

    for _, row in summary.iterrows():
        print(
            f"{row['dataset'].upper():<8} eps={row['eps']:<4g} "
            f"{row['attack']:<5} {row['model']:<8} "
            f"runs={int(row['runs']):<2} "
            f"ASR={row['asr_med']:>5.1f}+/-{row['asr_std']:>4.1f}% "
            f"F1={row['f1_clean_med']:.3f}->{row['f1_adv_med']:.3f} "
            f"Recall={row['rec_adv_med']:.3f}"
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        "-dataset",
        default="swat",
        choices=DATASETS,
        help="Dataset a tracer.",
    )
    parser.add_argument(
        "--eps",
        "-eps",
        type=float,
        default=0.1,
        choices=EPSILONS,
        help="Epsilon a tracer.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Dossier de sortie des figures.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    datasets = [args.dataset]
    epsilons = [args.eps]

    # Keep global plot order stable even if argparse receives subsets.
    global DATASETS, EPSILONS
    DATASETS = datasets
    EPSILONS = epsilons

    output_dir = (
        Path(args.output_dir).expanduser()
        if args.output_dir
        else Path(f"~/{args.dataset}/results").expanduser()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_results(datasets, epsilons)

    print("Generating whitebox multi-run figures...")
    plot_asr_grid(df, output_dir)
    plot_f1_grid(df, output_dir)
    plot_recall_grid(df, output_dir)
    print_article_values(df)
    print(f"\nDone. Figures saved in: {output_dir}")


if __name__ == "__main__":
    main()
