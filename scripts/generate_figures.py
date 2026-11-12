#!/usr/bin/env python3
"""Regenerate manuscript vector figures from the supplied criterion results.

Run from any directory: python scripts/generate_figures.py
Requires matplotlib, numpy and pandas. The script performs no model calls and
uses only the three aggregate CSV files in data/results/. Regenerates the four
numerical figures; the author-edited workflow diagrams are not regenerated.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "results"
OUT = ROOT / "build" / "figures"
ORDER = ["gpt_full_context", "gpt_rag", "gemini_full_context", "gemini_rag"]
TITLES = ["GPT\nFull context", "GPT\nRAG", "Gemini\nFull context", "Gemini\nRAG"]
COLORS = ["#28567A", "#B26827", "#297D78", "#80559A"]
LABELS = {
    "Q001": "Sustainability consideration",
    "Q002": "Societal sustainability pressure",
    "Q003": "Stakeholder consideration",
    "Q004": "Social problem priority",
    "Q005": "Environmental problem priority",
    "Q006": "Economic growth priority",
    "Q007": "Societal value priority",
    "Q008": "Profit / shareholder priority",
    "Q009": "Environmental value priority",
    "Q010": "Social vision",
    "Q011": "Social business-model benefit",
    "Q012": "Social product / service value",
    "Q013": "Social impact tracking",
    "Q014": "Founder and employee diversity",
    "Q015": "Negative social effects",
    "Q016": "Environmental vision",
    "Q017": "Environmental business benefit",
    "Q018": "Environmental product benefit",
    "Q019": "Environmental impact tracking",
    "Q020": "Negative environmental effects",
    "Q021": "Circular economy principles",
    "Q022": "Financial value vision",
    "Q023": "Economic business-model value",
    "Q024": "Economic product / service value",
    "Q025": "Economic impact tracking",
    "Q026": "Negative economic effects",
    "Q027": "Geographic scaling intention",
    "Q028": "Transferability to other settings",
    "Q029": "Employment growth expectation",
    "Q030": "Growth beyond home market",
    "Q031": "SDG consideration",
    "Q032": "Core solution competencies",
    "Q033": "Sustainability competencies",
    "Q034": "Environmental impact knowledge",
    "Q035": "Social impact knowledge",
    "Q036": "Economic impact knowledge",
    "Q037": "Investment competencies",
}


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.5,
        "axes.titlesize": 9, "axes.labelsize": 8.5,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#6E7479", "axes.linewidth": 0.6,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.facecolor": "white", "figure.facecolor": "white",
        "svg.fonttype": "none",
    })


def save(fig, name):
    # Fixed dimensions are intentional: do not crop using bbox_inches='tight'.
    fig.savefig(OUT / f"{name}.pdf", metadata={"Title": name.replace("_", " ")})
    fig.savefig(OUT / f"{name}.svg")
    plt.close(fig)


def heatmap(results, catalog, metric):
    ids = catalog.sort_values("question_id").question_id.tolist()
    values = results.pivot(index="question_id", columns="config_id", values=metric).loc[ids, ORDER]
    if metric == "mae":
        cmap = plt.get_cmap("YlOrRd")
        norm = Normalize(vmin=0.45, vmax=1.5)
        ticks = [0.5, 0.75, 1.0, 1.25, 1.5]
        bar_label = "MAE (rating points; lower is better)"
        name = "mae_heatmap"
    else:
        cmap = LinearSegmentedColormap.from_list("human_relative", ["#B56531", "#FCFCF8", "#177A68"])
        norm = TwoSlopeNorm(vmin=0.85, vcenter=1.0, vmax=1.45)
        ticks = [0.85, 1.0, 1.15, 1.30, 1.45]
        bar_label = "HRMAE (1 = equal disagreement; higher is better)"
        name = "hrmae_heatmap"
    assert np.nanmin(values) >= norm.vmin and np.nanmax(values) <= norm.vmax
    fig = plt.figure(figsize=(6.5, 8.1))
    ax = fig.add_axes([0.47, 0.055, 0.445, 0.88])
    img = ax.imshow(values.values, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_xticks(range(4), TITLES)
    ax.tick_params(axis="x", top=True, bottom=False, labeltop=True, labelbottom=False, length=0, pad=7)
    ax.set_yticks(range(37), [f"{q}  {LABELS[q]}" for q in ids], fontsize=7.5)
    ax.tick_params(axis="y", length=0, pad=5)
    ax.set_xticks(np.arange(-0.5, 4), minor=True)
    ax.set_yticks(np.arange(-0.5, 37), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)
    for boundary in [2.5, 8.5, 14.5, 20.5, 25.5, 29.5, 30.5]:
        ax.axhline(boundary, color="white", linewidth=1.8)
    for row in range(37):
        for col in range(4):
            value = values.iloc[row, col]
            rgba = cmap(norm(value))
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            ax.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=7.5,
                    color="white" if luminance < 0.48 else "#14222A")
    for spine in ax.spines.values():
        spine.set_visible(False)
    cax = fig.add_axes([0.930, 0.21, 0.014, 0.57])
    cb = fig.colorbar(img, cax=cax, ticks=ticks)
    cb.ax.tick_params(labelsize=7.5, width=0.5, length=2)
    fig.text(0.70, 0.022, bar_label, fontsize=7.5, ha="center", va="center")
    save(fig, name)


def disagreement(results):
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 6.5), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.105, right=0.98, top=0.95, bottom=0.09, wspace=0.20, hspace=0.23)
    panel_names = ["GPT full context", "GPT RAG", "Gemini full context", "Gemini RAG"]
    for ax, cfg, title, color in zip(axes.flat, ORDER, panel_names, COLORS):
        data = results[results.config_id == cfg].set_index("question_id")
        ax.plot([0, 1.75], [0, 1.75], "--", color="#7C858C", linewidth=0.8, zorder=1)
        ax.scatter(data.human_human_mae, data.model_human_mae, s=17, color=color,
                   edgecolor="white", linewidth=0.3, zorder=3)
        offsets = {"Q005": (-19, -12), "Q026": (-21, 8), "Q035": (8, 9)}
        for q, offset in offsets.items():
            row = data.loc[q]
            ax.annotate(q, (row.human_human_mae, row.model_human_mae), xytext=offset,
                        textcoords="offset points", fontsize=7.5, color="#35424B",
                        arrowprops={"arrowstyle": "-", "lw": 0.45, "color": "#89939B"})
        ax.set_title(title, fontweight="bold", pad=7)
        ax.set_xlim(0, 1.75)
        ax.set_ylim(0, 1.75)
        ax.set_xticks([0, 0.5, 1.0, 1.5])
        ax.set_yticks([0, 0.5, 1.0, 1.5])
        ax.set_aspect("equal", adjustable="box")
        ax.grid(color="#E6E9EB", linewidth=0.5)
        ax.set_axisbelow(True)
    for ax in axes[1, :]:
        ax.set_xlabel("Expert–expert MAE", labelpad=5)
    for ax in axes[:, 0]:
        ax.set_ylabel("Model–expert MAE", labelpad=5)
    save(fig, "disagreement")


def delta(comparisons):
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 8.2), sharex=True)
    fig.subplots_adjust(left=0.105, right=0.98, top=0.95, bottom=0.075, wspace=0.36)
    for ax, family in zip(axes, ["gpt", "gemini"]):
        data = comparisons[comparisons.comparison_id == family].sort_values(
            ["delta_mae", "question_id"], ascending=[False, True])
        data = data.reset_index(drop=True)
        ys = np.arange(len(data))
        for i in range(0, len(data), 2):
            ax.axhspan(i - 0.5, i + 0.5, facecolor="#F4F6F7", zorder=0)
        ax.axvline(0, color="#6D767D", linestyle="--", linewidth=0.8, zorder=1)
        for y, row in data.iterrows():
            color = "#725B9E" if row.delta_mae > 0 else "#247D67"
            if row.delta_mae == 0:
                color = "#65727A"
            ax.errorbar(row.delta_mae, y,
                        xerr=[[row.delta_mae-row.delta_mae_ci_low],
                              [row.delta_mae_ci_high-row.delta_mae]],
                        fmt="o", color=color, ecolor=color, markersize=3.0,
                        linewidth=0.8, capsize=1.6, capthick=0.65, zorder=3)
        ax.set_yticks(ys, data.question_id, fontsize=7.5)
        ax.set_ylim(len(data) - 0.5, -0.5)
        ax.set_xlim(-0.30, 0.50)
        ax.set_xticks([-0.2, 0, 0.2, 0.4])
        ax.tick_params(axis="y", length=0, pad=4)
        ax.set_title(family.upper() if family == "gpt" else "Gemini", fontweight="bold", pad=7)
        ax.set_xlabel("RAG MAE − full-context MAE", fontsize=8, labelpad=7)
        ax.grid(axis="x", color="#E1E6E9", linewidth=0.5)
        ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)
    save(fig, "delta_mae")


def main():
    style()
    OUT.mkdir(parents=True, exist_ok=True)
    results = pd.read_csv(DATA / "criterion_results.csv")
    comparisons = pd.read_csv(DATA / "full_context_rag_comparisons.csv")
    catalog = pd.read_csv(DATA / "criterion_catalog.csv")
    assert len(catalog) == 37 and set(catalog.question_id) == set(LABELS)
    assert len(results) == 148 and len(comparisons) == 74
    assert not results.duplicated(["config_id", "question_id"]).any()
    heatmap(results, catalog, "mae")
    heatmap(results, catalog, "hrmae")
    disagreement(results)
    delta(comparisons)
    print("Wrote four numerical figures as PDF and SVG.")


if __name__ == "__main__":
    main()
