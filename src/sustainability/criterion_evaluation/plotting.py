"""Publication figures for criterion-level model/expert comparisons."""

from __future__ import annotations

import math
import re
import textwrap
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
import numpy as np
import pandas as pd


LOW_N = 5
_COLORS = ["#2166ac", "#b35806", "#35978f", "#762a83"]


def _number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _sample_size(value: Any) -> int:
    number = _number(value)
    return int(number) if math.isfinite(number) else 0


def _criterion_label(row: pd.Series, *, width: int = 104) -> str:
    question = " ".join(str(row["question_text"]).split())
    short = textwrap.shorten(question, width=width, placeholder="…")
    return f"{row['question_id']}  {short}"


def _save(fig: plt.Figure, directory: Path, stem: str) -> list[Path]:
    paths = []
    for extension in ("png", "svg"):
        path = directory / f"{stem}.{extension}"
        fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
        paths.append(path)
    plt.close(fig)
    return paths


def _config_results(results: pd.DataFrame, config_id: str, ids: list[str]) -> pd.DataFrame:
    selected = results.loc[results["config_id"] == config_id]
    if selected["question_id"].duplicated().any():
        raise ValueError(f"Duplicate criterion results for configuration {config_id}")
    return selected.set_index("question_id").reindex(ids)


def _heatmap(
    catalog: pd.DataFrame,
    results: pd.DataFrame,
    configurations: list[dict[str, Any]],
    metric: str,
    directory: Path,
) -> list[Path]:
    ids = catalog["question_id"].tolist()
    values = np.full((len(ids), len(configurations)), np.nan)
    sizes = np.zeros(values.shape, dtype=int)
    for column, configuration in enumerate(configurations):
        rows = _config_results(results, configuration["id"], ids)
        values[:, column] = [_number(value) for value in rows[metric]]
        sizes[:, column] = [_sample_size(value) for value in rows["n"]]

    finite = values[np.isfinite(values)]
    if metric == "hrmae":
        vmax = max(2.0, float(finite.max())) if finite.size else 2.0
        norm = TwoSlopeNorm(vmin=0.0, vcenter=1.0, vmax=vmax)
        cmap = plt.get_cmap("RdYlGn").copy()
        title = "Human-relative MAE by sustainability criterion"
        subtitle = "Human–human MAE ÷ model–human MAE · 1 = equal disagreement · higher is better"
        color_label = "HRMAE"
    else:
        vmax = max(1.0, float(finite.max())) if finite.size else 1.0
        norm = Normalize(vmin=0.0, vmax=vmax)
        cmap = plt.get_cmap("YlOrRd").copy()
        title = "Model error by sustainability criterion"
        subtitle = "MAE against the mean of two experts · ratings on the 1–5 scale · lower is better"
        color_label = "MAE (rating points)"
    cmap.set_bad("#eeeeee")

    fig, ax = plt.subplots(figsize=(22, max(10, len(ids) * 0.43 + 3.4)))
    fig.subplots_adjust(left=0.49, right=0.92, top=0.91, bottom=0.065)
    shown = ax.imshow(np.ma.masked_invalid(values), aspect="auto", cmap=cmap, norm=norm)
    ax.set_yticks(np.arange(len(ids)), [_criterion_label(row) for _, row in catalog.iterrows()], fontsize=8.7)
    ax.set_xticks(
        np.arange(len(configurations)),
        ["\n".join(textwrap.wrap(c["label"], width=22)) for c in configurations],
        fontsize=10,
    )
    ax.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False, length=0, pad=10)
    ax.tick_params(axis="y", length=0, pad=9)
    ax.set_xticks(np.arange(-0.5, len(configurations), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(ids), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.4)
    ax.tick_params(which="minor", left=False, bottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for row in range(len(ids)):
        for column in range(len(configurations)):
            value, n = values[row, column], sizes[row, column]
            label = f"{value:.2f}" if math.isfinite(value) else "NA"
            low_n = 0 < n < LOW_N
            if low_n:
                label += f"\n(n={n})*"
            color = "#222222"
            if math.isfinite(value):
                red, green, blue, _ = cmap(norm(value))
                if 0.2126 * red + 0.7152 * green + 0.0722 * blue < 0.46:
                    color = "white"
            ax.text(column, row, label, ha="center", va="center", fontsize=8.5 if low_n else 10, color=color)

    bar = fig.colorbar(shown, ax=ax, fraction=0.038, pad=0.025, shrink=0.58)
    bar.set_label(color_label, fontsize=11)
    if metric == "hrmae":
        ticks = sorted(set([0.0, 1.0, vmax]))
        bar.set_ticks(ticks)
    fig.suptitle(title, fontsize=19, x=0.56, y=0.985)
    fig.text(0.56, 0.958, subtitle, ha="center", fontsize=11)
    fig.text(
        0.49,
        0.022,
        "* Fewer than 5 complete startup records; sample size is shown in the cell.\n"
        "NA = unavailable or undefined. Each cell uses startups with both experts and that model.\n"
        "Criterion wording is shortened; full text and per-cell sample sizes are available in the CSV tables.",
        fontsize=9,
        va="bottom",
    )
    return _save(fig, directory, f"{metric}_heatmap")


def _annotate_points(ax: plt.Axes, points: list[tuple[float, float, str]]) -> None:
    """Place short criterion IDs with deterministic overlap avoidance."""
    scale = ax.figure.dpi / 72
    boundary = ax.get_window_extent()
    occupied: list[tuple[float, float, float, float]] = []
    for x, y, _ in points:
        px, py = ax.transData.transform((x, y))
        radius = 4.5 * scale
        occupied.append((px - radius, py - radius, px + radius, py + radius))
    offsets = [(x, y) for x in (7, -7, 23, -23, 40, -40, 60, -60) for y in (7, -7, 20, -20, 34, -34, 50, -50)]
    offsets.sort(key=lambda offset: offset[0] ** 2 + offset[1] ** 2)

    def overlap(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
        return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))

    for x, y, label in points:
        px, py = ax.transData.transform((x, y))
        width, height = (len(label) * 4.0 + 3) * scale, 11 * scale
        best: tuple[float, tuple[int, int], tuple[float, float, float, float]] | None = None
        for dx, dy in offsets:
            left = px + dx * scale if dx > 0 else px + dx * scale - width
            bottom = py + dy * scale if dy > 0 else py + dy * scale - height
            padding = 2 * scale
            box = (left - padding, bottom - padding, left + width + padding, bottom + height + padding)
            collision = sum(overlap(box, used) for used in occupied)
            outside = max(boundary.x0 - box[0], 0) + max(box[2] - boundary.x1, 0) + max(boundary.y0 - box[1], 0) + max(box[3] - boundary.y1, 0)
            score = collision * 100 + outside * 1000 + (dx * dx + dy * dy) * 0.02
            if best is None or score < best[0]:
                best = (score, (dx, dy), box)
        assert best is not None
        _, (dx, dy), box = best
        occupied.append(box)
        ax.annotate(
            label,
            xy=(x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="left" if dx > 0 else "right",
            va="bottom" if dy > 0 else "top",
            fontsize=7,
            color="#303030",
            arrowprops={"arrowstyle": "-", "color": "#888888", "linewidth": 0.45},
        )


def _disagreement_scatter(
    catalog: pd.DataFrame,
    results: pd.DataFrame,
    configurations: list[dict[str, Any]],
    directory: Path,
) -> list[Path]:
    ids = catalog["question_id"].tolist()
    frame_by_config = {c["id"]: _config_results(results, c["id"], ids) for c in configurations}
    all_values = [
        _number(value)
        for rows in frame_by_config.values()
        for metric in ("human_human_mae", "model_human_mae")
        for value in rows[metric]
    ]
    finite = [value for value in all_values if math.isfinite(value)]
    limit = max(1.0, max(finite, default=1.0) * 1.16)
    nrows = max(1, math.ceil(len(configurations) / 2))
    fig, axes = plt.subplots(nrows, 2, figsize=(18, nrows * 7.5 + 0.9), squeeze=False)
    fig.subplots_adjust(left=0.065, right=0.97, bottom=0.08, top=0.92, hspace=0.25, wspace=0.19)
    annotations: list[tuple[plt.Axes, list[tuple[float, float, str]]]] = []
    for index, configuration in enumerate(configurations):
        ax = axes.flat[index]
        rows = frame_by_config[configuration["id"]]
        color = _COLORS[index % len(_COLORS)]
        ax.set(xlim=(-0.025 * limit, limit), ylim=(-0.025 * limit, limit), aspect="equal")
        ax.plot([0, limit], [0, limit], color="#555555", linestyle="--", linewidth=1.1, label="Equal disagreement")
        ax.set_title(configuration["label"], fontsize=14, pad=12)
        ax.set_xlabel("Expert–expert MAE", fontsize=11)
        ax.set_ylabel("Model–human MAE", fontsize=11)
        ax.grid(color="#e5e5e5", linewidth=0.65)
        points = []
        unavailable = []
        for question_id, row in rows.iterrows():
            x, y = _number(row["human_human_mae"]), _number(row["model_human_mae"])
            n = _sample_size(row["n"])
            if not math.isfinite(x) or not math.isfinite(y):
                unavailable.append(str(question_id))
                continue
            low_n = n < LOW_N
            ax.scatter(x, y, s=45 if low_n else 34, marker="D" if low_n else "o", facecolors="white" if low_n else color, edgecolors=color, linewidth=1.2, zorder=3)
            label = str(question_id) + (f" (n={n})*" if low_n else "")
            points.append((x, y, label))
        annotations.append((ax, points))
        ax.text(0.025, 0.975, "Above line: higher model disagreement\nBelow line: lower model disagreement", transform=ax.transAxes, va="top", fontsize=8, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9})
        if unavailable:
            ax.text(0.975, 0.025, "NA: " + ", ".join(unavailable), transform=ax.transAxes, ha="right", fontsize=8)
    for index in range(len(configurations), axes.size):
        axes.flat[index].set_visible(False)
    fig.canvas.draw()
    for ax, points in annotations:
        _annotate_points(ax, points)
    fig.suptitle("Human disagreement versus model error across sustainability criteria", fontsize=19, y=0.985)
    fig.text(0.5, 0.028, "Each point is one criterion; Q IDs match the criterion catalogue. Dashed line: y = x.\n* Hollow diamonds mark fewer than 5 complete startup records; the sample size is printed beside the criterion.", ha="center", fontsize=10)
    return _save(fig, directory, "human_disagreement_vs_model_error")


def _delta_plot(
    catalog: pd.DataFrame,
    comparisons: pd.DataFrame,
    comparison: dict[str, Any],
    directory: Path,
) -> list[Path]:
    selected = comparisons.loc[comparisons["comparison_id"] == comparison["id"]]
    if selected["question_id"].duplicated().any():
        raise ValueError(f"Duplicate criterion comparison for {comparison['id']}")
    columns = ["question_id", "n", "delta_mae", "delta_mae_ci_low", "delta_mae_ci_high"]
    rows = catalog.merge(selected[columns], on="question_id", how="left", validate="one_to_one")
    rows["delta_mae"] = pd.to_numeric(rows["delta_mae"], errors="coerce")
    rows = rows.sort_values(
        ["delta_mae", "question_id"], ascending=[False, True], na_position="last", kind="stable"
    ).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(21, max(10, len(rows) * 0.43 + 3.3)))
    fig.subplots_adjust(left=0.54, right=0.95, top=0.915, bottom=0.095)
    extent = [0.0]
    labels = []
    for position, row in rows.iterrows():
        n = _sample_size(row["n"])
        low_n = 0 < n < LOW_N
        labels.append(f"{_criterion_label(row, width=96)}  [n={n}{'*' if low_n else ''}]")
        value = _number(row["delta_mae"])
        if not math.isfinite(value):
            ax.text(0.015, position, "NA", transform=ax.get_yaxis_transform(), va="center", fontsize=9, color="#777777")
            continue
        low, high = _number(row["delta_mae_ci_low"]), _number(row["delta_mae_ci_high"])
        color = "#238b45" if value < 0 else "#756bb1"
        if math.isfinite(low) and math.isfinite(high):
            ax.hlines(position, low, high, color=color, linewidth=1.4)
            ax.plot([low, high], [position, position], linestyle="none", marker="|", markersize=7, color=color)
            extent.extend([low, high])
        ax.scatter(value, position, color="white" if low_n else color, edgecolor=color, marker="D" if low_n else "o", s=43, linewidth=1.3, zorder=3)
        extent.append(value)
    spread = max(0.25, max(abs(value) for value in extent)) * 1.16
    ax.set_xlim(-spread, spread)
    ax.set_ylim(len(rows) - 0.4, -0.7)
    ax.set_yticks(np.arange(len(rows)), labels, fontsize=8.7)
    ax.tick_params(axis="y", length=0, pad=10)
    ax.axvline(0, color="#555555", linestyle="--", linewidth=1.1)
    ax.grid(axis="x", color="#e5e5e5", linewidth=0.65)
    ax.set_axisbelow(True)
    for position in range(len(rows)):
        if position % 2 == 0:
            ax.axhspan(position - 0.5, position + 0.5, color="#f7f7f7", zorder=0)
    for spine in ("left", "top", "right"):
        ax.spines[spine].set_visible(False)
    ax.set_xlabel("ΔMAE = MAE(RAG) − MAE(Full Context)\n← RAG has lower error                  Full Context has lower error →", fontsize=11, labelpad=12)
    fig.suptitle(f"Full Context versus RAG: {comparison['label']}", fontsize=19, x=0.57, y=0.985)
    fig.text(0.57, 0.957, "Criteria ranked by paired ΔMAE · intervals show 95% startup bootstrap CIs", ha="center", fontsize=11)
    fig.text(0.54, 0.017, "n counts startups with both expert ratings and both methods for the same business-plan version.\n* Fewer than 5 paired startups; hollow diamonds indicate low sample size. NA = unavailable.\nNo interval is drawn when a bootstrap CI is undefined. Full criterion wording is in the CSV tables.", fontsize=9, va="bottom")
    stem = re.sub(r"[^a-zA-Z0-9_-]", "_", comparison["id"])
    return _save(fig, directory, f"delta_mae_{stem}")


def generate_figures(
    catalog: pd.DataFrame,
    results: pd.DataFrame,
    comparisons: pd.DataFrame,
    config: dict,
    output_dir: Path,
) -> list[Path]:
    """Write PNG and SVG figures, retaining all canonical criterion rows.

    Inputs are the unrounded result tables. All files are placed under
    ``output_dir / 'figures'``; the returned paths include both formats.
    """
    directory = Path(output_dir) / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    configurations = config["configurations"]
    if not configurations:
        raise ValueError("At least one model configuration is required to draw figures")
    if catalog["question_id"].duplicated().any():
        raise ValueError("The criterion catalogue contains duplicate question IDs")
    paths: list[Path] = []
    with plt.rc_context({"font.family": "DejaVu Sans", "svg.fonttype": "none", "axes.titleweight": "bold"}):
        paths.extend(_heatmap(catalog, results, configurations, "mae", directory))
        paths.extend(_heatmap(catalog, results, configurations, "hrmae", directory))
        paths.extend(_disagreement_scatter(catalog, results, configurations, directory))
        for comparison in config["comparisons"]:
            paths.extend(_delta_plot(catalog, comparisons, comparison, directory))
    return paths
