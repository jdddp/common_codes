import csv
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


def _legend_names(names: Sequence[str], valid: np.ndarray, total_classes: int):
    if names and len(names) == total_classes:
        return [names[idx] for idx in np.where(valid)[0]]
    return [str(idx) for idx in np.where(valid)[0]]


def _plot_curve(
    x: np.ndarray,
    curves: np.ndarray,
    valid: np.ndarray,
    out_path: Path,
    xlabel: str,
    ylabel: str,
    title: str,
    names: Sequence[str],
    max_legend: int = 20,
):
    plt.figure(figsize=(9, 6), dpi=160)
    valid_curves = curves[valid]
    if valid_curves.size == 0:
        plt.plot(x, np.zeros_like(x), color="tab:blue", linewidth=2.5)
    else:
        if valid.sum() <= max_legend:
            for curve, name in zip(valid_curves, _legend_names(names, valid, curves.shape[0])):
                plt.plot(x, curve, linewidth=1.0, alpha=0.8, label=name)
        else:
            for curve in valid_curves:
                plt.plot(x, curve, color="gray", linewidth=0.8, alpha=0.3)
        plt.plot(x, valid_curves.mean(axis=0), color="tab:blue", linewidth=2.8, label="all classes")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    if valid.sum() and valid.sum() <= max_legend:
        plt.legend(loc="best", fontsize=8)
    elif valid.sum():
        plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_detection_curves(curves: dict, out_dir: str, names: Sequence[str]):
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    valid = curves["valid"]
    _plot_curve(
        curves["conf_x"],
        curves["p_curve"],
        valid,
        output_dir / "P_curve.png",
        xlabel="Confidence",
        ylabel="Precision",
        title="Precision-Confidence Curve",
        names=names,
    )
    _plot_curve(
        curves["conf_x"],
        curves["r_curve"],
        valid,
        output_dir / "R_curve.png",
        xlabel="Confidence",
        ylabel="Recall",
        title="Recall-Confidence Curve",
        names=names,
    )
    _plot_curve(
        curves["recall_x"],
        curves["pr_curve"],
        valid,
        output_dir / "PR_curve.png",
        xlabel="Recall",
        ylabel="Precision",
        title="Precision-Recall Curve",
        names=names,
    )


def plot_results(results_csv: str, out_path: str):
    csv_path = Path(results_csv)
    if not csv_path.exists():
        return

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return

    epochs = [int(row["epoch"]) for row in rows]
    series = {
        "Box Loss": [float(row["box_loss"]) for row in rows],
        "Cls Loss": [float(row["cls_loss"]) for row in rows],
        "DFL Loss": [float(row["dfl_loss"]) for row in rows],
        "Precision": [float(row["precision"]) for row in rows],
        "Recall": [float(row["recall"]) for row in rows],
        "mAP50": [float(row["map50"]) for row in rows],
        "mAP50-95": [float(row["map50_95"]) for row in rows],
        "Val Loss": [float(row["val_loss"]) for row in rows],
    }

    fig, axes = plt.subplots(2, 4, figsize=(18, 9), dpi=160)
    for ax, (title, values) in zip(axes.flat, series.items()):
        ax.plot(epochs, values, color="tab:blue", linewidth=2.0, marker="o", markersize=3)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def plot_per_class_results(results_csv: str, out_path: str):
    csv_path = Path(results_csv)
    if not csv_path.exists():
        return

    with csv_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return

    class_names = [row["class_name"] for row in rows]
    x = np.arange(len(rows))
    metrics = {
        "Precision": [float(row["precision"]) for row in rows],
        "Recall": [float(row["recall"]) for row in rows],
        "mAP50": [float(row["map50"]) for row in rows],
        "mAP50-95": [float(row["map50_95"]) for row in rows],
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=160)
    for ax, (title, values) in zip(axes.flat, metrics.items()):
        ax.plot(x, values, color="tab:blue", linewidth=2.0, marker="o", markersize=5)
        ax.set_title(title)
        ax.set_xlabel("Class")
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=30, ha="right")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)
