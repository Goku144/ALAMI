from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parents[1]
DEFAULT_OUTPUT = PROJECT_DIR / "public" / "checkpoints" / "img" / "comparison.png"


@dataclass
class TrainingPoint:
    iteration: int
    total_iterations: int
    loss: float
    accuracy: float
    confidence: float


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def parse_matrix_row(line: str) -> list[int] | None:
    if "[" not in line or "]" not in line:
        return None

    start = line.rfind("[")
    end = line.rfind("]")
    if end <= start:
        return None

    values = [int(value) for value in re.findall(r"-?\d+", line[start + 1 : end])]
    if len(values) < 10:
        return None

    return values[:10]


def parse_confusion_matrix(text: str, marker: str) -> np.ndarray:
    marker_index = text.rfind(marker)
    if marker_index == -1:
        raise ValueError(f"Could not find marker: {marker}")

    rows: list[list[int]] = []
    for line in text[marker_index:].splitlines()[1:]:
        row = parse_matrix_row(line)
        if row is None:
            if rows:
                break
            continue

        rows.append(row)
        if len(rows) == 10:
            break

    if len(rows) != 10:
        raise ValueError(f"Could not parse a 10x10 confusion matrix after: {marker}")

    return np.asarray(rows, dtype=np.int64)


def parse_training_points(text: str) -> list[TrainingPoint]:
    pattern = re.compile(
        r"TRAIN iter=(\d+)/(\d+) loss=([0-9.eE+-]+) "
        r"accuracy=([0-9.eE+-]+)% confidence=([0-9.eE+-]+)%"
    )

    points: list[TrainingPoint] = []
    for match in pattern.finditer(text):
        points.append(
            TrainingPoint(
                iteration=int(match.group(1)),
                total_iterations=int(match.group(2)),
                loss=float(match.group(3)),
                accuracy=float(match.group(4)),
                confidence=float(match.group(5)),
            )
        )

    return points


def accuracy(matrix: np.ndarray) -> float:
    total = matrix.sum()
    if total == 0:
        return 0.0
    return float(np.trace(matrix) / total)


def per_class_recall(matrix: np.ndarray) -> np.ndarray:
    totals = matrix.sum(axis=1)
    return np.divide(
        np.diag(matrix),
        totals,
        out=np.zeros(matrix.shape[0], dtype=np.float64),
        where=totals != 0,
    )


def per_class_precision(matrix: np.ndarray) -> np.ndarray:
    totals = matrix.sum(axis=0)
    return np.divide(
        np.diag(matrix),
        totals,
        out=np.zeros(matrix.shape[0], dtype=np.float64),
        where=totals != 0,
    )


def macro_f1(matrix: np.ndarray) -> float:
    precision = per_class_precision(matrix)
    recall = per_class_recall(matrix)
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros(matrix.shape[0], dtype=np.float64),
        where=(precision + recall) != 0,
    )
    return float(f1.mean())


def print_report(dl_matrix: np.ndarray, ml_matrix: np.ndarray) -> None:
    print("Model comparison")
    print("----------------")
    print(f"DL accuracy : {accuracy(dl_matrix):.4f}")
    print(f"ML accuracy : {accuracy(ml_matrix):.4f}")
    print(f"DL macro F1 : {macro_f1(dl_matrix):.4f}")
    print(f"ML macro F1 : {macro_f1(ml_matrix):.4f}")
    print()
    print("Per-class recall")
    print("class    DL      ML      diff")
    for label, dl_score, ml_score in zip(
        range(10), per_class_recall(dl_matrix), per_class_recall(ml_matrix)
    ):
        print(f"{label:>5}  {dl_score:0.4f}  {ml_score:0.4f}  {dl_score - ml_score:+0.4f}")


def plot_comparison(
    dl_matrix: np.ndarray,
    ml_matrix: np.ndarray,
    training_points: list[TrainingPoint],
    output: Path,
) -> None:
    import matplotlib.pyplot as plt

    labels = np.arange(10)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

    for ax, title, matrix in (
        (axes[0, 0], "DL confusion matrix", dl_matrix),
        (axes[0, 1], "ML confusion matrix", ml_matrix),
    ):
        image = ax.imshow(matrix, cmap="Blues")
        ax.set_title(f"{title} | accuracy={accuracy(matrix):.4f}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks(labels)
        ax.set_yticks(labels)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    width = 0.38
    axes[1, 0].bar(labels - width / 2, per_class_recall(dl_matrix), width, label="DL")
    axes[1, 0].bar(labels + width / 2, per_class_recall(ml_matrix), width, label="ML")
    axes[1, 0].set_title("Per-class recall")
    axes[1, 0].set_xlabel("Class")
    axes[1, 0].set_ylim(0.0, 1.05)
    axes[1, 0].set_xticks(labels)
    axes[1, 0].legend()

    ax = axes[1, 1]
    if training_points:
        iterations = [point.iteration for point in training_points]
        losses = [point.loss for point in training_points]
        accuracies = [point.accuracy for point in training_points]

        ax.plot(iterations, losses, label="DL batch loss")
        ax.set_title("DL training progress")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")

        ax_accuracy = ax.twinx()
        ax_accuracy.plot(iterations, accuracies, color="tab:green", label="DL batch accuracy")
        ax_accuracy.set_ylabel("Batch accuracy %")

        lines, names = ax.get_legend_handles_labels()
        more_lines, more_names = ax_accuracy.get_legend_handles_labels()
        ax.legend(lines + more_lines, names + more_names, loc="best")
    else:
        names = ["DL accuracy", "ML accuracy", "DL macro F1", "ML macro F1"]
        values = [
            accuracy(dl_matrix),
            accuracy(ml_matrix),
            macro_f1(dl_matrix),
            macro_f1(ml_matrix),
        ]
        ax.bar(names, values, color=["tab:blue", "tab:orange", "tab:blue", "tab:orange"])
        ax.set_title("Summary")
        ax.set_ylim(0.0, 1.05)
        ax.tick_params(axis="x", rotation=20)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare existing DL and ML confusion matrices without training a model."
    )
    parser.add_argument("--dl-log", type=Path, required=True, help="Text log from the C++ DL run.")
    parser.add_argument("--ml-log", type=Path, required=True, help="Text output from app/src/ml.py.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="PNG chart output path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dl_text = read_text(args.dl_log)
    ml_text = read_text(args.ml_log)

    dl_matrix = parse_confusion_matrix(dl_text, "CONFUSION MATRIX")
    ml_matrix = parse_confusion_matrix(ml_text, "Confusion matrix")
    training_points = parse_training_points(dl_text)

    print_report(dl_matrix, ml_matrix)
    plot_comparison(dl_matrix, ml_matrix, training_points, args.output)
    print(f"\nSaved comparison chart to: {args.output}")


if __name__ == "__main__":
    main()
