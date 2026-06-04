from __future__ import annotations

import argparse
import csv
from pathlib import Path

import joblib
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parents[1]

META_DIR = PROJECT_DIR / "public" / "target" / "meta"


def resolve_image_path(filepath: str) -> Path:
    """Resolve a filepath from the CSV to an actual image path."""
    candidates = [
        META_DIR / filepath,
        BASE_DIR / filepath,
        Path(filepath),
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(f"Could not find image for CSV filepath: {filepath}")


def load_image_as_features(image_path: Path) -> np.ndarray:
    image = Image.open(image_path).convert("L")

    if image.size != (28, 28):
        image = image.resize((28, 28))

    pixels = np.asarray(image, dtype=np.float32) / 255.0
    return pixels.reshape(-1)


def load_dataset(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    features = []
    labels = []

    with csv_path.open(newline="") as file:
        reader = csv.DictReader(file)
        required_columns = {"filepath", "label"}
        missing_columns = required_columns.difference(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(
                f"{csv_path} is missing required columns: {sorted(missing_columns)}"
            )

        for row in reader:
            image_path = resolve_image_path(row["filepath"])
            features.append(load_image_as_features(image_path))
            labels.append(int(row["label"]))

    return np.vstack(features), np.asarray(labels, dtype=np.int64)


def print_metrics(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    print(f"\n{name} metrics")
    print("-" * (len(name) + 8))
    print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")

    print("Confusion matrix")
    print(confusion_matrix(y_true, y_pred))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a Random Forest classifier on MNIST PNG data."
    )
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=META_DIR / "train.csv",
        help="Path to the training CSV with filepath,label columns.",
    )
    parser.add_argument(
        "--test-csv",
        type=Path,
        default=META_DIR / "test.csv",
        help="Path to the evaluation CSV with filepath,label columns.",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=PROJECT_DIR / "public" / "checkpoints" / "random_forest_mnist.joblib",
        help="Where to save the trained model.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=200,
        help="Number of trees in the random forest.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model = None
    if args.model_output.exists():
        print(f"Loading trained model from: {args.model_output}")
        model = joblib.load(args.model_output)

    print("Loading and preparing images...")
    if model is None:
        x_train, y_train = load_dataset(args.train_csv)
    else:
        x_train = y_train = None
    x_test, y_test = load_dataset(args.test_csv)

    if x_train is not None:
        print(f"Training samples  : {x_train.shape[0]}")
    else:
        print("Training samples  : skipped, loaded checkpoint")
    print(f"Test samples      : {x_test.shape[0]}")
    print(f"Features per image: {x_test.shape[1]}")

    if model is None:
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        )

        print("\nTraining Random Forest classifier...")
        model.fit(x_train, y_train)

        args.model_output.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, args.model_output)
        print(f"\nSaved trained model to: {args.model_output}")
    else:
        print("\nUsing saved Random Forest classifier.")

    test_predictions = model.predict(x_test)
    print_metrics("Test", y_test, test_predictions)


if __name__ == "__main__":
    main()
