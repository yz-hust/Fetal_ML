"""Run batch inference with a trained fetal disease classifier.

The default files are resolved relative to this script, so this directory can be
copied and used on another computer. Only load pickle model files that come from
a trusted source: unpickling an untrusted file can execute arbitrary code.
"""

from __future__ import annotations

import argparse
import pickle
import re
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_PATH = SCRIPT_DIR / "test.xlsx"
DEFAULT_OUTPUT_PATH = SCRIPT_DIR / "inference_predictions.csv"
DEFAULT_MODEL_VERSION = "Model_2"
MODEL_PATHS = {
    "Model_1": SCRIPT_DIR / "Model_1.pkl",
    "Model_2": SCRIPT_DIR / "Model_2.pkl",
}
TRUE_LABEL_COLUMN = "CoA/VSD Group"
CLASS_NAME_MAP = {0: "VSD", 1: "CoA/VSD"}


class InferenceError(RuntimeError):
    """Raised when the input data or model cannot be used for inference."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load an Excel file and a trusted scikit-learn model, then output "
            "the predicted class and class probabilities for every sample."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Input Excel file (default: test.xlsx beside this script).",
    )
    parser.add_argument(
        "--model-version",
        choices=tuple(MODEL_PATHS),
        default=DEFAULT_MODEL_VERSION,
        help=(
            "Bundled model to use when --model is not supplied "
            "(default: Model_2). Model_1 is the original parameter model; "
            "Model_2 is the new-parameter model."
        ),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help=(
            "Path to a trusted pickle model. If supplied, this takes precedence "
            "over --model-version."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=(
            "Output CSV file (default: inference_predictions.csv beside this "
            "script)."
        ),
    )
    return parser.parse_args(argv)


def resolve_model_path(
    explicit_model_path: Path | None, model_version: str
) -> Path:
    """Return an explicit model path or the selected bundled model path."""
    if explicit_model_path is not None:
        return explicit_model_path
    return MODEL_PATHS[model_version]


def _require_file(path: Path, description: str) -> None:
    if not path.exists():
        raise InferenceError(f"{description} does not exist: {path}")
    if not path.is_file():
        raise InferenceError(f"{description} is not a file: {path}")


def load_input_data(path: Path) -> pd.DataFrame:
    _require_file(path, "Input file")
    try:
        data = pd.read_excel(path)
    except Exception as exc:
        raise InferenceError(f"Failed to read input Excel file '{path}': {exc}") from exc

    if data.empty:
        raise InferenceError(f"Input Excel file contains no samples: {path}")
    return data


def load_model(path: Path) -> Any:
    _require_file(path, "Model file")
    try:
        with path.open("rb") as model_file:
            model = pickle.load(model_file)
    except Exception as exc:
        raise InferenceError(f"Failed to load model '{path}': {exc}") from exc

    missing_methods = [
        method
        for method in ("predict", "predict_proba")
        if not callable(getattr(model, method, None))
    ]
    if missing_methods:
        raise InferenceError(
            "Loaded model does not provide required method(s): "
            + ", ".join(missing_methods)
        )
    if not hasattr(model, "feature_names_in_"):
        raise InferenceError(
            "Loaded model has no feature_names_in_; the required feature order cannot be determined safely."
        )
    if not hasattr(model, "classes_"):
        raise InferenceError(
            "Loaded model has no classes_; probability columns cannot be mapped to classes safely."
        )
    return model


def prepare_features(data: pd.DataFrame, model: Any) -> pd.DataFrame:
    feature_names = [str(name) for name in model.feature_names_in_]
    if not feature_names:
        raise InferenceError("The model contains an empty feature list.")
    if len(feature_names) != len(set(feature_names)):
        raise InferenceError("The model contains duplicate feature names.")

    missing_features = [name for name in feature_names if name not in data.columns]
    if missing_features:
        raise InferenceError(
            "Input data is missing required feature column(s): "
            + ", ".join(missing_features)
        )

    duplicated_features = [
        name for name in feature_names if list(data.columns).count(name) > 1
    ]
    if duplicated_features:
        raise InferenceError(
            "Input data contains duplicate required feature column(s): "
            + ", ".join(duplicated_features)
        )

    features = data.loc[:, feature_names].copy()
    null_columns = features.columns[features.isna().any()].tolist()
    if null_columns:
        raise InferenceError(
            "Required feature column(s) contain missing values: " + ", ".join(null_columns)
        )

    non_numeric_columns = [
        name for name in feature_names if not is_numeric_dtype(features[name])
    ]
    if non_numeric_columns:
        raise InferenceError(
            "Required feature column(s) must be numeric: " + ", ".join(non_numeric_columns)
        )

    feature_values = features.to_numpy(dtype=float)
    if not np.isfinite(feature_values).all():
        invalid_columns = features.columns[~np.isfinite(feature_values).all(axis=0)].tolist()
        raise InferenceError(
            "Required feature column(s) contain infinite values: " + ", ".join(invalid_columns)
        )
    return features


def _python_scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def class_display_name(class_value: Any) -> str:
    class_value = _python_scalar(class_value)
    return CLASS_NAME_MAP.get(class_value, str(class_value))


def _probability_column_name(class_value: Any) -> str:
    display_name = class_display_name(class_value)
    safe_name = re.sub(r"[^0-9A-Za-z]+", "_", display_name).strip("_")
    if not safe_name:
        safe_name = str(_python_scalar(class_value))
    return f"Probability_{safe_name}"


def build_prediction_results(
    data: pd.DataFrame, model: Any, features: pd.DataFrame
) -> pd.DataFrame:
    predictions = np.asarray(model.predict(features))
    probabilities = np.asarray(model.predict_proba(features))
    classes = np.asarray(model.classes_)
    sample_count = len(features)

    if predictions.ndim != 1 or len(predictions) != sample_count:
        raise InferenceError("Model returned an unexpected prediction shape.")
    if probabilities.shape != (sample_count, len(classes)):
        raise InferenceError(
            "Model probability output shape does not match the sample count and model classes."
        )
    if not np.isfinite(probabilities).all():
        raise InferenceError("Model returned non-finite probability values.")
    if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=1e-6, atol=1e-8):
        raise InferenceError("Model probability values do not sum to 1 for every sample.")

    result_data: dict[str, Any] = {
        "Sample_Number": np.arange(1, sample_count + 1),
    }
    if TRUE_LABEL_COLUMN in data.columns:
        result_data["True_Label"] = data[TRUE_LABEL_COLUMN].to_numpy()

    result_data["Predicted_Class"] = predictions
    result_data["Predicted_Class_Name"] = [
        class_display_name(value) for value in predictions
    ]

    used_probability_columns: set[str] = set()
    for class_index, class_value in enumerate(classes):
        column_name = _probability_column_name(class_value)
        if column_name in used_probability_columns:
            column_name = f"{column_name}_Class_{class_index}"
        used_probability_columns.add(column_name)
        result_data[column_name] = probabilities[:, class_index]

    return pd.DataFrame(result_data)


def run_inference(
    input_path: Path, model_path: Path, output_path: Path
) -> pd.DataFrame:
    data = load_input_data(input_path)
    model = load_model(model_path)
    features = prepare_features(data, model)
    results = build_prediction_results(data, model, features)

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(output_path, index=False, encoding="utf-8-sig")
    except Exception as exc:
        raise InferenceError(f"Failed to write output CSV '{output_path}': {exc}") from exc

    return results


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    model_path = resolve_model_path(args.model, args.model_version)
    try:
        results = run_inference(args.input, model_path, args.output)
    except InferenceError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(results.to_string(index=False))
    print(f"\nModel loaded from: {model_path}")
    print(f"\nPredictions saved to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
