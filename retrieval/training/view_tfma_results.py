"""Load and print TFMA Evaluator results for the latest local pipeline run."""

from __future__ import annotations

import argparse
import glob
import os
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tensorflow as tf
import tensorflow_model_analysis as tfma
from tensorflow_model_analysis.proto import validation_result_pb2

from retrieval import config


def latest_subdir(parent: Path) -> Path:
    """Return the latest run directory, preferring numeric directory names."""
    subdirs = [Path(d) for d in glob.glob(str(parent / "*")) if os.path.isdir(d)]
    if not subdirs:
        raise FileNotFoundError(f"No subdirectories found under: {parent}")

    numeric = []
    for subdir in subdirs:
        if subdir.name.isdigit():
            numeric.append((int(subdir.name), subdir))
    if numeric:
        return sorted(numeric, key=lambda item: item[0])[-1][1]
    return sorted(subdirs)[-1]


def flatten_metrics(node, prefix: str = "") -> dict[str, float]:
    """Flatten nested TFMA metric dicts into name -> float values."""
    flat: dict[str, float] = {}
    if isinstance(node, dict):
        if "doubleValue" in node:
            flat[prefix.strip("/") if prefix else "value"] = float(node["doubleValue"])
            return flat
        for key, value in node.items():
            child_prefix = f"{prefix}/{key}" if prefix else str(key)
            flat.update(flatten_metrics(value, child_prefix))
        return flat
    if isinstance(node, (int, float)):
        flat[prefix.strip("/") if prefix else "value"] = float(node)
    return flat


def overall_slice_metrics(eval_result):
    """Return metrics for the overall slice."""
    for slice_key, metrics_dict in eval_result.slicing_metrics:
        if slice_key == () or slice_key == ((),):
            return metrics_dict
    if eval_result.slicing_metrics:
        return eval_result.slicing_metrics[0][1]
    return None


def extract_slice_columns(slice_key) -> set[str]:
    """Extract slice column names from a TFMA slice key payload."""
    columns: set[str] = set()

    def walk(node) -> None:
        if isinstance(node, tuple):
            if len(node) == 2 and isinstance(node[0], str):
                columns.add(node[0])
                return
            for item in node:
                walk(item)
        elif isinstance(node, list):
            for item in node:
                walk(item)
        elif isinstance(node, dict):
            for key, value in node.items():
                if isinstance(key, str):
                    columns.add(key)
                walk(value)

    walk(slice_key)
    return columns


def available_slice_columns(eval_result) -> list[str]:
    columns: set[str] = set()
    for slice_key, _ in eval_result.slicing_metrics:
        columns.update(extract_slice_columns(slice_key))
    return sorted(column for column in columns if column)


def has_renderable_plot_data(eval_result) -> bool:
    for _, plot_payload in getattr(eval_result, "plots", []) or []:
        if plot_payload:
            return True
    return False


def render_plot_if_available(eval_result) -> None:
    print("\nRendering TFMA plot view when plot data is available...")
    if not has_renderable_plot_data(eval_result):
        print("Skipping TFMA plot view: this evaluation contains metrics but no plot payloads.")
        return
    try:
        tfma.view.render_plot(eval_result)
    except Exception as exc:
        print(f"Could not render TFMA plot in this environment: {exc}")


def is_delta_metric(metric_name: str) -> bool:
    leaf = metric_name.lower().rsplit("/", 1)[-1]
    return leaf.endswith(("_diff", "_change", "_delta")) or leaf in {
        "diff",
        "change",
        "delta",
    }


def delta_candidates(metric_name: str) -> list[str]:
    if "/" in metric_name:
        prefix, leaf = metric_name.rsplit("/", 1)
        return [
            f"{metric_name}_diff",
            f"{prefix}/{leaf}_diff",
            f"{metric_name}/diff",
            f"{metric_name}_change",
            f"{metric_name}_delta",
        ]
    return [
        f"{metric_name}_diff",
        f"{metric_name}/diff",
        f"{metric_name}_change",
        f"{metric_name}_delta",
    ]


def find_delta(metric_name: str, flat_metrics: dict[str, float]) -> float | None:
    lower_to_key = {key.lower(): key for key in flat_metrics}
    for candidate in delta_candidates(metric_name):
        key = lower_to_key.get(candidate.lower())
        if key is not None:
            return flat_metrics[key]
    return None


def print_baseline_comparison(eval_result) -> None:
    """Print current, inferred baseline, and delta metrics for the overall slice."""
    overall_metrics = overall_slice_metrics(eval_result)
    print("\n" + "=" * 80)
    print("CURRENT VS BASELINE METRICS (overall slice)")
    print("=" * 80)
    if overall_metrics is None:
        print("No overall-slice metrics available.")
        return

    flat = flatten_metrics(overall_metrics)
    metric_items = {key: value for key, value in flat.items() if not is_delta_metric(key)}
    if not metric_items:
        print("No candidate metric values found in metrics payload.")
        return

    print("Baseline values are inferred as current - delta when TFMA provides *_diff metrics.")
    print(f"{'Metric':<38} {'Current':>12} {'Baseline':>12} {'Delta':>12}")
    print(f"{'-' * 38} {'-' * 12} {'-' * 12} {'-' * 12}")
    for name, current_value in sorted(metric_items.items()):
        delta = find_delta(name, flat)
        if delta is None:
            print(f"{name:<38} {current_value:>12.6f} {'n/a':>12} {'n/a':>12}")
        else:
            print(
                f"{name:<38} {current_value:>12.6f} "
                f"{current_value - delta:>12.6f} {delta:>12.6f}"
            )


def slice_key_to_text(slice_key) -> str:
    """Convert TFMA single_slice_keys proto to readable text across proto versions."""
    parts = []
    for item in slice_key.single_slice_keys:
        column = ""
        value = "overall"
        for field_desc, field_val in item.ListFields():
            if field_desc.name == "column":
                column = field_val
            elif isinstance(field_val, bytes):
                value = field_val.decode("utf-8", errors="replace")
            else:
                value = str(field_val)
        parts.append(f"{column}={value}" if column else str(value))
    return ", ".join(parts) if parts else "overall"


def print_validation_failures(eval_dir: Path) -> None:
    """Parse validations.tfrecord and print failing slices/metrics."""
    val_path = eval_dir / "validations.tfrecord"
    print("\n" + "=" * 80)
    print("VALIDATION FAILURES")
    print("=" * 80)
    if not val_path.exists():
        print(f"No validations.tfrecord found at: {val_path}")
        return

    validation_result = validation_result_pb2.ValidationResult()
    for raw in tf.data.TFRecordDataset(str(val_path)).take(1):
        validation_result.ParseFromString(raw.numpy())

    print(f"validation_ok: {validation_result.validation_ok}")
    print(f"failed_slices: {len(validation_result.metric_validations_per_slice)}")
    if validation_result.validation_ok:
        print("No validation failures detected.")
        return

    failure_counter = Counter()
    rows = []
    for metrics_per_slice in validation_result.metric_validations_per_slice:
        slice_text = slice_key_to_text(metrics_per_slice.slice_key)
        metric_names = []
        for failure in metrics_per_slice.failures:
            metric = failure.metric_key.name
            if failure.metric_key.is_diff:
                metric = f"{metric}_diff"
            metric_names.append(metric)
            failure_counter[metric] += 1
        rows.append((slice_text, ", ".join(metric_names)))

    print(f"{'Slice':<55} {'Failed metrics':<30}")
    print(f"{'-' * 55} {'-' * 30}")
    for slice_text, metrics_text in rows:
        print(f"{slice_text[:55]:<55} {metrics_text[:30]:<30}")

    print("\nFailure counts by metric:")
    for metric, count in sorted(failure_counter.items()):
        print(f"  {metric}: {count}")


def view_tfma_results(evaluator_root: Path) -> None:
    """Load TFMA eval result and print/render evaluation summaries."""
    eval_dir = latest_subdir(evaluator_root / "evaluation")
    bless_dir = latest_subdir(evaluator_root / "blessing")

    print("=" * 80)
    print("TFMA EVALUATOR RESULTS")
    print("=" * 80)
    print(f"Evaluation dir: {eval_dir}")
    print(f"Blessing dir  : {bless_dir}")
    if (bless_dir / "BLESSED").exists():
        print("Model blessing: BLESSED")
    elif (bless_dir / "NOT_BLESSED").exists():
        print("Model blessing: NOT_BLESSED")
    else:
        print("Model blessing: Unknown")

    eval_result = tfma.load_eval_result(output_path=str(eval_dir))

    print("\nRendering TFMA slicing metrics (all slices)...")
    tfma.view.render_slicing_metrics(eval_result)

    columns = set(available_slice_columns(eval_result))
    print(f"\nAvailable slice columns: {sorted(columns) if columns else 'none'}")
    for column in ("gender", "occupation"):
        if column in columns:
            print(f"\nRendering TFMA slicing metrics by {column}...")
            tfma.view.render_slicing_metrics(eval_result, slicing_column=column)
        else:
            print(f"\nSkipping '{column}' slice view: no slices found for this column.")

    render_plot_if_available(eval_result)
    print_baseline_comparison(eval_result)
    print_validation_failures(eval_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluator-root",
        type=Path,
        default=config.PIPELINE_ROOT / "Evaluator",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.evaluator_root.is_dir():
        raise FileNotFoundError(
            f"Evaluator output not found at {args.evaluator_root}. Run the pipeline first."
        )
    view_tfma_results(args.evaluator_root)


if __name__ == "__main__":
    main()
