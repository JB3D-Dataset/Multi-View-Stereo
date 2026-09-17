from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Sequence

import cv2
import numpy as np


ABSOLUTE_THRESHOLDS = (1.0, 0.5, 0.25, 0.10, 0.05, 0.01)
RELATIVE_THRESHOLDS = (1.25, 1.20, 1.15, 1.10, 1.05, 1.01)
DEPTH_SUFFIXES = {".tif", ".tiff"}


@dataclass(frozen=True)
class DepthMetrics:
    l1: float
    accuracy: tuple[float, ...]
    completeness: tuple[float, ...]

    def values(self) -> tuple[float, ...]:
        values = [self.l1]
        for accuracy, completeness in zip(self.accuracy, self.completeness):
            values.extend((accuracy, completeness))
        return tuple(values)


@dataclass(frozen=True)
class EvaluationResult:
    filename: str
    metrics: DepthMetrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate estimated metric depth maps against ground truth.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-ns",
        "--noscale",
        "--median-scale",
        dest="median_scale",
        action="store_true",
        help=(
            "Median-scale each estimate using pixels with valid estimated and "
            "ground truth depth."
        ),
    )
    parser.add_argument(
        "-abs",
        "--absolute_error",
        "--absolute-error",
        dest="absolute_error",
        action="store_true",
        help="Use metric absolute error instead of relative error.",
    )
    parser.add_argument(
        "--resize-estimate",
        action="store_true",
        help=(
            "Resize estimates to the ground truth resolution with nearest-neighbor "
            "sampling. Without this option, dimensions must match exactly."
        ),
    )
    parser.add_argument(
        "--error-map-directory",
        "--error-map-dir",
        type=Path,
        default=None,
        help=(
            "Write Turbo color-coded PNG error maps to this directory. Directory input "
            "paths are preserved relative to the estimate root."
        ),
    )
    parser.add_argument(
        "--error-map-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=None,
        help=(
            "Fixed absolute or relative error range used for all Turbo maps. By "
            "default, each map uses zero to its 99th-percentile valid error."
        ),
    )
    parser.add_argument(
        "estimate",
        type=Path,
        help="Estimated TIFF depth map or directory of TIFF depth maps.",
    )
    parser.add_argument(
        "ground_truth",
        type=Path,
        help="Ground truth TIFF depth map or directory of TIFF depth maps.",
    )
    return parser


def is_depth_map(path: Path) -> bool:
    return path.suffix.lower() in DEPTH_SUFFIXES


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.estimate.exists():
        parser.error(f"Estimate path does not exist: {args.estimate}")
    if not args.ground_truth.exists():
        parser.error(f"Ground truth path does not exist: {args.ground_truth}")

    both_directories = args.estimate.is_dir() and args.ground_truth.is_dir()
    both_files = args.estimate.is_file() and args.ground_truth.is_file()
    if not both_directories and not both_files:
        parser.error("Estimate and ground truth must both be files or both be directories")
    if both_files and (
        not is_depth_map(args.estimate) or not is_depth_map(args.ground_truth)
    ):
        parser.error("Single-file inputs must use the .tif or .tiff extension")
    if args.error_map_directory is not None:
        if args.error_map_directory.exists() and not args.error_map_directory.is_dir():
            parser.error(
                f"Error-map destination exists and is not a directory: "
                f"{args.error_map_directory}"
            )
    if args.error_map_range is not None:
        minimum_error, maximum_error = args.error_map_range
        if args.error_map_directory is None:
            parser.error("--error-map-range requires --error-map-directory")
        if (
            not np.isfinite(minimum_error)
            or not np.isfinite(maximum_error)
            or minimum_error < 0.0
            or maximum_error <= minimum_error
        ):
            parser.error("--error-map-range requires 0 <= MIN < MAX")
    return args


def read_depth_map(path: Path) -> np.ndarray:
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise ValueError(f"Could not read depth map: {path}")
    if depth.ndim != 2:
        raise ValueError(
            f"Depth map must have one channel, found shape {depth.shape}: {path}"
        )

    depth = np.asarray(depth, dtype=np.float64)
    depth[~np.isfinite(depth) | (depth <= 0.0)] = 0.0
    return depth


def load_depth_pair(
    estimate_path: Path,
    ground_truth_path: Path,
    resize_estimate: bool,
) -> tuple[np.ndarray, np.ndarray]:
    estimate = read_depth_map(estimate_path)
    ground_truth = read_depth_map(ground_truth_path)
    if estimate.shape == ground_truth.shape:
        return estimate, ground_truth
    if not resize_estimate:
        raise ValueError(
            f"Depth map dimensions differ for '{estimate_path.name}': estimate "
            f"{estimate.shape[1]}x{estimate.shape[0]}, ground truth "
            f"{ground_truth.shape[1]}x{ground_truth.shape[0]}. Use --resize-estimate "
            "only when both maps represent the same camera geometry."
        )
    resized = cv2.resize(
        estimate,
        (ground_truth.shape[1], ground_truth.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    return resized, ground_truth


def median_scale_estimate(estimate: np.ndarray, ground_truth: np.ndarray) -> np.ndarray:
    overlap = (estimate > 0.0) & (ground_truth > 0.0)
    if not np.any(overlap):
        raise ValueError("Cannot median-scale depth maps without overlapping valid pixels")

    scale = float(np.median(ground_truth[overlap]) / np.median(estimate[overlap]))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"Median scaling produced an invalid scale factor: {scale}")
    scaled = estimate.copy()
    scaled[scaled > 0.0] *= scale
    return scaled


def valid_pixel_counts(
    estimate: np.ndarray,
    ground_truth: np.ndarray,
) -> tuple[np.ndarray, int, int]:
    estimate_valid = estimate > 0.0
    ground_truth_valid = ground_truth > 0.0
    overlap = estimate_valid & ground_truth_valid
    estimate_count = int(np.count_nonzero(estimate_valid))
    ground_truth_count = int(np.count_nonzero(ground_truth_valid))
    overlap_count = int(np.count_nonzero(overlap))

    if estimate_count == 0:
        raise ValueError("Estimate contains no valid positive depth pixels")
    if ground_truth_count == 0:
        raise ValueError("Ground truth contains no valid positive depth pixels")
    if overlap_count == 0:
        raise ValueError("Estimate and ground truth have no overlapping valid pixels")
    return overlap, estimate_count, ground_truth_count


def colorize_error_turbo(
    estimate: np.ndarray,
    ground_truth: np.ndarray,
    absolute_error: bool,
    error_range: Sequence[float] | None = None,
) -> np.ndarray:
    overlap, _, _ = valid_pixel_counts(estimate, ground_truth)
    estimated_depth = estimate[overlap]
    ground_truth_depth = ground_truth[overlap]
    errors = np.abs(estimated_depth - ground_truth_depth)
    if not absolute_error:
        errors /= ground_truth_depth

    if error_range is None:
        minimum_error = 0.0
        maximum_error = float(np.percentile(errors, 99.0))
    else:
        minimum_error, maximum_error = error_range

    scaled = np.zeros(estimate.shape, dtype=np.uint8)
    if maximum_error > minimum_error:
        normalized = np.clip(
            (errors - minimum_error) / (maximum_error - minimum_error),
            0.0,
            1.0,
        )
        scaled[overlap] = np.rint(normalized * 255.0).astype(np.uint8)
    colored = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
    colored[~overlap] = 0
    return colored


def error_map_destination(root: Path, filename: str) -> Path:
    relative_path = Path(filename)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"Unsafe error-map filename: {filename}")
    return root / relative_path.with_name(f"{relative_path.stem}_error_turbo.png")


def write_error_map(
    destination: Path,
    estimate: np.ndarray,
    ground_truth: np.ndarray,
    absolute_error: bool,
    error_range: Sequence[float] | None,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    colored = colorize_error_turbo(
        estimate,
        ground_truth,
        absolute_error,
        error_range,
    )
    if not cv2.imwrite(str(destination), colored):
        raise RuntimeError(f"Could not write Turbo error map: {destination}")


def threshold_metrics(
    errors: np.ndarray,
    thresholds: tuple[float, ...],
    estimate_count: int,
    ground_truth_count: int,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    inlier_counts = [
        int(np.count_nonzero(errors < threshold)) for threshold in thresholds
    ]
    accuracy = tuple(count / estimate_count for count in inlier_counts)
    completeness = tuple(count / ground_truth_count for count in inlier_counts)
    return accuracy, completeness


def compute_absolute_metrics(
    estimate: np.ndarray,
    ground_truth: np.ndarray,
) -> DepthMetrics:
    overlap, estimate_count, ground_truth_count = valid_pixel_counts(
        estimate,
        ground_truth,
    )
    errors = np.abs(estimate[overlap] - ground_truth[overlap])
    accuracy, completeness = threshold_metrics(
        errors,
        ABSOLUTE_THRESHOLDS,
        estimate_count,
        ground_truth_count,
    )
    return DepthMetrics(float(np.mean(errors)), accuracy, completeness)


def compute_relative_metrics(
    estimate: np.ndarray,
    ground_truth: np.ndarray,
) -> DepthMetrics:
    overlap, estimate_count, ground_truth_count = valid_pixel_counts(
        estimate,
        ground_truth,
    )
    estimated_depth = estimate[overlap]
    ground_truth_depth = ground_truth[overlap]
    relative_errors = np.abs(estimated_depth - ground_truth_depth) / ground_truth_depth
    ratios = np.maximum(
        estimated_depth / ground_truth_depth,
        ground_truth_depth / estimated_depth,
    )
    accuracy, completeness = threshold_metrics(
        ratios,
        RELATIVE_THRESHOLDS,
        estimate_count,
        ground_truth_count,
    )
    return DepthMetrics(float(np.mean(relative_errors)), accuracy, completeness)


def evaluate_file(
    estimate_path: Path,
    ground_truth_path: Path,
    filename: str,
    median_scale: bool,
    absolute_error: bool,
    resize_estimate: bool,
    error_map_path: Path | None = None,
    error_map_range: Sequence[float] | None = None,
) -> EvaluationResult:
    estimate, ground_truth = load_depth_pair(
        estimate_path,
        ground_truth_path,
        resize_estimate,
    )
    if median_scale:
        estimate = median_scale_estimate(estimate, ground_truth)
    if error_map_path is not None:
        write_error_map(
            error_map_path,
            estimate,
            ground_truth,
            absolute_error,
            error_map_range,
        )
    metrics = (
        compute_absolute_metrics(estimate, ground_truth)
        if absolute_error
        else compute_relative_metrics(estimate, ground_truth)
    )
    return EvaluationResult(filename, metrics)


def find_depth_maps(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and is_depth_map(path)
    )


def evaluate_directory(args: argparse.Namespace) -> list[EvaluationResult]:
    estimate_paths = find_depth_maps(args.estimate)
    if not estimate_paths:
        raise ValueError(f"Estimate directory contains no TIFF depth maps: {args.estimate}")

    results = []
    for index, estimate_path in enumerate(estimate_paths, start=1):
        print(
            f"> Processing file {index} / {len(estimate_paths)}",
            end="\r",
            flush=True,
        )
        relative_path = estimate_path.relative_to(args.estimate)
        ground_truth_path = args.ground_truth / relative_path
        if not ground_truth_path.is_file():
            raise ValueError(f"Ground truth depth map not found: {ground_truth_path}")
        error_map_path = (
            error_map_destination(args.error_map_directory, relative_path.as_posix())
            if args.error_map_directory is not None
            else None
        )
        results.append(
            evaluate_file(
                estimate_path,
                ground_truth_path,
                relative_path.as_posix(),
                args.median_scale,
                args.absolute_error,
                args.resize_estimate,
                error_map_path,
                args.error_map_range,
            )
        )
    print("")
    return results


def print_header(args: argparse.Namespace) -> None:
    input_mode = "directories" if args.estimate.is_dir() else "files"
    error_measure = "absolute" if args.absolute_error else "relative"
    print("################################################################################")
    print(f"# > Input mode: {input_mode}")
    print(f"# > Estimate: {args.estimate}")
    print(f"# > Ground truth: {args.ground_truth}")
    print(f"# > Median scaling: {args.median_scale}")
    print(f"# > Error measure: {error_measure}")
    print(f"# > Resize estimates: {args.resize_estimate}")
    if args.error_map_directory is not None:
        error_range = args.error_map_range or "0 to per-map 99th percentile"
        print(f"# > Turbo error maps: {args.error_map_directory}")
        print(f"# > Error-map range: {error_range}")
    print("# -------------------------------------------------------------------------------")


def result_labels(absolute_error: bool) -> list[str]:
    thresholds = ABSOLUTE_THRESHOLDS if absolute_error else RELATIVE_THRESHOLDS
    labels = ["L1-abs" if absolute_error else "L1-rel"]
    for threshold in thresholds:
        labels.extend((f"Acc_{threshold}", f"Cpl_{threshold}"))
    return labels


def format_metric(value: float) -> str:
    return f"{value:.10g}"


def print_results(results: list[EvaluationResult], absolute_error: bool) -> None:
    print("# " + ";".join(("Filename", *result_labels(absolute_error))))
    for result in results:
        values = (format_metric(value) for value in result.metrics.values())
        print(";".join((result.filename, *values)))

    if len(results) > 1:
        mean_values = np.mean(
            np.asarray(
                [result.metrics.values() for result in results],
                dtype=np.float64,
            ),
            axis=0,
        )
        print("# -------------------------------------------------------------------------------")
        print(";".join(("Mean", *(format_metric(value) for value in mean_values))))


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    print_header(args)
    try:
        if args.estimate.is_dir():
            results = evaluate_directory(args)
        else:
            results = [
                evaluate_file(
                    args.estimate,
                    args.ground_truth,
                    args.estimate.name,
                    args.median_scale,
                    args.absolute_error,
                    args.resize_estimate,
                    (
                        error_map_destination(
                            args.error_map_directory,
                            args.estimate.name,
                        )
                        if args.error_map_directory is not None
                        else None
                    ),
                    args.error_map_range,
                )
            ]
        print_results(results, args.absolute_error)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())