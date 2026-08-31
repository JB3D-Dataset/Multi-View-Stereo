from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib
import os
from pathlib import Path
import struct
import sys
from typing import Any, BinaryIO, Sequence

import cv2
import numpy as np


PINHOLE_CAMERA_MODEL_SPECS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
}
PINHOLE_CAMERA_MODEL_PARAM_COUNTS = {
    model_name: param_count
    for model_name, param_count in PINHOLE_CAMERA_MODEL_SPECS.values()
}


@dataclass(frozen=True)
class ColmapCamera:
    camera_id: int
    model: str
    width: int
    height: int
    params: np.ndarray

    def pinhole_parameters(self) -> tuple[float, float, float, float]:
        if self.model == "SIMPLE_PINHOLE":
            focal_length, principal_x, principal_y = self.params[:3]
            return float(focal_length), float(focal_length), float(principal_x), float(principal_y)
        if self.model == "PINHOLE":
            focal_x, focal_y, principal_x, principal_y = self.params[:4]
            return float(focal_x), float(focal_y), float(principal_x), float(principal_y)
        raise ValueError(
            f"Camera {self.camera_id} uses model '{self.model}'. The COLMAP model must "
            "already be undistorted and use SIMPLE_PINHOLE or PINHOLE cameras."
        )


@dataclass(frozen=True)
class ColmapImage:
    image_id: int
    qvec: np.ndarray
    tvec: np.ndarray
    camera_id: int
    name: str


@dataclass(frozen=True)
class ColmapModel:
    cameras: dict[int, ColmapCamera]
    images: dict[int, ColmapImage]
    storage_format: str


@dataclass(frozen=True)
class RenderSummary:
    rendered_views: int
    depth_maps_written: int
    rgb_maps_written: int
    skipped_views: int


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Render metric ground truth depth maps from a registered point cloud "
            "at every camera pose in an undistorted pinhole COLMAP model."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-transform",
        "--path_to_transformation_matrix",
        type=Path,
        default=None,
        help=(
            "Optional 4x4 matrix mapping COLMAP world coordinates into the "
            "ground truth point cloud frame."
        ),
    )
    parser.add_argument(
        "-p",
        "--polygon_for_crop",
        type=Path,
        default=None,
        help="Optional XY polygon used to crop the ground truth point cloud.",
    )
    parser.add_argument(
        "-vox",
        "--voxel_size",
        type=positive_float,
        default=None,
        help="Optional voxel size in meters for ground truth downsampling.",
    )
    parser.add_argument(
        "--point-size",
        type=positive_float,
        default=4.0,
        help="Open3D point size used for depth rendering.",
    )
    parser.add_argument(
        "--max-images",
        type=positive_int,
        default=None,
        help="Render at most this many registered images.",
    )
    parser.add_argument(
        "--render-rgb",
        action="store_true",
        help="Also write a Turbo-colorized RGB PNG beside each metric TIFF.",
    )
    parser.add_argument(
        "--rgb-depth-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=None,
        help=(
            "Fixed metric depth range for all RGB maps. By default, each view uses "
            "its positive-depth 1st and 99th percentiles."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace depth maps that already exist in the destination.",
    )
    parser.add_argument(
        "colmap_model",
        type=Path,
        help=(
            "Undistorted sparse COLMAP model directory containing SIMPLE_PINHOLE or "
            "PINHOLE cameras and images in .bin or .txt format."
        ),
    )
    parser.add_argument(
        "ground_truth",
        type=Path,
        help="Registered ground truth point cloud readable by Open3D.",
    )
    parser.add_argument(
        "destination",
        type=Path,
        help="Directory for metric float32 TIFF depth maps.",
    )
    return parser


def model_file_paths(model_path: Path) -> tuple[Path, Path, str]:
    binary_cameras = model_path / "cameras.bin"
    binary_images = model_path / "images.bin"
    if binary_cameras.is_file() and binary_images.is_file():
        return binary_cameras, binary_images, "binary"

    text_cameras = model_path / "cameras.txt"
    text_images = model_path / "images.txt"
    if text_cameras.is_file() and text_images.is_file():
        return text_cameras, text_images, "text"

    raise FileNotFoundError(
        f"'{model_path}' must contain cameras.bin and images.bin, or cameras.txt and images.txt"
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.colmap_model.is_dir():
        parser.error(f"COLMAP model directory does not exist: {args.colmap_model}")
    try:
        model_file_paths(args.colmap_model)
    except FileNotFoundError as exc:
        parser.error(str(exc))
    if not args.ground_truth.is_file():
        parser.error(f"Ground truth point cloud does not exist: {args.ground_truth}")
    if args.destination.exists() and not args.destination.is_dir():
        parser.error(f"Destination exists and is not a directory: {args.destination}")
    for label, path in (
        ("Transformation matrix", args.path_to_transformation_matrix),
        ("Crop polygon", args.polygon_for_crop),
    ):
        if path is not None and not path.is_file():
            parser.error(f"{label} does not exist: {path}")
    if args.rgb_depth_range is not None:
        minimum_depth, maximum_depth = args.rgb_depth_range
        if not args.render_rgb:
            parser.error("--rgb-depth-range requires --render-rgb")
        if minimum_depth < 0.0 or maximum_depth <= minimum_depth:
            parser.error("--rgb-depth-range requires 0 <= MIN < MAX")
    return args


def read_next_bytes(
    file: BinaryIO,
    num_bytes: int,
    format_char_sequence: str,
    endian_character: str = "<",
) -> tuple[Any, ...]:
    data = file.read(num_bytes)
    if len(data) != num_bytes:
        raise EOFError(f"Expected {num_bytes} bytes, found {len(data)}")
    return struct.unpack(endian_character + format_char_sequence, data)


def read_c_string(file: BinaryIO) -> str:
    value = bytearray()
    while True:
        character = file.read(1)
        if not character:
            raise EOFError("Unexpected end of file while reading a COLMAP image name")
        if character == b"\x00":
            return value.decode("utf-8")
        value.extend(character)


def read_cameras_binary(path: Path) -> dict[int, ColmapCamera]:
    cameras: dict[int, ColmapCamera] = {}
    with path.open("rb") as file:
        camera_count = read_next_bytes(file, 8, "Q")[0]
        for _ in range(camera_count):
            camera_id, model_id, width, height = read_next_bytes(file, 24, "iiQQ")
            try:
                model_name, param_count = PINHOLE_CAMERA_MODEL_SPECS[model_id]
            except KeyError as exc:
                raise ValueError(
                    f"COLMAP camera model ID {model_id} is not pinhole. The model must "
                    "already be undistorted and use SIMPLE_PINHOLE or PINHOLE cameras."
                ) from exc
            params = np.asarray(
                read_next_bytes(file, 8 * param_count, "d" * param_count),
                dtype=np.float64,
            )
            cameras[camera_id] = ColmapCamera(
                camera_id=camera_id,
                model=model_name,
                width=width,
                height=height,
                params=params,
            )
    return cameras


def read_images_binary(path: Path) -> dict[int, ColmapImage]:
    images: dict[int, ColmapImage] = {}
    with path.open("rb") as file:
        image_count = read_next_bytes(file, 8, "Q")[0]
        for _ in range(image_count):
            properties = read_next_bytes(file, 64, "idddddddi")
            image_id = properties[0]
            image_name = read_c_string(file)
            point_count = read_next_bytes(file, 8, "Q")[0]
            file.seek(24 * point_count, 1)
            images[image_id] = ColmapImage(
                image_id=image_id,
                qvec=np.asarray(properties[1:5], dtype=np.float64),
                tvec=np.asarray(properties[5:8], dtype=np.float64),
                camera_id=properties[8],
                name=image_name,
            )
    return images


def data_lines(path: Path):
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                yield stripped


def read_cameras_text(path: Path) -> dict[int, ColmapCamera]:
    cameras: dict[int, ColmapCamera] = {}
    for line in data_lines(path):
        fields = line.split()
        if len(fields) < 5:
            raise ValueError(f"Malformed COLMAP camera line: {line}")
        camera_id = int(fields[0])
        model_name = fields[1]
        try:
            expected_params = PINHOLE_CAMERA_MODEL_PARAM_COUNTS[model_name]
        except KeyError as exc:
            raise ValueError(
                f"COLMAP camera model '{model_name}' is not pinhole. The model must "
                "already be undistorted and use SIMPLE_PINHOLE or PINHOLE cameras."
            ) from exc
        params = np.asarray(fields[4:], dtype=np.float64)
        if params.size != expected_params:
            raise ValueError(
                f"Camera {camera_id} model {model_name} expects {expected_params} parameters, "
                f"found {params.size}"
            )
        cameras[camera_id] = ColmapCamera(
            camera_id=camera_id,
            model=model_name,
            width=int(fields[2]),
            height=int(fields[3]),
            params=params,
        )
    return cameras


def read_images_text(path: Path) -> dict[int, ColmapImage]:
    images: dict[int, ColmapImage] = {}
    with path.open("r", encoding="utf-8") as file:
        lines = iter(file)
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split(maxsplit=9)
            if len(fields) != 10:
                raise ValueError(f"Malformed COLMAP image line: {stripped}")
            image_id = int(fields[0])
            images[image_id] = ColmapImage(
                image_id=image_id,
                qvec=np.asarray(fields[1:5], dtype=np.float64),
                tvec=np.asarray(fields[5:8], dtype=np.float64),
                camera_id=int(fields[8]),
                name=fields[9],
            )
            next(lines, None)
    return images


def read_colmap_model(model_path: Path) -> ColmapModel:
    cameras_path, images_path, storage_format = model_file_paths(model_path)
    if storage_format == "binary":
        cameras = read_cameras_binary(cameras_path)
        images = read_images_binary(images_path)
    else:
        cameras = read_cameras_text(cameras_path)
        images = read_images_text(images_path)
    if not cameras:
        raise ValueError(f"COLMAP model contains no cameras: {model_path}")
    if not images:
        raise ValueError(f"COLMAP model contains no registered images: {model_path}")
    return ColmapModel(cameras=cameras, images=images, storage_format=storage_format)


def load_transform_4x4(path: Path | None) -> np.ndarray:
    if path is None:
        return np.eye(4, dtype=np.float64)
    transform = np.loadtxt(path, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 transform at '{path}', got {transform.shape}")
    if not np.all(np.isfinite(transform)):
        raise ValueError(f"Transformation matrix contains non-finite values: {path}")
    if not np.allclose(transform[3], (0.0, 0.0, 0.0, 1.0)):
        raise ValueError(f"Transformation matrix must be affine: {path}")
    return transform


def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(qvec)
    if norm == 0.0:
        raise ValueError("COLMAP quaternion must not be zero")
    qw, qx, qy, qz = qvec / norm
    return np.asarray(
        [
            [1 - 2 * qy**2 - 2 * qz**2, 2 * qx * qy - 2 * qw * qz, 2 * qz * qx + 2 * qw * qy],
            [2 * qx * qy + 2 * qw * qz, 1 - 2 * qx**2 - 2 * qz**2, 2 * qy * qz - 2 * qw * qx],
            [2 * qz * qx - 2 * qw * qy, 2 * qy * qz + 2 * qw * qx, 1 - 2 * qx**2 - 2 * qy**2],
        ],
        dtype=np.float64,
    )


def world_to_camera_in_ground_truth_frame(
    image: ColmapImage,
    colmap_to_ground_truth: np.ndarray,
) -> np.ndarray:
    world_to_camera = np.eye(4, dtype=np.float64)
    world_to_camera[:3, :3] = qvec_to_rotmat(image.qvec)
    world_to_camera[:3, 3] = image.tvec
    return world_to_camera @ np.linalg.inv(colmap_to_ground_truth)


def load_crop_polygon(path: Path) -> np.ndarray:
    polygon = np.loadtxt(path, dtype=np.float64)
    polygon = np.atleast_2d(polygon)
    if polygon.shape[0] < 3 or polygon.shape[1] < 2:
        raise ValueError(f"Crop polygon must contain at least three XY points: {path}")
    polygon = polygon[:, :2]
    if np.allclose(polygon[0], polygon[-1]):
        polygon = polygon[:-1]
    if polygon.shape[0] < 3:
        raise ValueError(f"Crop polygon must contain at least three distinct XY points: {path}")
    return polygon


def points_inside_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    point_x = points[:, 0]
    point_y = points[:, 1]
    inside = np.zeros(points.shape[0], dtype=bool)
    previous_x, previous_y = polygon[-1]
    for current_x, current_y in polygon:
        if current_y != previous_y:
            crosses_y = (current_y > point_y) != (previous_y > point_y)
            edge_x = (
                (previous_x - current_x)
                * (point_y - current_y)
                / (previous_y - current_y)
                + current_x
            )
            inside ^= crosses_y & (point_x < edge_x)
        previous_x, previous_y = current_x, current_y
    return inside


def crop_point_cloud(point_cloud: Any, polygon: np.ndarray, chunk_size: int = 1_000_000):
    points = np.asarray(point_cloud.points)
    selected_chunks = []
    for start in range(0, points.shape[0], chunk_size):
        stop = min(start + chunk_size, points.shape[0])
        local_indices = np.flatnonzero(points_inside_polygon(points[start:stop, :2], polygon))
        selected_chunks.append(local_indices + start)
    if not selected_chunks:
        return point_cloud.select_by_index([])
    selected_indices = np.concatenate(selected_chunks)
    return point_cloud.select_by_index(selected_indices)


def depth_destination(root: Path, image_name: str) -> Path:
    relative_image = Path(image_name)
    if relative_image.is_absolute() or ".." in relative_image.parts:
        raise ValueError(f"Unsafe COLMAP image name: {image_name}")
    return root / relative_image.with_suffix(".tiff")


def rgb_destination(root: Path, image_name: str) -> Path:
    relative_image = Path(image_name)
    if relative_image.is_absolute() or ".." in relative_image.parts:
        raise ValueError(f"Unsafe COLMAP image name: {image_name}")
    return root / relative_image.with_name(f"{relative_image.stem}_turbo.png")


def colorize_depth_turbo(
    depth: np.ndarray,
    depth_range: Sequence[float] | None = None,
) -> np.ndarray:
    clean_depth = np.nan_to_num(
        depth.astype(np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    valid = clean_depth > 0.0
    if not np.any(valid):
        return np.zeros((*clean_depth.shape, 3), dtype=np.uint8)

    if depth_range is None:
        minimum_depth, maximum_depth = np.percentile(clean_depth[valid], (1.0, 99.0))
        if maximum_depth <= minimum_depth:
            minimum_depth = float(np.min(clean_depth[valid]))
            maximum_depth = float(np.max(clean_depth[valid]))
    else:
        minimum_depth, maximum_depth = depth_range

    if maximum_depth <= minimum_depth:
        scaled = np.zeros(clean_depth.shape, dtype=np.uint8)
    else:
        normalized = np.clip(
            (clean_depth - minimum_depth) / (maximum_depth - minimum_depth),
            0.0,
            1.0,
        )
        scaled = np.rint(normalized * 255.0).astype(np.uint8)
    colored = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
    colored[~valid] = 0
    return colored


def load_open3d():
    os.environ.setdefault("EGL_PLATFORM", "surfaceless")
    try:
        return importlib.import_module("open3d")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Open3D is required for depth rendering. Install it with 'pip install open3d'."
        ) from exc


class DepthRenderer:
    def __init__(
        self,
        open3d: Any,
        point_cloud: Any,
        width: int,
        height: int,
        point_size: float,
    ) -> None:
        self.open3d = open3d
        self.width = width
        self.height = height
        try:
            self.renderer = open3d.visualization.rendering.OffscreenRenderer(width, height)
        except Exception as exc:
            raise RuntimeError("Open3D could not create an offscreen rendering context") from exc
        self.renderer.scene.set_background(np.zeros(4, dtype=np.float32))
        material = open3d.visualization.rendering.MaterialRecord()
        material.shader = "defaultUnlit"
        material.point_size = point_size
        self.renderer.scene.add_geometry("ground_truth", point_cloud, material)

    def render(self, camera: ColmapCamera, extrinsic: np.ndarray) -> np.ndarray:
        focal_x, focal_y, principal_x, principal_y = camera.pinhole_parameters()
        intrinsic = np.asarray(
            [
                [focal_x, 0.0, principal_x],
                [0.0, focal_y, principal_y],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        self.renderer.setup_camera(intrinsic, extrinsic, camera.width, camera.height)
        depth = np.asarray(
            self.renderer.render_to_depth_image(z_in_view_space=True),
            dtype=np.float32,
        ).copy()
        if depth.shape != (self.height, self.width):
            raise RuntimeError(
                f"Rendered depth has shape {depth.shape}, expected {(self.height, self.width)}"
            )
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        return depth

    def close(self) -> None:
        self.renderer.scene.clear_geometry()


def validate_colmap_model(model: ColmapModel) -> tuple[int, int]:
    resolutions = set()
    for image in model.images.values():
        try:
            camera = model.cameras[image.camera_id]
        except KeyError as exc:
            raise ValueError(
                f"Image {image.image_id} references missing camera {image.camera_id}"
            ) from exc
        camera.pinhole_parameters()
        resolutions.add((camera.width, camera.height))
    if len(resolutions) != 1:
        formatted = ", ".join(f"{width}x{height}" for width, height in sorted(resolutions))
        raise ValueError(f"All cameras must share one image resolution; found {formatted}")
    return next(iter(resolutions))


def print_header(args: argparse.Namespace, model: ColmapModel, image_count: int) -> None:
    print("################################################################################")
    print(f"# > COLMAP model: {args.colmap_model} ({model.storage_format})")
    print(f"# > Ground truth: {args.ground_truth}")
    print(f"# > Destination: {args.destination}")
    print(f"# > Registered images: {image_count}")
    if args.path_to_transformation_matrix is not None:
        print(f"# > Transformation matrix: {args.path_to_transformation_matrix}")
    if args.polygon_for_crop is not None:
        print(f"# > Crop polygon: {args.polygon_for_crop}")
    if args.voxel_size is not None:
        print(f"# > Voxel size: {args.voxel_size}")
    print(f"# > Point size: {args.point_size}")
    print("# > Camera geometry: undistorted pinhole")
    if args.render_rgb:
        color_range = args.rgb_depth_range or "per-view 1st-99th percentiles"
        print(f"# > Turbo RGB maps: {color_range}")
    print("# -------------------------------------------------------------------------------")


def prepare_point_cloud(open3d: Any, args: argparse.Namespace):
    point_cloud = open3d.io.read_point_cloud(str(args.ground_truth))
    if point_cloud.is_empty():
        raise ValueError(f"Ground truth point cloud is empty: {args.ground_truth}")
    if args.polygon_for_crop is not None:
        point_cloud = crop_point_cloud(
            point_cloud,
            load_crop_polygon(args.polygon_for_crop),
        )
    if args.voxel_size is not None:
        point_cloud = point_cloud.voxel_down_sample(args.voxel_size)
    if point_cloud.is_empty():
        raise ValueError("Ground truth point cloud is empty after preprocessing")
    return point_cloud


def render_depth_maps(args: argparse.Namespace) -> RenderSummary:
    model = read_colmap_model(args.colmap_model)
    width, height = validate_colmap_model(model)
    images = sorted(model.images.values(), key=lambda image: (image.name, image.image_id))
    if args.max_images is not None:
        images = images[: args.max_images]
    print_header(args, model, len(images))

    depth_destinations = [depth_destination(args.destination, image.name) for image in images]
    if len(set(depth_destinations)) != len(depth_destinations):
        raise ValueError("COLMAP image names map to duplicate depth map destinations")
    rgb_destinations = [
        rgb_destination(args.destination, image.name) if args.render_rgb else None
        for image in images
    ]
    concrete_rgb_destinations = [path for path in rgb_destinations if path is not None]
    if len(set(concrete_rgb_destinations)) != len(concrete_rgb_destinations):
        raise ValueError("COLMAP image names map to duplicate RGB depth map destinations")
    pending = [
        (image, depth_path, rgb_path)
        for image, depth_path, rgb_path in zip(
            images,
            depth_destinations,
            rgb_destinations,
        )
        if args.overwrite
        or not depth_path.exists()
        or (rgb_path is not None and not rgb_path.exists())
    ]
    skipped = len(images) - len(pending)
    if not pending:
        return RenderSummary(0, 0, 0, skipped)

    open3d = load_open3d()
    open3d.utility.set_verbosity_level(open3d.utility.VerbosityLevel.Error)
    point_cloud = prepare_point_cloud(open3d, args)
    colmap_to_ground_truth = load_transform_4x4(args.path_to_transformation_matrix)
    args.destination.mkdir(parents=True, exist_ok=True)

    renderer = DepthRenderer(open3d, point_cloud, width, height, args.point_size)
    depth_maps_written = 0
    rgb_maps_written = 0
    try:
        for index, (image, depth_path, rgb_path) in enumerate(pending, start=1):
            print(f"> Rendering image {index} / {len(pending)}", end="\r", flush=True)
            camera = model.cameras[image.camera_id]
            extrinsic = world_to_camera_in_ground_truth_frame(
                image,
                colmap_to_ground_truth,
            )
            depth = renderer.render(camera, extrinsic)
            depth_path.parent.mkdir(parents=True, exist_ok=True)
            if args.overwrite or not depth_path.exists():
                if not cv2.imwrite(str(depth_path), depth):
                    raise RuntimeError(f"Could not write depth map: {depth_path}")
                depth_maps_written += 1
            if rgb_path is not None and (args.overwrite or not rgb_path.exists()):
                rgb_path.parent.mkdir(parents=True, exist_ok=True)
                rgb_depth = colorize_depth_turbo(depth, args.rgb_depth_range)
                if not cv2.imwrite(str(rgb_path), rgb_depth):
                    raise RuntimeError(f"Could not write RGB depth map: {rgb_path}")
                rgb_maps_written += 1
    finally:
        renderer.close()
    print("")
    return RenderSummary(
        rendered_views=len(pending),
        depth_maps_written=depth_maps_written,
        rgb_maps_written=rgb_maps_written,
        skipped_views=skipped,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = render_depth_maps(args)
    except (EOFError, OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Rendered views: {summary.rendered_views}")
    print(f"Metric depth maps written: {summary.depth_maps_written}")
    if args.render_rgb:
        print(f"Turbo RGB maps written: {summary.rgb_maps_written}")
    print(f"Skipped complete views: {summary.skipped_views}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())