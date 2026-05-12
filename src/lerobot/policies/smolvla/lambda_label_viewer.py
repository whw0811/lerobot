from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot.datasets import LeRobotDataset

from .lambda_labels import LambdaLabelLookup, load_lambda_sidecar


def _scalar_int(value: Any) -> int:
    if torch.is_tensor(value):
        return int(value.item())
    if isinstance(value, np.ndarray):
        return int(value.item())
    return int(value)


def to_hwc_uint8(image: Any) -> np.ndarray:
    array = image.detach().cpu().numpy() if torch.is_tensor(image) else np.asarray(image)

    if array.ndim != 3:
        raise ValueError(f"Expected image with 3 dimensions, got shape {array.shape}")

    if array.shape[0] in {1, 3, 4} and array.shape[-1] not in {1, 3, 4}:
        array = np.transpose(array, (1, 2, 0))

    if np.issubdtype(array.dtype, np.floating):
        array = np.clip(array, 0.0, 1.0) * 255.0
    else:
        array = np.clip(array, 0, 255)

    return array.astype(np.uint8)


def format_lambda_line(episode: int, frame: int, index: int, lookup: LambdaLabelLookup) -> str:
    values, valid = lookup.lookup(torch.tensor([index], dtype=torch.long))
    return (
        f"ep={episode} frame={frame} index={index} "
        f"lambda={float(values[0].item()):.6f} valid={bool(valid[0].item())}"
    )


def overlay_text(image: np.ndarray, text: str) -> np.ndarray:
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "'cv2' is required but not installed. Install it with: "
            "pip install 'lerobot[opencv]' (or uv pip install 'lerobot[opencv]')"
        ) from exc

    annotated = image.copy()
    height, width = annotated.shape[:2]
    font_scale = 0.38
    thickness = 1
    padding = 5
    parts = text.split()
    lambda_parts = [part for part in parts if part.startswith("lambda=")]
    metadata = " ".join(part for part in parts if not part.startswith("lambda="))
    lines = [metadata, *lambda_parts] if lambda_parts else [text]
    line_sizes = [
        cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness) for line in lines
    ]
    line_height = max(size[0][1] + size[1] for size in line_sizes)
    box_width = min(width, max(size[0][0] for size in line_sizes) + 2 * padding)
    box_height = min(height, line_height * len(lines) + 2 * padding)
    background = annotated.copy()
    cv2.rectangle(background, (0, 0), (box_width, box_height), (0, 0, 0), thickness=-1)
    cv2.addWeighted(background, 0.55, annotated, 0.45, 0.0, annotated)
    for line_idx, line in enumerate(lines):
        text_size, _baseline = line_sizes[line_idx]
        y = padding + text_size[1] + line_height * line_idx
        cv2.putText(
            annotated,
            line,
            (padding, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
    return annotated


def _choose_camera_key(dataset: LeRobotDataset, camera_key: str | None) -> str:
    if camera_key is not None:
        if camera_key not in dataset.meta.camera_keys:
            raise ValueError(
                f"Camera key '{camera_key}' is not in dataset cameras: {dataset.meta.camera_keys}"
            )
        return camera_key
    if not dataset.meta.camera_keys:
        raise ValueError("Dataset has no camera keys to display")
    return dataset.meta.camera_keys[0]


def view_lambda_labels(
    repo_id: str,
    labels_path: str | Path,
    episode: int,
    root: str | Path | None = None,
    camera_key: str | None = None,
    fps: float | None = None,
    default_value: float = 0.0,
    print_every: int = 1,
    output_video: str | Path | None = None,
) -> None:
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "'cv2' is required but not installed. Install it with: "
            "pip install 'lerobot[opencv]' (or uv pip install 'lerobot[opencv]')"
        ) from exc

    dataset = LeRobotDataset(repo_id, root=root, episodes=[episode])
    selected_camera = _choose_camera_key(dataset, camera_key)
    lookup = LambdaLabelLookup(load_lambda_sidecar(labels_path), default_value=default_value)
    playback_fps = float(fps if fps is not None else dataset.fps)
    delay_s = 0.0 if playback_fps <= 0 else 1.0 / playback_fps
    writer = None
    output_path = Path(output_video) if output_video is not None else None

    window_name = f"{repo_id} episode {episode} lambda"
    try:
        for frame in range(dataset.num_frames):
            start_t = time.perf_counter()
            item = dataset.get_raw_item(frame)
            index = _scalar_int(item["index"])
            line = format_lambda_line(episode=episode, frame=frame, index=index, lookup=lookup)
            if print_every > 0 and frame % print_every == 0:
                print(line, flush=True)

            image = to_hwc_uint8(item[selected_camera])
            display_image = overlay_text(image, line)
            bgr_image = cv2.cvtColor(display_image, cv2.COLOR_RGB2BGR)

            if output_path is not None:
                if writer is None:
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    height, width = bgr_image.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(str(output_path), fourcc, playback_fps, (width, height))
                    if not writer.isOpened():
                        raise RuntimeError(f"Failed to open video writer for {output_path}")
                writer.write(bgr_image)
            else:
                cv2.imshow(window_name, bgr_image)
                key = cv2.waitKey(1)
                if key in {ord("q"), 27}:
                    break

                elapsed_s = time.perf_counter() - start_t
                if delay_s > elapsed_s:
                    time.sleep(delay_s - elapsed_s)
    finally:
        if writer is not None:
            writer.release()
        if output_path is None:
            cv2.destroyWindow(window_name)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Play a LeRobot episode while printing SmolVLA lambda labels.")
    parser.add_argument("--repo-id", required=True, help="Dataset repository id, e.g. HuggingFaceVLA/libero.")
    parser.add_argument("--root", default=None, help="Local dataset root. Defaults to the LeRobot cache.")
    parser.add_argument("--labels-path", required=True, help="Path to lambda_labels.pt.")
    parser.add_argument("--episode", type=int, required=True, help="Episode index to visualize.")
    parser.add_argument("--camera-key", default=None, help="Camera key to display. Defaults to first camera.")
    parser.add_argument("--fps", type=float, default=None, help="Playback fps. Defaults to dataset fps.")
    parser.add_argument("--default-value", type=float, default=0.0, help="Lambda value for missing labels.")
    parser.add_argument("--print-every", type=int, default=1, help="Print every N frames. Use 1 for every frame.")
    parser.add_argument(
        "--output-video",
        default=None,
        help="Write an annotated video to this path instead of opening an OpenCV display window.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    view_lambda_labels(
        repo_id=args.repo_id,
        root=args.root,
        labels_path=args.labels_path,
        episode=args.episode,
        camera_key=args.camera_key,
        fps=args.fps,
        default_value=args.default_value,
        print_every=args.print_every,
        output_video=args.output_video,
    )


if __name__ == "__main__":
    main()
