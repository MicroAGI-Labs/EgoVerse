"""Convert V2 common-format MCAP captures to EgoVerse Zarr v3 episodes.

Mirrors the structure of `aria_to_zarr.py` so the rest of the EgoVerse
toolchain (ray_helper, run_conversion, etc.) can dispatch this converter
the same way.

Usage::

    python -m egomimic.scripts.mcap_process.mcap_to_zarr \\
        --raw-path /path/to/episode.mcap \\
        --output-dir /path/to/zarr_out \\
        --fps 30 --arm both \\
        --task-name "demo" --task-description "..." \\
        --episode-hash my_episode

The microagi-schemas submodule must be checked out — it is registered as
an optional submodule (`update = none`) and not auto-fetched. Initialise
it explicitly::

    git submodule update --init -- third_party/microagi-schemas
"""

import argparse
import gc
import logging
import traceback
from pathlib import Path

import numpy as np

from egomimic.rldb.zarr.validate import log_validation
from egomimic.rldb.zarr.zarr_writer import ZarrWriter
from egomimic.scripts.mcap_process.mcap_v2_extractor import MCAPV2Extractor
from egomimic.utils.aws.aws_sql import timestamp_ms_to_episode_hash
from egomimic.utils.egomimicUtils import str2bool
from egomimic.utils.video_utils import save_preview_mp4

logger = logging.getLogger(__name__)


_ARM_TO_EMBODIMENT = {
    "both": "microagi_bimanual",
    "left": "microagi_left_arm",
    "right": "microagi_right_arm",
}


class DatasetConverter:
    """Convert V2 common-format MCAP files to EgoVerse Zarr episodes."""

    def __init__(
        self,
        raw_path: Path | str,
        fps: int,
        arm: str = "both",
        save_mp4: bool = False,
        validate: bool = True,
        debug: bool = False,
    ):
        self.raw_path = raw_path if isinstance(raw_path, Path) else Path(raw_path)
        self.fps = fps
        self.arm = arm
        self.save_mp4 = save_mp4
        self.validate = validate
        self.debug = debug

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(
                logging.Formatter("%(asctime)s - [%(name)s] - %(message)s")
            )
            self.logger.addHandler(console_handler)

        if arm not in _ARM_TO_EMBODIMENT:
            raise ValueError(
                f"--arm must be one of {list(_ARM_TO_EMBODIMENT)}, got {arm!r}"
            )
        self.embodiment = _ARM_TO_EMBODIMENT[arm]

        if self.raw_path.is_dir():
            self.episode_list = sorted(self.raw_path.glob("*.mcap"))
        else:
            self.episode_list = [self.raw_path]

        self.logger.info(f"V2 MCAP → Zarr converter")
        self.logger.info(f"  raw_path:   {self.raw_path}")
        self.logger.info(f"  episodes:   {len(self.episode_list)}")
        self.logger.info(f"  fps:        {self.fps}")
        self.logger.info(f"  arm:        {self.arm}  (embodiment={self.embodiment})")
        self.logger.info(f"  save_mp4:   {self.save_mp4}")
        self.logger.info(f"  validate:   {self.validate}")

    def extract_episode(
        self,
        episode_path: Path,
        task_name: str = "",
        task_description: str = "",
        output_dir: Path = Path("."),
        dataset_name: str = "",
        chunk_timesteps: int = 100,
    ) -> tuple[Path, Path | None]:
        episode_feats, metadata, annotations = MCAPV2Extractor.process_episode(
            episode_path=episode_path,
            arm=self.arm,
        )

        # Episode hash convention (CONTRIBUTING_DATA.md §3): the UTC wall-clock
        # at the start of recording, rendered YYYY-MM-DD-HH-MM-SS-ffffff. Derive
        # it from the first color timestamp unless explicitly overridden.
        if not dataset_name:
            ts0_ms = int(episode_feats["obs_rgb_timestamps_ns"][0]) // 1_000_000
            dataset_name = timestamp_ms_to_episode_hash(ts0_ms)
            self.logger.info(f"Derived episode hash from recording start: {dataset_name}")

        numeric_data: dict[str, np.ndarray] = {}
        image_data: dict[str, np.ndarray] = {}
        for key, value in episode_feats.items():
            if key.startswith("images."):
                image_data[key] = value
            else:
                numeric_data[key] = value

        # mecka/scale-style attrs: per-episode `intrinsics`, task/objects, and
        # camera/operator metadata extracted from the MCAP, plus the episode id.
        metadata = dict(metadata)
        metadata["episode_id"] = dataset_name
        # Prefer the task title from the MCAP as the episode's task_name.
        effective_task_name = metadata.get("task") or task_name

        zarr_path = ZarrWriter.create_and_write(
            episode_path=output_dir / f"{dataset_name}.zarr",
            numeric_data=numeric_data if numeric_data else None,
            image_data=image_data if image_data else None,
            fps=self.fps,
            embodiment=self.embodiment,
            task_name=effective_task_name,
            task_description=task_description,
            annotations=annotations if annotations else None,
            chunk_timesteps=chunk_timesteps,
            metadata_override=metadata,
        )

        # Validate against the format contract (CONTRIBUTING_DATA.md §10.1)
        # before the episode is considered done. A malformed store should fail
        # loudly here rather than be uploaded.
        if self.validate:
            errors = log_validation(zarr_path, self.logger)
            if errors:
                raise ValueError(
                    f"{zarr_path}: failed validation with {len(errors)} error(s); "
                    f"first: {errors[0]}"
                )

        mp4_path: Path | None = None
        if self.save_mp4 and "images.front_1" in image_data:
            mp4_path = output_dir / f"{dataset_name}.mp4"
            images_tchw = np.asarray(image_data["images.front_1"]).transpose(0, 3, 1, 2)
            save_preview_mp4(images_tchw, mp4_path, self.fps, half_res=False)
        return zarr_path, mp4_path


def main(args) -> tuple[Path, Path | None] | None:
    """Convert a V2 MCAP file (or directory of them) into Zarr episode(s)."""
    try:
        raw_path = Path(args.raw_path)
        output_dir = Path(args.output_dir) if args.output_dir else raw_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        converter = DatasetConverter(
            raw_path=raw_path,
            fps=args.fps,
            arm=args.arm,
            save_mp4=args.save_mp4,
            validate=args.validate,
            debug=args.debug,
        )

        if not converter.episode_list:
            raise FileNotFoundError(f"No .mcap files found at {raw_path}")

        results: list[tuple[Path, Path | None]] = []
        for ep_path in converter.episode_list:
            # None → extract_episode derives the hash from the recording start
            # timestamp per CONTRIBUTING_DATA.md §3.
            episode_hash = args.episode_hash
            gc.collect()
            zarr_path, mp4_path = converter.extract_episode(
                episode_path=ep_path,
                task_name=args.task_name,
                task_description=args.task_description,
                output_dir=output_dir,
                dataset_name=episode_hash,
                chunk_timesteps=args.chunk_timesteps,
            )
            results.append((zarr_path, mp4_path))

        return results[0] if len(results) == 1 else results
    except Exception:
        logger.error(
            "Error converting %s:\n%s", args.raw_path, traceback.format_exc()
        )
        return None


def argument_parse():
    parser = argparse.ArgumentParser(
        description="Convert V2 common-format MCAP to EgoVerse Zarr."
    )
    parser.add_argument(
        "--raw-path",
        type=Path,
        required=True,
        help="Path to a .mcap file or a directory containing .mcap files.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        required=True,
        help="Capture frame rate (must match actual color/0 rate).",
    )
    parser.add_argument(
        "--task-name",
        type=str,
        default="microagi_v2",
        help="Task name attribute on the produced episode.",
    )
    parser.add_argument(
        "--task-description",
        type=str,
        default="",
        help="Free-text task description.",
    )
    parser.add_argument(
        "--arm",
        type=str,
        choices=["left", "right", "both"],
        default="both",
        help="Which hand topics to emit.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where the .zarr episode is written. Defaults to raw_path's parent.",
    )
    parser.add_argument(
        "--episode-hash",
        type=str,
        default=None,
        help="Override episode-hash (zarr filename stem). Defaults to the UTC "
        "recording-start timestamp (YYYY-MM-DD-HH-MM-SS-ffffff) per "
        "CONTRIBUTING_DATA.md §3.",
    )
    parser.add_argument(
        "--chunk-timesteps",
        type=int,
        default=100,
        help="Numeric-array chunk size in frames (matches CONTRIBUTING_DATA.md §5.4).",
    )
    parser.add_argument(
        "--save-mp4",
        type=str2bool,
        default=False,
        help="If true, write a side-car MP4 preview alongside the Zarr episode.",
    )
    parser.add_argument(
        "--validate",
        type=str2bool,
        default=True,
        help="If true (default), run the CONTRIBUTING_DATA.md §10.1 format checks "
        "on each written episode and fail if any error is found.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Process only the first episode for debugging.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = argument_parse()
    result = main(args)
    print(result)
