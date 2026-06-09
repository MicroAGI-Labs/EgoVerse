"""V2 common-format MCAP → EgoVerse episode-feats extractor.

Reads a single V2 MCAP file (one capture session) and returns an `episode_feats`
dict in the same shape that `aria_to_zarr` consumes:

    episode_feats = {
        "images.front_1":           (T, H, W, 3) uint8,
        "obs_rgb_timestamps_ns":    (T,)  int64,
        "obs_head_pose":            (T, 7) float64,
        "left.obs_wrist_pose":      (T, 7) float64,    # if /hands/left present
        "right.obs_wrist_pose":     (T, 7) float64,    # if /hands/right present
        "left.obs_ee_pose":         (T, 7) float64,    # = wrist pose for tracked hands
        "right.obs_ee_pose":        (T, 7) float64,
        "left.obs_keypoints":       (T, 63) float64,   # 21 * (x,y,z), world frame
        "right.obs_keypoints":      (T, 63) float64,
    }

All poses are in the SLAM world frame (`world`). Pose rows are
`[tx, ty, tz, qw, qx, qy, qz]` per CONTRIBUTING_DATA.md §5.2.

Per V2 spec ("Don't compute for /task/health == False; messages should exist
but can be defaults/0/identity"), frames where `/task/health` is False are
emitted as identity poses with zero translation rather than dropped.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import av
import cv2
import numpy as np
from mcap.reader import make_reader
from scipy.spatial.transform import Rotation

# Resolve microagi-schemas via the optional submodule. Imported lazily so the
# rest of egomimic doesn't fail to import when the schemas aren't checked out.
_SCHEMAS_PATH = (
    Path(__file__).resolve().parents[3] / "third_party" / "microagi-schemas" / "python"
)


def _import_schemas():
    if not _SCHEMAS_PATH.exists():
        raise RuntimeError(
            f"microagi-schemas not found at {_SCHEMAS_PATH}.\n"
            "The submodule is optional and not auto-installed. To enable the\n"
            "MCAP→Zarr converter, run from the EgoVerse repo root:\n\n"
            "    git submodule update --init -- third_party/microagi-schemas\n\n"
            "Or clone it manually:\n"
            "    git clone https://github.com/MicroAGI-Labs/microagi-schemas.git \\\n"
            "        third_party/microagi-schemas"
        )
    if str(_SCHEMAS_PATH) not in sys.path:
        sys.path.insert(0, str(_SCHEMAS_PATH))
    from foxglove import (
        CameraCalibration_pb2,
        CompressedVideo_pb2,
        FrameTransforms_pb2,
        PosesInFrame_pb2,
    )
    from microagi import (
        health_pb2,
        task_pb2,
    )
    return (
        CameraCalibration_pb2,
        CompressedVideo_pb2,
        FrameTransforms_pb2,
        PosesInFrame_pb2,
        health_pb2,
        task_pb2,
    )


# Topic constants (V2 spec)
_TOPIC_COLOR_IMAGE = "/camera/color/0/image"
_TOPIC_COLOR_INFO = "/camera/color/0/info"
_TOPIC_TF_STATIC = "/camera/tf_static"
_TOPIC_SLAM_TF = "/slam/tf"
_TOPIC_HANDS_TF = "/hands/tf"
_TOPIC_HANDS_LEFT = "/hands/left"
_TOPIC_HANDS_RIGHT = "/hands/right"
_TOPIC_TASK_HEALTH = "/task/health"
_TOPIC_TASK = "/task"
_TOPIC_SUBTASK = "/task/subtask"

# Consortium episodes are stored downscaled (aria/scale 640x480, mecka 640x360).
# We match that convention: rectify color frames to this width, preserving aspect
# ratio. World-frame poses/keypoints are resolution-independent and untouched.
_DOWNSCALE_WIDTH = 640

# Canonical rectified MicroAGI color-0 camera, at full sensor resolution. Each
# episode's color frames are undistorted and resampled onto this single fixed
# pinhole (scaled to the stored resolution), so every episode shares one
# distortion-free K regardless of per-unit intrinsic variation — mirroring how
# aria_to_zarr warps to a fixed linear camera. This absorbs the per-episode K/D
# differences into the pixel resampling instead of leaking them downstream.
#
# MUST stay in sync with egomimic.utils.egomimicUtils.MICROAGI_INTRINSICS (which
# is this matrix scaled to the stored resolution); the visualizers project onto
# the stored frames with that scaled K.
_MICROAGI_CANONICAL_SIZE = (1920, 1080)  # (w, h) the canonical K is defined at
_MICROAGI_CANONICAL_K = np.array(
    [
        [1042.562744140625, 0.0, 969.295654296875],
        [0.0, 1042.52001953125, 532.9319458007812],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

# Distortion models the standard OpenCV pinhole undistort path can rectify.
# Fisheye/equidistant need cv2.fisheye and are rejected loudly rather than
# silently mis-rectified.
_PINHOLE_DISTORTION_MODELS = {
    "",
    "none",
    "plumb_bob",
    "radtan",
    "rational_polynomial",
    "opencv",
}

# Parent frame of the per-frame SLAM transform in /slam/tf. The V2 captures
# emit `world → camera`; the child optical frame is composed on top via the
# static `camera → color0` edge.
_SLAM_WORLD_FRAME = "world"

# V2 captures publish `camera` as a REP-103 body frame (x-forward, y-left,
# z-up) and an *identity* `camera→color0` static transform, while the color-0
# intrinsics `K` follow the optical convention (z-forward, x-right, y-down).
# Without correcting for this, world points land at depth ≈ 0 in the stored
# `obs_head_pose` frame and projection explodes. Compose this fixed
# body→optical rotation (R_body_optical = R_optical_body.T) on top of whatever
# `camera→color0` the file provides so `obs_head_pose` is a true optical frame
# and `K @ inv(T_world_color0) @ p_world` projects onto the visible hands.
_T_COLOR0_OPTICAL = np.array(
    [
        [0.0, 0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

_HAND_KEYPOINT_COUNT = 21


def _quat_xyzw_to_wxyz(q_xyzw: np.ndarray) -> np.ndarray:
    """Reorder a (4,) quaternion from foxglove (x,y,z,w) to zarr (w,x,y,z)."""
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float64)


def _identity_pose_row() -> np.ndarray:
    """Return a (7,) zero-translation, identity-rotation pose in zarr layout."""
    return np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)


def _build_rectify_maps(
    K: np.ndarray,
    D: np.ndarray,
    distortion_model: str,
    new_K: np.ndarray,
    out_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Build (map1, map2) that undistort source pixels and reproject onto ``new_K``.

    ``out_size`` is the (w, h) of the rectified output. Source pixels are sampled
    from the full-resolution decoded frame described by (``K``, ``D``,
    ``distortion_model``); the rotation between source and target frames is
    identity, so only intrinsics + distortion change. The maps depend solely on
    these per-episode constants, so they are built once and reused for every
    frame.

    Only standard pinhole / radial-tangential distortion models are supported.
    Fisheye-family models raise ``NotImplementedError`` so a wrong rectification
    fails loudly instead of silently producing misaligned overlays.
    """
    model = (distortion_model or "").lower()
    if model not in _PINHOLE_DISTORTION_MODELS:
        raise NotImplementedError(
            f"Rectification for distortion_model {distortion_model!r} is not "
            "implemented. Extend _build_rectify_maps (e.g. via cv2.fisheye) "
            "before converting captures with this model."
        )
    D = np.asarray(D, dtype=np.float64).reshape(-1)
    if D.size == 0:
        D = np.zeros(5, dtype=np.float64)
    map1, map2 = cv2.initUndistortRectifyMap(
        K, D, np.eye(3), new_K, out_size, cv2.CV_32FC1
    )
    return map1, map2


def _pose_row(translation: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
    """Build a (7,) zarr-layout pose row from translation and (x,y,z,w) quat."""
    qw = _quat_xyzw_to_wxyz(quat_xyzw)
    return np.array(
        [translation[0], translation[1], translation[2], qw[0], qw[1], qw[2], qw[3]],
        dtype=np.float64,
    )


def _transform_to_matrix(translation: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
    """Build a 4x4 homogeneous matrix from translation + (x,y,z,w) quat."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Rotation.from_quat(quat_xyzw).as_matrix()
    T[:3, 3] = translation
    return T


def _matrix_to_pose_row(T: np.ndarray) -> np.ndarray:
    """Decompose a 4x4 homogeneous matrix back into a (7,) zarr pose row."""
    quat_xyzw = Rotation.from_matrix(T[:3, :3]).as_quat()
    return _pose_row(T[:3, 3], quat_xyzw)


def _subtasks_to_annotations(
    subtasks: list[tuple[int, str]], color_ts_ns: list[int]
) -> list[tuple[str, int, int]]:
    """Turn (timestamp_ns, title) subtask markers into (text, start_idx, end_idx).

    Each subtask spans from its own color-frame index up to the next subtask's
    start (or the end of the episode for the last one) — matching the mecka /
    scale ``annotations`` segment convention.
    """
    if not subtasks or not color_ts_ns:
        return []
    subtasks = sorted(subtasks, key=lambda x: x[0])
    ts_arr = np.asarray(color_ts_ns, dtype=np.int64)
    T = len(color_ts_ns)
    starts = [int(np.searchsorted(ts_arr, ts, side="left")) for ts, _ in subtasks]
    out: list[tuple[str, int, int]] = []
    for i, (_ts, title) in enumerate(subtasks):
        s = min(max(starts[i], 0), T)
        e = starts[i + 1] if i + 1 < len(subtasks) else T
        e = min(max(e, s), T)
        out.append((str(title), s, e))
    return out


class MCAPV2Extractor:
    """Streaming reader/extractor for V2 common-format MCAP captures."""

    TAGS = ["microagi", "mcap", "v2"]

    @staticmethod
    def process_episode(
        episode_path: Path | str,
        arm: str,
    ) -> tuple[dict[str, np.ndarray], dict, list[tuple[str, int, int]]]:
        """Extract all features from one V2 MCAP.

        Returns ``(episode_feats, metadata, annotations)``:

        * ``episode_feats`` — the (T, ...) arrays (images, poses, keypoints).
        * ``metadata`` — mecka-style ``zarr.attrs`` payload: ``intrinsics``
          (scaled to the downscaled resolution), ``task``, ``objects``,
          ``duration``, ``source_mcap``.
        * ``annotations`` — ``(text, start_idx, end_idx)`` subtask segments.

        Args:
            episode_path: path to a V2 MCAP file.
            arm: "left", "right", or "both" — selects which hand topics to emit.
        """
        episode_path = Path(episode_path)
        (
            CameraCalibration_pb2,
            CompressedVideo_pb2,
            FrameTransforms_pb2,
            PosesInFrame_pb2,
            health_pb2,
            task_pb2,
        ) = _import_schemas()

        # Pass 1: scan for color timestamps + collect raw payloads of every
        # ancillary topic, keyed by ts_ns. Payload bytes stay tiny per message
        # (a few KB each); only color images are large and are decoded later.
        color_packet_bytes: list[bytes] = []
        color_ts_ns: list[int] = []
        color_format: str | None = None
        slam_tf_by_ts: dict[int, bytes] = {}
        hands_tf_by_ts: dict[int, bytes] = {}
        hands_left_by_ts: dict[int, bytes] = {}
        hands_right_by_ts: dict[int, bytes] = {}
        task_health_by_ts: dict[int, bool] = {}
        color_info_msg = None
        tf_static_msg = None
        task_msg = None
        subtasks: list[tuple[int, str]] = []

        with open(episode_path, "rb") as f:
            reader = make_reader(f)
            for _schema, channel, message in reader.iter_messages():
                topic = channel.topic
                ts_ns = int(message.log_time)
                if topic == _TOPIC_COLOR_IMAGE:
                    cv = CompressedVideo_pb2.CompressedVideo()
                    cv.ParseFromString(message.data)
                    color_packet_bytes.append(bytes(cv.data))
                    color_ts_ns.append(ts_ns)
                    if color_format is None:
                        color_format = str(cv.format)
                elif topic == _TOPIC_COLOR_INFO and color_info_msg is None:
                    color_info_msg = CameraCalibration_pb2.CameraCalibration()
                    color_info_msg.ParseFromString(message.data)
                elif topic == _TOPIC_TF_STATIC and tf_static_msg is None:
                    tf_static_msg = FrameTransforms_pb2.FrameTransforms()
                    tf_static_msg.ParseFromString(message.data)
                elif topic == _TOPIC_SLAM_TF:
                    slam_tf_by_ts[ts_ns] = bytes(message.data)
                elif topic == _TOPIC_HANDS_TF:
                    hands_tf_by_ts[ts_ns] = bytes(message.data)
                elif topic == _TOPIC_HANDS_LEFT:
                    hands_left_by_ts[ts_ns] = bytes(message.data)
                elif topic == _TOPIC_HANDS_RIGHT:
                    hands_right_by_ts[ts_ns] = bytes(message.data)
                elif topic == _TOPIC_TASK_HEALTH:
                    h = health_pb2.Health()
                    h.ParseFromString(message.data)
                    task_health_by_ts[ts_ns] = bool(h.valid)
                elif topic == _TOPIC_TASK and task_msg is None:
                    task_msg = task_pb2.Task()
                    task_msg.ParseFromString(message.data)
                elif topic == _TOPIC_SUBTASK:
                    st = task_pb2.Task()
                    st.ParseFromString(message.data)
                    subtasks.append((ts_ns, str(st.title)))

        if color_info_msg is None:
            raise ValueError(
                f"{episode_path}: missing required {_TOPIC_COLOR_INFO} (CameraCalibration)"
            )
        if tf_static_msg is None:
            raise ValueError(
                f"{episode_path}: missing required {_TOPIC_TF_STATIC}"
            )

        # Compose static camera→color0 so we can re-express the head pose in
        # the optical color frame (matches the aria convention where
        # `obs_head_pose = T_world_color`).
        cam_to_color0 = _find_tf_in_message(
            tf_static_msg, "camera", "color0"
        )
        if cam_to_color0 is None:
            raise ValueError(
                f"{episode_path}: /camera/tf_static missing camera→color0 edge"
            )
        T_camera_color0 = _transform_to_matrix(*cam_to_color0) @ _T_COLOR0_OPTICAL

        # Rectify to the canonical pinhole at the stored resolution. The raw
        # per-episode (K, D) differ across capture units; rather than leak that
        # variation downstream, we undistort + resample every frame onto the one
        # fixed _MICROAGI_CANONICAL_K (scaled to the target resolution). After
        # this, all episodes share a single distortion-free K and project with
        # egomimicUtils.MICROAGI_INTRINSICS, no per-episode patching required.
        orig_w = int(color_info_msg.width)
        orig_h = int(color_info_msg.height)
        target_w = min(_DOWNSCALE_WIDTH, orig_w)
        target_h = int(round(orig_h * target_w / orig_w))

        K_raw = np.array(color_info_msg.K, dtype=np.float64).reshape(3, 3)
        D_raw = np.array(color_info_msg.D, dtype=np.float64)
        distortion_model = str(color_info_msg.distortion_model)

        canon_w, canon_h = _MICROAGI_CANONICAL_SIZE
        K_canon = _MICROAGI_CANONICAL_K.copy()
        K_canon[0, :] *= target_w / canon_w
        K_canon[1, :] *= target_h / canon_h

        rectify_maps = _build_rectify_maps(
            K_raw, D_raw, distortion_model, K_canon, (target_w, target_h)
        )

        intrinsics = {
            # canonical, identical across all rectified episodes
            "K": K_canon.reshape(-1).tolist(),
            "D": [0.0, 0.0, 0.0, 0.0, 0.0],
            "distortion_model": "none",
            "width": target_w,
            "height": target_h,
            "frame_id": str(color_info_msg.frame_id),
            "T_camera_color0": T_camera_color0.tolist(),
            "rectified": True,
            # raw per-episode calibration kept for provenance / re-rectification
            "K_raw": K_raw.reshape(-1).tolist(),
            "D_raw": D_raw.reshape(-1).tolist(),
            "raw_distortion_model": distortion_model,
            "raw_width": orig_w,
            "raw_height": orig_h,
        }

        if not color_ts_ns:
            raise ValueError(f"No /camera/color/0/image messages found in {episode_path}")

        T = len(color_ts_ns)
        rgb_ts_ns = np.array(color_ts_ns, dtype=np.int64)

        # Decode H.264/H.265 → (T, H, W, 3) uint8 using PyAV, rectifying each
        # frame onto the canonical pinhole at the target resolution. Spec keyframe
        # interval is 30 with repeat-headers=1 / annexb=1 / no B-frames, so
        # streaming decode is a one-pass operation in input order. The codec is
        # chosen from the per-stream format field.
        images = MCAPV2Extractor._decode_color(
            color_packet_bytes, T, color_format, rectify_maps=rectify_maps
        )

        # Build per-frame head pose: T_world_color0 = T_world_camera @ T_camera_color0.
        # V2 spec requires per-frame messages aligned exactly to color-0 timestamps,
        # so we expect dict[color_ts] → bytes.
        head_pose = MCAPV2Extractor._build_head_pose(
            color_ts_ns,
            slam_tf_by_ts,
            task_health_by_ts,
            T_camera_color0,
            FrameTransforms_pb2,
        )

        episode_feats: dict[str, np.ndarray] = {
            "images.front_1": images,
            "obs_rgb_timestamps_ns": rgb_ts_ns,
            "obs_head_pose": head_pose,
        }

        has_hands_left = bool(hands_left_by_ts)
        has_hands_right = bool(hands_right_by_ts)
        has_hands_tf = bool(hands_tf_by_ts)

        wants_left = arm in ("left", "both") and has_hands_left and has_hands_tf
        wants_right = arm in ("right", "both") and has_hands_right and has_hands_tf

        if wants_left or wants_right:
            (
                left_wrist_pose,
                right_wrist_pose,
                left_kpts,
                right_kpts,
            ) = MCAPV2Extractor._build_hand_arrays(
                color_ts_ns=color_ts_ns,
                slam_tf_by_ts=slam_tf_by_ts,
                hands_tf_by_ts=hands_tf_by_ts,
                hands_left_by_ts=hands_left_by_ts if wants_left else {},
                hands_right_by_ts=hands_right_by_ts if wants_right else {},
                task_health_by_ts=task_health_by_ts,
                FrameTransforms_pb2=FrameTransforms_pb2,
                PosesInFrame_pb2=PosesInFrame_pb2,
            )
            if wants_left:
                episode_feats["left.obs_wrist_pose"] = left_wrist_pose
                episode_feats["left.obs_ee_pose"] = left_wrist_pose.copy()
                episode_feats["left.obs_keypoints"] = left_kpts
            if wants_right:
                episode_feats["right.obs_wrist_pose"] = right_wrist_pose
                episode_feats["right.obs_ee_pose"] = right_wrist_pose.copy()
                episode_feats["right.obs_keypoints"] = right_kpts

        annotations = _subtasks_to_annotations(subtasks, color_ts_ns)

        # mecka/scale-style metadata payload for zarr.attrs. All fields are
        # optional — only those present in the MCAP are emitted.
        metadata: dict = {
            "intrinsics": intrinsics,
            "source_mcap": episode_path.name,
            "duration": (color_ts_ns[-1] - color_ts_ns[0]) / 1e9,
        }
        if task_msg is not None:
            metadata["task"] = str(task_msg.title)
            metadata["task_success"] = bool(task_msg.success)
            metadata["task_dexterous"] = bool(task_msg.dexterous)
            if task_msg.confidence:
                metadata["task_confidence"] = float(task_msg.confidence)
            if list(task_msg.tools):
                metadata["objects"] = [str(t) for t in task_msg.tools]

        return episode_feats, metadata, annotations

    # Maps the CompressedVideo.format string to the PyAV/ffmpeg codec name.
    _FORMAT_TO_CODEC = {
        "h264": "h264",
        "avc": "h264",
        "h265": "hevc",
        "hevc": "hevc",
    }

    @staticmethod
    def _decode_color(
        packets: list[bytes],
        expected_count: int,
        fmt: str | None,
        rectify_maps: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> np.ndarray:
        """Decode a sequence of H.264/H.265 Annex-B packets into a (T, H, W, 3) RGB uint8 array.

        ZarrWriter encodes images via simplejpeg with `colorspace="RGB"`, and
        every other EgoVerse consumer decodes with the same convention, so
        frames must be RGB at this boundary. If ``rectify_maps=(map1, map2)`` is
        given (from ``_build_rectify_maps``), each frame is undistorted and
        resampled onto the canonical pinhole at the maps' output resolution.
        """
        codec = MCAPV2Extractor._FORMAT_TO_CODEC.get((fmt or "").lower())
        if codec is None:
            raise ValueError(
                f"Unsupported CompressedVideo.format {fmt!r}; "
                f"expected one of {sorted(MCAPV2Extractor._FORMAT_TO_CODEC)}"
            )

        def _maybe_rectify(arr: np.ndarray) -> np.ndarray:
            if rectify_maps is None:
                return arr
            return cv2.remap(
                arr, rectify_maps[0], rectify_maps[1], interpolation=cv2.INTER_LINEAR
            )

        codec_ctx = av.CodecContext.create(codec, "r")
        frames: list[np.ndarray] = []
        for pkt_bytes in packets:
            pkt = av.Packet(pkt_bytes)
            for frame in codec_ctx.decode(pkt):
                frames.append(_maybe_rectify(frame.to_ndarray(format="rgb24")))
        for frame in codec_ctx.decode(None):
            frames.append(_maybe_rectify(frame.to_ndarray(format="rgb24")))

        if len(frames) != expected_count:
            raise ValueError(
                f"H.265 decode produced {len(frames)} frames, expected {expected_count}"
            )
        return np.stack(frames, axis=0)

    @staticmethod
    def _build_head_pose(
        color_ts_ns: list[int],
        slam_tf_by_ts: dict[int, bytes],
        task_health_by_ts: dict[int, bool],
        T_camera_color0: np.ndarray,
        FrameTransforms_pb2,
    ) -> np.ndarray:
        """One (7,) row per color frame: T_world_color0 in zarr layout.

        Composed as `T_world_color0 = T_world_camera @ T_camera_color0`. This
        matches the aria convention where ``obs_head_pose`` is the optical
        color frame in world, so consumers can project a world point onto the
        image as ``K @ inv(T_world_color0) @ p_world``.
        """
        out = np.zeros((len(color_ts_ns), 7), dtype=np.float64)
        for i, ts in enumerate(color_ts_ns):
            if not task_health_by_ts.get(ts, True):
                out[i] = _identity_pose_row()
                continue
            payload = slam_tf_by_ts.get(ts)
            if payload is None:
                out[i] = _identity_pose_row()
                continue
            edge = _find_tf_edge(payload, FrameTransforms_pb2, _SLAM_WORLD_FRAME, "camera")
            if edge is None:
                out[i] = _identity_pose_row()
                continue
            T_world_camera = _transform_to_matrix(*edge)
            T_world_color0 = T_world_camera @ T_camera_color0
            out[i] = _matrix_to_pose_row(T_world_color0)
        return out

    @staticmethod
    def _build_hand_arrays(
        color_ts_ns: list[int],
        slam_tf_by_ts: dict[int, bytes],
        hands_tf_by_ts: dict[int, bytes],
        hands_left_by_ts: dict[int, bytes],
        hands_right_by_ts: dict[int, bytes],
        task_health_by_ts: dict[int, bool],
        FrameTransforms_pb2,
        PosesInFrame_pb2,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Build (T,7) wrist poses and (T,63) keypoints for both hands in world frame."""
        T = len(color_ts_ns)
        left_wrist = np.zeros((T, 7), dtype=np.float64)
        right_wrist = np.zeros((T, 7), dtype=np.float64)
        left_kpts = np.zeros((T, _HAND_KEYPOINT_COUNT * 3), dtype=np.float64)
        right_kpts = np.zeros((T, _HAND_KEYPOINT_COUNT * 3), dtype=np.float64)

        identity_row = _identity_pose_row()

        for i, ts in enumerate(color_ts_ns):
            healthy = task_health_by_ts.get(ts, True)
            slam_payload = slam_tf_by_ts.get(ts)
            hands_tf_payload = hands_tf_by_ts.get(ts)

            if not healthy or slam_payload is None or hands_tf_payload is None:
                left_wrist[i] = identity_row
                right_wrist[i] = identity_row
                continue

            slam_edge = _find_tf_edge(
                slam_payload, FrameTransforms_pb2, _SLAM_WORLD_FRAME, "camera"
            )
            if slam_edge is None:
                left_wrist[i] = identity_row
                right_wrist[i] = identity_row
                continue
            T_world_camera = _transform_to_matrix(*slam_edge)

            for side, hands_dict, wrist_arr, kpts_arr, child_frame in (
                ("left", hands_left_by_ts, left_wrist, left_kpts, "left_wrist"),
                ("right", hands_right_by_ts, right_wrist, right_kpts, "right_wrist"),
            ):
                if not hands_dict:
                    continue
                cam_to_wrist = _find_tf_edge(
                    hands_tf_payload, FrameTransforms_pb2, "camera", child_frame
                )
                if cam_to_wrist is None:
                    wrist_arr[i] = identity_row
                    continue
                T_camera_wrist = _transform_to_matrix(*cam_to_wrist)
                T_world_wrist = T_world_camera @ T_camera_wrist
                wrist_arr[i] = _matrix_to_pose_row(T_world_wrist)

                pose_payload = hands_dict.get(ts)
                if pose_payload is None:
                    continue
                kpts_world = _keypoints_to_world(
                    pose_payload, T_world_wrist, PosesInFrame_pb2
                )
                if kpts_world is not None:
                    kpts_arr[i] = kpts_world

        return left_wrist, right_wrist, left_kpts, right_kpts


def _find_tf_edge(
    payload: bytes,
    FrameTransforms_pb2,
    parent: str,
    child: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Locate a (parent→child) transform inside a FrameTransforms message (raw bytes)."""
    msg = FrameTransforms_pb2.FrameTransforms()
    msg.ParseFromString(payload)
    return _find_tf_in_message(msg, parent, child)


def _find_tf_in_message(msg, parent: str, child: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Locate a (parent→child) transform inside an already-parsed FrameTransforms message."""
    for tf in msg.transforms:
        if tf.parent_frame_id == parent and tf.child_frame_id == child:
            t = np.array(
                [tf.translation.x, tf.translation.y, tf.translation.z], dtype=np.float64
            )
            q = np.array(
                [tf.rotation.x, tf.rotation.y, tf.rotation.z, tf.rotation.w],
                dtype=np.float64,
            )
            return t, q
    return None


def _keypoints_to_world(
    payload: bytes,
    T_world_wrist: np.ndarray,
    PosesInFrame_pb2,
) -> np.ndarray | None:
    """Transform 21 wrist-frame keypoints into world frame and flatten to (63,)."""
    msg = PosesInFrame_pb2.PosesInFrame()
    msg.ParseFromString(payload)
    if len(msg.poses) != _HAND_KEYPOINT_COUNT:
        return None
    pts_wrist = np.empty((_HAND_KEYPOINT_COUNT, 3), dtype=np.float64)
    for j, pose in enumerate(msg.poses):
        pts_wrist[j, 0] = pose.position.x
        pts_wrist[j, 1] = pose.position.y
        pts_wrist[j, 2] = pose.position.z
    pts_wrist_h = np.concatenate(
        [pts_wrist, np.ones((_HAND_KEYPOINT_COUNT, 1), dtype=np.float64)], axis=1
    )
    pts_world = (T_world_wrist @ pts_wrist_h.T).T[:, :3]
    return pts_world.reshape(-1)
