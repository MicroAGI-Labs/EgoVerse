"""Pre-submission checker for EgoVerse episode contributions.

Verifies a local ``<episode_hash>.zarr`` store against the requirements in
CONTRIBUTING_DATA.md (sections 3, 5, 7, 8, 11). Offline checks run by
default; database and S3 checks require credentials and are opt-in.

Usage:
    python -m egomimic.scripts.check_contribution <path/to/episode.zarr> [...]
    python -m egomimic.scripts.check_contribution <dir-of-zarrs> --db --s3

Exit code is non-zero if any check FAILs.
"""

import argparse
import io
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import zarr

# §8 Embodiment Identifiers
VALID_EMBODIMENTS = {
    "aria_bimanual", "aria_right_arm", "aria_left_arm",
    "eva_bimanual", "eva_right_arm", "eva_left_arm",
    "mecka_bimanual", "mecka_right_arm", "mecka_left_arm",
    "scale_bimanual", "scale_right_arm", "scale_left_arm",
    "microagi_bimanual", "microagi_right_arm", "microagi_left_arm",
}

# §3 Episode hash: YYYY-MM-DD-HH-MM-SS-ffffff (microseconds zero-padded)
HASH_RE = re.compile(r"^\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}-\d{6}$")

POSE_KEYS = (
    "obs_head_pose",
    "left.obs_ee_pose", "right.obs_ee_pose",
    "left.obs_wrist_pose", "right.obs_wrist_pose",
    "left.cmd_ee_pose", "right.cmd_ee_pose",
)
GRIPPER_KEYS = (
    "left.obs_gripper", "right.obs_gripper",
    "left.cmd_gripper", "right.cmd_gripper",
)
KEYPOINT_KEYS = ("left.obs_keypoints", "right.obs_keypoints")

GREEN, RED, YELLOW, BLUE, DIM, RESET = (
    "\033[32m", "\033[31m", "\033[33m", "\033[36m", "\033[2m", "\033[0m"
)


class Report:
    def __init__(self):
        self.results = []  # (status, section, message)

    def _add(self, status, section, msg):
        self.results.append((status, section, msg))

    def ok(self, section, msg):
        self._add("PASS", section, msg)

    def fail(self, section, msg):
        self._add("FAIL", section, msg)

    def warn(self, section, msg):
        self._add("WARN", section, msg)

    def skip(self, section, msg):
        self._add("SKIP", section, msg)

    def print(self):
        colors = {"PASS": GREEN, "FAIL": RED, "WARN": YELLOW, "SKIP": DIM}
        section = None
        for status, sec, msg in self.results:
            if sec != section:
                section = sec
                print(f"\n  {BLUE}{section}{RESET}")
            c = colors[status]
            print(f"    {c}{status:4}{RESET}  {msg}")

    @property
    def num_failed(self):
        return sum(1 for s, _, _ in self.results if s == "FAIL")

    @property
    def num_warned(self):
        return sum(1 for s, _, _ in self.results if s == "WARN")


def estimate_jpeg_quality(jpeg_bytes):
    """Estimate libjpeg quality from the luminance quantization table.

    Returns None if PIL is unavailable or the table can't be read. The
    estimate is approximate (encoders vary), so callers should only WARN.
    """
    try:
        from PIL import Image
    except ImportError:
        return None
    std_luminance = np.array([
        16, 11, 10, 16, 24, 40, 51, 61,
        12, 12, 14, 19, 26, 58, 60, 55,
        14, 13, 16, 24, 40, 57, 69, 56,
        14, 17, 22, 29, 51, 87, 80, 62,
        18, 22, 37, 56, 68, 109, 103, 77,
        24, 35, 55, 64, 81, 104, 113, 92,
        49, 64, 78, 87, 103, 121, 120, 101,
        72, 92, 95, 98, 112, 100, 103, 99,
    ], dtype=np.float64)
    try:
        im = Image.open(io.BytesIO(jpeg_bytes))
        actual_sum = sum(im.quantization[0])
    except Exception:
        return None
    best_q, best_diff = None, None
    for q in range(1, 101):
        scale = 5000 / q if q < 50 else 200 - 2 * q
        tbl = np.clip(np.floor((std_luminance * scale + 50) / 100), 1, 255)
        diff = abs(tbl.sum() - actual_sum)
        if best_diff is None or diff < best_diff:
            best_q, best_diff = q, diff
    return best_q


def check_episode(zarr_path: Path, rep: Report, db_engine=None, check_s3=False):
    hash_str = zarr_path.name.removesuffix(".zarr")

    # ── Episode hash (§3) ────────────────────────────────────────────────
    sec = "Episode hash (§3)"
    if HASH_RE.match(hash_str):
        try:
            datetime.strptime(hash_str, "%Y-%m-%d-%H-%M-%S-%f")
            rep.ok(sec, f"hash '{hash_str}' is a valid UTC timestamp string")
        except ValueError:
            rep.fail(sec, f"hash '{hash_str}' matches the pattern but is not a valid date/time")
    else:
        rep.fail(sec, f"hash '{hash_str}' does not match YYYY-MM-DD-HH-MM-SS-ffffff")

    # ── Open store / Zarr v3 (§5) ────────────────────────────────────────
    sec = "Zarr format (§5)"
    zj = zarr_path / "zarr.json"
    if not zj.is_file():
        rep.fail(sec, "zarr.json missing — not a Zarr v3 store")
        return
    root_meta = json.loads(zj.read_text())
    if root_meta.get("zarr_format") == 3:
        rep.ok(sec, "zarr_format is 3")
    else:
        rep.fail(sec, f"zarr_format is {root_meta.get('zarr_format')}, expected 3")

    try:
        store = zarr.open_group(str(zarr_path), mode="r")
    except Exception as e:
        rep.fail(sec, f"failed to open store: {e}")
        return
    attrs = dict(store.attrs)
    array_keys = [k for k in store.keys() if isinstance(store[k], zarr.Array)]

    # ── Top-level attributes (§5.3) ──────────────────────────────────────
    sec = "Top-level attributes (§5.3)"
    for field in ("embodiment", "total_frames", "fps", "task_name", "task_description", "features"):
        if field in attrs:
            rep.ok(sec, f"attrs['{field}'] present")
        else:
            rep.fail(sec, f"attrs['{field}'] missing")
    T = attrs.get("total_frames")
    if not isinstance(T, int) or T <= 0:
        rep.fail(sec, f"total_frames={T!r} is not a positive int — skipping frame-dependent checks")
        return
    fps = attrs.get("fps")
    if fps in (30, 60):
        rep.ok(sec, f"fps={fps} is valid")
    else:
        rep.fail(sec, f"fps={fps!r}, expected 30 or 60")

    embodiment = attrs.get("embodiment", "")
    if embodiment in VALID_EMBODIMENTS:
        rep.ok(sec, f"embodiment '{embodiment}' is a registered identifier (§8)")
    else:
        rep.fail(sec, f"embodiment '{embodiment}' not in the §8 identifier list")

    features = attrs.get("features", {})
    for key in array_keys:
        if key not in features:
            rep.fail(sec, f"array '{key}' has no entry in attrs['features']")
    for key in features:
        if key not in array_keys:
            rep.fail(sec, f"features entry '{key}' has no corresponding array in the store")
    if all(k in features for k in array_keys) and all(k in array_keys for k in features):
        rep.ok(sec, f"features dict has exactly one entry per array key ({len(array_keys)} keys)")

    json_keys = {k for k, f in features.items() if f.get("dtype") == "json"}
    jpeg_keys = [k for k, f in features.items() if f.get("dtype") == "jpeg" and k in array_keys]

    # ── Required arrays (§5.2) ───────────────────────────────────────────
    sec = "Required arrays (§5.2)"
    for key in ("images.front_1", "obs_head_pose", "obs_rgb_timestamps_ns"):
        if key in array_keys:
            rep.ok(sec, f"required key present: {key}")
        else:
            rep.fail(sec, f"missing required key: {key}")
    has_keypoints = any(k in array_keys for k in KEYPOINT_KEYS)
    for side in ("left", "right"):
        key = f"{side}.obs_ee_pose"
        if key in array_keys:
            rep.ok(sec, f"key present: {key}")
        elif has_keypoints:
            rep.fail(sec, f"{key} missing but hand keypoints are present — required when hand tracking is available")
        else:
            rep.warn(sec, f"{key} missing — OK only if no hand tracking and no {side} arm")
    for kp, wp in zip(KEYPOINT_KEYS, ("left.obs_wrist_pose", "right.obs_wrist_pose")):
        if kp in array_keys and wp not in array_keys:
            rep.warn(sec, f"{kp} present but {wp} missing — wrist pose is required if hand tracking is available")

    # ── Frame counts (§5.2 / §5.3) ───────────────────────────────────────
    # ZarrWriter pads arrays to the next chunk_timesteps=100 boundary;
    # total_frames is the valid (unpadded) count, so length in [T, padded_T]
    # is expected (matches the §10.1 validator, which only checks >= T).
    sec = "Frame counts"
    padded_T = -(-T // 100) * 100
    bad = 0
    for key in array_keys:
        if key in json_keys:
            continue
        n = store[key].shape[0]
        if n < T:
            rep.fail(sec, f"{key}: length {n} < total_frames {T}")
            bad += 1
        elif n > padded_T:
            rep.warn(sec, f"{key}: length {n} exceeds chunk-padded total_frames "
                          f"({T} padded to {padded_T})")
    if bad == 0:
        rep.ok(sec, f"all {len(array_keys) - len(json_keys)} frame-indexed arrays "
                    f"have >= total_frames ({T}) entries")

    # ── Pose arrays (§5.2 / §6) ──────────────────────────────────────────
    sec = "Poses & coordinate frames (§5.2, §6)"
    for key in POSE_KEYS:
        if key not in array_keys:
            continue
        arr = store[key][:T]
        if arr.ndim != 2 or arr.shape[1] != 7:
            rep.fail(sec, f"{key}: expected shape (T, 7), got {arr.shape}")
            continue
        rep.ok(sec, f"{key}: shape (T, 7) OK, dtype {arr.dtype}")
        if arr.dtype != np.float64:
            rep.warn(sec, f"{key}: dtype {arr.dtype}, spec says float64")
        norms = np.linalg.norm(arr[:, 3:7], axis=1)
        if np.allclose(norms, 1.0, atol=1e-4):
            rep.ok(sec, f"{key}: all quaternions unit-norm (XYZWXYZ order assumed)")
        else:
            bad_idx = np.where(np.abs(norms - 1.0) > 1e-4)[0]
            rep.fail(sec, f"{key}: {len(bad_idx)} frames with non-unit quaternions "
                          f"(e.g. frame {bad_idx[0]}, norm={norms[bad_idx[0]]:.6f}) — "
                          f"check XYZWXYZ ordering and normalization")
        tmax = np.abs(arr[:, :3]).max()
        if tmax > 100:
            rep.warn(sec, f"{key}: max |translation| = {tmax:.1f} — looks too large for meters (mm?)")
    for key in KEYPOINT_KEYS:
        if key not in array_keys:
            continue
        arr = store[key]
        if arr.shape[-1] == 63:
            rep.ok(sec, f"{key}: shape (T, 63) OK")
        else:
            rep.fail(sec, f"{key}: expected last dim 63 (21 MANO landmarks x 3), got {arr.shape}")
    for key in GRIPPER_KEYS:
        if key not in array_keys:
            continue
        arr = store[key][:T]
        if arr.ndim != 2 or arr.shape[1] != 1:
            rep.fail(sec, f"{key}: expected shape (T, 1), got {arr.shape}")
            continue
        rep.ok(sec, f"{key}: shape (T, 1) OK")
        if arr.min() < -1e-6 or arr.max() > 1 + 1e-6:
            rep.fail(sec, f"{key}: values outside [0, 1] (min={arr.min():.3f}, max={arr.max():.3f})")
    if "obs_eye_gaze" in array_keys:
        arr = store["obs_eye_gaze"][:T]
        if arr.ndim != 2 or arr.shape[1] != 3:
            rep.fail(sec, f"obs_eye_gaze: expected shape (T, 3), got {arr.shape}")
        else:
            norms = np.linalg.norm(arr, axis=1)
            if np.allclose(norms, 1.0, atol=1e-3):
                rep.ok(sec, "obs_eye_gaze: shape (T, 3) OK, unit direction vectors")
            else:
                rep.warn(sec, "obs_eye_gaze: not all vectors are unit-norm")

    # ── Timestamps ───────────────────────────────────────────────────────
    sec = "Timestamps"
    if "obs_rgb_timestamps_ns" in array_keys:
        ts = store["obs_rgb_timestamps_ns"][:T]
        if ts.dtype != np.int64:
            rep.fail(sec, f"obs_rgb_timestamps_ns: dtype {ts.dtype}, expected int64")
        else:
            rep.ok(sec, "obs_rgb_timestamps_ns: dtype int64 OK")
        dt = np.diff(ts.astype(np.float64))
        if len(dt) and (dt <= 0).any():
            rep.fail(sec, f"obs_rgb_timestamps_ns: {(dt <= 0).sum()} non-increasing steps")
        elif len(dt):
            actual_fps = 1e9 / np.median(dt)
            if fps and abs(actual_fps - fps) / fps > 0.1:
                rep.warn(sec, f"measured rate {actual_fps:.1f} fps differs >10% from attrs fps={fps} "
                              f"(§5.3: fps must be the actual capture rate)")
            else:
                rep.ok(sec, f"monotonic timestamps, measured rate {actual_fps:.1f} fps matches attrs fps={fps}")

    # ── Images (§5.2, §11) ───────────────────────────────────────────────
    sec = "Images"
    try:
        import simplejpeg
    except ImportError:
        simplejpeg = None
        rep.skip(sec, "simplejpeg not installed — skipping decode checks")
    for key in jpeg_keys:
        if simplejpeg is None:
            break
        try:
            raw = store[key][0:1][0]
            while isinstance(raw, np.ndarray):
                raw = raw.item() if raw.shape == () else raw.flat[0]
            raw = bytes(raw)
            frame = simplejpeg.decode_jpeg(raw, colorspace="RGB")
        except Exception as e:
            rep.fail(sec, f"{key}: failed to decode frame 0 as JPEG: {e}")
            continue
        if frame.ndim != 3 or frame.shape[2] != 3:
            rep.fail(sec, f"{key}: decoded frame has unexpected shape {frame.shape}")
            continue
        rep.ok(sec, f"{key}: frame 0 decodes OK, shape={frame.shape}")
        feat_shape = features.get(key, {}).get("shape")
        if feat_shape is not None:
            if list(frame.shape) == list(feat_shape):
                rep.ok(sec, f"{key}: decoded shape matches features['{key}']['shape']")
            else:
                rep.fail(sec, f"{key}: decoded shape {list(frame.shape)} != features shape {feat_shape}")
        q = estimate_jpeg_quality(raw)
        if q is None:
            rep.skip(sec, f"{key}: could not estimate JPEG quality")
        elif abs(q - 85) <= 3:
            rep.ok(sec, f"{key}: estimated JPEG quality ~{q} (spec: 85)")
        else:
            rep.warn(sec, f"{key}: estimated JPEG quality ~{q}, spec requires 85")

    # ── Annotations (§7) ─────────────────────────────────────────────────
    sec = "Annotations (§7)"
    ann_keys = [k for k in json_keys if k in array_keys]
    if "annotations" not in array_keys:
        rep.fail(sec, "'annotations' key missing — must be present even if empty (shape (0,))")
    for key in ann_keys:
        node = store[key]
        n = node.shape[0]
        if n == 0:
            rep.ok(sec, f"{key}: empty annotation array (valid, but annotations are strongly encouraged)")
            continue
        bad, first_err, non_ascii = 0, None, 0
        for i in range(n):
            raw = node[i]
            while isinstance(raw, np.ndarray):
                raw = raw.item() if raw.shape == () else raw.flat[0]
            if isinstance(raw, np.bytes_):
                raw = bytes(raw)
            try:
                rec = json.loads(raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw)
                if not isinstance(rec, dict):
                    raise ValueError(f"record is {type(rec).__name__}, expected dict")
                for field, expected in (("text", str), ("start_idx", int), ("end_idx", int)):
                    if not isinstance(rec.get(field), expected):
                        raise ValueError(f"field '{field}' missing or wrong type")
                if not (0 <= rec["start_idx"] < rec["end_idx"] <= T):
                    raise ValueError(f"span invalid: [{rec['start_idx']}, {rec['end_idx']}) with T={T}")
                if not rec["text"].isascii():
                    non_ascii += 1
            except Exception as e:
                bad += 1
                if first_err is None:
                    first_err = (i, str(e))
        if bad:
            rep.fail(sec, f"{key}: {bad}/{n} annotations malformed (e.g. index {first_err[0]}: {first_err[1]})")
        else:
            rep.ok(sec, f"{key}: all {n} annotations well-formed with valid spans")
        if non_ascii:
            rep.warn(sec, f"{key}: {non_ascii}/{n} annotations contain non-ASCII text — verify they are English")

    # ── Preview MP4 (§5.5) ───────────────────────────────────────────────
    sec = "Preview MP4 (§5.5)"
    mp4 = zarr_path.parent / f"{hash_str}.mp4"
    if mp4.is_file():
        rep.ok(sec, f"sibling preview {mp4.name} exists ({mp4.stat().st_size / 1e6:.1f} MB)")
    else:
        rep.warn(sec, f"no sibling {hash_str}.mp4 — the Mecka AI dataset viz requires it")

    # ── Database (§4, §11) ───────────────────────────────────────────────
    sec = "Database registry (§4)"
    if db_engine is None:
        rep.skip(sec, "skipped (run with --db to check the episode registry)")
    else:
        from egomimic.utils.aws.aws_sql import episode_hash_to_table_row
        row = episode_hash_to_table_row(db_engine, hash_str)
        if row is None:
            rep.warn(sec, "no DB row for this hash — OK pre-registration, but the row "
                          "must be inserted before upload (§4)")
        else:
            rep.ok(sec, "DB row exists for this episode_hash")
            if row.embodiment == embodiment:
                rep.ok(sec, f"DB embodiment matches zarr.attrs ('{embodiment}')")
            else:
                rep.fail(sec, f"DB embodiment '{row.embodiment}' != attrs '{embodiment}'")
            if row.task == attrs.get("task_name"):
                rep.ok(sec, f"DB task matches attrs task_name ('{row.task}')")
            else:
                rep.fail(sec, f"DB task '{row.task}' != attrs task_name '{attrs.get('task_name')}'")
            if row.num_frames == T:
                rep.ok(sec, f"DB num_frames matches total_frames ({T})")
            else:
                rep.fail(sec, f"DB num_frames {row.num_frames} != total_frames {T}")
            if re.fullmatch(r"[0-9a-f]{64}", row.operator or ""):
                rep.ok(sec, "operator field looks like a SHA-256 hash (no raw PII)")
            else:
                rep.warn(sec, f"operator '{row.operator}' is not a SHA-256 hex digest — "
                              "operator IDs must be hashed (§4.1)")

    # ── S3 (§9, §11) ─────────────────────────────────────────────────────
    sec = "S3 upload (§9)"
    if not check_s3:
        rep.skip(sec, "skipped (run with --s3 to check the uploaded copy)")
    else:
        import boto3
        from egomimic.utils.aws.aws_data_utils import load_env
        load_env()
        prefix = embodiment.split("_")[0] if embodiment else "unknown"
        s3_key = f"processed_v3/{prefix}/{hash_str}.zarr/zarr.json"
        try:
            s3 = boto3.client("s3", endpoint_url=__import__("os").environ.get("AWS_ENDPOINT_URL_S3"))
            s3.head_object(Bucket="rldb", Key=s3_key)
            rep.ok(sec, f"episode accessible at s3://rldb/{s3_key.rsplit('/', 1)[0]}/")
        except Exception as e:
            rep.warn(sec, f"s3://rldb/{s3_key} not accessible ({type(e).__name__}) — "
                          "OK if not uploaded yet")


MANUAL_ITEMS = [
    "Poses are in the SLAM world frame, not head/camera frame (§6.1)",
    "Images are RGB, not BGR (§11)",
    "Annotation text is English, imperative/present-continuous (§7.1)",
    "Store was produced with ZarrWriter, not a custom writer (§5.4)",
    "task name reuses an existing registry task where possible (§4.1)",
    "End-to-end load test through MultiDataset passes (§10.2)",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", nargs="+", help=".zarr episode dir(s), or a directory containing them")
    ap.add_argument("--db", action="store_true", help="also check the PostgreSQL episode registry (needs credentials)")
    ap.add_argument("--s3", action="store_true", help="also check the episode exists on S3 (needs credentials)")
    args = ap.parse_args()

    episodes = []
    for p in map(Path, args.paths):
        if p.name.endswith(".zarr"):
            episodes.append(p)
        elif p.is_dir():
            episodes.extend(sorted(p.glob("*.zarr")))
        else:
            print(f"{RED}error:{RESET} {p} is not a .zarr store or a directory", file=sys.stderr)
            sys.exit(2)
    if not episodes:
        print(f"{RED}error:{RESET} no .zarr episodes found", file=sys.stderr)
        sys.exit(2)

    engine = None
    if args.db:
        from egomimic.utils.aws.aws_sql import create_default_engine
        engine = create_default_engine()

    total_fail = 0
    for ep in episodes:
        print(f"\n{'=' * 70}\n{ep}\n{'=' * 70}")
        rep = Report()
        try:
            check_episode(ep, rep, db_engine=engine, check_s3=args.s3)
        except Exception as e:
            rep.fail("Checker", f"unhandled error: {type(e).__name__}: {e}")
        rep.print()
        status = (f"{RED}{rep.num_failed} FAILED{RESET}" if rep.num_failed
                  else f"{GREEN}all checks passed{RESET}")
        print(f"\n  => {status}, {rep.num_warned} warnings")
        total_fail += rep.num_failed

    print(f"\n{BLUE}Not automatically checkable — verify manually:{RESET}")
    for item in MANUAL_ITEMS:
        print(f"  [ ] {item}")

    sys.exit(1 if total_fail else 0)


if __name__ == "__main__":
    main()
