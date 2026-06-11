"""Visual / semantic companion to check_contribution.py.

Covers the CONTRIBUTING_DATA.md checklist items that need a human eye:

- SLAM world frame + XYZWXYZ order + meters (§6, §11): projects the stored
  hand keypoints into the image via inverse(obs_head_pose) and the episode's
  intrinsics. Dots land on the hands only if frame, quaternion order, and
  units are all correct.
- RGB vs BGR (§11): frames are decoded as RGB and saved as PNG — if colors
  look natural (skin is skin-colored, not blue), the data is RGB.
- Annotation text quality (§7): prints every annotation span for review.

Usage:
    python -m egomimic.scripts.check_contribution_visual <episode.zarr> \
        [--frames 6] [--out ./contribution_check]

Then open the PNGs in <out>/ — left hand keypoints are green, right are red.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import simplejpeg
import zarr
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation


def pose_to_matrix(pose7):
    """XYZWXYZ pose [tx,ty,tz,qw,qx,qy,qz] -> 4x4 world-from-device matrix."""
    T = np.eye(4)
    T[:3, 3] = pose7[:3]
    qw, qx, qy, qz = pose7[3:7]
    T[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
    return T


def project(points_world, T_world_device, K):
    """World-frame points -> pixel coords using the head/camera pose."""
    T_device_world = np.linalg.inv(T_world_device)
    pts = (T_device_world[:3, :3] @ points_world.T).T + T_device_world[:3, 3]
    z = pts[:, 2]
    valid = z > 1e-6
    uv = np.full((len(pts), 2), np.nan)
    uv[valid, 0] = K[0, 0] * pts[valid, 0] / z[valid] + K[0, 2]
    uv[valid, 1] = K[1, 1] * pts[valid, 1] / z[valid] + K[1, 2]
    return uv


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("zarr_path", type=Path)
    ap.add_argument("--frames", type=int, default=6, help="number of frames to render")
    ap.add_argument("--out", type=Path, default=Path("./contribution_check"))
    args = ap.parse_args()

    store = zarr.open_group(str(args.zarr_path), mode="r")
    attrs = dict(store.attrs)
    T_total = attrs["total_frames"]
    hash_str = args.zarr_path.name.removesuffix(".zarr")

    K = np.array(attrs.get("intrinsics", []), dtype=np.float64)
    if K.shape != (3, 3):
        sys.exit("no 3x3 'intrinsics' in zarr.attrs — cannot reproject")

    args.out.mkdir(parents=True, exist_ok=True)
    frame_ids = np.linspace(0, T_total - 1, args.frames, dtype=int)

    # ── Keypoint reprojection overlays ───────────────────────────────────
    head = store["obs_head_pose"]
    sides = {}
    for side, color in (("left", (0, 255, 0)), ("right", (255, 0, 0))):
        key = f"{side}.obs_keypoints"
        if key in store:
            sides[side] = (store[key], color)
    on_screen = 0
    total_pts = 0
    for fi in frame_ids:
        raw = store["images.front_1"][fi : fi + 1][0]
        while isinstance(raw, np.ndarray):
            raw = raw.item() if raw.shape == () else raw.flat[0]
        img = simplejpeg.decode_jpeg(bytes(raw), colorspace="RGB")
        H, W = img.shape[:2]
        pil = Image.fromarray(img)
        draw = ImageDraw.Draw(pil)
        T_wd = pose_to_matrix(head[fi])
        for side, (arr, color) in sides.items():
            kps = arr[fi].reshape(21, 3)
            uv = project(kps, T_wd, K)
            for u, v in uv:
                if np.isnan(u):
                    continue
                total_pts += 1
                if 0 <= u < W and 0 <= v < H:
                    on_screen += 1
                    draw.ellipse([u - 3, v - 3, u + 3, v + 3], fill=color)
        out = args.out / f"{hash_str}_frame{fi:06d}.png"
        pil.save(out)
        print(f"wrote {out}")
    if total_pts:
        print(f"\n{on_screen}/{total_pts} projected keypoints fall inside the image "
              f"({100 * on_screen / total_pts:.0f}%). If the dots sit on the hands in the "
              f"PNGs, the SLAM frame, XYZWXYZ order, and meter units are all consistent.")

    # ── Hand-to-head distance sanity (meters) ────────────────────────────
    print("\nhand-to-head distances (should be ~0.2-1.2 m for arm's reach):")
    head_t = head[:T_total, :3]
    for side in sides:
        ee = store[f"{side}.obs_ee_pose"][:T_total, :3]
        d = np.linalg.norm(ee - head_t, axis=1)
        print(f"  {side}: min={d.min():.2f} m  median={np.median(d):.2f} m  max={d.max():.2f} m")

    # ── Annotations dump (§7) ────────────────────────────────────────────
    if "annotations" in store:
        node = store["annotations"]
        print(f"\nannotations ({node.shape[0]}) — verify English, imperative/present-continuous:")
        for i in range(node.shape[0]):
            raw = node[i]
            while isinstance(raw, np.ndarray):
                raw = raw.item() if raw.shape == () else raw.flat[0]
            rec = json.loads(raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw)
            print(f"  [{rec['start_idx']:>6}, {rec['end_idx']:>6})  {rec['text']}")


if __name__ == "__main__":
    main()
