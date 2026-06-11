"""Register converted episodes in the DB and upload them to S3 (R2).

Follows the CONTRIBUTING_DATA.md order: insert the app.episodes row first
(§4), then upload the .zarr store and sibling .mp4 preview (§9), then update
the row with zarr_processed_path / zarr_mp4_path / num_frames (§4.3), then
verify the upload is reachable.

Local episodes may be nested in per-session subdirectories; the S3 layout is
always FLAT: s3://rldb/processed_v3/<embodiment_prefix>/<episode_hash>.zarr/
(plus <episode_hash>.mp4 alongside) — local subfolder names never appear in
the destination key.

The live app.episodes table can lag behind the TableRow dataclass (e.g. no
robot_name column yet), so all DB reads/writes here are restricted to the
columns that actually exist; dropped fields are reported once.

DRY-RUN BY DEFAULT: prints exactly what would be inserted and uploaded.
Nothing touches the DB or bucket without --execute.

Usage:
    python -m egomimic.scripts.register_and_upload ~/egoverse-data/Converted \
        --operator <your-operator-id> --lab microagi            # dry run
    python -m egomimic.scripts.register_and_upload ... --execute
"""

import argparse
import hashlib
import re
import sys
from pathlib import Path

import zarr

BUCKET = "rldb"
REMOTE_ROOT = "processed_v3"

# §3: the episode hash (== .zarr dir name == DB primary key) must be a zero-padded
# UTC timestamp YYYY-MM-DD-HH-MM-SS-ffffff. strptime (in _hash_problem) additionally
# rejects impossible calendar values; this regex enforces the exact shape.
HASH_RE = re.compile(r"\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}-\d{6}")

GREEN, RED, YELLOW, DIM, RESET = (
    "\033[32m",
    "\033[31m",
    "\033[33m",
    "\033[2m",
    "\033[0m",
)


def collect_episodes(paths):
    """Find all *.zarr stores at any depth under the given paths."""
    episodes = []
    for p in map(Path, paths):
        if p.name.endswith(".zarr"):
            episodes.append(p.resolve())
        elif p.is_dir():
            episodes.extend(
                sorted(q.resolve() for q in p.rglob("*.zarr") if q.is_dir())
            )
        else:
            sys.exit(f"error: {p} is not a .zarr store or a directory")
    if not episodes:
        sys.exit("error: no .zarr episodes found")
    return episodes


def plan_episode(zarr_path: Path, operator_hash: str, lab: str, robot_name: str | None):
    """Build the TableRow fields and S3 destinations for one episode."""
    attrs = dict(zarr.open_group(str(zarr_path), mode="r").attrs)
    hash_str = zarr_path.name.removesuffix(".zarr")
    embodiment = attrs["embodiment"]
    prefix = embodiment.split("_")[0]
    objects = attrs.get("objects", "")
    if isinstance(objects, (list, tuple)):
        objects = ",".join(map(str, objects))
    mp4 = zarr_path.parent / f"{hash_str}.mp4"
    zarr_key = f"{REMOTE_ROOT}/{prefix}/{hash_str}.zarr"
    mp4_key = f"{REMOTE_ROOT}/{prefix}/{hash_str}.mp4"
    return {
        "hash": hash_str,
        "local_zarr": zarr_path,
        "local_mp4": mp4 if mp4.is_file() else None,
        "zarr_key": zarr_key,
        "mp4_key": mp4_key if mp4.is_file() else None,
        "row_fields": dict(
            episode_hash=hash_str,
            operator=operator_hash,
            lab=lab,
            task=attrs["task_name"],
            embodiment=embodiment,
            robot_name=robot_name or embodiment,
            task_description=attrs.get("task_description", ""),
            objects=objects,
            num_frames=int(attrs["total_frames"]),
        ),
    }


# ── Pre-flight validation (§3 hash, §8 embodiment, §10 content) ─────────────
# Runs locally before anything touches the DB or bucket. The hash and embodiment
# guards are cheap and protect the DB primary key + S3 routing, so they ALWAYS
# run. --skip-validation only bypasses the expensive §10 content checks.

_VALID_EMBODIMENTS = None


def valid_embodiments():
    """Lowercased set of the §8 embodiment enum strings (cached)."""
    global _VALID_EMBODIMENTS
    if _VALID_EMBODIMENTS is None:
        from egomimic.rldb.embodiment.embodiment import EMBODIMENT

        _VALID_EMBODIMENTS = {m.name.lower() for m in EMBODIMENT}
    return _VALID_EMBODIMENTS


def _hash_problem(h):
    """Return a problem string if the hash is not a valid UTC timestamp, else None."""
    if not HASH_RE.fullmatch(h):
        return f"hash '{h}' is not a UTC timestamp YYYY-MM-DD-HH-MM-SS-ffffff (§3)"
    from egomimic.utils.aws.aws_sql import episode_hash_to_timestamp_ms

    try:
        episode_hash_to_timestamp_ms(h)
    except Exception as e:
        return f"hash '{h}' is not a valid calendar timestamp: {e}"
    return None


def preflight(plan, skip_validation):
    """Return (problems, warnings) for one episode.

    `problems` block registration/upload (empty list = ready); `warnings` are
    advisory (e.g. non-canonical task_name, no annotations) and never block.
    """
    problems = []
    warnings = []

    p = _hash_problem(plan["hash"])
    if p:
        problems.append(p)

    emb = plan["row_fields"]["embodiment"]
    try:
        if emb not in valid_embodiments():
            problems.append(
                f"embodiment '{emb}' is not in the §8 enum "
                f"({', '.join(sorted(valid_embodiments()))})"
            )
    except Exception as e:
        problems.append(f"could not load embodiment enum to validate '{emb}': {e}")

    if not skip_validation:
        try:
            from egomimic.test_zarr import validate_episode

            errors, warns, _ = validate_episode(str(plan["local_zarr"]))
            problems.extend(f"§10: {e}" for e in errors)
            warnings.extend(f"§10: {w}" for w in warns)
        except Exception as e:
            problems.append(f"§10 validation could not run: {type(e).__name__}: {e}")

    return problems, warnings


# ── Schema-tolerant DB access ──────────────────────────────────────────────
# The repo's add_episode/update_episode/episode_hash_to_table_row write or
# validate every TableRow field and fail when the live table is missing a
# column, so this script talks to the table directly with the intersection
# of plan fields and live columns.


def live_columns(engine):
    from egomimic.utils.aws.aws_sql import _episodes_table

    return _episodes_table(engine), {c.name for c in _episodes_table(engine).columns}


def fetch_row(engine, table, episode_hash):
    from sqlalchemy import select

    stmt = select(table).where(table.c.episode_hash == episode_hash).limit(1)
    with engine.connect() as conn:
        rec = conn.execute(stmt).mappings().first()
    return dict(rec) if rec is not None else None


def insert_row(engine, table, cols, fields):
    from sqlalchemy import insert

    values = {k: v for k, v in fields.items() if k in cols}
    with engine.begin() as conn:
        conn.execute(insert(table).values(**values))


def update_row(engine, table, cols, episode_hash, fields):
    from sqlalchemy import update

    values = {k: v for k, v in fields.items() if k in cols and k != "episode_hash"}
    with engine.begin() as conn:
        conn.execute(
            update(table).where(table.c.episode_hash == episode_hash).values(**values)
        )


def related_tasks(task, registry_tasks, limit=5):
    """Registry task names sharing a meaningful word with the planned task."""
    words = {w for w in task.split("_") if len(w) > 3}
    hits = [t for t in registry_tasks if any(w in t for w in words)]
    hits.sort(key=lambda t: -registry_tasks[t])
    return hits[:limit]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "paths",
        nargs="+",
        help=".zarr episode dir(s) or directories to search recursively",
    )
    ap.add_argument(
        "--operator",
        required=True,
        help="raw operator identifier; stored as its SHA-256 hash, never in plain text",
    )
    ap.add_argument(
        "--lab",
        default="microagi",
        help="short lowercase lab string (stable, used in filters)",
    )
    ap.add_argument(
        "--robot-name",
        default=None,
        help="<platform>_<config>; defaults to the embodiment string",
    )
    ap.add_argument(
        "--execute",
        action="store_true",
        help="actually insert rows and upload (default: dry run)",
    )
    ap.add_argument(
        "--skip-validation",
        action="store_true",
        help="skip the §10 content validation (shapes/quaternions/JPEG/annotations). "
        "The §3 hash-format and §8 embodiment guards always run.",
    )
    ap.add_argument(
        "--allow-warnings",
        action="store_true",
        help="upload even if pre-flight raised warnings (default: warnings block "
        "upload, like errors). Errors always block regardless of this flag.",
    )
    args = ap.parse_args()

    operator_hash = hashlib.sha256(args.operator.encode()).hexdigest()
    episodes = collect_episodes(args.paths)
    plans = [
        plan_episode(p, operator_hash, args.lab, args.robot_name) for p in episodes
    ]

    missing_mp4 = [p["hash"] for p in plans if p["local_mp4"] is None]
    if missing_mp4:
        print(
            f"{YELLOW}warning:{RESET} {len(missing_mp4)} episode(s) have no sibling .mp4 preview: "
            + ", ".join(missing_mp4)
        )

    # ── Pre-flight validation (§3/§8/§10) — local, before DB or bucket ──
    if args.skip_validation:
        print(
            f"{YELLOW}note:{RESET} --skip-validation set — §10 content checks skipped "
            "(hash-format and embodiment guards still enforced)"
        )
    print(f"{DIM}running pre-flight validation on {len(plans)} episode(s)…{RESET}")
    _pf = {p["hash"]: preflight(p, args.skip_validation) for p in plans}
    problems = {h: r[0] for h, r in _pf.items()}
    warnings_map = {h: r[1] for h, r in _pf.items()}

    # ── Connect and inspect registry state (read-only) ──────────────────
    engine = table = None
    cols = set()
    existing = {}
    registry_tasks = {}
    try:
        from egomimic.utils.aws.aws_sql import (
            create_default_engine,
            episode_table_to_df,
        )

        engine = create_default_engine()
        table, cols = live_columns(engine)
        registry_tasks = episode_table_to_df(engine).groupby("task").size().to_dict()
        for p in plans:
            existing[p["hash"]] = fetch_row(engine, table, p["hash"])
    except Exception as e:
        if args.execute:
            sys.exit(f"error: DB connection required for --execute: {e}")
        print(
            f"{YELLOW}warning:{RESET} no DB connection ({type(e).__name__}: {e}) — "
            "collision/task checks skipped in this dry run"
        )

    dropped = set(plans[0]["row_fields"]) - cols if cols else set()
    if dropped:
        print(
            f"{YELLOW}note:{RESET} app.episodes has no column(s) {sorted(dropped)} — "
            "these fields will not be stored"
        )

    # ── Plan summary ────────────────────────────────────────────────────
    print(f"\n{'hash':<28} {'task':<26} {'frames':>7}  destination")
    new_tasks = {}
    for p in plans:
        f = p["row_fields"]
        h = p["hash"]
        row = existing.get(h)
        probs = problems[h]
        warns = warnings_map[h]
        if probs:
            status = f"{RED}INVALID — will not register/upload{RESET}"
        elif warns and not args.allow_warnings:
            status = (
                f"{RED}BLOCKED by {len(warns)} warning(s) — "
                f"re-run with --allow-warnings to upload{RESET}"
            )
        elif row is not None and (row.get("zarr_processed_path") or "").strip():
            status = f"{DIM}SKIP (already uploaded){RESET}"
        elif row is not None:
            status = f"{YELLOW}row exists, will upload + update{RESET}"
        else:
            status = "register + upload"
        print(
            f"{h:<28} {f['task']:<26} {f['num_frames']:>7}  "
            f"s3://{BUCKET}/{p['zarr_key']}/  [{status}]"
        )
        for pr in probs:
            print(f"    {RED}✗{RESET} {pr}")
        for w in warns:
            print(f"    {YELLOW}⚠{RESET} {w}")
        if not probs and registry_tasks and f["task"] not in registry_tasks:
            new_tasks.setdefault(f["task"], related_tasks(f["task"], registry_tasks))
    if new_tasks:
        print(
            f"\n{YELLOW}new task names{RESET} (not yet in the registry of "
            f"{len(registry_tasks)} tasks — reuse an existing name if one fits, §4.1):"
        )
        for t, similar in new_tasks.items():
            hint = (
                ", ".join(f"{s} ({registry_tasks[s]})" for s in similar)
                or "no similar existing tasks"
            )
            print(f"  {t}  {DIM}similar: {hint}{RESET}")
    n_invalid = sum(1 for h in problems if problems[h])
    if n_invalid:
        print(
            f"\n{RED}{n_invalid}/{len(plans)} episode(s) failed pre-flight validation "
            f"and will be skipped (see ✗ above).{RESET}"
        )
    n_warn_blocked = sum(
        1 for p in plans if not problems[p["hash"]] and warnings_map[p["hash"]]
    )
    if n_warn_blocked:
        if args.allow_warnings:
            print(
                f"\n{YELLOW}{n_warn_blocked}/{len(plans)} episode(s) have warnings but "
                f"will be uploaded (--allow-warnings set; see ⚠ above).{RESET}"
            )
        else:
            print(
                f"\n{RED}{n_warn_blocked}/{len(plans)} episode(s) blocked by warnings "
                f"and will be skipped — re-run with --allow-warnings to upload them "
                f"(see ⚠ above).{RESET}"
            )

    print(
        f"\nshared row fields: lab='{args.lab}', operator=sha256:{operator_hash[:12]}…, "
        f"robot_name='{plans[0]['row_fields']['robot_name']}'"
    )

    if not args.execute:
        print(
            f"\n{YELLOW}DRY RUN{RESET} — nothing was inserted or uploaded. "
            "Re-run with --execute to proceed."
        )
        return

    # ── Execute: register -> upload -> update -> verify ────────────────
    from egomimic.utils.aws.aws_data_utils import get_boto3_s3_client

    s3 = get_boto3_s3_client()
    failures = []
    for p in plans:
        h = p["hash"]
        if problems[h]:
            failures.append(h)
            print(
                f"\n[SKIP] {h}: {RED}failed pre-flight validation{RESET} "
                f"({len(problems[h])} problem(s)) — not registered or uploaded"
            )
            continue
        if warnings_map[h] and not args.allow_warnings:
            failures.append(h)
            print(
                f"\n[SKIP] {h}: {RED}blocked by {len(warnings_map[h])} warning(s){RESET} "
                f"— re-run with --allow-warnings to upload"
            )
            continue
        row = existing.get(h)
        if row is not None and (row.get("zarr_processed_path") or "").strip():
            print(f"\n[SKIP] {h}: already uploaded")
            continue
        print(f"\n[{h}]")
        try:
            if row is None:
                insert_row(engine, table, cols, p["row_fields"])
                print("  registered row in app.episodes")
            else:
                print("  row already registered")

            print(
                f"  uploading {p['local_zarr'].name} -> s3://{BUCKET}/{p['zarr_key']}/"
            )
            upload_dir_flat(s3, p["local_zarr"], p["zarr_key"])
            if p["local_mp4"] is not None:
                s3.upload_file(str(p["local_mp4"]), BUCKET, p["mp4_key"])
                print(f"  uploaded preview -> s3://{BUCKET}/{p['mp4_key']}")

            s3.head_object(Bucket=BUCKET, Key=f"{p['zarr_key']}/zarr.json")

            update_row(
                engine,
                table,
                cols,
                h,
                {
                    "num_frames": p["row_fields"]["num_frames"],
                    "zarr_processed_path": f"s3://{BUCKET}/{p['zarr_key']}",
                    "zarr_mp4_path": f"s3://{BUCKET}/{p['mp4_key']}"
                    if p["mp4_key"]
                    else "",
                    "zarr_processing_error": "",
                },
            )
            print(f"  {GREEN}done{RESET} — row updated with zarr_processed_path")
        except Exception as e:
            failures.append(h)
            print(f"  {RED}FAILED:{RESET} {type(e).__name__}: {e}")

    print(f"\n{len(plans) - len(failures)}/{len(plans)} episodes completed")
    if failures:
        print(f"{RED}failed:{RESET} " + ", ".join(failures))
        sys.exit(1)


def upload_dir_flat(s3, local_dir: Path, key_prefix: str):
    """Upload a directory tree to BUCKET/key_prefix/, preserving the tree
    relative to local_dir (but nothing of the path above it)."""
    from boto3.s3.transfer import TransferConfig

    cfg = TransferConfig(
        max_concurrency=32,
        multipart_threshold=64 * 1024 * 1024,
        multipart_chunksize=64 * 1024 * 1024,
        use_threads=True,
    )
    files = sorted(p for p in local_dir.rglob("*") if p.is_file())
    total_bytes = sum(p.stat().st_size for p in files)
    done = 0
    for i, lp in enumerate(files, 1):
        key = f"{key_prefix}/{lp.relative_to(local_dir).as_posix()}"
        s3.upload_file(str(lp), BUCKET, key, Config=cfg)
        done += lp.stat().st_size
        if i % 200 == 0 or i == len(files):
            print(
                f"    {i}/{len(files)} files ({done / 1e6:.0f}/{total_bytes / 1e6:.0f} MB)"
            )


if __name__ == "__main__":
    main()
