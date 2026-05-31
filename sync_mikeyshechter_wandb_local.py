#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping
from urllib.parse import urlparse


DEFAULT_PROJECT_PATH = "mikeyshechter/modded-nanogpt-moe"
DEFAULT_OUTPUT_DIR = Path("/home/ycarmon/no_backup/users/sachter/wandb_mikeyshechter_local")
SCAN_LOG_INTERVAL = 100
MATCH_LOG_INTERVAL = 25


def _import_pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "pandas is required for sync_mikeyshechter_wandb_local.py."
        ) from exc
    return pd


def _import_wandb():
    try:
        import wandb
    except ImportError as exc:
        raise ImportError(
            "wandb is required for sync_mikeyshechter_wandb_local.py."
        ) from exc
    return wandb


def _pythonize(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _pythonize(nested_value) for key, nested_value in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_pythonize(item) for item in value]

    try:
        item_method = getattr(value, "item")
    except Exception:
        item_method = None
    if callable(item_method):
        try:
            return _pythonize(item_method())
        except Exception:
            pass

    try:
        tolist_method = getattr(value, "tolist")
    except Exception:
        tolist_method = None
    if callable(tolist_method):
        try:
            return _pythonize(tolist_method())
        except Exception:
            pass

    try:
        json.dumps(value)
    except TypeError:
        return str(value)
    return value


def _flatten_mapping(mapping: Mapping[str, Any], prefix: str = "") -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for key, value in mapping.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flat.update(_flatten_mapping(value, prefix=full_key))
        else:
            flat[full_key] = _pythonize(value)
    return flat


def _display_name(run: Any) -> str:
    name = getattr(run, "name", None)
    if name:
        return str(name)
    attrs = getattr(run, "_attrs", {}) or {}
    display_name = attrs.get("displayName")
    if display_name:
        return str(display_name)
    return str(getattr(run, "id", "unknown"))


def _run_path(run: Any) -> str:
    path = getattr(run, "path", None)
    if isinstance(path, (list, tuple)):
        return "/".join(str(part) for part in path)
    return str(path or "")


def _run_url(run: Any) -> str:
    path = _run_path(run)
    if path:
        entity, project, run_id = path.split("/", 2)
        return f"https://wandb.ai/{entity}/{project}/runs/{run_id}"
    run_id = getattr(run, "id", "")
    return f"https://wandb.ai/runs/{run_id}" if run_id else ""


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _normalize_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _parse_datetime_arg(raw_value: str) -> datetime:
    value = raw_value.strip()
    for date_format in ("%Y-%m-%d", "%d/%m/%Y"):
        try:
            return datetime.strptime(value, date_format).replace(tzinfo=timezone.utc)
        except ValueError:
            pass

    iso_value = value.replace("Z", "+00:00")
    try:
        return _normalize_datetime(datetime.fromisoformat(iso_value))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Expected a date like 2026-03-30 or 30/03/2026, or an ISO timestamp."
        ) from exc


def _parse_project_path(raw_value: str) -> str:
    value = raw_value.strip()
    if not value:
        raise argparse.ArgumentTypeError("Expected a W&B project path like entity/project.")

    parsed = urlparse(value)
    if parsed.scheme or parsed.netloc:
        if parsed.netloc not in {"wandb.ai", "www.wandb.ai"}:
            raise argparse.ArgumentTypeError("Expected a wandb.ai URL or entity/project path.")
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) < 2:
            raise argparse.ArgumentTypeError(
                "Expected a W&B URL like https://wandb.ai/entity/project."
            )
        return "/".join(parts[:2])

    parts = [part for part in value.strip("/").split("/") if part]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Expected a W&B project path like entity/project.")
    return "/".join(parts)


def _run_created_at(run: Any) -> datetime | None:
    created_at = getattr(run, "created_at", None)
    if created_at is None:
        return None
    if isinstance(created_at, datetime):
        return _normalize_datetime(created_at)
    if isinstance(created_at, str):
        try:
            return _parse_datetime_arg(created_at)
        except argparse.ArgumentTypeError:
            return None
    return None


def _log(message: str) -> None:
    timestamp = _now_utc().strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{timestamp}] {message}", flush=True)


def _iter_history_rows(run: Any, page_size: int) -> Iterable[Dict[str, Any]]:
    for row in run.scan_history(page_size=page_size):
        yield {key: _pythonize(value) for key, value in row.items()}


def _write_history_file(run: Any, history_path: Path, page_size: int) -> int:
    pd = _import_pandas()
    records = list(_iter_history_rows(run, page_size=page_size))
    history_df = pd.DataFrame.from_records(records)
    history_df.to_csv(history_path, index=False, compression="gzip")
    return int(len(history_df))


def _build_run_record(run: Any, flat_config: Mapping[str, Any]) -> Dict[str, Any]:
    summary = {str(key): _pythonize(value) for key, value in dict(run.summary).items()}
    attrs = getattr(run, "_attrs", {}) or {}

    record: Dict[str, Any] = {
        "run_id": getattr(run, "id", None),
        "run_name": _display_name(run),
        "run_path": _run_path(run),
        "run_url": _run_url(run),
        "state": getattr(run, "state", None),
        "group": getattr(run, "group", None),
        "created_at": getattr(run, "created_at", None),
        "history_line_count": attrs.get("historyLineCount"),
    }

    for key, value in flat_config.items():
        record[f"config.{key}"] = value
    for key, value in summary.items():
        record[f"summary.{key}"] = value
    return record


def _write_manifest(
    manifest_path: Path,
    *,
    project_path: str,
    output_dir: Path,
    scanned_runs: int,
    matched_runs: int,
    excluded_runs: int,
    exclude_substrings: list,
    created_at_on_or_after: datetime | None,
    history_downloaded_runs: int,
    history_skipped_runs: int,
    metadata_only: bool,
    page_size: int,
) -> None:
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_path": project_path,
        "output_dir": str(output_dir),
        "client_side_filter": {
            "exclude_name_substrings": exclude_substrings or None,
            "created_at_on_or_after_utc": (
                created_at_on_or_after.isoformat() if created_at_on_or_after else None
            ),
        },
        "scanned_runs": scanned_runs,
        "matched_runs": matched_runs,
        "excluded_runs": excluded_runs,
        "history_downloaded_runs": history_downloaded_runs,
        "history_skipped_runs": history_skipped_runs,
        "metadata_only": metadata_only,
        "history_page_size": page_size,
        "metadata_csv": str(output_dir / "selected_runs.csv"),
        "metadata_jsonl": str(output_dir / "selected_runs.jsonl"),
        "history_dir": str(output_dir / "histories"),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Client-side sync of W&B runs into a local directory. "
            "This downloads run metadata and, by default, full W&B history for each run."
        )
    )
    parser.add_argument(
        "--project-path",
        type=_parse_project_path,
        default=DEFAULT_PROJECT_PATH,
        help=(
            "W&B project path as entity/project, or a https://wandb.ai/entity/project URL. "
            f"Default: {DEFAULT_PROJECT_PATH}"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Local output directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=1000,
        help="W&B history scan page size.",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Only save run metadata; do not download per-run W&B history.",
    )
    parser.add_argument(
        "--force-history",
        action="store_true",
        help="Redownload history even if the local history file already exists.",
    )
    parser.add_argument(
        "--max-matched-runs",
        type=int,
        default=None,
        help="Optional cap on the number of runs to save. Useful for testing.",
    )
    parser.add_argument(
        "--exclude-name-substring",
        action="append",
        default=[],
        help=(
            "Skip runs whose display name contains this substring. "
            "Can be specified multiple times."
        ),
    )
    parser.add_argument(
        "--created-at-on-or-after",
        type=_parse_datetime_arg,
        default=None,
        help=(
            "Only keep runs created at or after this UTC date/time. "
            "Examples: 2026-03-30, 30/03/2026, 2026-03-30T00:00:00Z."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pd = _import_pandas()
    wandb = _import_wandb()
    start_time = _now_utc()

    output_dir = args.output_dir.expanduser().resolve()
    histories_dir = output_dir / "histories"
    output_dir.mkdir(parents=True, exist_ok=True)
    histories_dir.mkdir(parents=True, exist_ok=True)

    api = wandb.Api(timeout=600)
    records = []
    scanned_runs = 0
    matched_runs = 0
    excluded_runs = 0
    history_downloaded_runs = 0
    history_skipped_runs = 0

    project_path = args.project_path
    exclude_substrings = list(args.exclude_name_substring or [])
    created_at_cutoff = args.created_at_on_or_after

    _log("Starting W&B local sync.")
    _log(f"Project: {project_path}")
    _log(f"Output dir: {output_dir}")
    if created_at_cutoff is not None:
        _log(f"Keeping only runs with created_at >= {created_at_cutoff.isoformat()}")
    if exclude_substrings:
        _log(f"Excluding runs whose name contains any of: {exclude_substrings}")
    else:
        _log("Name-based filter: none.")
    _log(
        f"Options: metadata_only={args.metadata_only}, force_history={args.force_history}, "
        f"page_size={args.page_size}, max_matched_runs={args.max_matched_runs}"
    )
    _log("Scanning W&B runs...")
    for run in api.runs(project_path, per_page=1000, order="+created_at"):
        scanned_runs += 1
        raw_config = {k: v for k, v in dict(run.config).items() if not str(k).startswith("_")}
        flat_config = _flatten_mapping(raw_config)

        run_name = _display_name(run)
        run_created_at = _run_created_at(run)
        if created_at_cutoff is not None:
            if run_created_at is None:
                excluded_runs += 1
                _log(f"Excluding run {run.id} {run_name} (missing or invalid created_at)")
                continue
            if run_created_at < created_at_cutoff:
                excluded_runs += 1
                continue
        if any(sub in run_name for sub in exclude_substrings):
            excluded_runs += 1
            _log(f"Excluding run {run.id} {run_name} (matches exclude filter)")
            continue

        matched_runs += 1
        history_path = histories_dir / f"{run.id}.csv.gz"
        record = _build_run_record(run, flat_config)
        record["history_file"] = str(history_path)

        if args.metadata_only:
            record["history_rows"] = None
            _log(f"[{matched_runs}] {run.id} {run_name} (metadata only)")
        else:
            if history_path.exists() and not args.force_history:
                history_skipped_runs += 1
                record["history_rows"] = None
                _log(
                    f"[{matched_runs}] Skipping existing history for {run.id} {run_name}",
                )
            else:
                _log(
                    f"[{matched_runs}] Downloading history for {run.id} {run_name}",
                )
                history_rows = _write_history_file(run, history_path, page_size=args.page_size)
                history_downloaded_runs += 1
                record["history_rows"] = history_rows
                _log(
                    f"[{matched_runs}] Saved history for {run.id} with {history_rows} rows "
                    f"to {history_path}"
                )

        records.append(record)

        if args.max_matched_runs is not None and matched_runs >= args.max_matched_runs:
            _log(f"Reached --max-matched-runs={args.max_matched_runs}. Stopping early.")
            break

        if matched_runs % MATCH_LOG_INTERVAL == 0:
            elapsed_seconds = max((_now_utc() - start_time).total_seconds(), 1e-9)
            _log(
                f"Saved {matched_runs} runs so far. "
                f"History downloaded={history_downloaded_runs}, skipped={history_skipped_runs}, "
                f"elapsed={elapsed_seconds:.1f}s."
            )

    if records:
        runs_df = pd.DataFrame.from_records(records)
        runs_df = runs_df.sort_values(by=["created_at", "run_id"], kind="mergesort").reset_index(drop=True)
    else:
        runs_df = pd.DataFrame()

    metadata_csv_path = output_dir / "selected_runs.csv"
    metadata_jsonl_path = output_dir / "selected_runs.jsonl"
    _log(f"Writing metadata for {len(runs_df)} runs...")
    runs_df.to_csv(metadata_csv_path, index=False)
    jsonl_text = runs_df.to_json(orient="records", lines=True)
    metadata_jsonl_path.write_text(jsonl_text + ("\n" if jsonl_text else ""))

    _write_manifest(
        output_dir / "manifest.json",
        project_path=project_path,
        output_dir=output_dir,
        scanned_runs=scanned_runs,
        matched_runs=matched_runs,
        excluded_runs=excluded_runs,
        exclude_substrings=exclude_substrings,
        created_at_on_or_after=created_at_cutoff,
        history_downloaded_runs=history_downloaded_runs,
        history_skipped_runs=history_skipped_runs,
        metadata_only=args.metadata_only,
        page_size=args.page_size,
    )

    elapsed_seconds = (_now_utc() - start_time).total_seconds()
    _log(
        "Finished sync: "
        f"scanned_runs={scanned_runs}, matched_runs={matched_runs}, "
        f"excluded_runs={excluded_runs}, "
        f"history_downloaded_runs={history_downloaded_runs}, "
        f"history_skipped_runs={history_skipped_runs}, "
        f"elapsed={elapsed_seconds:.1f}s"
    )
    _log(f"Metadata saved to {metadata_csv_path}")
    _log(f"Manifest saved to {output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
