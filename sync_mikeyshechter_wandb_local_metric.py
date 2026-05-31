#!/usr/bin/env python3
"""On-demand, per-metric W&B local sync.

Layout:
    <output_dir>/summary/selected_runs.csv
    <output_dir>/summary/selected_runs.jsonl
    <output_dir>/summary/manifest.json
    <output_dir>/histories/<metric>/<run_id>.csv.gz

History downloads use ``run.scan_history(keys=['_step', metric])``, which
returns the full (un-sampled, un-duplicated) per-step series. Each metric
is requested separately, parallelised across threads.
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Iterable, List, Mapping, Optional
from urllib.parse import urlparse


DEFAULT_PROJECT_PATH = "mikeyshechter/modded-nanogpt-moe"
DEFAULT_OUTPUT_DIR = Path("/home/ycarmon/no_backup/users/sachter/wandb_mikeyshechter_100b_local")
DEFAULT_NUM_WORKERS = 8
DEFAULT_PAGE_SIZE = 10_000
MAX_DOWNLOAD_ATTEMPTS = 4

_LOG_LOCK = Lock()


def _import_pandas():
    import pandas as pd
    return pd


def _import_wandb():
    import wandb
    return wandb


def _pythonize(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _pythonize(nested) for key, nested in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_pythonize(item) for item in value]
    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            return _pythonize(item_method())
        except Exception:
            pass
    tolist_method = getattr(value, "tolist", None)
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
            raise argparse.ArgumentTypeError("Expected a W&B URL like https://wandb.ai/entity/project.")
        return "/".join(parts[:2])
    parts = [part for part in value.strip("/").split("/") if part]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Expected a W&B project path like entity/project.")
    return "/".join(parts)


def _run_created_at(run: Any) -> Optional[datetime]:
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
    with _LOG_LOCK:
        print(f"[{timestamp}] {message}", flush=True)


def _metric_dir(histories_dir: Path, metric: str) -> Path:
    # Metric paths like "val/ce_loss" become nested directories. Disallow
    # parent traversal just in case.
    parts = [part for part in metric.split("/") if part not in ("", ".", "..")]
    if not parts:
        raise ValueError(f"Invalid metric name: {metric!r}")
    return histories_dir.joinpath(*parts)


def _history_path(histories_dir: Path, metric: str, run_id: str) -> Path:
    return _metric_dir(histories_dir, metric) / f"{run_id}.csv.gz"


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


def _download_metric_history(
    run: Any,
    metric: str,
    history_path: Path,
    page_size: int,
) -> int:
    pd = _import_pandas()
    keys = ["_step", metric] if metric != "_step" else ["_step"]
    last_error: Optional[Exception] = None
    for attempt in range(1, MAX_DOWNLOAD_ATTEMPTS + 1):
        try:
            rows: List[Dict[str, Any]] = []
            for row in run.scan_history(keys=keys, page_size=page_size):
                rows.append({k: _pythonize(v) for k, v in row.items()})
            if rows:
                history_df = pd.DataFrame.from_records(rows)
                # Keep only the requested columns, in stable order.
                ordered = [c for c in ("_step", metric) if c in history_df.columns]
                history_df = history_df[ordered]
            else:
                history_df = pd.DataFrame(columns=keys)
            history_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = history_path.with_suffix(history_path.suffix + ".tmp")
            history_df.to_csv(tmp_path, index=False, compression="gzip")
            tmp_path.replace(history_path)
            return int(len(history_df))
        except Exception as exc:  # noqa: BLE001 - we want broad retry on network errors
            last_error = exc
            backoff = min(60.0, 2.0 ** attempt)
            _log(
                f"  attempt {attempt}/{MAX_DOWNLOAD_ATTEMPTS} failed for "
                f"{getattr(run, 'id', '?')} metric={metric!r}: {exc!r}; sleeping {backoff:.1f}s"
            )
            time.sleep(backoff)
    assert last_error is not None
    raise last_error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "On-demand, per-metric W&B local sync. "
            "Downloads run metadata and the full history for the requested metric(s)."
        )
    )
    parser.add_argument(
        "metrics",
        nargs="+",
        help="One or more W&B metric paths to download (e.g. 'val/ce_loss').",
    )
    parser.add_argument(
        "--project-path",
        type=_parse_project_path,
        default=DEFAULT_PROJECT_PATH,
        help=f"W&B project path or URL. Default: {DEFAULT_PROJECT_PATH}",
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
        default=DEFAULT_PAGE_SIZE,
        help="W&B history scan page size.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help="Number of concurrent download threads.",
    )
    parser.add_argument(
        "--force-history",
        action="store_true",
        help="Redownload metric history even if the local file already exists.",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Only refresh summary/metadata; skip metric histories entirely.",
    )
    parser.add_argument(
        "--max-matched-runs",
        type=int,
        default=None,
        help="Optional cap on the number of runs to consider. Useful for testing.",
    )
    parser.add_argument(
        "--exclude-name-substring",
        action="append",
        default=[],
        help="Skip runs whose display name contains this substring. Repeatable.",
    )
    parser.add_argument(
        "--created-at-on-or-after",
        type=_parse_datetime_arg,
        default=None,
        help="Only keep runs created at or after this UTC date/time.",
    )
    return parser.parse_args()


def _write_manifest(
    manifest_path: Path,
    *,
    project_path: str,
    output_dir: Path,
    metrics: List[str],
    scanned_runs: int,
    matched_runs: int,
    excluded_runs: int,
    exclude_substrings: List[str],
    created_at_on_or_after: Optional[datetime],
    page_size: int,
    num_workers: int,
    download_counts: Dict[str, int],
    skipped_counts: Dict[str, int],
    failed_counts: Dict[str, int],
    metadata_only: bool,
) -> None:
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_path": project_path,
        "output_dir": str(output_dir),
        "layout": "per_metric",
        "metrics": metrics,
        "client_side_filter": {
            "exclude_name_substrings": exclude_substrings or None,
            "created_at_on_or_after_utc": (
                created_at_on_or_after.isoformat() if created_at_on_or_after else None
            ),
        },
        "scanned_runs": scanned_runs,
        "matched_runs": matched_runs,
        "excluded_runs": excluded_runs,
        "metadata_only": metadata_only,
        "history_page_size": page_size,
        "num_workers": num_workers,
        "history_downloaded_runs_per_metric": download_counts,
        "history_skipped_runs_per_metric": skipped_counts,
        "history_failed_runs_per_metric": failed_counts,
        "summary_dir": str(output_dir / "summary"),
        "history_dir": str(output_dir / "histories"),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    pd = _import_pandas()
    wandb = _import_wandb()
    start_time = _now_utc()

    output_dir = args.output_dir.expanduser().resolve()
    summary_dir = output_dir / "summary"
    histories_dir = output_dir / "histories"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    histories_dir.mkdir(parents=True, exist_ok=True)
    for metric in args.metrics:
        _metric_dir(histories_dir, metric).mkdir(parents=True, exist_ok=True)

    project_path = args.project_path
    exclude_substrings = list(args.exclude_name_substring or [])
    created_at_cutoff = args.created_at_on_or_after

    _log("Starting per-metric W&B local sync.")
    _log(f"Project: {project_path}")
    _log(f"Output dir: {output_dir}")
    _log(f"Metrics: {args.metrics}")
    _log(f"Workers: {args.num_workers}, page_size: {args.page_size}")
    if created_at_cutoff is not None:
        _log(f"Keeping only runs created_at >= {created_at_cutoff.isoformat()}")
    if exclude_substrings:
        _log(f"Excluding runs whose name contains any of: {exclude_substrings}")

    api = wandb.Api(timeout=600)

    # ----- Phase 1: scan project, collect metadata for matching runs. -----
    _log("Scanning W&B runs (metadata phase)...")
    records: List[Dict[str, Any]] = []
    matched_runs_list: List[Any] = []
    scanned_runs = 0
    matched_runs = 0
    excluded_runs = 0

    for run in api.runs(project_path, per_page=1000, order="+created_at"):
        scanned_runs += 1
        run_name = _display_name(run)
        run_created_at = _run_created_at(run)
        if created_at_cutoff is not None:
            if run_created_at is None or run_created_at < created_at_cutoff:
                excluded_runs += 1
                continue
        if any(sub in run_name for sub in exclude_substrings):
            excluded_runs += 1
            _log(f"Excluding run {run.id} {run_name} (matches exclude filter)")
            continue

        matched_runs += 1
        raw_config = {k: v for k, v in dict(run.config).items() if not str(k).startswith("_")}
        flat_config = _flatten_mapping(raw_config)
        record = _build_run_record(run, flat_config)
        for metric in args.metrics:
            record[f"history_file.{metric}"] = str(_history_path(histories_dir, metric, run.id))
        records.append(record)
        matched_runs_list.append(run)

        if args.max_matched_runs is not None and matched_runs >= args.max_matched_runs:
            _log(f"Reached --max-matched-runs={args.max_matched_runs}. Stopping early.")
            break

    _log(
        f"Metadata phase complete: scanned={scanned_runs}, matched={matched_runs}, "
        f"excluded={excluded_runs}."
    )

    # Persist metadata immediately, before kicking off the (long) history phase.
    if records:
        runs_df = pd.DataFrame.from_records(records)
        runs_df = runs_df.sort_values(by=["created_at", "run_id"], kind="mergesort").reset_index(drop=True)
    else:
        runs_df = pd.DataFrame()

    metadata_csv_path = summary_dir / "selected_runs.csv"
    metadata_jsonl_path = summary_dir / "selected_runs.jsonl"
    runs_df.to_csv(metadata_csv_path, index=False)
    jsonl_text = runs_df.to_json(orient="records", lines=True) if not runs_df.empty else ""
    metadata_jsonl_path.write_text(jsonl_text + ("\n" if jsonl_text else ""))
    _log(f"Metadata saved to {metadata_csv_path}")

    download_counts: Dict[str, int] = {metric: 0 for metric in args.metrics}
    skipped_counts: Dict[str, int] = {metric: 0 for metric in args.metrics}
    failed_counts: Dict[str, int] = {metric: 0 for metric in args.metrics}

    if args.metadata_only or not matched_runs_list:
        if args.metadata_only:
            _log("--metadata-only set; skipping history downloads.")
        _write_manifest(
            output_dir / "manifest.json",
            project_path=project_path,
            output_dir=output_dir,
            metrics=list(args.metrics),
            scanned_runs=scanned_runs,
            matched_runs=matched_runs,
            excluded_runs=excluded_runs,
            exclude_substrings=exclude_substrings,
            created_at_on_or_after=created_at_cutoff,
            page_size=args.page_size,
            num_workers=args.num_workers,
            download_counts=download_counts,
            skipped_counts=skipped_counts,
            failed_counts=failed_counts,
            metadata_only=args.metadata_only,
        )
        _log("Done.")
        return

    # ----- Phase 2: parallel per-metric history downloads. -----
    tasks: List[tuple] = []  # (metric, run)
    for metric in args.metrics:
        for run in matched_runs_list:
            history_path = _history_path(histories_dir, metric, run.id)
            if history_path.exists() and not args.force_history:
                skipped_counts[metric] += 1
                continue
            tasks.append((metric, run, history_path))

    total_tasks = len(tasks)
    _log(
        f"History phase: {total_tasks} download tasks "
        f"(skipped existing: { {m: skipped_counts[m] for m in args.metrics} })."
    )

    completed = 0
    if total_tasks > 0:
        with ThreadPoolExecutor(max_workers=args.num_workers) as pool:
            future_to_task = {
                pool.submit(
                    _download_metric_history,
                    run,
                    metric,
                    history_path,
                    args.page_size,
                ): (metric, run, history_path)
                for metric, run, history_path in tasks
            }
            for future in as_completed(future_to_task):
                metric, run, history_path = future_to_task[future]
                completed += 1
                try:
                    n_rows = future.result()
                    download_counts[metric] += 1
                    _log(
                        f"[{completed}/{total_tasks}] downloaded {metric} for "
                        f"{run.id} {_display_name(run)} -> {n_rows} rows"
                    )
                except Exception as exc:  # noqa: BLE001
                    failed_counts[metric] += 1
                    _log(
                        f"[{completed}/{total_tasks}] FAILED {metric} for "
                        f"{run.id} {_display_name(run)}: {exc!r}"
                    )
                    traceback.print_exc()

    _write_manifest(
        output_dir / "manifest.json",
        project_path=project_path,
        output_dir=output_dir,
        metrics=list(args.metrics),
        scanned_runs=scanned_runs,
        matched_runs=matched_runs,
        excluded_runs=excluded_runs,
        exclude_substrings=exclude_substrings,
        created_at_on_or_after=created_at_cutoff,
        page_size=args.page_size,
        num_workers=args.num_workers,
        download_counts=download_counts,
        skipped_counts=skipped_counts,
        failed_counts=failed_counts,
        metadata_only=args.metadata_only,
    )

    elapsed_seconds = (_now_utc() - start_time).total_seconds()
    _log(
        f"Finished sync: scanned_runs={scanned_runs}, matched_runs={matched_runs}, "
        f"excluded_runs={excluded_runs}, "
        f"downloaded={download_counts}, skipped={skipped_counts}, failed={failed_counts}, "
        f"elapsed={elapsed_seconds:.1f}s"
    )


if __name__ == "__main__":
    main()
