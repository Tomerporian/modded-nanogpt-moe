from __future__ import annotations

import csv
import gzip
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple


DEFAULT_OUTPUT_DIR = Path("/home/ycarmon/no_backup/users/sachter/wandb_tomer_local")
DEFAULT_MERGE_KEY_FIELDS: Tuple[str, ...] = (
    "group",
    "config.output_dir",
    "config.output",
    "run_name",
)
LEGACY_LOAD_BALANCE_FLAGS = {
    "fsq": "fsq_load_balance",
    "maxvio": "maxvio_load_balance",
    "maxviosq": "maxviosq_load_balance",
    "minmaxvio": "minmaxvio_load_balance",
    "totalvio": "totalvio_load_balance",
}
_CHECKPOINT_STEP_RE = re.compile(r"state_step(\d+)\.pt$")


def _import_pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "pandas is required for wandb_local.py. Install it in your notebook "
            "environment before using these helpers."
        ) from exc
    return pd


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    try:
        pd = _import_pandas()
        return bool(pd.isna(value))
    except Exception:
        return False


def _pythonize(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def _is_true_like(value: Any) -> bool:
    if _is_missing(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _coalesce_values(*values: Any) -> Any:
    for value in values:
        if not _is_missing(value):
            return value
    return None


def _normalize_output_path(value: Any) -> Optional[str]:
    if _is_missing(value):
        return None
    normalized = str(value).strip().rstrip("/")
    marker = "logs/"
    marker_index = normalized.find(marker)
    if marker_index >= 0:
        normalized = normalized[marker_index:]
    return normalized or None


def _resume_output_dir(resume_value: Any) -> Optional[str]:
    if _is_missing(resume_value):
        return None
    resume_text = str(resume_value).strip()
    if not resume_text or resume_text == "auto":
        return None
    checkpoint_match = _CHECKPOINT_STEP_RE.search(resume_text)
    if checkpoint_match:
        return _normalize_output_path(resume_text[: checkpoint_match.start()].rstrip("/"))
    return _normalize_output_path(Path(resume_text).parent)


def _resume_step(resume_value: Any) -> Optional[int]:
    if _is_missing(resume_value):
        return None
    checkpoint_match = _CHECKPOINT_STEP_RE.search(str(resume_value).strip())
    if checkpoint_match is None:
        return None
    return int(checkpoint_match.group(1))


def _source_label(manifest: Mapping[str, Any], output_dir: Path) -> str:
    project_path = manifest.get("project_path")
    if isinstance(project_path, str) and project_path.strip():
        return project_path
    return output_dir.name


def _split_run_path(run_path: Any) -> Tuple[Optional[str], Optional[str]]:
    if _is_missing(run_path):
        return None, None
    parts = str(run_path).split("/", 2)
    if len(parts) < 2:
        return None, None
    return parts[0], parts[1]


def _load_manifest(output_dir: Path) -> Dict[str, Any]:
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text())


def _detect_layout(output_dir: Path, manifest: Mapping[str, Any]) -> str:
    layout = manifest.get("layout") if isinstance(manifest, Mapping) else None
    if isinstance(layout, str) and layout:
        return layout
    if (output_dir / "summary" / "selected_runs.csv").exists():
        return "per_metric"
    return "flat"


def _metadata_csv_path(output_dir: Path, layout: str) -> Path:
    if layout == "per_metric":
        return output_dir / "summary" / "selected_runs.csv"
    return output_dir / "selected_runs.csv"


def _history_file_for_row(
    row: Mapping[str, Any],
    output_dir: Path,
    *,
    metric: Optional[str] = None,
) -> Path:
    if metric is not None:
        per_metric = row.get(f"history_file.{metric}")
        if not _is_missing(per_metric):
            return Path(str(per_metric)).expanduser()
        run_id = row.get("run_id")
        parts = [part for part in metric.split("/") if part not in ("", ".", "..")]
        return output_dir.joinpath("histories", *parts, f"{run_id}.csv.gz")
    history_file = row.get("history_file")
    if not _is_missing(history_file):
        return Path(str(history_file)).expanduser()
    run_id = row.get("run_id")
    return output_dir / "histories" / f"{run_id}.csv.gz"


def _auto_merge_key(record: Mapping[str, Any], merge_key_fields: Sequence[str]) -> str:
    for key in merge_key_fields:
        value = record.get(key)
        if not _is_missing(value):
            return f"{key}:{value}"
    return f"run_id:{record['run_id']}"


def _as_tuple(values: Iterable[Any]) -> Tuple[Any, ...]:
    normalized = []
    for value in values:
        if _is_missing(value):
            normalized.append(None)
        else:
            normalized.append(_pythonize(value))
    return tuple(normalized)


def _optional_series(frame: Any, column: str) -> Any:
    pd = _import_pandas()
    if column in frame.columns:
        return frame[column]
    return pd.Series([None] * len(frame), index=frame.index)


def _frame_column_values(frame: Any, column: str) -> Sequence[Any]:
    if column in frame.columns:
        return frame[column].tolist()
    return [None] * len(frame)


def _row_config_value(row: Mapping[str, Any], key: str) -> Any:
    for candidate in (key, f"config.{key}"):
        if candidate in row and not _is_missing(row[candidate]):
            return row[candidate]
    return None


def _infer_load_balance_loss(row: Mapping[str, Any]) -> str:
    explicit = _row_config_value(row, "load_balance_loss")
    if not _is_missing(explicit):
        return str(explicit)
    for loss_name, flag_key in LEGACY_LOAD_BALANCE_FLAGS.items():
        if _is_true_like(_row_config_value(row, flag_key)):
            return loss_name
    return "switch"


def _normalize_rect_ste_threshold(value: Any) -> str:
    if _is_missing(value):
        return "topk"
    return str(value)


def _normalize_approx_global_load_balance(value: Any) -> bool:
    if _is_missing(value):
        return False
    return _is_true_like(value)


def _safe_perplexity(value: Any) -> Any:
    if _is_missing(value):
        return None
    try:
        return math.exp(float(value))
    except (TypeError, ValueError, OverflowError):
        return None


def _build_component_runs(output_dir: Path, source_label: str, layout: str) -> Any:
    pd = _import_pandas()
    metadata_path = _metadata_csv_path(output_dir, layout)
    component_runs = pd.read_csv(metadata_path, low_memory=False)
    if component_runs.empty:
        return component_runs

    if "created_at" in component_runs.columns:
        component_runs["created_at"] = pd.to_datetime(component_runs["created_at"], utc=True)

    component_runs = component_runs.copy()
    component_runs["source_label"] = source_label
    component_runs["component_id"] = (
        component_runs["source_label"].astype(str) + "::" + component_runs["run_id"].astype(str)
    )
    entities = []
    projects = []
    for run_path in component_runs.get("run_path", []):
        entity, project = _split_run_path(run_path)
        entities.append(entity)
        projects.append(project)
    if entities:
        component_runs["entity"] = entities
        component_runs["project"] = projects

    component_runs["output"] = component_runs.apply(
        lambda row: _coalesce_values(row.get("config.output_dir"), row.get("config.output")),
        axis=1,
    )
    component_runs["output"] = component_runs["output"].map(_normalize_output_path)
    resume_series = _optional_series(component_runs, "config.resume")
    component_runs["resume_output_dir"] = resume_series.map(_resume_output_dir)
    component_runs["resume_step"] = resume_series.map(_resume_step)
    if layout == "flat":
        component_runs["history_file"] = component_runs.apply(
            lambda row: str(_history_file_for_row(row, output_dir)),
            axis=1,
        )
        component_runs["history_exists"] = component_runs["history_file"].map(
            lambda path: Path(path).exists()
        )
    else:
        # In per-metric layout, history existence depends on which metric is
        # being requested; compute lazily in the bundle helpers.
        component_runs["history_file"] = None
        component_runs["history_exists"] = False
    component_runs["load_balance_loss"] = component_runs.apply(_infer_load_balance_loss, axis=1)
    component_runs["rect_ste_threshold"] = _optional_series(
        component_runs,
        "config.rect_ste_threshold",
    ).map(_normalize_rect_ste_threshold)
    component_runs["approx_global_load_balance"] = _optional_series(
        component_runs,
        "config.approx_global_load_balance",
    ).map(_normalize_approx_global_load_balance)
    if "summary.val/ce_loss" in component_runs.columns:
        component_runs["summary.val/perplexity"] = component_runs["summary.val/ce_loss"].map(_safe_perplexity)
    for loss_name, flag_key in LEGACY_LOAD_BALANCE_FLAGS.items():
        config_column = f"config.{flag_key}"
        if config_column in component_runs.columns:
            component_runs[flag_key] = component_runs[config_column].map(_is_true_like)
        else:
            component_runs[flag_key] = component_runs["load_balance_loss"] == loss_name

    sort_columns = [column for column in ("created_at", "run_id") if column in component_runs.columns]
    if sort_columns:
        component_runs = component_runs.sort_values(by=sort_columns, kind="mergesort")
    return component_runs.reset_index(drop=True)


def _build_logical_runs(component_runs: Any, merge_key_fields: Sequence[str]) -> Tuple[Any, Any]:
    pd = _import_pandas()
    if component_runs.empty:
        return component_runs.copy(), component_runs.copy()

    component_runs = component_runs.copy()
    component_runs["merge_key"] = component_runs.apply(
        lambda row: _auto_merge_key(row, merge_key_fields),
        axis=1,
    )
    component_runs["logical_run_id"] = (
        component_runs["source_label"].astype(str) + "::" + component_runs["merge_key"].astype(str)
    )

    logical_records = []
    grouped = component_runs.groupby("logical_run_id", sort=False)
    for logical_run_id, group_df in grouped:
        ordered = group_df.sort_values(by=["created_at", "run_id"], kind="mergesort").reset_index(drop=True)
        if "summary._step" in ordered.columns:
            # Use the component that reached the highest step as the representative
            # for summary fields. A later wall-clock resume may have failed early
            # and reached fewer steps than an earlier component; picking by
            # created_at in that case yields stale summary values.
            step_series = pd.to_numeric(ordered["summary._step"], errors="coerce")
            max_idx = step_series.fillna(-1).idxmax()
            final_row = ordered.loc[max_idx].copy()
        else:
            final_row = ordered.iloc[-1].copy()
        logical_record = final_row.to_dict()
        logical_record["logical_run_id"] = logical_run_id
        logical_record["component_count"] = int(len(ordered))
        logical_record["component_ids"] = _as_tuple(ordered["component_id"].tolist())
        logical_record["component_run_ids"] = _as_tuple(ordered["run_id"].tolist())
        logical_record["component_run_paths"] = _as_tuple(_frame_column_values(ordered, "run_path"))
        logical_record["component_states"] = _as_tuple(_frame_column_values(ordered, "state"))
        logical_record["component_created_at"] = _as_tuple(ordered["created_at"].tolist())
        logical_record["component_history_files"] = _as_tuple(ordered["history_file"].tolist())
        logical_record["component_resume_steps"] = _as_tuple(ordered["resume_step"].tolist())
        logical_record["first_created_at"] = ordered.iloc[0]["created_at"]
        logical_record["last_created_at"] = ordered.iloc[-1]["created_at"]
        logical_record["history_exists"] = bool(ordered["history_exists"].any())
        logical_record["has_resume"] = bool(len(ordered) > 1 or ordered["resume_output_dir"].notna().any())
        logical_record["output"] = _coalesce_values(
            final_row.get("output"),
            final_row.get("config.output_dir"),
            final_row.get("config.output"),
        )
        logical_record["resume_output_dirs"] = _as_tuple(
            value for value in ordered["resume_output_dir"].tolist() if not _is_missing(value)
        )
        logical_records.append(logical_record)

    logical_runs = pd.DataFrame.from_records(logical_records)
    if logical_runs.empty:
        return component_runs.reset_index(drop=True), logical_runs

    logical_runs = logical_runs.sort_values(
        by=["first_created_at", "logical_run_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    return component_runs.reset_index(drop=True), logical_runs


def _history_usecols(columns: Optional[Sequence[str]]):
    if columns is None:
        return None
    requested = {column for column in columns if column != "step"}
    if "val/perplexity" in requested:
        requested.add("val/ce_loss")
    return lambda column: column == "_step" or column in requested


def _empty_history_frame(columns: Optional[Sequence[str]] = None) -> Any:
    pd = _import_pandas()
    requested = list(columns or ())
    ordered_columns = ["step", *[column for column in requested if column != "step"]]
    return pd.DataFrame(columns=ordered_columns)


def _collapse_step_duplicates(history_df: Any) -> Any:
    if history_df.empty:
        return history_df
    if not history_df["step"].duplicated().any():
        return history_df.sort_values(by="step", kind="mergesort").reset_index(drop=True)
    return history_df.groupby("step", as_index=False, sort=True).last().reset_index(drop=True)


def _read_history_file(path: Path, columns: Optional[Sequence[str]] = None) -> Any:
    pd = _import_pandas()
    if not path.exists():
        return _empty_history_frame(columns)

    try:
        history_df = pd.read_csv(path, usecols=_history_usecols(columns), low_memory=False)
    except pd.errors.EmptyDataError:
        return _empty_history_frame(columns)

    history_df = history_df.loc[:, ~history_df.columns.astype(str).str.startswith("Unnamed:")].copy()
    if "_step" not in history_df.columns:
        return _empty_history_frame(columns)

    history_df = history_df.rename(columns={"_step": "step"})
    history_df = history_df.dropna(subset=["step"]).copy()
    if history_df.empty:
        return _empty_history_frame(columns)

    history_df["step"] = history_df["step"].astype(int)
    if "val/ce_loss" in history_df.columns:
        history_df["val/perplexity"] = history_df["val/ce_loss"].map(_safe_perplexity)
    if columns is not None:
        requested = [column for column in columns if column != "step"]
        for column in requested:
            if column not in history_df.columns:
                history_df[column] = pd.NA
        history_df = history_df[["step", *requested]]

    return _collapse_step_duplicates(history_df)


def _ordered_history_columns(columns: Iterable[str]) -> Tuple[str, ...]:
    unique_columns = []
    seen = set()
    for column in columns:
        normalized = "step" if column == "_step" else column
        if normalized in seen:
            continue
        seen.add(normalized)
        unique_columns.append(normalized)
    if "step" in seen:
        unique_columns = ["step", *[column for column in unique_columns if column != "step"]]
    return tuple(unique_columns)


def _iter_history_header_columns(path: Path) -> Tuple[str, ...]:
    if not path.exists():
        return ("step",)
    with gzip.open(path, "rt", newline="") as handle:
        header_line = handle.readline()
    if not header_line:
        return ("step",)
    reader = csv.reader([header_line])
    try:
        header = next(reader)
    except StopIteration:
        return ("step",)
    return _ordered_history_columns(
        column for column in header if column and not str(column).startswith("Unnamed:")
    )


def _resolve_column(frame: Any, key: str) -> str:
    preferred_columns = {
        "output": ("output", "config.output_dir", "config.output"),
        "output_dir": ("config.output_dir", "output", "config.output"),
        "resume": ("config.resume",),
        "resume_output_dir": ("resume_output_dir",),
        "resume_step": ("resume_step",),
    }
    candidates = preferred_columns.get(key, ())
    candidates = (*candidates, key, f"config.{key}", f"summary.{key}")
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    available = ", ".join(sorted(str(column) for column in frame.columns[:25]))
    raise KeyError(f"Could not resolve column '{key}'. Sample available columns: {available}")


def _resolve_target_rows(frame: Any, target: str, *, logical: bool) -> Any:
    pd = _import_pandas()
    if logical:
        masks = []
        for column in (
            "component_id",
            "logical_run_id",
            "merge_key",
            "run_name",
            "output",
            "config.output_dir",
            "config.output",
        ):
            if column in frame.columns:
                masks.append(frame[column] == target)
        if "component_run_ids" in frame.columns:
            masks.append(frame["component_run_ids"].map(lambda ids: target in ids if isinstance(ids, tuple) else False))
    else:
        masks = []
        for column in (
            "component_id",
            "run_id",
            "run_name",
            "run_path",
            "output",
            "config.output_dir",
            "config.output",
        ):
            if column in frame.columns:
                masks.append(frame[column] == target)

    if not masks:
        return frame.iloc[0:0].copy()

    mask = pd.Series(False, index=frame.index)
    for partial_mask in masks:
        mask = mask | partial_mask.fillna(False)
    return frame.loc[mask].copy()


@dataclass
class LocalWandbBundle:
    output_dir: Path
    manifest: Mapping[str, Any]
    component_runs: Any
    logical_runs: Any
    merge_key_fields: Tuple[str, ...] = DEFAULT_MERGE_KEY_FIELDS
    layout: str = "flat"
    _history_cache: Dict[Tuple[str, Optional[Tuple[str, ...]]], Any] = field(default_factory=dict, repr=False)
    _history_columns_cache: Dict[str, Tuple[str, ...]] = field(default_factory=dict, repr=False)
    _all_history_columns_cache: Optional[Tuple[str, ...]] = field(default=None, repr=False)

    def _resolve_history_path(self, row: Mapping[str, Any], metric: Optional[str]) -> Path:
        if self.layout == "per_metric":
            if metric is None:
                raise ValueError(
                    "This bundle uses the per_metric layout; pass metric=... "
                    "(e.g. metric='val/ce_loss') to read history."
                )
            return _history_file_for_row(row, self.output_dir, metric=metric)
        return _history_file_for_row(row, self.output_dir)

    def _effective_columns(
        self,
        metric: Optional[str],
        columns: Optional[Sequence[str]],
    ) -> Optional[Tuple[str, ...]]:
        if self.layout == "per_metric":
            if metric is None:
                raise ValueError(
                    "This bundle uses the per_metric layout; pass metric=... "
                    "(e.g. metric='val/ce_loss') to read history."
                )
            return (metric,)
        if columns is None:
            return None
        return tuple(columns)

    def select_runs(
        self,
        *,
        component: bool = False,
        query: Optional[str] = None,
        **filters: Any,
    ) -> Any:
        frame = self.component_runs if component else self.logical_runs
        result = frame
        for key, expected in filters.items():
            column = _resolve_column(result, key)
            if isinstance(expected, Sequence) and not isinstance(expected, (str, bytes)):
                result = result[result[column].isin(list(expected))]
            else:
                result = result[result[column] == expected]
        if query:
            result = result.query(query)
        return result.copy().reset_index(drop=True)

    def component_history(
        self,
        target: str,
        *,
        columns: Optional[Sequence[str]] = None,
        metric: Optional[str] = None,
    ) -> Any:
        component_matches = _resolve_target_rows(self.component_runs, target, logical=False)
        if component_matches.empty:
            raise KeyError(f"Unknown component target '{target}'.")
        if len(component_matches) > 1:
            matches = component_matches[["component_id", "run_id", "run_name", "source_label"]].to_string(index=False)
            raise ValueError(f"Component target '{target}' is ambiguous:\n{matches}")
        component_id = str(component_matches.iloc[0]["component_id"])
        effective_columns = self._effective_columns(metric, columns)

        full_cache_key = (component_id, None)
        if self.layout == "flat" and full_cache_key in self._history_cache:
            history_df = self._history_cache[full_cache_key]
            if effective_columns is None:
                return history_df.copy()
            ordered_columns = ["step", *[c for c in effective_columns if c != "step"]]
            history_df = history_df.copy()
            for column in ordered_columns:
                if column not in history_df.columns:
                    history_df[column] = None
            return history_df[ordered_columns].copy()

        normalized_columns: Optional[Tuple[str, ...]] = None
        if effective_columns is not None:
            normalized_columns = tuple(c for c in effective_columns if c != "step")
        cache_key = (component_id, normalized_columns)
        if cache_key in self._history_cache:
            return self._history_cache[cache_key].copy()

        history_path = self._resolve_history_path(component_matches.iloc[0], metric)
        history_df = _read_history_file(history_path, columns=effective_columns)
        self._history_cache[cache_key] = history_df
        return history_df.copy()

    def history(
        self,
        target: str,
        *,
        columns: Optional[Sequence[str]] = None,
        metric: Optional[str] = None,
        include_component_columns: bool = False,
    ) -> Any:
        pd = _import_pandas()
        logical_matches = _resolve_target_rows(self.logical_runs, target, logical=True)
        if logical_matches.empty:
            component_matches = _resolve_target_rows(self.component_runs, target, logical=False)
            if component_matches.empty:
                raise KeyError(f"Could not find a logical run or component run matching '{target}'.")
            logical_run_id = component_matches.iloc[0]["logical_run_id"]
            logical_matches = self.logical_runs[self.logical_runs["logical_run_id"] == logical_run_id]

        if len(logical_matches) > 1:
            matches = logical_matches[["logical_run_id", "run_name", "merge_key"]].to_string(index=False)
            raise ValueError(f"Target '{target}' is ambiguous across logical runs:\n{matches}")

        logical_row = logical_matches.iloc[0]
        ordered_components = (
            self.component_runs[self.component_runs["logical_run_id"] == logical_row["logical_run_id"]]
            .sort_values(by=["created_at", "run_id"], kind="mergesort")
            .reset_index(drop=True)
        )

        effective_columns = self._effective_columns(metric, columns)

        merged_parts = []
        for component_index, component in ordered_components.iterrows():
            component_history = self.component_history(
                str(component["component_id"]),
                columns=columns,
                metric=metric,
            )
            if component_history.empty:
                continue
            component_history = component_history.copy()
            component_history["component_id"] = component["component_id"]
            component_history["component_run_id"] = component["run_id"]
            component_history["component_index"] = component_index
            component_history["component_created_at"] = component["created_at"]
            merged_parts.append(component_history)

        if not merged_parts:
            return _empty_history_frame(effective_columns)

        history_df = pd.concat(merged_parts, ignore_index=True, sort=False)
        history_df = history_df.sort_values(
            by=["step", "component_index", "component_run_id"],
            kind="mergesort",
        )
        history_df = history_df.groupby("step", as_index=False, sort=True).last().reset_index(drop=True)
        history_df["logical_run_id"] = logical_row["logical_run_id"]
        history_df["run_name"] = logical_row["run_name"]
        history_df["merge_key"] = logical_row["merge_key"]

        ordered_columns = [
            "logical_run_id",
            "run_name",
            "merge_key",
            "step",
        ]
        if include_component_columns:
            ordered_columns.extend(["component_id", "component_run_id", "component_index", "component_created_at"])
        metric_columns = [
            column
            for column in history_df.columns
            if column not in {
                "logical_run_id",
                "run_name",
                "merge_key",
                "step",
                "component_id",
                "component_run_id",
                "component_index",
                "component_created_at",
            }
        ]
        return history_df[[*ordered_columns, *metric_columns]].copy()

    def history_for_runs(
        self,
        logical_run_ids: Optional[Sequence[str]] = None,
        *,
        columns: Optional[Sequence[str]] = None,
        metric: Optional[str] = None,
        query: Optional[str] = None,
        allow_full_history: bool = False,
        **filters: Any,
    ) -> Any:
        pd = _import_pandas()
        if logical_run_ids is None:
            selected_runs = self.select_runs(query=query, **filters)
        else:
            selected_runs = self.logical_runs[
                self.logical_runs["logical_run_id"].isin(list(logical_run_ids))
            ].copy()
            if query:
                selected_runs = selected_runs.query(query)
            if filters:
                selected_ids = set(
                    self.select_runs(query=query, **filters)["logical_run_id"].tolist()
                )
                selected_runs = selected_runs[
                    selected_runs["logical_run_id"].isin(selected_ids)
                ].copy()

        effective_columns = self._effective_columns(metric, columns)

        if selected_runs.empty:
            return _empty_history_frame(effective_columns)

        if (
            self.layout == "flat"
            and effective_columns is None
            and len(selected_runs) > 1
            and not allow_full_history
        ):
            raise ValueError(
                "Refusing to load every history column for multiple runs. "
                "Pass columns=[...] or set allow_full_history=True."
            )

        history_frames = [
            self.history(logical_run_id, columns=columns, metric=metric)
            for logical_run_id in selected_runs["logical_run_id"].tolist()
        ]
        if not history_frames:
            return _empty_history_frame(effective_columns)

        history_df = pd.concat(history_frames, ignore_index=True, sort=False)
        return history_df.sort_values(
            by=["logical_run_id", "step"],
            kind="mergesort",
        ).reset_index(drop=True)

    def available_metrics(self, target: Optional[str] = None) -> Tuple[str, ...]:
        """List metrics that have history files in this per_metric bundle.

        If ``target`` is None, returns the union across all runs.
        For the flat layout, returns the columns from the (single) history files.
        """
        histories_root = self.output_dir / "histories"
        if not histories_root.exists():
            return ()
        if self.layout != "per_metric":
            return self.available_history_columns(target)

        if target is None:
            run_ids = set(self.component_runs["run_id"].astype(str).tolist())
        else:
            component_matches = _resolve_target_rows(self.component_runs, target, logical=False)
            if component_matches.empty:
                logical_matches = _resolve_target_rows(self.logical_runs, target, logical=True)
                if logical_matches.empty:
                    raise KeyError(f"Unknown target '{target}'.")
                logical_run_id = logical_matches.iloc[0]["logical_run_id"]
                run_ids = set(
                    self.component_runs.loc[
                        self.component_runs["logical_run_id"] == logical_run_id,
                        "run_id",
                    ].astype(str).tolist()
                )
            else:
                run_ids = set(component_matches["run_id"].astype(str).tolist())

        metrics: set = set()
        for history_path in histories_root.rglob("*.csv.gz"):
            run_id = history_path.stem.rsplit(".csv", 1)[0]
            if run_id not in run_ids:
                continue
            relative = history_path.parent.relative_to(histories_root)
            metric = "/".join(relative.parts)
            if metric:
                metrics.add(metric)
        return tuple(sorted(metrics))

    def available_history_columns(self, target: Optional[str] = None) -> Tuple[str, ...]:
        if self.layout == "per_metric":
            return self.available_metrics(target)
        if target is None and self._all_history_columns_cache is not None:
            return self._all_history_columns_cache

        if target is None:
            columns = set()
            for run_id in self.component_runs["run_id"].tolist():
                columns.update(self.available_history_columns(run_id))
            ordered = _ordered_history_columns(columns)
            self._all_history_columns_cache = ordered
            return ordered

        logical_matches = _resolve_target_rows(self.logical_runs, target, logical=True)
        if not logical_matches.empty:
            columns = set()
            logical_run_id = logical_matches.iloc[0]["logical_run_id"]
            component_rows = self.component_runs[
                self.component_runs["logical_run_id"] == logical_run_id
            ]
            for component_id in component_rows["component_id"].tolist():
                columns.update(self.available_history_columns(component_id))
            return _ordered_history_columns(columns)

        component_matches = _resolve_target_rows(self.component_runs, target, logical=False)
        if component_matches.empty:
            raise KeyError(f"Could not find a logical run or component run matching '{target}'.")
        if len(component_matches) > 1:
            matches = component_matches[["run_id", "run_name"]].to_string(index=False)
            raise ValueError(f"Target '{target}' is ambiguous across component runs:\n{matches}")

        component_id = str(component_matches.iloc[0]["component_id"])
        if component_id in self._history_columns_cache:
            return self._history_columns_cache[component_id]

        history_path = Path(component_matches.iloc[0]["history_file"])
        columns = _iter_history_header_columns(history_path)
        self._history_columns_cache[component_id] = columns
        return columns

    def merged_with(self, *others: "LocalWandbBundle") -> "LocalWandbBundle":
        return merge_local_wandb_runs(self, *others)


def merge_local_wandb_runs(*bundles: LocalWandbBundle) -> LocalWandbBundle:
    pd = _import_pandas()
    if not bundles:
        raise ValueError("merge_local_wandb_runs requires at least one bundle.")

    merge_key_fields = bundles[0].merge_key_fields
    for bundle in bundles[1:]:
        if bundle.merge_key_fields != merge_key_fields:
            raise ValueError(
                "All bundles must use the same merge_key_fields. "
                f"Got {merge_key_fields!r} and {bundle.merge_key_fields!r}."
            )

    layouts = {bundle.layout for bundle in bundles}
    if len(layouts) > 1:
        raise ValueError(
            "All bundles must share the same layout to merge. "
            f"Got mixed layouts: {sorted(layouts)}."
        )
    merged_layout = next(iter(layouts))

    component_frames = [bundle.component_runs for bundle in bundles if not bundle.component_runs.empty]
    logical_frames = [bundle.logical_runs for bundle in bundles if not bundle.logical_runs.empty]
    component_runs = pd.concat(component_frames, ignore_index=True, sort=False) if component_frames else pd.DataFrame()
    logical_runs = pd.concat(logical_frames, ignore_index=True, sort=False) if logical_frames else pd.DataFrame()

    if not component_runs.empty:
        component_runs = component_runs.sort_values(
            by=[column for column in ("source_label", "created_at", "run_id") if column in component_runs.columns],
            kind="mergesort",
        ).reset_index(drop=True)
    if not logical_runs.empty:
        logical_runs = logical_runs.sort_values(
            by=[column for column in ("source_label", "first_created_at", "logical_run_id") if column in logical_runs.columns],
            kind="mergesort",
        ).reset_index(drop=True)

    merged_manifest = {
        "merged": True,
        "bundle_count": len(bundles),
        "source_labels": tuple(bundle.manifest.get("project_path", bundle.output_dir.name) for bundle in bundles),
        "manifests": tuple(bundle.manifest for bundle in bundles),
        "output_dirs": tuple(str(bundle.output_dir) for bundle in bundles),
    }
    return LocalWandbBundle(
        output_dir=bundles[0].output_dir,
        manifest=merged_manifest,
        component_runs=component_runs,
        logical_runs=logical_runs,
        merge_key_fields=merge_key_fields,
        layout=merged_layout,
    )


def load_local_wandb_runs(
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    *,
    merge_key_fields: Sequence[str] = DEFAULT_MERGE_KEY_FIELDS,
) -> LocalWandbBundle:
    """Load a local W&B dump created by sync_*_wandb_local.py.

    Auto-detects layout:
      * ``flat`` (legacy) – one history CSV per run at
        ``<output_dir>/histories/<run_id>.csv.gz`` and metadata at
        ``<output_dir>/selected_runs.csv``.
      * ``per_metric`` – one CSV per (metric, run) at
        ``<output_dir>/histories/<metric>/<run_id>.csv.gz`` and metadata at
        ``<output_dir>/summary/selected_runs.csv``. In this layout the history
        helpers require ``metric=...``.

    Examples:
        # flat layout
        runs = load_local_wandb_runs("/path/to/wandb_tomer_local")
        history = runs.history(
            "003_26-03-26-rect_ste+top_ste_wid=1.0",
            columns=["train/loss", "val/ce_loss", "val/MaxVioglobal"],
        )

        # per-metric layout
        runs = load_local_wandb_runs("/path/to/wandb_mikeyshechter_100b_local")
        ce = runs.history_for_runs(
            logical_run_ids=[...], metric="val/ce_loss",
        )
    """

    resolved_output_dir = Path(output_dir).expanduser().resolve()
    manifest = _load_manifest(resolved_output_dir)
    source_label = _source_label(manifest, resolved_output_dir)
    layout = _detect_layout(resolved_output_dir, manifest)
    component_runs = _build_component_runs(resolved_output_dir, source_label, layout)
    component_runs, logical_runs = _build_logical_runs(component_runs, tuple(merge_key_fields))
    return LocalWandbBundle(
        output_dir=resolved_output_dir,
        manifest=manifest,
        component_runs=component_runs,
        logical_runs=logical_runs,
        merge_key_fields=tuple(merge_key_fields),
        layout=layout,
    )
