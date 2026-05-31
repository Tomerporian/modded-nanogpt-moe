from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple


DEFAULT_METRICS: Tuple[str, ...] = ("val/ce_loss", "val/MaxVioglobal")

LEGACY_LOAD_BALANCE_FLAGS = {
    "fsq": "fsq_load_balance",
    "maxvio": "maxvio_load_balance",
    "maxviosq": "maxviosq_load_balance",
    "minmaxvio": "minmaxvio_load_balance",
    "totalvio": "totalvio_load_balance",
}

NORMALIZED_CONFIG_ALIASES = {
    "aux_coeff_train": ("aux_coeff_train", "loss.aux_coeff_train"),
    "aux_coeff_val": ("aux_coeff_val", "loss.aux_coeff_val"),
    "global_load_balance": ("global_load_balance", "loss.global_load_balance"),
    "load_balance_loss": ("load_balance_loss", "loss.load_balance_loss"),
    "load_balance_ste_width": ("load_balance_ste_width", "loss.load_balance_ste_width"),
    "num_experts": ("num_experts", "model.num_experts"),
    "output": ("output", "training.output"),
    "output_dir": ("output_dir", "training.output_dir"),
    "router_activation": ("router_activation", "model.router_activation"),
    "router_depth": ("router_depth", "model.router_depth", "model_router_depth"),
    "router_type": ("router_type", "model.router_type", "model_router_type"),
    "seed": ("seed", "training.seed"),
    "top_k": ("top_k", "model.top_k", "model_router_top_k"),
}

DEFAULT_DIFF_IGNORE = {
    "component_count",
    "component_created_at",
    "component_run_ids",
    "component_run_paths",
    "component_states",
    "created_at",
    "entity",
    "group",
    "history_rows",
    "last_step",
    "logical_run_id",
    "merge_key",
    "output",
    "output_dir",
    "project",
    "project_path",
    "run_name",
    "run_url",
    "source_label",
}

_MISSING = object()


RunPredicate = Callable[[Any, Dict[str, Any], Dict[str, Any]], bool]


def _import_pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "pandas is required for wandb_compare.py. Install it in your notebook "
            "environment before using these helpers."
        ) from exc
    return pd


def _import_wandb():
    try:
        import wandb
    except ImportError as exc:
        raise ImportError(
            "wandb is required for wandb_compare.py. Install it in your notebook "
            "environment before using these helpers."
        ) from exc
    return wandb


def _flatten_mapping(mapping: Mapping[str, Any], prefix: str = "") -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for key, value in mapping.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flat.update(_flatten_mapping(value, prefix=full_key))
        else:
            flat[full_key] = value
    return flat


def _lookup_config_value(flat_config: Mapping[str, Any], key: str) -> Any:
    if key in flat_config:
        return flat_config[key]
    for alias in NORMALIZED_CONFIG_ALIASES.get(key, ()):
        if alias in flat_config:
            return flat_config[alias]
    return _MISSING


def _is_true_config_value(value: Any) -> bool:
    return value is not _MISSING and bool(value)


def _infer_load_balance_loss(flat_config: Mapping[str, Any]) -> Any:
    explicit = _lookup_config_value(flat_config, "load_balance_loss")
    if explicit is not _MISSING and explicit not in (None, ""):
        return explicit
    for loss_name, flag_key in LEGACY_LOAD_BALANCE_FLAGS.items():
        if _is_true_config_value(_lookup_config_value(flat_config, flag_key)):
            return loss_name
    return None


def _normalize_run_config(flat_config: Mapping[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    for key in NORMALIZED_CONFIG_ALIASES:
        value = _lookup_config_value(flat_config, key)
        if value is not _MISSING:
            normalized[key] = value

    normalized["load_balance_loss"] = _infer_load_balance_loss(flat_config)

    for loss_name, flag_key in LEGACY_LOAD_BALANCE_FLAGS.items():
        value = _lookup_config_value(flat_config, flag_key)
        if value is _MISSING:
            normalized[flag_key] = normalized["load_balance_loss"] == loss_name
        else:
            normalized[flag_key] = bool(value)

    return normalized


def _coerce_iterable(values: Sequence[Any]) -> Tuple[Any, ...]:
    if isinstance(values, tuple):
        return values
    return tuple(values)


def _pythonize(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


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
    if isinstance(path, Sequence) and not isinstance(path, str):
        return "/".join(str(part) for part in path)
    return str(path or "")


def _run_url(run: Any) -> str:
    path = _run_path(run)
    if path:
        entity, project, run_id = path.split("/", 2)
        return f"https://wandb.ai/{entity}/{project}/runs/{run_id}"
    run_id = getattr(run, "id", "")
    return f"https://wandb.ai/runs/{run_id}" if run_id else ""


@dataclass
class RunSelector:
    entity: str
    project: str
    label: str
    name_contains_any: Tuple[str, ...] = ()
    name_contains_all: Tuple[str, ...] = ()
    config_equals: Mapping[str, Any] = field(default_factory=dict)
    config_in: Mapping[str, Sequence[Any]] = field(default_factory=dict)
    config_true_any: Tuple[str, ...] = ()
    config_true_all: Tuple[str, ...] = ()
    exclude_tags: Tuple[str, ...] = ()
    api_filters: Optional[Mapping[str, Any]] = None
    predicate: Optional[RunPredicate] = None

    def project_path(self) -> str:
        return f"{self.entity}/{self.project}"


@dataclass
class WandbComparisonBundle:
    component_runs: Any
    logical_runs: Any
    history: Any
    varying_params: Any

    def history_long(self, metrics: Optional[Sequence[str]] = None):
        return history_to_long(self.history, metrics=metrics)


def _selector_value(flat_config: Mapping[str, Any], normalized_config: Mapping[str, Any], key: str) -> Any:
    if key in normalized_config:
        return normalized_config[key]
    value = _lookup_config_value(flat_config, key)
    if value is not _MISSING:
        return value
    return flat_config.get(key, _MISSING)


def _selector_matches(
    selector: RunSelector,
    run: Any,
    flat_config: Dict[str, Any],
    normalized_config: Dict[str, Any],
) -> bool:
    display_name = _display_name(run)

    if selector.exclude_tags:
        tags = set(getattr(run, "tags", ()) or ())
        if tags.intersection(selector.exclude_tags):
            return False

    if selector.name_contains_any and not any(token in display_name for token in selector.name_contains_any):
        return False
    if selector.name_contains_all and not all(token in display_name for token in selector.name_contains_all):
        return False

    for key, expected in selector.config_equals.items():
        value = _selector_value(flat_config, normalized_config, key)
        if value is _MISSING or value != expected:
            return False

    for key, allowed in selector.config_in.items():
        value = _selector_value(flat_config, normalized_config, key)
        if value is _MISSING or value not in set(_coerce_iterable(allowed)):
            return False

    if selector.config_true_any:
        matched_any = False
        for key in selector.config_true_any:
            value = _selector_value(flat_config, normalized_config, key)
            if _is_true_config_value(value):
                matched_any = True
                break
        if not matched_any:
            return False

    for key in selector.config_true_all:
        value = _selector_value(flat_config, normalized_config, key)
        if not _is_true_config_value(value):
            return False

    if selector.predicate is not None and not selector.predicate(run, flat_config, normalized_config):
        return False

    return True


def _api_filter_field(key: str) -> str:
    aliases = NORMALIZED_CONFIG_ALIASES.get(key)
    if aliases:
        return f"config.{aliases[0]}"
    if "." in key:
        return f"config.{key}"
    return f"config.{key}"


def _combine_api_filters(parts: Sequence[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    non_empty = [dict(part) for part in parts if part]
    if not non_empty:
        return None
    if len(non_empty) == 1:
        return non_empty[0]
    return {"$and": non_empty}


def _build_selector_api_filters(selector: RunSelector) -> Optional[Dict[str, Any]]:
    if selector.api_filters is not None:
        return dict(selector.api_filters)

    parts = []

    if selector.name_contains_any:
        pattern = "|".join(re.escape(token) for token in selector.name_contains_any)
        parts.append({"displayName": {"$regex": pattern}})
    if selector.name_contains_all:
        pattern = "".join(f"(?=.*{re.escape(token)})" for token in selector.name_contains_all)
        parts.append({"displayName": {"$regex": pattern}})

    for key, expected in selector.config_equals.items():
        parts.append({_api_filter_field(key): expected})

    for key, allowed in selector.config_in.items():
        parts.append({_api_filter_field(key): {"$in": list(_coerce_iterable(allowed))}})

    for key in selector.config_true_all:
        parts.append({_api_filter_field(key): True})

    if selector.config_true_any:
        parts.append(
            {
                "$or": [
                    {_api_filter_field(key): True}
                    for key in selector.config_true_any
                ]
            }
        )

    if selector.exclude_tags:
        parts.append({"tags": {"$nin": list(selector.exclude_tags)}})

    return _combine_api_filters(parts)


def _build_component_record(
    run: Any,
    selector: RunSelector,
    metrics: Sequence[str],
) -> Dict[str, Any]:
    pd = _import_pandas()
    raw_config = {k: v for k, v in dict(run.config).items() if not str(k).startswith("_")}
    flat_config = _flatten_mapping(raw_config)
    normalized_config = _normalize_run_config(flat_config)
    attrs = getattr(run, "_attrs", {}) or {}

    record: Dict[str, Any] = {
        "source_label": selector.label,
        "entity": selector.entity,
        "project": selector.project,
        "project_path": selector.project_path(),
        "run_id": getattr(run, "id", None),
        "run_name": _display_name(run),
        "run_path": _run_path(run),
        "run_url": _run_url(run),
        "state": getattr(run, "state", None),
        "group": getattr(run, "group", None),
        "created_at": pd.to_datetime(getattr(run, "created_at", None), utc=True),
        "history_line_count": attrs.get("historyLineCount"),
    }

    for key, value in normalized_config.items():
        record[key] = _pythonize(value)
    for metric in metrics:
        record[metric] = _pythonize(run.summary.get(metric))
    for key, value in flat_config.items():
        record[f"config.{key}"] = _pythonize(value)
    return record


def _auto_merge_key(record: Mapping[str, Any], merge_key_fields: Sequence[str]) -> str:
    for key in merge_key_fields:
        value = record.get(key)
        if value not in (None, "", (), []):
            return f"{key}:{value}"
    return f"run_id:{record['run_id']}"


def _collect_component_runs(
    selectors: Sequence[RunSelector],
    metrics: Sequence[str],
    api: Optional[Any],
) -> Any:
    pd = _import_pandas()
    wandb = _import_wandb()
    api = api or wandb.Api()

    records = []
    for selector in selectors:
        api_filters = _build_selector_api_filters(selector)
        for run in api.runs(selector.project_path(), filters=api_filters, per_page=1000):
            raw_config = {k: v for k, v in dict(run.config).items() if not str(k).startswith("_")}
            flat_config = _flatten_mapping(raw_config)
            normalized_config = _normalize_run_config(flat_config)
            if not _selector_matches(selector, run, flat_config, normalized_config):
                continue
            records.append(_build_component_record(run, selector, metrics=metrics))

    component_runs = pd.DataFrame.from_records(records)
    if component_runs.empty:
        return component_runs

    component_runs = component_runs.sort_values(
        by=["source_label", "run_name", "created_at", "run_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    return component_runs


def _fetch_run_history(
    api_run: Any,
    metrics: Sequence[str],
    page_size: int,
) -> Any:
    pd = _import_pandas()
    records = []
    history_keys = ["_step", *metrics]
    for row in api_run.scan_history(keys=history_keys, page_size=page_size):
        step = row.get("_step")
        if step is None:
            continue
        record = {"step": int(step)}
        has_metric = False
        for metric in metrics:
            value = row.get(metric)
            record[metric] = _pythonize(value)
            has_metric = has_metric or (value is not None)
        if has_metric:
            records.append(record)
    return pd.DataFrame.from_records(records)


def _serialize_variants(values: Iterable[Any]) -> Tuple[Any, ...]:
    unique: Dict[str, Any] = {}
    for value in values:
        if value is None:
            continue
        py_value = _pythonize(value)
        unique[repr(py_value)] = py_value
    ordered = [unique[key] for key in sorted(unique)]
    return tuple(ordered)


def _build_logical_runs(
    component_runs: Any,
    metrics: Sequence[str],
    merge_key_fields: Sequence[str],
    include_history: bool,
    history_page_size: int,
    api: Optional[Any],
) -> Tuple[Any, Any]:
    pd = _import_pandas()
    wandb = _import_wandb()
    api = api or wandb.Api()

    if component_runs.empty:
        empty_history = pd.DataFrame(columns=["logical_run_id", "step", *metrics])
        return component_runs.copy(), empty_history

    component_runs = component_runs.copy()
    component_runs["merge_key"] = component_runs.apply(
        lambda row: _auto_merge_key(row, merge_key_fields),
        axis=1,
    )
    component_runs["logical_run_id"] = (
        component_runs["source_label"].astype(str) + "::" + component_runs["merge_key"].astype(str)
    )

    logical_records = []
    history_frames = []
    grouped = component_runs.groupby("logical_run_id", sort=False)
    for logical_run_id, group_df in grouped:
        ordered = group_df.sort_values(by=["created_at", "run_id"], kind="mergesort").reset_index(drop=True)
        final_row = ordered.iloc[-1].copy()
        logical_record = final_row.to_dict()
        logical_record["component_count"] = int(len(ordered))
        logical_record["component_run_ids"] = tuple(ordered["run_id"].tolist())
        logical_record["component_run_paths"] = tuple(ordered["run_path"].tolist())
        logical_record["component_states"] = tuple(ordered["state"].tolist())
        logical_record["component_created_at"] = tuple(ordered["created_at"].tolist())

        history_rows = 0
        if include_history:
            merged_history_parts = []
            for component_index, component in ordered.iterrows():
                api_run = api.run(component["run_path"])
                component_history = _fetch_run_history(
                    api_run,
                    metrics=metrics,
                    page_size=history_page_size,
                )
                if component_history.empty:
                    continue
                component_history["component_run_id"] = component["run_id"]
                component_history["component_run_path"] = component["run_path"]
                component_history["component_index"] = component_index
                merged_history_parts.append(component_history)

            if merged_history_parts:
                logical_history = pd.concat(merged_history_parts, ignore_index=True, sort=False)
                logical_history = logical_history.sort_values(
                    by=["step", "component_index", "component_run_id"],
                    kind="mergesort",
                )
                logical_history = logical_history.drop_duplicates(subset=["step"], keep="last")
                logical_history = logical_history.sort_values(by=["step"], kind="mergesort").reset_index(drop=True)
                logical_history["logical_run_id"] = logical_run_id
                logical_history["source_label"] = logical_record["source_label"]
                logical_history["run_name"] = logical_record["run_name"]
                logical_history["merge_key"] = logical_record["merge_key"]
                history_rows = int(len(logical_history))
                final_history_row = logical_history.iloc[-1]
                logical_record["last_step"] = int(final_history_row["step"])
                for metric in metrics:
                    logical_record[metric] = _pythonize(final_history_row.get(metric))
                history_frames.append(logical_history)
            else:
                logical_record["last_step"] = None
        else:
            logical_record["last_step"] = None

        logical_record["history_rows"] = history_rows
        logical_records.append(logical_record)

    logical_runs = pd.DataFrame.from_records(logical_records)
    logical_runs = logical_runs.sort_values(
        by=["source_label", "run_name", "created_at"],
        kind="mergesort",
    ).reset_index(drop=True)

    if history_frames:
        history = pd.concat(history_frames, ignore_index=True, sort=False)
        history = history.sort_values(
            by=["source_label", "run_name", "step"],
            kind="mergesort",
        ).reset_index(drop=True)
    else:
        history = pd.DataFrame(columns=["logical_run_id", "source_label", "run_name", "merge_key", "step", *metrics])

    return logical_runs, history


def _comparison_config_columns(logical_runs: Any) -> Sequence[str]:
    canonical_columns = [
        column
        for column in NORMALIZED_CONFIG_ALIASES
        if column in logical_runs.columns
    ]
    canonical_columns.extend(
        flag_key for flag_key in LEGACY_LOAD_BALANCE_FLAGS.values() if flag_key in logical_runs.columns
    )

    raw_alias_columns = {
        f"config.{alias}"
        for aliases in NORMALIZED_CONFIG_ALIASES.values()
        for alias in aliases
    }
    raw_config_columns = [
        column
        for column in logical_runs.columns
        if column.startswith("config.") and column not in raw_alias_columns
    ]
    return [*canonical_columns, *sorted(raw_config_columns)]


def summarize_varying_parameters(
    logical_runs: Any,
    *,
    config_columns: Optional[Sequence[str]] = None,
    groupby: Optional[str] = "source_label",
    ignore: Optional[Iterable[str]] = None,
) -> Any:
    pd = _import_pandas()
    if logical_runs.empty:
        return pd.DataFrame(columns=["scope", "config_key", "num_values", "values", "value_counts"])

    config_columns = list(config_columns or _comparison_config_columns(logical_runs))
    ignore = set(ignore or ()) | DEFAULT_DIFF_IGNORE
    config_columns = [column for column in config_columns if column not in ignore]

    scopes = [("__all__", logical_runs)]
    if groupby is not None and groupby in logical_runs.columns:
        scopes.extend((str(scope), frame) for scope, frame in logical_runs.groupby(groupby, sort=False))

    rows = []
    for scope_name, frame in scopes:
        for column in config_columns:
            if column not in frame.columns:
                continue
            values = _serialize_variants(frame[column].tolist())
            if len(values) <= 1:
                continue
            counts: Dict[str, int] = {}
            for value in frame[column].tolist():
                if value is None:
                    continue
                key = repr(_pythonize(value))
                counts[key] = counts.get(key, 0) + 1
            rows.append(
                {
                    "scope": scope_name,
                    "config_key": column,
                    "num_values": len(values),
                    "values": values,
                    "value_counts": counts,
                }
            )

    varying = pd.DataFrame.from_records(rows)
    if varying.empty:
        return varying
    return varying.sort_values(by=["scope", "config_key"], kind="mergesort").reset_index(drop=True)


def history_to_long(history: Any, metrics: Optional[Sequence[str]] = None) -> Any:
    pd = _import_pandas()
    metrics = list(metrics or [column for column in history.columns if "/" in column])
    if history.empty:
        return pd.DataFrame(columns=["logical_run_id", "metric", "value"])

    id_vars = [column for column in history.columns if column not in metrics]
    long_df = history.melt(
        id_vars=id_vars,
        value_vars=metrics,
        var_name="metric",
        value_name="value",
    )
    return long_df.dropna(subset=["value"]).reset_index(drop=True)


def load_wandb_comparison(
    selectors: Sequence[RunSelector],
    *,
    metrics: Sequence[str] = DEFAULT_METRICS,
    api: Optional[Any] = None,
    include_history: bool = True,
    history_page_size: int = 1000,
    merge_key_fields: Sequence[str] = ("group", "output", "run_name"),
) -> WandbComparisonBundle:
    component_runs = _collect_component_runs(
        selectors=selectors,
        metrics=metrics,
        api=api,
    )
    logical_runs, history = _build_logical_runs(
        component_runs=component_runs,
        metrics=metrics,
        merge_key_fields=merge_key_fields,
        include_history=include_history,
        history_page_size=history_page_size,
        api=api,
    )
    varying_params = summarize_varying_parameters(logical_runs)
    return WandbComparisonBundle(
        component_runs=component_runs,
        logical_runs=logical_runs,
        history=history,
        varying_params=varying_params,
    )


def make_default_moe_compare_selectors() -> Tuple[RunSelector, RunSelector]:
    return (
        RunSelector(
            entity="mikeyshechter",
            project="modded-nanogpt-moe",
            label="mikeyshechter",
            name_contains_any=("centered_fsq_lb", "sq_metrics_lb"),
        ),
        RunSelector(
            entity="team-tomer",
            project="modded-nanogpt-moe",
            label="team-tomer",
            config_equals={
                "router_type": "switch",
                "global_load_balance": False,
                "num_experts": 64,
            },
            config_in={"top_k": (2, 16)},
            config_true_any=(
                "maxvio_load_balance",
                "minmaxvio_load_balance",
                "totalvio_load_balance",
            ),
            predicate=lambda run, flat, normalized: float(normalized.get("load_balance_ste_width") or 0.0) > 0.0,
        ),
    )


def load_default_moe_comparison(
    *,
    metrics: Sequence[str] = DEFAULT_METRICS,
    api: Optional[Any] = None,
    include_history: bool = True,
    history_page_size: int = 1000,
) -> WandbComparisonBundle:
    return load_wandb_comparison(
        make_default_moe_compare_selectors(),
        metrics=metrics,
        api=api,
        include_history=include_history,
        history_page_size=history_page_size,
    )
