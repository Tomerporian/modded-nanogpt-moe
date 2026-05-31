from __future__ import annotations

import math
import re
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np


RunLike = Any


def _import_wandb():
    try:
        import wandb
    except ImportError as exc:
        raise ImportError(
            "wandb is required for plots.py. Install it in your notebook environment "
            "or call this helper where wandb is available."
        ) from exc
    return wandb


def _import_matplotlib():
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for plots.py. Install it in your notebook environment "
            "to render heatmaps."
        ) from exc
    return plt, LogNorm, Normalize, TwoSlopeNorm


def _import_pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "pandas is required to build result tables. Install pandas "
            "or call this helper where pandas is available."
        ) from exc
    return pd


def _import_yaml():
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "pyyaml is required to read YAML results. Install pyyaml "
            "or call this helper where yaml is available."
        ) from exc
    return yaml


def _resolve_run(
    run: Union[str, RunLike],
    *,
    entity: Optional[str] = None,
    project: Optional[str] = None,
    api: Optional[Any] = None,
):
    wandb = _import_wandb()
    if hasattr(run, "scan_history") and hasattr(run, "config"):
        return run

    if not isinstance(run, str):
        raise TypeError(
            "run must be either a W&B run path string '<entity>/<project>/<run_id>' "
            "or a wandb public Run object."
        )

    api = api or wandb.Api()
    if run.count("/") == 2:
        run_path = run
    else:
        if entity is None or project is None:
            raise ValueError(
                "If run is not a full run path, entity and project must be provided."
            )
        run_path = f"{entity}/{project}/{run}"
    return api.run(run_path)


def _history_anchor_key(split: str) -> str:
    if split == "val":
        return "val/loss"
    if split == "train":
        return "train/loss"
    raise ValueError("split must be 'train' or 'val'")


def _extract_matrix_from_row(row: Dict[str, Any], n_layers: int, num_experts: int) -> np.ndarray:
    matrix = np.empty((n_layers, num_experts), dtype=np.float64)
    for li in range(n_layers):
        for ei in range(num_experts):
            key = f"Expert Balance/Layer {li}/{ei}"
            if key not in row:
                raise KeyError(f"Missing key '{key}' in W&B history row.")
            matrix[li, ei] = float(row[key])
    return matrix


def _fraction_tick_label(value: float) -> str:
    frac = Fraction(value).limit_denominator()
    if frac.numerator == frac.denominator:
        return "1"
    return f"{frac.numerator}/{frac.denominator}"


def _log_ticks(num_experts: int, vmin: float, vmax: float) -> Tuple[list[float], list[str]]:
    ticks = []
    numerator = 1
    while numerator <= num_experts:
        tick = numerator / float(num_experts)
        if vmin <= tick <= vmax:
            ticks.append(tick)
        numerator *= 2
    if not ticks:
        ticks = [vmin, vmax]
    labels = [_fraction_tick_label(tick) for tick in ticks]
    return ticks, labels


def _expert_violation_matrix(matrix: np.ndarray, normalize: bool = True) -> np.ndarray:
    num_experts = matrix.shape[1]
    target = 1.0 / float(num_experts)
    if normalize:
        return (matrix - target) / target
    return matrix - target


def fetch_wandb_expert_balance_matrix(
    run: Union[str, RunLike],
    *,
    split: str = "val",
    step: Union[str, int] = "last",
    entity: Optional[str] = None,
    project: Optional[str] = None,
    api: Optional[Any] = None,
    n_layers: Optional[int] = None,
    num_experts: Optional[int] = None,
) -> Tuple[np.ndarray, Dict[str, Any], Any]:
    """
    Fetch the per-layer expert-balance matrix logged by this repo.

    Args:
        run:
            Preferred form is the W&B run path '<entity>/<project>/<run_id>'.
            A public API Run object also works.
        split:
            'val' or 'train'. Validation is usually the right choice for stable figures.
        step:
            'last' for the final logged cycle of the chosen split, or an integer W&B step.
        entity, project:
            Optional if run is already a full run path.
        api:
            Optional pre-created wandb.Api() instance.
        n_layers, num_experts:
            Optional overrides. By default these are read from run.config.

    Returns:
        matrix:
            NumPy array of shape (n_layers, num_experts).
        row:
            The W&B history row used to build the matrix.
        resolved_run:
            The public API Run object.
    """
    resolved_run = _resolve_run(run, entity=entity, project=project, api=api)
    run_config = dict(resolved_run.config)
    n_layers = int(n_layers if n_layers is not None else run_config["n_layer"])
    num_experts = int(num_experts if num_experts is not None else run_config["num_experts"])

    anchor_key = _history_anchor_key(split)
    balance_keys = [
        f"Expert Balance/Layer {li}/{ei}"
        for li in range(n_layers)
        for ei in range(num_experts)
    ]
    history_keys = [anchor_key] + balance_keys

    selected_row = None
    for row in resolved_run.scan_history(keys=history_keys, page_size=1000):
        row_step = row.get("_step")
        if step == "last":
            selected_row = row
        elif row_step == step:
            selected_row = row
            break

    if selected_row is None and step == "last":
        summary = dict(resolved_run.summary)
        if all(key in summary for key in balance_keys):
            selected_row = summary

    if selected_row is None:
        raise ValueError(
            f"Could not find a {split} history row for step={step!r} with the full "
            "expert-balance matrix logged."
        )

    matrix = _extract_matrix_from_row(selected_row, n_layers=n_layers, num_experts=num_experts)
    return matrix, selected_row, resolved_run


def plot_wandb_expert_balance_heatmap(
    run: Union[str, RunLike],
    *,
    split: str = "val",
    step: Union[str, int] = "last",
    entity: Optional[str] = None,
    project: Optional[str] = None,
    api: Optional[Any] = None,
    n_layers: Optional[int] = None,
    num_experts: Optional[int] = None,
    ax: Optional[Any] = None,
    figsize: Tuple[float, float] = (7.2, 6.0),
    cmap: str = "coolwarm",
    log_scale: bool = True,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    title: Optional[str] = None,
    show_colorbar: bool = True,
):
    """
    Plot the per-layer expert-balance heatmap for a W&B run.

    The function uses the scalar keys already logged by this repo:
    'Expert Balance/Layer {layer}/{expert}'. Because train and val use the same
    matrix keys, the function filters history rows using either 'train/loss' or
    'val/loss' as an anchor key.

    Returns:
        fig, ax, matrix, row
    """
    plt, LogNorm, Normalize, _ = _import_matplotlib()
    matrix, row, resolved_run = fetch_wandb_expert_balance_matrix(
        run,
        split=split,
        step=step,
        entity=entity,
        project=project,
        api=api,
        n_layers=n_layers,
        num_experts=num_experts,
    )

    n_layers, num_experts = matrix.shape
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    default_vmin = 1.0 / num_experts
    positive_entries = matrix[matrix > 0.0]
    if positive_entries.size == 0:
        raise ValueError("The expert-balance matrix contains no positive values.")

    vmin = float(default_vmin if vmin is None else vmin)
    vmax = float(1.0 if vmax is None else vmax)

    plot_matrix = matrix.copy()
    if log_scale:
        plot_matrix = np.clip(plot_matrix, vmin, None)
        norm = LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = Normalize(vmin=vmin, vmax=vmax)

    image = ax.imshow(
        plot_matrix,
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        aspect="auto",
    )

    ax.set_xlabel("Expert ID", fontsize=14)
    ax.set_ylabel("Layer ID", fontsize=14)
    ax.set_xticks([])
    ax.set_yticks(np.arange(n_layers))
    ax.tick_params(top=False, labeltop=False, bottom=False, labelbottom=False)

    if title is None:
        run_label = getattr(resolved_run, "name", None) or getattr(resolved_run, "id", "run")
        title = f"{split.title()} Expert Balance: {run_label}"
        if step != "last":
            title += f" (step {step})"
    ax.set_title(title, fontsize=16, pad=14)

    if show_colorbar:
        cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        if log_scale:
            ticks, labels = _log_ticks(num_experts, vmin, vmax)
            cbar.set_ticks(ticks)
            cbar.set_ticklabels(labels)
            cbar.ax.minorticks_off()

    return fig, ax, matrix, row


def plot_wandb_expert_violation_heatmap(
    run: Union[str, RunLike],
    *,
    split: str = "val",
    step: Union[str, int] = "last",
    entity: Optional[str] = None,
    project: Optional[str] = None,
    api: Optional[Any] = None,
    n_layers: Optional[int] = None,
    num_experts: Optional[int] = None,
    ax: Optional[Any] = None,
    figsize: Tuple[float, float] = (7.2, 6.0),
    cmap: str = "RdBu_r",
    normalize: bool = True,
    vmax: Optional[float] = None,
    title: Optional[str] = None,
    show_colorbar: bool = True,
):
    """
    Plot a per-expert violation heatmap derived from the logged expert-balance matrix.

    By default this shows the signed normalized violation:
        (load - 1/E) / (1/E) = E * load - 1

    So:
    - 0 means perfectly balanced expert load
    - positive values mean overloaded experts
    - negative values mean underloaded experts

    Returns:
        fig, ax, violation_matrix, row
    """
    plt, _, _, TwoSlopeNorm = _import_matplotlib()
    matrix, row, resolved_run = fetch_wandb_expert_balance_matrix(
        run,
        split=split,
        step=step,
        entity=entity,
        project=project,
        api=api,
        n_layers=n_layers,
        num_experts=num_experts,
    )

    violation = _expert_violation_matrix(matrix, normalize=normalize)
    n_layers, num_experts = violation.shape

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    if vmax is None:
        vmax = float(np.max(np.abs(violation)))
    vmax = max(float(vmax), 1e-8)

    image = ax.imshow(
        violation,
        cmap=cmap,
        norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax),
        interpolation="nearest",
        aspect="auto",
    )

    ax.set_xlabel("Expert ID", fontsize=14)
    ax.set_ylabel("Layer ID", fontsize=14)
    ax.set_xticks([])
    ax.set_yticks(np.arange(n_layers))
    ax.tick_params(top=False, labeltop=False, bottom=False, labelbottom=False)

    if title is None:
        run_label = getattr(resolved_run, "name", None) or getattr(resolved_run, "id", "run")
        metric_name = "Normalized Expert Violation" if normalize else "Expert Violation"
        title = f"{split.title()} {metric_name}: {run_label}"
        if step != "last":
            title += f" (step {step})"
    ax.set_title(title, fontsize=16, pad=14)

    if show_colorbar:
        cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        if normalize:
            cbar.set_label("(load - 1/E) / (1/E)", rotation=90)
        else:
            cbar.set_label("load - 1/E", rotation=90)

    return fig, ax, violation, row


_DEFAULT_LM_EVAL_BENCHMARKS: Tuple[Dict[str, Any], ...] = (
    {
        "key": "arc_challenge",
        "label": "ARC Challenge [Clark et al., 2018]",
        "tasks": ("arc_challenge",),
    },
    {
        "key": "arc_easy",
        "label": "ARC Easy [Clark et al., 2018]",
        "tasks": ("arc_easy",),
    },
    {
        "key": "hellaswag",
        "label": "HellaSwag [Zellers et al., 2019]",
        "tasks": ("hellaswag",),
    },
    {
        "key": "piqa",
        "label": "PIQA [Bisk et al., 2019]",
        "tasks": ("piqa",),
    },
    {
        "key": "winogrande",
        "label": "WinoGrande [Sakaguchi et al., 2019]",
        "tasks": ("winogrande",),
    },
    {
        "key": "lambada_openai",
        "label": "LAMBADA [Paperno et al., 2016]",
        "tasks": ("lambada_openai",),
    },
    {
        "key": "glue",
        "label": "GLUE [Wang et al., 2018]",
        "tasks": (
            "cola",
            "mnli",
            "mnli_mismatch",
            "mrpc",
            "qnli",
            "qqp",
            "rte",
            "sst2",
            "wnli",
        ),
    },
    {
        "key": "blimp",
        "label": "BLiMP [Warstadt et al., 2020]",
        "prefix": "blimp_",
    },
)

_DCLM_CORE_LM_EVAL_TASKS: Tuple[str, ...] = (
    "arc_easy",
    "arc_challenge",
    "boolq",
    "commonsense_qa",
    "copa",
    "hellaswag",
    "openbookqa",
    "piqa",
    "winogrande",
    "wsc273",
    "lambada_openai",
    "coqa",
    "squadv2",
    "agieval_lsat_ar",
    "bigbench_language_identification_multiple_choice",
    "bigbench_qa_wikidata_generate_until",
    "bigbench_dyck_languages_generate_until",
    "bigbench_operators_generate_until",
    "bigbench_repeat_copy_logic_generate_until",
    "bigbench_cs_algorithms_generate_until",
)

_DEFAULT_LM_EVAL_TASK_METRICS: Dict[str, Tuple[str, ...]] = {
    "agieval_lsat_ar": ("acc_norm,none", "acc,none"),
    "arc_challenge": ("acc_norm,none", "acc,none"),
    "arc_easy": ("acc_norm,none", "acc,none"),
    "boolq": ("acc,none",),
    "commonsense_qa": ("acc,none",),
    "copa": ("acc,none",),
    "coqa": ("f1,none", "em,none"),
    "hellaswag": ("acc_norm,none", "acc,none"),
    "openbookqa": ("acc_norm,none", "acc,none"),
    "piqa": ("acc_norm,none", "acc,none"),
    "squadv2": ("f1,none", "exact,none", "best_f1,none", "best_exact,none"),
    "wsc273": ("acc,none",),
    "winogrande": ("acc,none",),
    "lambada_openai": ("acc,none",),
    "bigbench_cs_algorithms_generate_until": ("exact_match,none",),
    "bigbench_dyck_languages_generate_until": ("exact_match,none",),
    "bigbench_language_identification_multiple_choice": ("acc,none",),
    "bigbench_operators_generate_until": ("exact_match,none",),
    "bigbench_qa_wikidata_generate_until": ("exact_match,none",),
    "bigbench_repeat_copy_logic_generate_until": ("exact_match,none",),
    "cola": ("mcc,none", "acc,none"),
    "mrpc": ("f1,none", "acc,none"),
    "qqp": ("f1,none", "acc,none"),
}

_DEFAULT_LM_EVAL_METADATA_KEYS: Tuple[str, ...] = (
    "config_label",
    "load_balance_loss",
    "aux_coeff_train",
    "aux_coeff_val",
    "load_balance_ste_width",
    "loss_free_mode",
    "loss_free_update_rate",
    "approx_global_load_balance",
    "topk_activation",
)

_DEFAULT_TASK_LOSS_RESULTS_ROOTS: Tuple[str, ...] = (
    "/e/project1/laionize/shechter1/task_loss_results/26-05-14-baselines",
    "/e/project1/laionize/shechter1/task_loss_results/26-05-15-centered_fsq",
)

_DEFAULT_TASK_LOSS_METADATA_KEYS: Tuple[str, ...] = (
    "config_label",
    "load_balance_loss",
    "seed",
    "aux_coeff_train",
    "aux_coeff_val",
    "load_balance_ste_width",
    "loss_free_mode",
    "loss_free_update_rate",
    "approx_global_load_balance",
    "topk_activation",
)


def _as_path_list(value: Union[str, Path, Sequence[Union[str, Path]], None]) -> list[Path]:
    if value is None:
        return []
    if isinstance(value, (str, Path)):
        return [Path(value)]
    return [Path(item) for item in value]


def _natural_sort_key(value: Union[str, Path]) -> list[Any]:
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", str(value))
    ]


def _collect_lm_eval_result_paths(
    results_root: Union[str, Path, Sequence[Union[str, Path]]],
    *,
    result_filename: str,
    run_paths: Optional[Sequence[Union[str, Path]]],
    recursive: bool,
) -> list[Path]:
    if run_paths is not None:
        paths = []
        for run_path in run_paths:
            path = Path(run_path)
            if path.is_dir():
                path = path / result_filename
            paths.append(path)
    else:
        paths = []
        pattern = f"**/{result_filename}" if recursive else f"*/{result_filename}"
        for root in _as_path_list(results_root):
            root = root.expanduser()
            if root.is_file():
                paths.append(root)
            else:
                paths.extend(root.glob(pattern))

    unique_paths = {path.expanduser().resolve() for path in paths if path.exists()}
    return sorted(unique_paths, key=_natural_sort_key)


def _lm_eval_result_group_and_run(path: Union[str, Path]) -> Tuple[str, str]:
    path = Path(path)
    run_name = path.parent.name
    group_name = path.parent.parent.name
    return group_name, run_name


def _lm_eval_run_id(path: Union[str, Path]) -> str:
    group_name, run_name = _lm_eval_result_group_and_run(path)
    return f"{group_name}/{run_name}" if group_name else run_name


def _load_yaml_mapping(path: Union[str, Path]) -> Dict[str, Any]:
    yaml = _import_yaml()
    with Path(path).open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return loaded


def _find_checkpoint_config_path(
    result_path: Union[str, Path],
    checkpoint_roots: Sequence[Union[str, Path]],
) -> Optional[Path]:
    group_name, run_name = _lm_eval_result_group_and_run(result_path)
    for root in _as_path_list(checkpoint_roots):
        candidates = (
            root / group_name / run_name / "args.yaml",
            root / group_name / run_name / "spec.yaml",
            root / run_name / "args.yaml",
            root / run_name / "spec.yaml",
        )
        for candidate in candidates:
            if candidate.exists():
                return candidate
    return None


def _load_lm_eval_run_metadata(
    result_path: Union[str, Path],
    checkpoint_roots: Sequence[Union[str, Path]],
) -> Dict[str, Any]:
    config_path = _find_checkpoint_config_path(result_path, checkpoint_roots)
    if config_path is None:
        return {}
    metadata = _load_yaml_mapping(config_path)
    metadata["_config_path"] = str(config_path)
    return metadata


def _collect_done_checkpoint_run_dirs(
    checkpoint_roots: Sequence[Union[str, Path]],
) -> Dict[str, Path]:
    done_runs: Dict[str, Path] = {}
    for root in _as_path_list(checkpoint_roots):
        for done_file in sorted(root.glob("*/done"), key=_natural_sort_key):
            run_dir = done_file.parent
            done_runs[f"{root.name}/{run_dir.name}"] = run_dir
        for done_file in sorted(root.glob("*/*/done"), key=_natural_sort_key):
            run_dir = done_file.parent
            done_runs[f"{run_dir.parent.name}/{run_dir.name}"] = run_dir
    return done_runs


def _format_lm_eval_metadata_value(value: Any) -> Any:
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, (list, tuple)):
        return ",".join(str(item) for item in value)
    return value


def _lm_eval_metric_value_scale(task_name: str, metric_key: str) -> float:
    metric_name = metric_key.split(",", 1)[0]
    if task_name == "squadv2" and metric_name in {
        "exact",
        "f1",
        "HasAns_exact",
        "HasAns_f1",
        "NoAns_exact",
        "NoAns_f1",
        "best_exact",
        "best_f1",
    }:
        return 0.01
    return 1.0


def _load_lm_eval_results_block(path: Union[str, Path]) -> Dict[str, Dict[str, Any]]:
    yaml = _import_yaml()
    path = Path(path)
    block_lines: list[str] = []
    in_results = False

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not in_results:
                if line == "results:\n" or line == "results:\r\n":
                    in_results = True
                    block_lines.append(line)
                continue

            if line and not line[0].isspace() and line.strip():
                break
            block_lines.append(line)

    if not block_lines:
        raise ValueError(f"Could not find a top-level results block in {path}")

    loaded = yaml.safe_load("".join(block_lines)) or {}
    results = loaded.get("results", {})
    if not isinstance(results, dict):
        raise ValueError(f"Expected a mapping under results in {path}")
    return results


def _pretty_lm_eval_run_label(name: str) -> str:
    name = re.sub(r"^\d+_\d{2}-\d{2}-\d{2}-", "", name)
    name = name.removeprefix("large_scale_")
    label = name.replace("_", " ").title()
    return label.replace("Fsq", "FSQ").replace("Maxvio", "MaxVio")


def _lm_eval_stderr_key(metric_key: str) -> str:
    if "," in metric_key:
        metric, suffix = metric_key.split(",", 1)
        return f"{metric}_stderr,{suffix}"
    return f"{metric_key}_stderr"


def _select_lm_eval_metric_with_key(
    task_name: str,
    metrics: Mapping[str, Any],
    metric_preferences: Sequence[str],
) -> Tuple[float, Optional[float], str]:
    keys = list(_DEFAULT_LM_EVAL_TASK_METRICS.get(task_name, ()))
    keys.extend(key for key in metric_preferences if key not in keys)

    for key in keys:
        value = metrics.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            stderr = metrics.get(_lm_eval_stderr_key(key))
            stderr_value = (
                float(stderr)
                if isinstance(stderr, (int, float)) and math.isfinite(float(stderr))
                else None
            )
            scale = _lm_eval_metric_value_scale(task_name, key)
            return (
                float(value) * scale,
                stderr_value * scale if stderr_value is not None else None,
                key,
            )

    raise KeyError(
        f"Could not find any supported metric for {task_name}. "
        f"Available keys: {sorted(metrics)}"
    )


def _select_lm_eval_metric(
    task_name: str,
    metrics: Mapping[str, Any],
    metric_preferences: Sequence[str],
) -> Tuple[float, Optional[float]]:
    value, stderr, _ = _select_lm_eval_metric_with_key(
        task_name,
        metrics,
        metric_preferences,
    )
    return value, stderr


def _lm_eval_tasks_for_spec(
    results: Mapping[str, Mapping[str, Any]],
    spec: Mapping[str, Any],
) -> list[str]:
    if "prefix" in spec:
        prefix = str(spec["prefix"])
        return sorted(task for task in results if task.startswith(prefix))

    requested = spec.get("tasks", ())
    return [task for task in requested if task in results]


def _combine_lm_eval_scores(
    values: Sequence[float],
    stderrs: Sequence[Optional[float]],
) -> Tuple[float, Optional[float]]:
    if not values:
        raise ValueError("Cannot combine an empty score list.")

    mean = float(np.mean(values))
    if all(stderr is not None for stderr in stderrs):
        stderr = math.sqrt(sum(float(stderr) ** 2 for stderr in stderrs)) / len(stderrs)
    elif len(values) > 1:
        stderr = float(np.std(values, ddof=1) / math.sqrt(len(values)))
    else:
        stderr = None
    return mean, stderr


def _format_lm_eval_score(
    value: Optional[float],
    stderr: Optional[float],
    *,
    decimals: int,
    percent: bool,
    show_stderr: bool,
    pm_symbol: str,
) -> str:
    if value is None:
        return "-"

    scale = 100.0 if percent else 1.0
    suffix = "%" if percent else ""
    formatted = f"{value * scale:.{decimals}f}{suffix}"
    if show_stderr and stderr is not None:
        formatted += f" {pm_symbol} {stderr * scale:.{decimals}f}{suffix}"
    return formatted


def _lm_eval_score_cell(
    value: Optional[float],
    stderr: Optional[float],
    *,
    decimals: int,
    percent: bool,
    show_stderr: bool,
    pm_symbol: str,
    format_scores: bool,
) -> Any:
    if format_scores or show_stderr:
        return _format_lm_eval_score(
            value,
            stderr,
            decimals=decimals,
            percent=percent,
            show_stderr=show_stderr,
            pm_symbol=pm_symbol,
        )
    if value is None:
        return np.nan
    scale = 100.0 if percent else 1.0
    return round(value * scale, decimals)


def _ordered_lm_eval_tasks(
    result_sets: Sequence[Tuple[str, Path, Mapping[str, Mapping[str, Any]], Mapping[str, Any]]],
    tasks: Optional[Sequence[str]],
) -> list[str]:
    if tasks is not None:
        return [task for task in tasks]

    available = set()
    for _, _, results, _ in result_sets:
        available.update(results)

    ordered = [task for task in _DCLM_CORE_LM_EVAL_TASKS if task in available]
    ordered.extend(sorted(available.difference(ordered)))
    return ordered


def _load_task_loss_run_metadata(payload: Mapping[str, Any]) -> Dict[str, Any]:
    candidates = []
    run_dir = payload.get("run_dir")
    if run_dir:
        candidates.append(Path(str(run_dir)))
    checkpoint = payload.get("checkpoint")
    if checkpoint:
        candidates.append(Path(str(checkpoint)).parent)

    for run_path in candidates:
        run_path = run_path.expanduser()
        for filename in ("args.yaml", "spec.yaml"):
            config_path = run_path / filename
            if config_path.exists():
                metadata = _load_yaml_mapping(config_path)
                metadata["_config_path"] = str(config_path)
                return metadata
    return {}


def _ordered_task_loss_tasks(
    result_sets: Sequence[Tuple[str, Path, Mapping[str, Any], Mapping[str, Any]]],
    tasks: Optional[Sequence[str]],
) -> list[str]:
    if tasks is not None:
        return [task for task in tasks]

    ordered = []
    seen = set()
    for _, _, payload, _ in result_sets:
        for task in payload.get("tasks", ()) or ():
            if task not in seen:
                ordered.append(task)
                seen.add(task)

        results = payload.get("results", {})
        if isinstance(results, Mapping):
            for task in results:
                if task not in seen:
                    ordered.append(task)
                    seen.add(task)
    return ordered


def _numeric_task_metric(metrics: Mapping[str, Any], key: str) -> float:
    value = metrics.get(key)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return float("nan")


def _maybe_round(value: float, decimals: Optional[int]) -> float:
    if decimals is None or not math.isfinite(value):
        return value
    return round(value, decimals)


def make_task_loss_results_table(
    results_root: Union[str, Path, Sequence[Union[str, Path]]] = _DEFAULT_TASK_LOSS_RESULTS_ROOTS,
    *,
    result_filename: str = "task_loss_results.yaml",
    run_paths: Optional[Sequence[Union[str, Path]]] = None,
    run_labels: Optional[Mapping[str, str]] = None,
    recursive: bool = True,
    layout: str = "wide",
    tasks: Optional[Sequence[str]] = None,
    task_labels: Optional[Mapping[str, str]] = None,
    value_key: str = "task_loss",
    include_metadata: bool = True,
    metadata_keys: Sequence[str] = _DEFAULT_TASK_LOSS_METADATA_KEYS,
    include_mean: bool = True,
    include_missing: bool = False,
    decimals: Optional[int] = None,
    output_path: Optional[Union[str, Path]] = None,
):
    """
    Build a pandas table from saved task-loss result directories.

    By default this scans the two current task-loss result folders:
    `/e/project1/laionize/shechter1/task_loss_results/26-05-14-baselines`
    and
    `/e/project1/laionize/shechter1/task_loss_results/26-05-15-centered_fsq`.

    `layout="wide"` returns one row per run and one column per task. Use
    `layout="long"` for one row per `(run, task)` with task counts included.

    Args:
        results_root:
            Result root, result file, or sequence of roots/files.
        result_filename:
            Result YAML filename inside each run directory.
        run_paths:
            Optional explicit result directories or YAML files. If omitted, all
            matching files under results_root are used.
        run_labels:
            Optional mapping from run id (`group/run`) or run directory name to
            display label.
        layout:
            `"wide"` for run rows and task columns, or `"long"` for tidy rows.
        tasks:
            Optional task order. If omitted, task order is read from the result
            files, preserving the first-seen order.
        value_key:
            Metric to place in task columns. Defaults to `task_loss`; `task_bpb`
            also works with these result files.
        include_metadata:
            If true, read the matching checkpoint `args.yaml`/`spec.yaml` from
            the `run_dir` recorded in each result file.
        output_path:
            Optional `.tex`, `.csv`, or `.md` path to write the table.

    Returns:
        pandas.DataFrame. Raw numeric values are stored in
        `table.attrs["raw_values"]`.
    """
    pd = _import_pandas()
    layout = layout.lower()
    if layout in {"runs", "table"}:
        layout = "wide"
    if layout in {"tidy"}:
        layout = "long"
    if layout not in {"wide", "long"}:
        raise ValueError("layout must be 'wide' or 'long'")

    paths = _collect_lm_eval_result_paths(
        results_root,
        result_filename=result_filename,
        run_paths=run_paths,
        recursive=recursive,
    )

    if not paths:
        raise FileNotFoundError(
            f"No {result_filename} files found under {results_root}"
        )

    run_labels = dict(run_labels or {})
    result_sets: list[Tuple[str, Path, Dict[str, Any], Dict[str, Any]]] = []
    for path in paths:
        payload = _load_yaml_mapping(path)
        results = payload.get("results", {})
        if not isinstance(results, Mapping):
            raise ValueError(f"Expected a mapping under results in {path}")
        metadata = _load_task_loss_run_metadata(payload) if include_metadata else {}
        result_sets.append((_lm_eval_run_id(path), path, payload, metadata))

    result_sets = sorted(result_sets, key=lambda item: _natural_sort_key(item[0]))
    task_names = _ordered_task_loss_tasks(result_sets, tasks)
    task_labels = dict(task_labels or {})
    raw_values: Dict[str, Dict[str, float]] = {}

    def base_row(
        run_id: str,
        path: Path,
        payload: Mapping[str, Any],
        metadata: Mapping[str, Any],
    ) -> Dict[str, Any]:
        group_name, run_name = _lm_eval_result_group_and_run(path)
        row: Dict[str, Any] = {
            "group": group_name,
            "run": run_name,
            "run_id": run_id,
            "label": (
                run_labels.get(run_id)
                or run_labels.get(run_name)
                or metadata.get("config_label")
                or run_id
            ),
        }
        if "checkpoint_step" in payload:
            row["checkpoint_step"] = payload["checkpoint_step"]
        if include_metadata:
            for key in metadata_keys:
                if key in metadata and key not in row:
                    row[key] = _format_lm_eval_metadata_value(metadata[key])
        return row

    rows: list[Dict[str, Any]] = []
    if layout == "wide":
        mean_column = f"mean_{value_key}"
        for run_id, path, payload, metadata in result_sets:
            row = base_row(run_id, path, payload, metadata)
            results = payload["results"]
            raw_values[run_id] = {}
            mean_values = []
            for task_name in task_names:
                metrics = results.get(task_name, {})
                value = (
                    _numeric_task_metric(metrics, value_key)
                    if isinstance(metrics, Mapping)
                    else float("nan")
                )
                raw_values[run_id][task_name] = value
                if math.isfinite(value):
                    mean_values.append(value)
                row[task_labels.get(task_name, task_name)] = _maybe_round(value, decimals)

            if include_mean:
                mean_value = float(np.mean(mean_values)) if mean_values else float("nan")
                raw_values[run_id]["mean"] = mean_value
                row[mean_column] = _maybe_round(mean_value, decimals)
            rows.append(row)

        front_columns = [
            column
            for column in [
                "group",
                "run",
                "run_id",
                "label",
                "checkpoint_step",
                *metadata_keys,
                mean_column,
            ]
            if any(column in row for row in rows)
        ]
        task_columns = [task_labels.get(task_name, task_name) for task_name in task_names]
        ordered_columns = front_columns + [
            column for column in task_columns if column not in front_columns
        ]
    else:
        for run_id, path, payload, metadata in result_sets:
            results = payload["results"]
            raw_values[run_id] = {}
            for task_name in task_names:
                metrics = results.get(task_name, {})
                if not metrics and not include_missing:
                    continue
                if not isinstance(metrics, Mapping):
                    metrics = {}

                value = _numeric_task_metric(metrics, value_key)
                raw_values[run_id][task_name] = value
                row = base_row(run_id, path, payload, metadata)
                row["task"] = task_name
                row[value_key] = _maybe_round(value, decimals)
                for key in ("task_bpb", "n_examples", "n_tokens", "answer_bytes"):
                    if key in metrics and key != value_key:
                        row[key] = metrics[key]
                rows.append(row)

        front_columns = [
            column
            for column in [
                "group",
                "run",
                "run_id",
                "label",
                "checkpoint_step",
                *metadata_keys,
                "task",
                value_key,
                "task_bpb",
                "n_examples",
                "n_tokens",
                "answer_bytes",
            ]
            if any(column in row for row in rows)
        ]
        ordered_columns = front_columns

    table = pd.DataFrame(rows)
    table = table.reindex(columns=ordered_columns)
    table.attrs["raw_values"] = raw_values
    table.attrs["result_paths"] = [str(path) for _, path, _, _ in result_sets]
    table.attrs["tasks"] = task_names
    table.attrs["value_key"] = value_key

    if output_path is not None:
        output_path = Path(output_path)
        suffix = output_path.suffix.lower()
        if suffix == ".tex":
            output = table.to_latex(index=False, escape=False)
        elif suffix == ".csv":
            output = table.to_csv(index=False)
        elif suffix in {".md", ".markdown"}:
            output = table.to_markdown(index=False)
        else:
            output = table.to_string(index=False)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output, encoding="utf-8")

    return table


def make_lm_eval_results_table(
    results_root: Union[str, Path, Sequence[Union[str, Path]]] = "/e/project1/laionize/shechter1/lm_eval_results",
    *,
    result_filename: str = "lm_eval_results.yaml",
    run_paths: Optional[Sequence[Union[str, Path]]] = None,
    run_labels: Optional[Mapping[str, str]] = None,
    checkpoint_roots: Union[str, Path, Sequence[Union[str, Path]], None] = None,
    recursive: bool = True,
    layout: str = "runs",
    tasks: Optional[Sequence[str]] = None,
    task_labels: Optional[Mapping[str, str]] = None,
    include_metadata: bool = True,
    metadata_keys: Sequence[str] = _DEFAULT_LM_EVAL_METADATA_KEYS,
    benchmark_specs: Optional[Sequence[Mapping[str, Any]]] = None,
    metric_preferences: Sequence[str] = (
        "acc_norm,none",
        "acc,none",
        "f1,none",
        "exact_match,none",
        "em,none",
        "exact,none",
        "mcc,none",
    ),
    decimals: int = 2,
    percent: bool = True,
    show_stderr: bool = False,
    show_mean_stderr: bool = False,
    pm_symbol: str = "+/-",
    format_scores: bool = True,
    include_missing: bool = False,
    output_path: Optional[Union[str, Path]] = None,
):
    """
    Build a pandas table from lm-eval result directories.

    The default recursively scans
    `/e/project1/laionize/shechter1/lm_eval_results` for
    `lm_eval_results.yaml`. The reader only loads the top-level `results:`
    block, avoiding the large `samples:` block emitted by lm-eval.

    `layout="runs"` returns one row per run and one column per task, which is
    the useful shape for sweeps with many tasks. `layout="benchmarks"` keeps the
    older paper-style view with benchmark rows and run columns.

    Args:
        results_root:
            Result root, result file, or sequence of roots/files.
        result_filename:
            Result YAML filename inside each run directory.
        run_paths:
            Optional explicit result directories or YAML files. If omitted, all
            matching files under results_root are used.
        run_labels:
            Optional mapping from run id (`group/run`) or run directory name to
            display label.
        checkpoint_roots:
            Optional checkpoint roots used to find matching `args.yaml` metadata.
            For these runs, pass
            `/e/project1/laionize/shechter1/checkpoints/modded-nanogpt-moe`.
        layout:
            `"runs"` for run rows and task columns, or `"benchmarks"` for the
            older benchmark-summary layout.
        tasks:
            Optional task order for `layout="runs"`. If omitted, DCLM-core tasks
            are ordered first, followed by any extra tasks in sorted order.
        benchmark_specs:
            Optional benchmark definitions for `layout="benchmarks"`. Each
            mapping should contain `label` and either `tasks` or `prefix`.
        metric_preferences:
            Fallback metric priority for tasks without a task-specific default.
        format_scores:
            If true, score cells are formatted strings such as `45.52%`. If
            false and `show_stderr` is false, score cells are numeric.
        output_path:
            Optional `.tex`, `.csv`, or `.md` path to write the formatted table.

    Returns:
        pandas.DataFrame. Raw normalized scores in `[0, 1]` are stored in
        `table.attrs["raw_scores"]`, and selected metric keys are stored in
        `table.attrs["selected_metrics"]`.
    """
    pd = _import_pandas()
    layout = layout.lower()
    if layout in {"paper", "benchmark", "summary"}:
        layout = "benchmarks"
    if layout not in {"runs", "wide", "benchmarks"}:
        raise ValueError("layout must be 'runs' or 'benchmarks'")

    paths = _collect_lm_eval_result_paths(
        results_root,
        result_filename=result_filename,
        run_paths=run_paths,
        recursive=recursive,
    )

    if not paths:
        raise FileNotFoundError(
            f"No {result_filename} files found under {results_root}"
        )

    run_labels = dict(run_labels or {})
    checkpoint_root_paths = _as_path_list(checkpoint_roots)
    result_sets: list[Tuple[str, Path, Dict[str, Dict[str, Any]], Dict[str, Any]]] = []
    for path in paths:
        run_id = _lm_eval_run_id(path)
        metadata = (
            _load_lm_eval_run_metadata(path, checkpoint_root_paths)
            if checkpoint_root_paths and include_metadata
            else {}
        )
        result_sets.append((run_id, path, _load_lm_eval_results_block(path), metadata))

    found_run_ids = {run_id for run_id, _, _, _ in result_sets}
    found_groups = {run_id.split("/", 1)[0] for run_id in found_run_ids if "/" in run_id}
    missing_run_ids: list[str] = []
    if include_missing and checkpoint_root_paths:
        done_run_dirs = _collect_done_checkpoint_run_dirs(checkpoint_root_paths)
        for run_id, run_dir in sorted(done_run_dirs.items(), key=lambda item: _natural_sort_key(item[0])):
            if run_id in found_run_ids:
                continue
            group_name = run_id.split("/", 1)[0] if "/" in run_id else ""
            if found_groups and group_name not in found_groups:
                continue
            synthetic_path = run_dir / result_filename
            metadata_path = run_dir / "args.yaml"
            if not metadata_path.exists():
                metadata_path = run_dir / "spec.yaml"
            metadata = _load_yaml_mapping(metadata_path) if metadata_path.exists() else {}
            if metadata:
                metadata["_config_path"] = str(metadata_path)
            result_sets.append((run_id, synthetic_path, {}, metadata))
            missing_run_ids.append(run_id)

    result_sets = sorted(result_sets, key=lambda item: _natural_sort_key(item[0]))

    if layout in {"runs", "wide"}:
        task_names = _ordered_lm_eval_tasks(result_sets, tasks)
        task_labels = dict(task_labels or {})
        rows: list[Dict[str, Any]] = []
        raw_scores: Dict[str, Dict[str, Tuple[Optional[float], Optional[float]]]] = {}
        selected_metrics: Dict[str, Dict[str, str]] = {}

        for run_id, path, results, metadata in result_sets:
            group_name, run_name = _lm_eval_result_group_and_run(path)
            label = (
                run_labels.get(run_id)
                or run_labels.get(run_name)
                or metadata.get("config_label")
                or _pretty_lm_eval_run_label(run_id)
            )
            row: Dict[str, Any] = {
                "group": group_name,
                "run": run_name,
                "run_id": run_id,
                "label": label,
            }
            if include_metadata:
                for key in metadata_keys:
                    if key in metadata and key not in row:
                        row[key] = _format_lm_eval_metadata_value(metadata[key])

            raw_scores[run_id] = {}
            selected_metrics[run_id] = {}
            mean_values = []
            mean_stderrs = []
            for task_name in task_names:
                column = task_labels.get(task_name, task_name)
                if task_name not in results:
                    value = None
                    stderr = None
                else:
                    value, stderr, metric_key = _select_lm_eval_metric_with_key(
                        task_name,
                        results[task_name],
                        metric_preferences,
                    )
                    selected_metrics[run_id][task_name] = metric_key
                    mean_values.append(value)
                    mean_stderrs.append(stderr)

                raw_scores[run_id][task_name] = (value, stderr)
                row[column] = _lm_eval_score_cell(
                    value,
                    stderr,
                    decimals=decimals,
                    percent=percent,
                    show_stderr=show_stderr,
                    pm_symbol=pm_symbol,
                    format_scores=format_scores,
                )

            if mean_values:
                mean_value, mean_stderr = _combine_lm_eval_scores(mean_values, mean_stderrs)
            else:
                mean_value = None
                mean_stderr = None
            raw_scores[run_id]["mean"] = (mean_value, mean_stderr)
            row["mean"] = _lm_eval_score_cell(
                mean_value,
                mean_stderr,
                decimals=decimals,
                percent=percent,
                show_stderr=show_mean_stderr,
                pm_symbol=pm_symbol,
                format_scores=format_scores,
            )

            rows.append(row)

        front_columns = [
            column
            for column in ["group", "run", "run_id", "label", *metadata_keys, "mean"]
            if any(column in row for row in rows)
        ]
        score_columns = [task_labels.get(task_name, task_name) for task_name in task_names]
        ordered_columns = front_columns + [
            column for column in score_columns if column not in front_columns
        ]
        table = pd.DataFrame(rows)
        table = table.reindex(columns=ordered_columns)
        table.attrs["raw_scores"] = raw_scores
        table.attrs["selected_metrics"] = selected_metrics
        table.attrs["result_paths"] = [str(path) for _, path, results, _ in result_sets if results]
        table.attrs["missing_run_ids"] = missing_run_ids
    else:
        specs = tuple(benchmark_specs or _DEFAULT_LM_EVAL_BENCHMARKS)
        rows: list[Dict[str, Any]] = []
        raw_scores: Dict[str, Dict[str, Tuple[Optional[float], Optional[float]]]] = {}

        for spec in specs:
            benchmark_key = str(spec.get("key", spec.get("label", "benchmark")))
            benchmark_label = str(spec["label"])
            row: Dict[str, Any] = {"Benchmark": benchmark_label}
            raw_scores[benchmark_key] = {}
            has_any_score = False

            for run_id, path, results, metadata in result_sets:
                run_name = path.parent.name
                run_label = (
                    run_labels.get(run_id)
                    or run_labels.get(run_name)
                    or metadata.get("config_label")
                    or _pretty_lm_eval_run_label(run_id)
                )
                task_names = _lm_eval_tasks_for_spec(results, spec)
                if not task_names:
                    value = None
                    stderr = None
                else:
                    values = []
                    stderrs = []
                    for task_name in task_names:
                        value_i, stderr_i = _select_lm_eval_metric(
                            task_name,
                            results[task_name],
                            metric_preferences,
                        )
                        values.append(value_i)
                        stderrs.append(stderr_i)
                    value, stderr = _combine_lm_eval_scores(values, stderrs)
                    has_any_score = True

                raw_scores[benchmark_key][run_label] = (value, stderr)
                column = f"{run_label} {pm_symbol} stderr" if show_stderr else run_label
                row[column] = _lm_eval_score_cell(
                    value,
                    stderr,
                    decimals=decimals,
                    percent=percent,
                    show_stderr=show_stderr,
                    pm_symbol=pm_symbol,
                    format_scores=format_scores,
                )

            if has_any_score or include_missing:
                rows.append(row)

        mean_row: Dict[str, Any] = {"Benchmark": "mean"}
        raw_scores["mean"] = {}
        benchmark_keys = [str(spec.get("key", spec.get("label", "benchmark"))) for spec in specs]
        for run_id, path, _, metadata in result_sets:
            run_name = path.parent.name
            run_label = (
                run_labels.get(run_id)
                or run_labels.get(run_name)
                or metadata.get("config_label")
                or _pretty_lm_eval_run_label(run_id)
            )
            values = []
            stderrs = []
            for benchmark_key in benchmark_keys:
                score = raw_scores.get(benchmark_key, {}).get(run_label)
                if score is None or score[0] is None:
                    continue
                values.append(float(score[0]))
                stderrs.append(score[1])

            if values:
                value, stderr = _combine_lm_eval_scores(values, stderrs)
            else:
                value = None
                stderr = None
            raw_scores["mean"][run_label] = (value, stderr)
            column = f"{run_label} {pm_symbol} stderr" if show_mean_stderr else run_label
            mean_row[column] = _lm_eval_score_cell(
                value,
                stderr,
                decimals=decimals,
                percent=percent,
                show_stderr=show_mean_stderr,
                pm_symbol=pm_symbol,
                format_scores=format_scores,
            )
        rows.append(mean_row)

        table = pd.DataFrame(rows)
        table.attrs["raw_scores"] = raw_scores
        table.attrs["selected_metrics"] = {}
        table.attrs["result_paths"] = [str(path) for _, path, results, _ in result_sets if results]
        table.attrs["missing_run_ids"] = missing_run_ids if include_missing else []

    if output_path is not None:
        output_path = Path(output_path)
        suffix = output_path.suffix.lower()
        if suffix == ".tex":
            output = table.to_latex(index=False, escape=False)
        elif suffix == ".csv":
            output = table.to_csv(index=False)
        elif suffix in {".md", ".markdown"}:
            output = table.to_markdown(index=False)
        else:
            output = table.to_string(index=False)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output, encoding="utf-8")

    return table
