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
            "pandas is required to build lm-eval result tables. Install pandas "
            "or call this helper where pandas is available."
        ) from exc
    return pd


def _import_yaml():
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "pyyaml is required to read lm-eval YAML results. Install pyyaml "
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

_DEFAULT_LM_EVAL_TASK_METRICS: Dict[str, Tuple[str, ...]] = {
    "arc_challenge": ("acc_norm,none", "acc,none"),
    "arc_easy": ("acc_norm,none", "acc,none"),
    "hellaswag": ("acc_norm,none", "acc,none"),
    "piqa": ("acc_norm,none", "acc,none"),
    "winogrande": ("acc,none",),
    "lambada_openai": ("acc,none",),
    "cola": ("mcc,none", "acc,none"),
    "mrpc": ("f1,none", "acc,none"),
    "qqp": ("f1,none", "acc,none"),
}


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


def _select_lm_eval_metric(
    task_name: str,
    metrics: Mapping[str, Any],
    metric_preferences: Sequence[str],
) -> Tuple[float, Optional[float]]:
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
            return float(value), stderr_value

    raise KeyError(
        f"Could not find any supported metric for {task_name}. "
        f"Available keys: {sorted(metrics)}"
    )


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


def make_lm_eval_results_table(
    results_root: Union[str, Path] = "/e/scratch/reformo/shechter1/lm_eval_results",
    *,
    result_filename: str = "lm_eval_results.yaml",
    run_paths: Optional[Sequence[Union[str, Path]]] = None,
    run_labels: Optional[Mapping[str, str]] = None,
    benchmark_specs: Optional[Sequence[Mapping[str, Any]]] = None,
    metric_preferences: Sequence[str] = (
        "acc_norm,none",
        "acc,none",
        "exact_match,none",
        "f1,none",
        "mcc,none",
    ),
    decimals: int = 2,
    percent: bool = True,
    show_stderr: bool = True,
    show_mean_stderr: bool = False,
    pm_symbol: str = "+/-",
    include_missing: bool = False,
    output_path: Optional[Union[str, Path]] = None,
):
    """
    Build a paper-style summary table from lm-eval result directories.

    The default scans `/e/scratch/reformo/shechter1/lm_eval_results` for
    `*/lm_eval_results.yaml`, which currently matches the three eval runs. The
    function reads only the top-level `results:` block, avoiding the large
    `samples:` block emitted by lm-eval.

    Scores are displayed as percentages with standard error when lm-eval reports
    one. Multi-subtask benchmarks such as GLUE or BLiMP are averaged across
    available subtasks and get a combined standard error when possible.

    Args:
        results_root:
            Directory containing one subdirectory per run.
        result_filename:
            Result YAML filename inside each run directory.
        run_paths:
            Optional explicit result directories or YAML files. If omitted, all
            matching subdirectories under results_root are used.
        run_labels:
            Optional mapping from run directory name to display label.
        benchmark_specs:
            Optional benchmark definitions. Each mapping should contain `label`
            and either `tasks` or `prefix`.
        metric_preferences:
            Fallback metric priority for tasks without a task-specific default.
        output_path:
            Optional `.tex`, `.csv`, or `.md` path to write the formatted table.

    Returns:
        pandas.DataFrame with formatted table cells. Raw numeric values are also
        stored in `table.attrs["raw_scores"]`.
    """
    pd = _import_pandas()
    results_root = Path(results_root)
    specs = tuple(benchmark_specs or _DEFAULT_LM_EVAL_BENCHMARKS)

    if run_paths is None:
        paths = sorted(results_root.glob(f"*/{result_filename}"))
    else:
        paths = []
        for run_path in run_paths:
            path = Path(run_path)
            if path.is_dir():
                path = path / result_filename
            paths.append(path)

    if not paths:
        raise FileNotFoundError(
            f"No {result_filename} files found under {results_root}"
        )

    run_labels = dict(run_labels or {})
    result_sets: list[Tuple[str, Dict[str, Dict[str, Any]]]] = []
    for path in paths:
        run_name = path.parent.name
        label = run_labels.get(run_name, _pretty_lm_eval_run_label(run_name))
        result_sets.append((label, _load_lm_eval_results_block(path)))

    rows: list[Dict[str, str]] = []
    raw_scores: Dict[str, Dict[str, Tuple[Optional[float], Optional[float]]]] = {}

    for spec in specs:
        benchmark_key = str(spec.get("key", spec.get("label", "benchmark")))
        benchmark_label = str(spec["label"])
        row: Dict[str, str] = {"Benchmark": benchmark_label}
        raw_scores[benchmark_key] = {}
        has_any_score = False

        for run_label, results in result_sets:
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
            column = f"{run_label} {pm_symbol} stderr"
            row[column] = _format_lm_eval_score(
                value,
                stderr,
                decimals=decimals,
                percent=percent,
                show_stderr=show_stderr,
                pm_symbol=pm_symbol,
            )

        if has_any_score or include_missing:
            rows.append(row)

    mean_row: Dict[str, str] = {"Benchmark": "mean"}
    raw_scores["mean"] = {}
    benchmark_keys = [str(spec.get("key", spec.get("label", "benchmark"))) for spec in specs]
    for run_label, _ in result_sets:
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
        column = f"{run_label} {pm_symbol} stderr"
        mean_row[column] = _format_lm_eval_score(
            value,
            stderr,
            decimals=decimals,
            percent=percent,
            show_stderr=show_mean_stderr,
            pm_symbol=pm_symbol,
        )
    rows.append(mean_row)

    table = pd.DataFrame(rows)
    table.attrs["raw_scores"] = raw_scores

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
