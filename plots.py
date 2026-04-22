from __future__ import annotations

import math
from fractions import Fraction
from typing import Any, Dict, Optional, Tuple, Union

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
