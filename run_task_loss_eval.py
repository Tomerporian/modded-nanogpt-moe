"""Standalone downstream task loss/BPB evaluation for finished checkpoints."""

from __future__ import annotations

import argparse
import logging
import math
import os
import re
import time
from contextlib import nullcontext
from dataclasses import MISSING, fields
from pathlib import Path
from typing import Any

import torch
import yaml
from transformers import AutoTokenizer

from gpt_moe_model import GPT, GPTConfig
from task_loss_eval import build_task_batches, configure_hf_caches, resolve_task_names


DTYPES = {
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return int(value)


def _load_run_args(run_dir: Path) -> dict[str, Any]:
    for candidate in ("args.yaml", "spec.yaml"):
        path = run_dir / candidate
        if path.exists():
            logging.info("Loading training arguments from %s", path)
            with open(path, "r", encoding="utf-8") as handle:
                data = yaml.safe_load(handle) or {}
            if not isinstance(data, dict):
                raise ValueError(f"Expected a mapping inside {path}, found {type(data)}")
            return data
    raise FileNotFoundError(f"Could not find args.yaml or spec.yaml inside {run_dir}")


def _checkpoint_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"state_step(\d+)\.pt$", path.name)
    if match:
        return int(match.group(1)), path.name
    return -1, path.name


def _resolve_checkpoint(run_dir: Path, checkpoint: str | None) -> Path:
    if checkpoint:
        candidate = Path(checkpoint).expanduser()
        if not candidate.is_file():
            candidate = run_dir / checkpoint
        if not candidate.is_file():
            raise FileNotFoundError(f"Checkpoint {checkpoint} not found")
        return candidate.resolve()

    checkpoints = sorted(run_dir.glob("state_step*.pt"), key=_checkpoint_sort_key)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint files named state_step*.pt found inside {run_dir}")
    return checkpoints[-1].resolve()


def _build_gpt_config(train_args: dict[str, Any]) -> GPTConfig:
    cfg_kwargs: dict[str, Any] = {}
    for field in fields(GPTConfig):
        value = train_args.get(field.name, MISSING)
        if value is not MISSING:
            cfg_kwargs[field.name] = value
        elif field.default is not MISSING:
            cfg_kwargs[field.name] = field.default
        elif field.default_factory is not MISSING:  # type: ignore[attr-defined]
            cfg_kwargs[field.name] = field.default_factory()  # type: ignore[attr-defined]
        else:
            raise ValueError(f"Missing required config value: {field.name}")
    return GPTConfig(**cfg_kwargs)


def _strip_prefix(state_dict: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    if not prefix:
        return state_dict
    return {
        (key[len(prefix):] if key.startswith(prefix) else key): value
        for key, value in state_dict.items()
    }


def _normalize_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict
    if any(key.startswith("_orig_mod.") for key in state_dict):
        return _strip_prefix(state_dict, "_orig_mod.")
    if any(key.startswith("module.") for key in state_dict):
        return _strip_prefix(state_dict, "module.")
    return state_dict


def _load_checkpoint(checkpoint_path: Path) -> tuple[dict[str, torch.Tensor], int | None]:
    logging.info("Loading checkpoint from %s", checkpoint_path)
    state = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_step = None
    if isinstance(state, dict):
        raw_step = state.get("step")
        if isinstance(raw_step, int):
            checkpoint_step = raw_step
        elif isinstance(raw_step, torch.Tensor) and raw_step.numel() == 1:
            checkpoint_step = int(raw_step.item())

        if "model" in state and isinstance(state["model"], dict):
            return _normalize_state_dict(state["model"]), checkpoint_step
        if "state_dict" in state and isinstance(state["state_dict"], dict):
            return _normalize_state_dict(state["state_dict"]), checkpoint_step
        if state and all(isinstance(key, str) for key in state) and all(
            isinstance(value, torch.Tensor) for value in state.values()
        ):
            return _normalize_state_dict(state), checkpoint_step

    raise ValueError(f"Unrecognized checkpoint structure in {checkpoint_path}")


def _checkpoint_step_from_name(checkpoint_path: Path) -> int | None:
    match = re.search(r"state_step(\d+)\.pt$", checkpoint_path.name)
    if match:
        return int(match.group(1))
    return None


def _scheduled_diff_weight(train_args: dict[str, Any], step: int | None) -> float:
    if step is None:
        return 1.0
    start = int(train_args.get("transition_start_iter", -1))
    end = int(train_args.get("transition_end_iter", -1))
    if start == -1 or step < start:
        return 1.0
    if end <= start or step > end:
        return 0.0
    return (end - step) / (end - start)


def _to_serializable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {key: _to_serializable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        converted = [_to_serializable(value) for value in obj]
        return converted if isinstance(obj, list) else tuple(converted)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, torch.dtype):
        return str(obj)
    if isinstance(obj, torch.device):
        return str(obj)
    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return obj.item()
        return obj.detach().cpu().tolist()
    try:
        import numpy as np  # type: ignore
    except ImportError:
        np = None
    if np is not None:
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    return obj


def _atomic_write_yaml(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with open(tmp_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(_to_serializable(payload), handle, sort_keys=False)
    os.replace(tmp_path, path)


def _read_yaml(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _sync_id() -> str:
    return (
        os.environ.get("SLURM_JOB_ID")
        or os.environ.get("MASTER_PORT")
        or f"manual_{os.getpid()}"
    )


def _wait_for_rank_parts(part_dir: Path, world_size: int, timeout_sec: float) -> list[dict[str, Any]]:
    deadline = time.monotonic() + timeout_sec
    part_paths = [part_dir / f"rank_{rank:05d}.yaml" for rank in range(world_size)]
    while True:
        missing = [path for path in part_paths if not path.exists()]
        if not missing:
            return [_read_yaml(path) for path in part_paths]
        if time.monotonic() > deadline:
            missing_names = ", ".join(path.name for path in missing[:8])
            if len(missing) > 8:
                missing_names += ", ..."
            raise TimeoutError(f"Timed out waiting for task-loss rank outputs: {missing_names}")
        time.sleep(1.0)


def _make_autocast(device: torch.device, dtype_name: str):
    if device.type == "cuda":
        dtype = DTYPES[dtype_name]
        return torch.amp.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def _evaluate_task_batches(
    *,
    model: GPT,
    task_batches,
    device: torch.device,
    ctx,
    diff_weight: float,
    batch_size: int,
) -> tuple[dict[str, dict[str, float | int]], dict[str, float]]:
    model.eval()
    results: dict[str, dict[str, float | int]] = {}
    flat_metrics: dict[str, float] = {}

    with torch.no_grad():
        for task_name, batch in task_batches.items():
            total_nll = 0.0
            total_tokens = 0
            total_bytes = sum(batch.answer_bytes)

            for start in range(0, batch.n_examples, batch_size):
                idx_b = batch.idx[start:start + batch_size].to(device, non_blocking=True)
                tgt_b = batch.targets[start:start + batch_size].to(device, non_blocking=True)
                n_tokens = int((tgt_b != -1).sum().item())
                if n_tokens == 0:
                    continue

                with ctx:
                    _, _, ce_loss, *_ = model(
                        idx_b,
                        tgt_b,
                        return_logits=False,
                        aux_coeff=0.0,
                        diff_topk_reg_coeff=0.0,
                        diff_weight=diff_weight,
                    )

                total_nll += float(ce_loss.item()) * n_tokens
                total_tokens += n_tokens

            if total_tokens > 0:
                task_loss = total_nll / total_tokens
                task_bpb = (total_nll / math.log(2.0)) / total_bytes if total_bytes > 0 else float("nan")
            else:
                task_loss = float("nan")
                task_bpb = float("nan")

            result = {
                "task_loss": task_loss,
                "task_bpb": task_bpb,
                "n_examples": batch.n_examples,
                "n_tokens": total_tokens,
                "answer_bytes": total_bytes,
            }
            results[task_name] = result
            flat_metrics[f"task_loss/{task_name}"] = task_loss
            flat_metrics[f"task_bpb/{task_name}"] = task_bpb
            logging.info(
                "%s task_loss=%.6f task_bpb=%.6f examples=%d tokens=%d",
                task_name,
                task_loss,
                task_bpb,
                batch.n_examples,
                total_tokens,
            )

    return results, flat_metrics


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained checkpoint with task_loss and task_bpb metrics."
    )
    parser.add_argument("--run_dir", type=str, required=True, help="Training run directory.")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint filename or path (default: latest state_step*.pt).",
    )
    parser.add_argument(
        "--tasks",
        default="dclm-core-22",
        help="Comma-separated lm-eval task names, or dclm-core-22.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=0,
        help="Maximum examples per task. 0 means all available examples.",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Eval batch size.")
    parser.add_argument(
        "--tokenizer-name",
        default=None,
        help="Tokenizer name or path (default: task_eval_tokenizer from args.yaml, then EleutherAI/gpt-neox-20b).",
    )
    parser.add_argument(
        "--shared-hf-home",
        default=None,
        help="Read-only HF cache root with hub/ and datasets/ (default: task_eval_hf_home from args.yaml).",
    )
    parser.add_argument(
        "--writable-hf-home",
        default=None,
        help="Writable HF cache root for offline symlinks (default: task_eval_writable_hf_home, HF_HOME, or XDG cache).",
    )
    parser.add_argument(
        "--results-file",
        default=None,
        help="Output YAML path (default: task_loss_results.yaml under the run directory).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device override (default: cuda:<LOCAL_RANK> if CUDA is available, else cpu).",
    )
    parser.add_argument(
        "--dtype",
        default=None,
        choices=sorted(DTYPES),
        help="Autocast dtype on CUDA (default: ops_dtype from args.yaml, then bfloat16).",
    )
    parser.add_argument(
        "--diff-weight",
        type=float,
        default=None,
        help="Override diff/switch interpolation weight (default: infer from checkpoint step and training schedule).",
    )
    parser.add_argument(
        "--sync-timeout-sec",
        type=float,
        default=7200.0,
        help="Timeout for file-based rank gather when WORLD_SIZE > 1.",
    )
    parser.add_argument("--log-level", default="INFO", help="Python log level.")
    return parser


def main(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    rank = _env_int("RANK", 0)
    local_rank = _env_int("LOCAL_RANK", 0)
    world_size = _env_int("WORLD_SIZE", 1)

    run_path = Path(args.run_dir).expanduser().resolve()
    if not run_path.is_dir():
        raise FileNotFoundError(f"{run_path} is not a directory")

    checkpoint_path = _resolve_checkpoint(run_path, args.checkpoint)
    train_args = _load_run_args(run_path)
    task_list = resolve_task_names(args.tasks)
    if not task_list:
        raise ValueError("At least one task must be provided")
    rank_task_list = task_list[rank::world_size]

    results_path = (
        Path(args.results_file).expanduser().resolve()
        if args.results_file
        else run_path / "task_loss_results.yaml"
    )
    if not results_path.is_absolute():
        results_path = run_path / results_path
    results_path.parent.mkdir(parents=True, exist_ok=True)

    shared_hf_home = (
        args.shared_hf_home
        or train_args.get("task_eval_hf_home")
        or os.environ.get("SHARED_HF_HOME")
    )
    writable_hf_home = (
        args.writable_hf_home
        or train_args.get("task_eval_writable_hf_home")
        or os.environ.get("HF_HOME")
    )
    configured_hf_home = configure_hf_caches(
        shared_hf_home=shared_hf_home or None,
        writable_hf_home=writable_hf_home or None,
        task_names=rank_task_list,
    )
    logging.info(
        "Rank %d/%d local_rank=%d using HF_HOME=%s",
        rank,
        world_size,
        local_rank,
        configured_hf_home,
    )

    context_length = int(train_args.get("sequence_length", 1024))
    tokenizer_name = args.tokenizer_name or train_args.get("task_eval_tokenizer") or "EleutherAI/gpt-neox-20b"
    logging.info(
        "Rank %d/%d assigned %d task(s): %s",
        rank,
        world_size,
        len(rank_task_list),
        ",".join(rank_task_list) if rank_task_list else "<none>",
    )

    if args.device is not None:
        target_device = torch.device(args.device)
    elif torch.cuda.is_available():
        target_device = torch.device(f"cuda:{local_rank}")
    else:
        target_device = torch.device("cpu")

    if target_device.type == "cuda":
        torch.cuda.set_device(target_device)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    dtype_name = args.dtype or train_args.get("ops_dtype") or "bfloat16"
    if dtype_name not in DTYPES:
        raise ValueError(f"Unsupported dtype {dtype_name}; choose one of {sorted(DTYPES)}")

    state_dict, checkpoint_step = _load_checkpoint(checkpoint_path)
    if checkpoint_step is None:
        checkpoint_step = _checkpoint_step_from_name(checkpoint_path)
    diff_weight = args.diff_weight
    if diff_weight is None:
        diff_weight = _scheduled_diff_weight(train_args, checkpoint_step)

    logging.info("Using device=%s dtype=%s diff_weight=%s", target_device, dtype_name, diff_weight)

    start = time.time()
    if rank_task_list:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=True)
        logging.info(
            "Preparing %d task(s), max_examples=%s, seq_len=%d",
            len(rank_task_list),
            "all" if args.max_examples <= 0 else args.max_examples,
            context_length,
        )
        task_batches = build_task_batches(
            rank_task_list,
            tokenizer,
            context_length,
            max_examples=args.max_examples,
        )
        if not task_batches:
            raise RuntimeError(f"Rank {rank} built no usable task batches")

        model = GPT(_build_gpt_config(train_args))
        missing = model.load_state_dict(state_dict, strict=True)
        if missing.unexpected_keys or missing.missing_keys:
            logging.warning(
                "Issues while loading model state: missing=%s unexpected=%s",
                missing.missing_keys,
                missing.unexpected_keys,
            )
        model.eval().to(target_device)
        for param in model.parameters():
            param.requires_grad_(False)

        results, flat_metrics = _evaluate_task_batches(
            model=model,
            task_batches=task_batches,
            device=target_device,
            ctx=_make_autocast(target_device, dtype_name),
            diff_weight=float(diff_weight),
            batch_size=args.batch_size,
        )
    else:
        results = {}
        flat_metrics = {}
    elapsed = time.time() - start
    logging.info("Rank %d task loss evaluation finished in %.2fs", rank, elapsed)

    rank_payload = {
        "rank": rank,
        "world_size": world_size,
        "tasks": rank_task_list,
        "elapsed_sec": elapsed,
        "results": results,
        "metrics": flat_metrics,
    }

    if world_size > 1:
        part_dir = results_path.parent / ".task_loss_eval_parts" / _sync_id()
        _atomic_write_yaml(part_dir / f"rank_{rank:05d}.yaml", rank_payload)
        if rank != 0:
            return

        rank_payloads = _wait_for_rank_parts(part_dir, world_size, args.sync_timeout_sec)
        results = {}
        flat_metrics = {}
        per_rank = []
        for payload in rank_payloads:
            results.update(payload.get("results", {}))
            flat_metrics.update(payload.get("metrics", {}))
            per_rank.append({
                "rank": payload.get("rank"),
                "tasks": payload.get("tasks", []),
                "elapsed_sec": payload.get("elapsed_sec", 0.0),
            })
        missing_tasks = [task for task in task_list if task not in results]
        if missing_tasks:
            raise RuntimeError(f"Missing task-loss outputs for tasks: {missing_tasks}")
        results = {task: results[task] for task in task_list}
        flat_metrics = {
            key: flat_metrics[key]
            for task in task_list
            for key in (f"task_loss/{task}", f"task_bpb/{task}")
            if key in flat_metrics
        }
        elapsed = max((float(item["elapsed_sec"]) for item in per_rank), default=elapsed)
    else:
        per_rank = [rank_payload]

    payload = {
        "run_dir": run_path,
        "checkpoint": checkpoint_path,
        "checkpoint_step": checkpoint_step,
        "tasks": task_list,
        "batch_size": args.batch_size,
        "max_examples": args.max_examples,
        "sequence_length": context_length,
        "tokenizer": tokenizer_name,
        "dtype": dtype_name,
        "diff_weight": diff_weight,
        "elapsed_sec": elapsed,
        "world_size": world_size,
        "per_rank": per_rank,
        "results": results,
        "metrics": flat_metrics,
    }

    _atomic_write_yaml(results_path, payload)
    logging.info("Saved task loss results to %s", results_path)


if __name__ == "__main__":
    parser = _build_parser()
    main(parser.parse_args())
