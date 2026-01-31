import re
import sys
import time
from dataclasses import MISSING, fields
from pathlib import Path
from typing import Any

import argparse
import torch
import yaml
from loguru import logger
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from transformers import AutoTokenizer, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from gpt_moe_model import GPT, GPTConfig


def _load_run_args(run_dir: Path) -> dict[str, Any]:
    for candidate in ("args.yaml", "spec.yaml"):
        path = run_dir / candidate
        if path.exists():
            logger.info(f"Loading training arguments from {path}")
            with open(path, "r", encoding="utf-8") as handle:
                data = yaml.safe_load(handle) or {}
                if isinstance(data, dict):
                    return data
                msg = f"Expected a mapping inside {path}, found {type(data)}"
                raise ValueError(msg)
    raise FileNotFoundError(
        f"Could not find args.yaml or spec.yaml inside {run_dir}"
    )


def _checkpoint_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"state_step(\d+)\.pt$", path.name)
    if match:
        return int(match.group(1)), path.name
    return -1, path.name


def _resolve_checkpoint(run_dir: Path, checkpoint: str | None) -> Path:
    if checkpoint:
        candidate = Path(checkpoint)
        if not candidate.is_file():
            candidate = run_dir / checkpoint
        if not candidate.is_file():
            raise FileNotFoundError(f"Checkpoint {checkpoint} not found")
        return candidate

    checkpoints = sorted(run_dir.glob("state_step*.pt"), key=_checkpoint_sort_key)
    if not checkpoints:
        raise FileNotFoundError(
            f"No checkpoint files named state_step*.pt found inside {run_dir}"
        )
    return checkpoints[-1]


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


def _load_model_state(checkpoint_path: Path, map_location: str | torch.device) -> dict:
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location=map_location)
    if isinstance(state, dict):
        if "model" in state and isinstance(state["model"], dict):
            return state["model"]
        if "state_dict" in state and isinstance(state["state_dict"], dict):
            return state["state_dict"]
    if isinstance(state, dict):
        return state
    raise ValueError(f"Unrecognized checkpoint structure in {checkpoint_path}")


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


def _to_serializable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {key: _to_serializable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        converted = [_to_serializable(value) for value in obj]
        return converted if isinstance(obj, list) else tuple(converted)
    if callable(obj):
        name = getattr(obj, "__name__", None)
        if name:
            return f"<callable {name}>"
        return "<callable>"
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


class CustomConfig(PretrainedConfig):
    model_type = "modded_nanogpt_moe"

    def __init__(
        self,
        vocab_size: int,
        n_positions: int,
        hidden_size: int,
        num_attention_heads: int,
        num_hidden_layers: int,
        bos_token_id: int,
        eos_token_id: int,
        pad_token_id: int,
    ):
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.max_position_embeddings = n_positions
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers


class CustomModel(PreTrainedModel):
    config_class = CustomConfig

    def __init__(self, config: CustomConfig, model: GPT):
        super().__init__(config)
        self.model = model

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **_: Any,
    ) -> CausalLMOutputWithPast:
        assert input_ids is not None, "input_ids must be provided"
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)

        if labels is not None:
            labels = labels.to(device)
            hf_loss_labels = labels.clone()
            model_targets = labels.masked_fill(labels == -100, -1)
        else:
            hf_loss_labels = None
            model_targets = torch.full_like(input_ids, fill_value=-1)

        logits, *_ = self.model(
            idx=input_ids,
            targets=model_targets,
            return_logits=True,
            aux_coeff=0.0,
            diff_topk_reg_coeff=0.0,
        )

        loss = None
        if hf_loss_labels is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                hf_loss_labels.reshape(-1),
                ignore_index=-100,
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )


def _configure_tokenizer(name: str, context_length: int) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = context_length
    tokenizer.padding_side = "left"
    return tokenizer


def _default_results_path(run_dir: Path, tasks: list[str]) -> Path:
    if len(tasks) == 1:
        filename = f"{tasks[0]}.yaml"
    else:
        filename = "lm_eval_results.yaml"
    return run_dir / filename


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained checkpoint on lm-eval tasks such as hellaswag."
    )
    parser.add_argument("--run_dir", type=str, required=True,
        help="Training run directory containing args.yaml/spec.yaml.")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint filename or path (default: latest state_step*.pt).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit number of eval examples.")
    parser.add_argument("--log-level", default="INFO", help="Loguru log level (default: INFO).")
    parser.add_argument("--device", default=None, help="Device override (e.g. cpu, cuda).")
    parser.add_argument(
        "--tasks",
        default="hellaswag",
        help="Comma-separated lm-eval tasks (default: hellaswag).",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Eval batch size.")
    parser.add_argument("--tokenizer-name", default="gpt2", help="Tokenizer name or path.")
    parser.add_argument(
        "--results-file",
        default=None,
        help="Output YAML filename (default: <task>.yaml or lm_eval_results.yaml).",
    )
    return parser


def main(args: argparse.Namespace) -> None:
    """
    Evaluate a trained checkpoint on lm-eval tasks such as hellaswag.
    """

    logger.remove()
    logger.add(sys.stderr, level=args.log_level.upper())

    run_path = Path(args.run_dir).expanduser().resolve()
    if not run_path.is_dir():
        raise FileNotFoundError(f"{run_path} is not a directory")

    checkpoint_path = _resolve_checkpoint(run_path, args.checkpoint)
    train_args = _load_run_args(run_path)
    gpt_config = _build_gpt_config(train_args)

    context_length = int(train_args.get("sequence_length", 1024))
    tokenizer = _configure_tokenizer(args.tokenizer_name, context_length)

    hf_config = CustomConfig(
        vocab_size=gpt_config.vocab_size,
        n_positions=context_length,
        hidden_size=gpt_config.n_embd,
        num_attention_heads=gpt_config.n_head,
        num_hidden_layers=gpt_config.n_layer,
        bos_token_id=tokenizer.bos_token_id or tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    target_device = torch.device(
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    logger.info(f"Using device {target_device}")

    state_dict = _normalize_state_dict(_load_model_state(checkpoint_path, map_location="cpu"))
    base_model = GPT(gpt_config)
    missing = base_model.load_state_dict(state_dict, strict=True)
    if missing.unexpected_keys or missing.missing_keys:
        logger.warning(
            f"Issues while loading model state: missing={missing.missing_keys}, unexpected={missing.unexpected_keys}"
        )
    base_model.eval().to(target_device)
    for param in base_model.parameters():
        param.requires_grad_(False)

    hf_model = CustomModel(hf_config, base_model).to(target_device)
    hf_model.eval()

    task_list = [task.strip() for task in args.tasks.split(",") if task.strip()]
    if not task_list:
        raise ValueError("At least one task must be provided")

    wrapped_model = HFLM(
        pretrained=hf_model,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        device=str(target_device),
    )

    start = time.time()
    logger.info(f"Running lm-eval on tasks: {task_list}")
    results_raw = evaluator.simple_evaluate(
        model=wrapped_model,
        tasks=task_list,
        limit=args.limit,
        verbosity="DEBUG",
    )
    elapsed = time.time() - start
    logger.info(f"Evaluation finished in {elapsed:.2f}s")

    for task in task_list:
        task_metrics = results_raw["results"].get(task, {})
        if task_metrics:
            acc = task_metrics.get("acc,none")
            if acc is not None:
                logger.success(f"{task} accuracy: {acc:.4f}")
            else:
                logger.success(f"{task} metrics: {task_metrics}")

    results_path = (
        Path(args.results_file).expanduser().resolve()
        if args.results_file
        else _default_results_path(run_path, task_list)
    )
    if not results_path.is_absolute():
        results_path = run_path / results_path
    results_path.parent.mkdir(parents=True, exist_ok=True)

    payload: Any
    if len(task_list) == 1:
        payload = results_raw["results"][task_list[0]]
    else:
        payload = results_raw
    payload = _to_serializable(payload)

    with open(results_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle)
    logger.info(f"Saved results to {results_path}")


if __name__ == "__main__":
    parser = _build_parser()
    main(parser.parse_args())
