"""Rank-0 downstream task loss evaluation for training-time logging."""

from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import dataclass
from typing import Callable

import torch


DCLM_CORE_22_TASKS = [
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
]


TASK_DATASET_CACHE_DIRS = {
    "arc_easy": ["allenai___ai2_arc"],
    "arc_challenge": ["allenai___ai2_arc"],
    "boolq": ["aps___super_glue"],
    "commonsense_qa": ["tau___commonsense_qa"],
    "copa": ["aps___super_glue"],
    "hellaswag": ["Rowan___hellaswag"],
    "openbookqa": ["allenai___openbookqa"],
    "piqa": ["baber___piqa"],
    "winogrande": ["allenai___winogrande"],
    "wsc273": ["winograd_wsc"],
    "lambada_openai": ["EleutherAI___lambada_openai"],
    "coqa": ["EleutherAI___coqa"],
    "squadv2": ["lighteval___squad_v2"],
    "agieval_lsat_ar": ["hails___agieval-lsat-ar"],
    "bigbench_language_identification_multiple_choice": ["hails___bigbench"],
    "bigbench_qa_wikidata_generate_until": ["hails___bigbench"],
    "bigbench_dyck_languages_generate_until": ["hails___bigbench"],
    "bigbench_operators_generate_until": ["hails___bigbench"],
    "bigbench_repeat_copy_logic_generate_until": ["hails___bigbench"],
    "bigbench_cs_algorithms_generate_until": ["hails___bigbench"],
}


@dataclass
class TaskBatch:
    name: str
    idx: torch.Tensor
    targets: torch.Tensor
    answer_bytes: list[int]

    @property
    def n_examples(self) -> int:
        return int(self.idx.size(0))


def resolve_task_names(value: str) -> list[str]:
    value = (value or "").strip()
    if not value or value in {"dclm-core-22", "dclm_core_22"}:
        return list(DCLM_CORE_22_TASKS)
    return [item.strip() for item in value.split(",") if item.strip()]


def _default_writable_hf_home() -> str:
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache:
        return os.path.join(xdg_cache, "huggingface")
    return os.path.join(os.path.expanduser("~"), ".cache", "huggingface")


def _skip_cache_file(name: str) -> bool:
    return name.endswith(".lock") or name.endswith(".incomplete") or name.endswith(".tmp")


def _mirror_cache_tree(src: str, dst: str) -> None:
    if not src or not os.path.isdir(src):
        return
    if os.path.islink(dst):
        os.unlink(dst)
    os.makedirs(dst, exist_ok=True)

    try:
        names = os.listdir(src)
    except OSError as err:
        logging.warning("task_loss_eval: cannot list cache %s: %s", src, err)
        return

    for name in names:
        src_path = os.path.join(src, name)
        dst_path = os.path.join(dst, name)
        if _skip_cache_file(name):
            if os.path.islink(dst_path):
                os.unlink(dst_path)
            continue
        if os.path.isdir(src_path) and not os.path.islink(src_path):
            _mirror_cache_tree(src_path, dst_path)
            continue
        if os.path.lexists(dst_path):
            continue
        try:
            os.symlink(src_path, dst_path)
        except OSError as err:
            logging.debug("task_loss_eval: symlink failed for %s: %s", src_path, err)


def _cache_dirs_for_tasks(task_names: list[str]) -> list[str]:
    dirs = set()
    for task_name in task_names:
        dirs.update(TASK_DATASET_CACHE_DIRS.get(task_name, []))
    return sorted(dirs)


def configure_hf_caches(
    shared_hf_home: str | None,
    writable_hf_home: str | None,
    task_names: list[str] | None = None,
) -> str:
    writable_hf_home = writable_hf_home or _default_writable_hf_home()
    writable_hf_home = os.path.abspath(os.path.expanduser(writable_hf_home))
    os.makedirs(writable_hf_home, exist_ok=True)

    os.environ["HF_HOME"] = writable_hf_home
    os.environ["HF_HUB_CACHE"] = os.path.join(writable_hf_home, "hub")
    os.environ["TRANSFORMERS_CACHE"] = os.path.join(writable_hf_home, "transformers")
    os.environ["HF_DATASETS_CACHE"] = os.path.join(writable_hf_home, "datasets_task_loss")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    for path in (
        os.environ["HF_HUB_CACHE"],
        os.environ["TRANSFORMERS_CACHE"],
        os.environ["HF_DATASETS_CACHE"],
    ):
        os.makedirs(path, exist_ok=True)

    if shared_hf_home:
        shared_hf_home = os.path.abspath(os.path.expanduser(shared_hf_home))
        shared_datasets = os.path.join(shared_hf_home, "datasets")
        cache_dirs = _cache_dirs_for_tasks(task_names or [])
        if cache_dirs:
            for cache_dir in cache_dirs:
                _mirror_cache_tree(
                    os.path.join(shared_datasets, cache_dir),
                    os.path.join(os.environ["HF_DATASETS_CACHE"], cache_dir),
                )
        else:
            _mirror_cache_tree(shared_datasets, os.environ["HF_DATASETS_CACHE"])

    return writable_hf_home


def _wait_for_file(path: str, timeout_sec: float, poll_sec: float = 0.25) -> None:
    start = time.monotonic()
    while not os.path.exists(path):
        if time.monotonic() - start > timeout_sec:
            raise TimeoutError(f"Timed out waiting for {path}")
        time.sleep(poll_sec)


def _atomic_write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    os.replace(tmp_path, path)


def _resolve_answer(task_obj, doc) -> str:
    target = task_obj.doc_to_target(doc)
    choices = None
    config = getattr(task_obj, "config", None)
    if getattr(config, "doc_to_choice", None) is not None:
        try:
            choices = task_obj.doc_to_choice(doc)
        except Exception:
            choices = None

    if isinstance(target, (list, tuple)):
        target = target[0] if target else ""

    if choices is not None:
        try:
            if isinstance(target, int):
                return str(choices[target])
            if isinstance(target, str) and target.strip().isdigit():
                return str(choices[int(target.strip())])
        except Exception:
            pass

    return str(target)


def _find_answer_boundary(prompt_ids: list[int], full_ids: list[int]) -> int:
    boundary = 0
    upper = min(len(prompt_ids), len(full_ids))
    while boundary < upper and prompt_ids[boundary] == full_ids[boundary]:
        boundary += 1
    return boundary


def _make_tensors(pairs: list[tuple[str, str]], tokenizer, seq_len: int) -> TaskBatch | None:
    idx_list = []
    target_list = []
    answer_bytes = []

    for prompt, answer in pairs:
        if not answer:
            continue

        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        full_ids = tokenizer.encode(prompt + answer, add_special_tokens=False)
        boundary = _find_answer_boundary(prompt_ids, full_ids)

        if len(full_ids) > seq_len + 1:
            offset = len(full_ids) - (seq_len + 1)
            full_ids = full_ids[offset:]
            boundary -= offset

        if boundary < 1:
            continue

        n = min(len(full_ids), seq_len + 1)
        hi = min(n - 1, seq_len)
        lo = boundary - 1
        if hi <= lo:
            continue

        idx = torch.zeros(seq_len, dtype=torch.long)
        targets = torch.full((seq_len,), -1, dtype=torch.long)
        input_len = min(n, seq_len)
        idx[:input_len] = torch.tensor(full_ids[:input_len], dtype=torch.long)
        for pos in range(lo, hi):
            targets[pos] = full_ids[pos + 1]

        idx_list.append(idx)
        target_list.append(targets)
        answer_bytes.append(len(answer.encode("utf-8")))

    if not idx_list:
        return None
    return TaskBatch(
        name="",
        idx=torch.stack(idx_list),
        targets=torch.stack(target_list),
        answer_bytes=answer_bytes,
    )


def build_task_batches(
    task_names: list[str],
    tokenizer,
    seq_len: int,
    max_examples: int,
) -> dict[str, TaskBatch]:
    from lm_eval.tasks import TaskManager, get_task_dict

    task_manager = TaskManager()
    task_batches = {}

    for task_name in task_names:
        try:
            task_dict = get_task_dict([task_name], task_manager)
            task_obj = task_dict.get(task_name)
            if task_obj is None:
                logging.warning("task_loss_eval: task %s not found, skipping", task_name)
                continue
            task_obj.set_config(key="num_fewshot", value=0)

            if task_obj.has_test_docs():
                docs_iter = task_obj.test_docs()
            elif task_obj.has_validation_docs():
                docs_iter = task_obj.validation_docs()
            else:
                logging.warning("task_loss_eval: no docs for %s, skipping", task_name)
                continue

            pairs = []
            for doc in docs_iter:
                try:
                    pairs.append((str(task_obj.doc_to_text(doc)), _resolve_answer(task_obj, doc)))
                except Exception as err:
                    logging.debug("task_loss_eval: bad doc for %s: %s", task_name, err)
                if max_examples > 0 and len(pairs) >= max_examples:
                    break

            batch = _make_tensors(pairs, tokenizer, seq_len)
            if batch is None:
                logging.warning("task_loss_eval: no usable examples for %s", task_name)
                continue
            batch.name = task_name
            task_batches[task_name] = batch
            logging.info(
                "task_loss_eval: prepared %d/%d examples for %s",
                batch.n_examples,
                len(pairs),
                task_name,
            )
        except Exception as err:
            logging.warning("task_loss_eval: failed to prepare %s: %s", task_name, err)

    return task_batches


class TaskLossEvaluator:
    def __init__(
        self,
        *,
        enabled_by_config: bool,
        task_names: list[str],
        output_dir: str,
        rank: int,
        local_rank: int,
        world_size: int,
        every: int,
        sync_timeout_sec: float,
    ):
        self.enabled = enabled_by_config
        self.task_names = task_names
        self.task_batches: dict[str, TaskBatch] = {}
        self.output_dir = output_dir
        self.rank = rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.every = every
        self.sync_timeout_sec = sync_timeout_sec
        self.sync_id = os.environ.get("SLURM_JOB_ID") or os.environ.get("MASTER_PORT") or "local"
        self.sync_dir = os.path.join(output_dir, ".task_loss_eval_sync")

    @classmethod
    def from_args(
        cls,
        args,
        *,
        rank: int,
        local_rank: int,
        world_size: int,
        master_process: bool,
        seq_len: int,
    ) -> "TaskLossEvaluator":
        every = int(args.task_eval_every)
        enabled = every >= 0
        evaluator = cls(
            enabled_by_config=enabled,
            task_names=resolve_task_names(args.task_eval_tasks),
            output_dir=args.output,
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            every=every,
            sync_timeout_sec=float(args.task_eval_sync_timeout_sec),
        )

        if master_process and enabled:
            try:
                hf_home = configure_hf_caches(
                    args.task_eval_hf_home or None,
                    args.task_eval_writable_hf_home or None,
                    evaluator.task_names,
                )
                from transformers import AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(
                    args.task_eval_tokenizer,
                    local_files_only=True,
                )
                evaluator.task_batches = build_task_batches(
                    evaluator.task_names,
                    tokenizer,
                    seq_len,
                    max_examples=int(args.task_eval_max_examples),
                )
                evaluator.enabled = bool(evaluator.task_batches)
                logging.info(
                    "task_loss_eval: setup %s with %d tasks using HF_HOME=%s",
                    "enabled" if evaluator.enabled else "disabled",
                    len(evaluator.task_batches),
                    hf_home,
                )
            except Exception as err:
                evaluator.enabled = False
                logging.warning("task_loss_eval: setup failed, disabling: %s", err)

        evaluator.sync_setup(master_process)
        return evaluator

    def _setup_status_path(self) -> str:
        return os.path.join(self.sync_dir, f"setup_{self.sync_id}.json")

    def _done_path(self, step: int) -> str:
        return os.path.join(self.sync_dir, f"done_{self.sync_id}_step{step:08d}.json")

    def sync_setup(self, master_process: bool) -> None:
        if self.world_size <= 1:
            return
        status_path = self._setup_status_path()
        if master_process:
            _atomic_write_json(
                status_path,
                {"enabled": self.enabled, "tasks": list(self.task_batches.keys())},
            )
        else:
            _wait_for_file(status_path, self.sync_timeout_sec)
            with open(status_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            self.enabled = bool(payload.get("enabled"))
            self.task_names = list(payload.get("tasks", []))

    def should_run(self, step: int, last_step: bool) -> bool:
        if not self.enabled:
            return False
        if last_step:
            return True
        if self.every == 0:
            return True
        return self.every > 0 and step % self.every == 0

    def run_or_wait(
        self,
        *,
        step: int,
        last_step: bool,
        master_process: bool,
        raw_model,
        device: str,
        ctx,
        diff_weight: float,
        batch_size: int,
        log_metrics: Callable[[dict[str, float]], None] | None = None,
    ) -> None:
        if not self.should_run(step, last_step):
            return

        done_path = self._done_path(step)
        if master_process:
            try:
                if os.path.exists(done_path):
                    os.remove(done_path)
                metrics = self.evaluate(
                    raw_model=raw_model,
                    device=device,
                    ctx=ctx,
                    diff_weight=diff_weight,
                    batch_size=batch_size,
                )
                if metrics and log_metrics is not None:
                    log_metrics(metrics)
                logging.info("task_loss_eval: logged %d metrics at step %d", len(metrics), step)
            except Exception as err:
                logging.warning("task_loss_eval: eval failed at step %d: %s", step, err)
            finally:
                try:
                    torch.cuda.synchronize()
                finally:
                    _atomic_write_json(done_path, {"step": step, "time": time.time()})
        else:
            _wait_for_file(done_path, self.sync_timeout_sec)

    def evaluate(
        self,
        *,
        raw_model,
        device: str,
        ctx,
        diff_weight: float,
        batch_size: int,
    ) -> dict[str, float]:
        was_training = raw_model.training
        raw_model.eval()
        metrics = {}

        try:
            with torch.no_grad():
                for task_name, batch in self.task_batches.items():
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
                            _, _, ce_loss, *_ = raw_model(
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
                        nll = total_nll / total_tokens
                        metrics[f"task_loss/{task_name}"] = nll
                        if total_bytes > 0:
                            metrics[f"task_bpb/{task_name}"] = (total_nll / math.log(2.0)) / total_bytes
                    else:
                        metrics[f"task_loss/{task_name}"] = float("nan")
        finally:
            if was_training:
                raw_model.train()
        return metrics
