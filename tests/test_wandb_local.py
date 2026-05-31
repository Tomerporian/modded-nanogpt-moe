import math
from pathlib import Path
import sys

import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wandb_local import load_local_wandb_runs, merge_local_wandb_runs


def _write_history(path: Path, rows) -> None:
    pd.DataFrame(rows).to_csv(path, index=False, compression="gzip")


def test_load_local_wandb_runs_merges_resumed_components(tmp_path: Path) -> None:
    output_dir = tmp_path / "wandb_dump"
    histories_dir = output_dir / "histories"
    histories_dir.mkdir(parents=True)

    _write_history(
        histories_dir / "run_a.csv.gz",
        [
            {"_step": 0, "train/loss": 10.0},
            {"_step": 1, "train/loss": 9.0},
            {"_step": 2, "train/loss": 8.0},
            {"_step": 2, "val/ce_loss": 0.80},
        ],
    )
    _write_history(
        histories_dir / "run_b.csv.gz",
        [
            {"_step": 2, "train/loss": 7.95},
            {"_step": 2, "val/ce_loss": 0.79},
            {"_step": 3, "train/loss": 7.0},
            {"_step": 3, "val/ce_loss": 0.70},
        ],
    )
    _write_history(
        histories_dir / "run_c.csv.gz",
        [
            {"_step": 0, "train/loss": 11.0},
            {"_step": 1, "train/loss": 10.5},
        ],
    )

    pd.DataFrame(
        [
            {
                "run_id": "run_a",
                "run_name": "exp-001",
                "run_path": "team-tomer/modded-nanogpt-moe/run_a",
                "state": "crashed",
                "created_at": "2026-04-01T00:00:00Z",
                "config.output": "logs/experiments/exp-001",
                "config.output_dir": "logs/experiments/exp-001",
                "config.resume": "auto",
                "config.top_k": 16,
                "config.router_type": "switch",
                "config.maxvio_load_balance": True,
                "config.rect_ste_threshold": None,
                "config.approx_global_load_balance": None,
                "summary.val/ce_loss": 0.80,
            },
            {
                "run_id": "run_b",
                "run_name": "exp-001",
                "run_path": "team-tomer/modded-nanogpt-moe/run_b",
                "state": "finished",
                "created_at": "2026-04-01T01:00:00Z",
                "config.output": "logs/experiments/exp-001",
                "config.output_dir": "logs/experiments/exp-001",
                "config.resume": "logs/experiments/exp-001/state_step000002.pt",
                "config.top_k": 16,
                "config.router_type": "switch",
                "config.rect_ste_threshold": "midpoint",
                "config.approx_global_load_balance": True,
                "summary.val/ce_loss": 0.70,
            },
            {
                "run_id": "run_c",
                "run_name": "exp-002",
                "run_path": "team-tomer/modded-nanogpt-moe/run_c",
                "state": "finished",
                "created_at": "2026-04-01T02:00:00Z",
                "config.output": "logs/experiments/exp-002",
                "config.output_dir": "logs/experiments/exp-002",
                "config.resume": "auto",
                "config.top_k": 16,
                "config.router_type": "switch",
                "config.rect_ste_threshold": None,
                "config.approx_global_load_balance": None,
                "summary.val/ce_loss": 1.50,
            },
        ]
    ).to_csv(output_dir / "selected_runs.csv", index=False)

    (output_dir / "manifest.json").write_text('{"project_path": "team-tomer/modded-nanogpt-moe"}\n')

    bundle = load_local_wandb_runs(output_dir)

    assert len(bundle.component_runs) == 3
    assert len(bundle.logical_runs) == 2

    selected = bundle.select_runs(top_k=16, router_type="switch")
    assert len(selected) == 2
    assert len(bundle.select_runs(load_balance_loss="switch")) == 2
    assert len(bundle.select_runs(component=True, load_balance_loss="maxvio")) == 1
    assert len(bundle.select_runs(component=True, rect_ste_threshold="topk")) == 2
    assert len(bundle.select_runs(component=True, approx_global_load_balance=False)) == 2

    logical_row = bundle.select_runs(output="logs/experiments/exp-001").iloc[0]
    assert logical_row["component_count"] == 2
    assert logical_row["has_resume"]
    assert logical_row["component_run_ids"] == ("run_a", "run_b")
    assert logical_row["component_resume_steps"] == (None, 2)
    assert logical_row["load_balance_loss"] == "switch"
    assert not logical_row["maxvio_load_balance"]
    assert logical_row["rect_ste_threshold"] == "midpoint"
    assert bool(logical_row["approx_global_load_balance"]) is True
    assert logical_row["summary.val/perplexity"] == pytest.approx(math.exp(0.70))

    component_row = bundle.select_runs(component=True, run_id="run_a").iloc[0]
    assert component_row["load_balance_loss"] == "maxvio"
    assert component_row["maxvio_load_balance"]
    assert component_row["rect_ste_threshold"] == "topk"
    assert bool(component_row["approx_global_load_balance"]) is False
    assert component_row["summary.val/perplexity"] == pytest.approx(math.exp(0.80))

    history = bundle.history(
        "exp-001",
        columns=["train/loss", "val/ce_loss", "val/perplexity"],
        include_component_columns=True,
    )
    assert history["step"].tolist() == [0, 1, 2, 3]

    step_two = history[history["step"] == 2].iloc[0]
    assert step_two["train/loss"] == pytest.approx(7.95)
    assert step_two["val/ce_loss"] == pytest.approx(0.79)
    assert step_two["val/perplexity"] == pytest.approx(math.exp(0.79))
    assert step_two["component_run_id"] == "run_b"

    with pytest.raises(ValueError):
        bundle.history_for_runs()

    multi_history = bundle.history_for_runs(columns=["train/loss"])
    assert set(multi_history["logical_run_id"]) == set(bundle.logical_runs["logical_run_id"])


def test_merge_local_wandb_runs_combines_bundles(tmp_path: Path) -> None:
    base_a = tmp_path / "bundle_a"
    base_b = tmp_path / "bundle_b"
    (base_a / "histories").mkdir(parents=True)
    (base_b / "histories").mkdir(parents=True)

    _write_history(
        base_a / "histories" / "shared.csv.gz",
        [
            {"_step": 0, "train/loss": 1.0},
            {"_step": 1, "train/loss": 0.9},
        ],
    )
    _write_history(
        base_b / "histories" / "shared.csv.gz",
        [
            {"_step": 0, "train/loss": 2.0},
            {"_step": 1, "train/loss": 1.9},
        ],
    )

    pd.DataFrame(
        [
            {
                "run_id": "shared",
                "run_name": "team-run",
                "run_path": "team-tomer/modded-nanogpt-moe/shared",
                "state": "finished",
                "created_at": "2026-04-01T00:00:00Z",
                "config.output": "logs/team-run",
                "config.output_dir": "logs/team-run",
                "config.resume": "auto",
                "config.top_k": 16,
                "config.router_type": "switch",
            }
        ]
    ).to_csv(base_a / "selected_runs.csv", index=False)
    (base_a / "manifest.json").write_text('{"project_path": "team-tomer/modded-nanogpt-moe"}\n')

    pd.DataFrame(
        [
            {
                "run_id": "shared",
                "run_name": "mikey-run",
                "run_path": "mikeyshechter/modded-nanogpt-moe/shared",
                "state": "finished",
                "created_at": "2026-04-01T00:00:00Z",
                "config.output": "logs/mikey-run",
                "config.output_dir": "logs/mikey-run",
                "config.resume": "auto",
                "config.top_k": 16,
                "config.router_type": "switch",
            }
        ]
    ).to_csv(base_b / "selected_runs.csv", index=False)
    (base_b / "manifest.json").write_text('{"project_path": "mikeyshechter/modded-nanogpt-moe"}\n')

    team_bundle = load_local_wandb_runs(base_a)
    mikey_bundle = load_local_wandb_runs(base_b)
    merged = merge_local_wandb_runs(team_bundle, mikey_bundle)

    assert len(merged.logical_runs) == 2
    assert len(merged.component_runs) == 2
    assert len(merged.select_runs(top_k=16, router_type="switch")) == 2

    team_row = merged.select_runs(output="logs/team-run").iloc[0]
    mikey_row = merged.select_runs(output="logs/mikey-run").iloc[0]
    assert team_row["source_label"] == "team-tomer/modded-nanogpt-moe"
    assert mikey_row["source_label"] == "mikeyshechter/modded-nanogpt-moe"

    team_history = merged.history("logs/team-run", columns=["train/loss"], include_component_columns=True)
    mikey_history = merged.history("logs/mikey-run", columns=["train/loss"], include_component_columns=True)
    assert team_history["train/loss"].tolist() == [1.0, 0.9]
    assert mikey_history["train/loss"].tolist() == [2.0, 1.9]
    assert team_history["component_id"].iloc[0] != mikey_history["component_id"].iloc[0]
