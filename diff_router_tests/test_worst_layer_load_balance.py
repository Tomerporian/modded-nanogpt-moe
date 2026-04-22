from pathlib import Path
import sys
import types

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if "wandb" not in sys.modules:
    sys.modules["wandb"] = types.SimpleNamespace(
        log=lambda *args, **kwargs: None,
        init=lambda *args, **kwargs: None,
        finish=lambda *args, **kwargs: None,
    )

from gpt_moe_model import _aggregate_load_balance_aux
from params import parse_args
from wandb_logging import log_max_vio


def main():
    args, _ = parse_args(["--worst-layer-load-balance"])
    assert args.worst_layer_load_balance

    per_layer_aux = torch.tensor([1.0, 3.0, 2.0], requires_grad=True)
    total_aux_sum = _aggregate_load_balance_aux(per_layer_aux, worst_layer_only=False)
    total_aux_worst = _aggregate_load_balance_aux(per_layer_aux, worst_layer_only=True)
    assert torch.allclose(total_aux_sum, torch.tensor(6.0))
    assert torch.allclose(total_aux_worst, torch.tensor(3.0))

    grad_sum = torch.autograd.grad(total_aux_sum, per_layer_aux, retain_graph=True)[0]
    grad_worst = torch.autograd.grad(total_aux_worst, per_layer_aux)[0]
    assert torch.allclose(grad_sum, torch.tensor([1.0, 1.0, 1.0]))
    assert torch.allclose(grad_worst, torch.tensor([0.0, 1.0, 0.0]))

    balance = torch.tensor([
        [0.40, 0.30, 0.10, 0.20],
        [0.25, 0.25, 0.25, 0.25],
    ], dtype=torch.float32)
    log = log_max_vio(
        balance,
        'train/MaxViobatch',
        'train/MaxViobatch',
        worst_key='train/MaxViobatchWorstLayer',
    )
    assert abs(log['train/MaxViobatch'] - 0.3) < 1e-6
    assert abs(log['train/MaxViobatchWorstLayer'] - 0.6) < 1e-6

    print("Worst-layer load-balance checks passed.")


if __name__ == "__main__":
    main()
