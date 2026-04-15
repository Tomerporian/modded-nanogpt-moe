from pathlib import Path
import contextlib
import io
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gpt_moe_model import GPTConfig, MoE
from params import parse_args


def _expected_frac(balance_tensor):
    return balance_tensor.new_tensor(1.0 / balance_tensor.size(-1))


def maxvio_per_layer(balance_tensor):
    expected_frac = _expected_frac(balance_tensor)
    return (balance_tensor.max(dim=-1).values - expected_frac) / expected_frac


def minvio_per_layer(balance_tensor):
    expected_frac = _expected_frac(balance_tensor)
    return (expected_frac - balance_tensor.min(dim=-1).values) / expected_frac


def totalvio_per_layer(balance_tensor):
    expected_frac = _expected_frac(balance_tensor)
    return torch.abs(balance_tensor - expected_frac).sum(dim=-1) / expected_frac


LOSS_FLAG_FIELDS = {
    "maxvio": "maxvio_load_balance",
    "minmaxvio": "minmaxvio_load_balance",
    "totalvio": "totalvio_load_balance",
}


def _build_moe(loss_name, load_balance_ste_width, global_load_balance):
    loss_flags = {field: False for field in LOSS_FLAG_FIELDS.values()}
    loss_flags[LOSS_FLAG_FIELDS[loss_name]] = True
    cfg = GPTConfig(
        n_embd=1,
        hidden_dim_scale_factor=1.0,
        num_experts=4,
        top_k=2,
        router_type="switch",
        topk_activation="softmax",
        global_load_balance=global_load_balance,
        topk_ste_width=0.0,
        load_balance_ste_width=load_balance_ste_width,
        **loss_flags,
    )
    moe = MoE(cfg)
    moe.router = torch.nn.Linear(1, 4, bias=False)
    with torch.no_grad():
        moe.router.weight.copy_(torch.tensor([[2.7], [2.5], [2.4], [0.1]]))
    return moe


def _objective_value_and_grad(loss_name, load_balance_ste_width, global_frac=None):
    moe = _build_moe(
        loss_name=loss_name,
        load_balance_ste_width=load_balance_ste_width,
        global_load_balance=global_frac is not None,
    )
    x = torch.ones(1, 1, 1)
    router_context = None
    if global_frac is not None:
        router_context = {
            "mode": "use",
            "global_frac": global_frac.unsqueeze(0),
        }
    _, aux, *_ = moe(x, router_context=router_context)
    if aux.requires_grad:
        grad = torch.autograd.grad(aux, moe.router.weight)[0][:, 0].detach()
    else:
        grad = torch.zeros_like(moe.router.weight[:, 0].detach())
    return aux.detach(), grad


def _assert_parser_rejects(argv):
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            parse_args(argv)
    except SystemExit as exc:
        assert exc.code != 0
        return
    raise AssertionError(f"Expected parse_args({argv}) to fail")


def main():
    args, _ = parse_args(["--minmaxvio-load-balance"])
    assert args.minmaxvio_load_balance
    _assert_parser_rejects(["--maxvio-load-balance", "--totalvio-load-balance"])
    _assert_parser_rejects(["--totalvio-load-balance", "--theta-load-balance-coeff", "0.1"])

    expected_local = {
        "maxvio": torch.tensor(1.0),
        "minmaxvio": torch.tensor(2.0),
        "totalvio": torch.tensor(4.0),
    }
    global_frac = torch.tensor([0.40, 0.30, 0.10, 0.20], dtype=torch.float32)
    expected_global = {
        "maxvio": torch.tensor(0.6),
        "minmaxvio": torch.tensor(1.2),
        "totalvio": torch.tensor(1.6),
    }

    for loss_name, expected_value in expected_local.items():
        aux_no_ste, grad_no_ste = _objective_value_and_grad(loss_name, 0.0)
        aux_ste, grad_ste = _objective_value_and_grad(loss_name, 0.5)
        assert torch.allclose(aux_no_ste, expected_value), f"Unexpected local {loss_name} value without STE"
        assert torch.allclose(aux_ste, aux_no_ste), f"{loss_name} STE changed the local forward value"
        assert torch.allclose(grad_no_ste, torch.zeros_like(grad_no_ste)), f"{loss_name} should have zero local gradient without STE"
        if loss_name == "maxvio":
            assert grad_ste[0].abs() > 1e-4, "Local MaxVio STE should update the max-load expert"
            assert grad_ste[1].abs() > 1e-4, "Local MaxVio STE should propagate through the threshold term"
            assert grad_ste[2].abs() < 1e-7, "Local MaxVio STE should not update non-max experts directly"
            assert grad_ste[3].abs() < 1e-7, "Local MaxVio STE should not perturb experts outside the rectangle window"

    for loss_name, expected_value in expected_global.items():
        aux_no_ste, grad_no_ste = _objective_value_and_grad(loss_name, 0.0, global_frac=global_frac)
        aux_ste, grad_ste = _objective_value_and_grad(loss_name, 0.5, global_frac=global_frac)
        assert torch.allclose(aux_no_ste, expected_value), f"Unexpected global {loss_name} value without STE"
        assert torch.allclose(aux_ste, aux_no_ste), f"{loss_name} STE changed the global forward value"
        assert torch.allclose(grad_no_ste, torch.zeros_like(grad_no_ste)), f"{loss_name} should have zero global gradient without STE"
        if loss_name == "maxvio":
            assert grad_ste[0].abs() > 1e-4, "Global MaxVio STE should update the global max-load expert"
            assert grad_ste[1].abs() > 1e-4, "Global MaxVio STE should propagate through the threshold term"
            assert grad_ste[2].abs() < 1e-7, "Global MaxVio STE should not update non-max experts directly"
            assert grad_ste[3].abs() < 1e-7, "Global MaxVio STE should not perturb experts outside the rectangle window"
        else:
            assert grad_ste[0].abs() > 1e-4, f"Global {loss_name} STE should update the overloaded expert"
            assert grad_ste[2].abs() > 1e-4, f"Global {loss_name} STE should update the near-boundary underloaded expert"
            assert grad_ste[1].abs() < 1e-7, f"Global {loss_name} STE should not retain a net threshold-expert gradient in this setup"
            assert grad_ste[3].abs() < 1e-7, f"Global {loss_name} STE should not perturb experts outside the rectangle window"

    balance = torch.tensor([
        [0.40, 0.30, 0.10, 0.20],
        [0.25, 0.25, 0.25, 0.25],
    ], dtype=torch.float32)
    assert torch.allclose(maxvio_per_layer(balance), torch.tensor([0.6, 0.0]))
    assert torch.allclose(minvio_per_layer(balance), torch.tensor([0.6, 0.0]))
    assert torch.allclose(totalvio_per_layer(balance), torch.tensor([1.6, 0.0]))

    print("Violation load-balance checks passed.")


if __name__ == "__main__":
    main()
