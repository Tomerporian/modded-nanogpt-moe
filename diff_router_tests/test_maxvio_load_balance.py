from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gpt_moe_model import GPTConfig, MoE


def _build_moe(load_balance_ste_width, global_load_balance):
    cfg = GPTConfig(
        n_embd=1,
        hidden_dim_scale_factor=1.0,
        num_experts=4,
        top_k=2,
        router_type="switch",
        topk_activation="softmax",
        global_load_balance=global_load_balance,
        maxvio_load_balance=True,
        topk_ste_width=0.0,
        load_balance_ste_width=load_balance_ste_width,
    )
    moe = MoE(cfg)
    moe.router = torch.nn.Linear(1, 4, bias=False)
    with torch.no_grad():
        moe.router.weight.copy_(torch.tensor([[2.7], [2.5], [2.4], [0.1]]))
    return moe


def _maxvio_value_and_grad(load_balance_ste_width, global_frac=None):
    moe = _build_moe(
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


def _global_mismatch_value_and_grad(load_balance_ste_width):
    moe = _build_moe(
        load_balance_ste_width=load_balance_ste_width,
        global_load_balance=True,
    )
    with torch.no_grad():
        moe.router.weight.copy_(torch.tensor([[3.0], [2.5], [2.4], [0.1]]))
    x = torch.ones(1, 1, 1)
    router_context = {
        "mode": "use",
        "global_frac": torch.tensor([[0.20, 0.20, 0.40, 0.20]]),
    }
    _, aux, *_ = moe(x, router_context=router_context)
    grad = torch.autograd.grad(aux, moe.router.weight)[0][:, 0].detach()
    return aux.detach(), grad


def main():
    local_no_ste, grad_local_no_ste = _maxvio_value_and_grad(0.0)
    local_ste, grad_local_ste = _maxvio_value_and_grad(0.5)

    assert torch.allclose(local_no_ste, torch.tensor(2.0)), "Unexpected local MaxVio value without STE"
    assert torch.allclose(local_ste, local_no_ste), "MaxVio STE changed the local forward value"
    assert torch.allclose(grad_local_no_ste, torch.zeros_like(grad_local_no_ste)), "Direct MaxVio should have zero local gradient without STE"
    assert grad_local_ste[0].abs() > 1e-4, "Local MaxVio STE should update the max-load expert"
    assert grad_local_ste[1].abs() > 1e-4, "Local MaxVio STE should propagate through the threshold term"
    assert grad_local_ste[3].abs() < 1e-7, "Local MaxVio STE should not perturb experts outside the rectangle window"
    assert grad_local_ste[2].abs() < 1e-7, "Local MaxVio STE should not update non-max experts directly"

    global_frac = torch.tensor([0.40, 0.30, 0.20, 0.10], dtype=torch.float32)
    global_no_ste, grad_global_no_ste = _maxvio_value_and_grad(0.0, global_frac=global_frac)
    global_ste, grad_global_ste = _maxvio_value_and_grad(0.5, global_frac=global_frac)

    assert torch.allclose(global_no_ste, torch.tensor(1.6)), "Unexpected global MaxVio value without STE"
    assert torch.allclose(global_ste, global_no_ste), "MaxVio STE changed the global forward value"
    assert torch.allclose(grad_global_no_ste, torch.zeros_like(grad_global_no_ste)), "Direct MaxVio should have zero global gradient without STE"
    assert grad_global_ste[0].abs() > 1e-4, "Global MaxVio STE should update the global max-load expert"
    assert grad_global_ste[1].abs() > 1e-4, "Global MaxVio STE should propagate through the threshold term"
    assert grad_global_ste[3].abs() < 1e-7, "Global MaxVio STE should not perturb experts outside the rectangle window"
    assert grad_global_ste[2].abs() < 1e-7, "Global MaxVio STE should not update non-max experts directly"

    mismatch_aux, mismatch_grad = _global_mismatch_value_and_grad(0.5)
    assert torch.allclose(mismatch_aux, torch.tensor(1.6)), "Unexpected global MaxVio value in mismatch case"
    assert mismatch_grad[0].abs() < 1e-7, "Global MaxVio should not give gradient to a local-only max expert"
    assert mismatch_grad[1].abs() > 1e-4, "Global MaxVio mismatch case should still propagate through the local threshold term"
    assert mismatch_grad[2].abs() > 1e-4, "Global MaxVio mismatch case should update the true global max expert"
    assert mismatch_grad[3].abs() < 1e-7, "Global MaxVio mismatch case should not perturb experts outside the rectangle window"

    print("Local MaxVio forward:", local_ste.item())
    print("Local MaxVio grad without STE:", grad_local_no_ste)
    print("Local MaxVio grad with STE:", grad_local_ste)
    print("Global MaxVio forward:", global_ste.item())
    print("Global MaxVio grad without STE:", grad_global_no_ste)
    print("Global MaxVio grad with STE:", grad_global_ste)
    print("Global mismatch forward:", mismatch_aux.item())
    print("Global mismatch grad with STE:", mismatch_grad)
    print("MaxVio load-balance checks passed.")


if __name__ == "__main__":
    main()
