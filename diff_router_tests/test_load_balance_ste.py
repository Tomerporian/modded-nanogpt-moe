from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gpt_moe_model import GPTConfig, MoE


def _build_moe(load_balance_ste_width):
    cfg = GPTConfig(
        n_embd=1,
        hidden_dim_scale_factor=1.0,
        num_experts=4,
        top_k=2,
        router_type="switch",
        topk_activation="softmax",
        topk_ste_width=0.0,
        load_balance_ste_width=load_balance_ste_width,
    )
    moe = MoE(cfg)
    moe.router = torch.nn.Linear(1, 4, bias=False)
    with torch.no_grad():
        moe.router.weight.copy_(torch.tensor([[3.0], [2.5], [2.4], [0.1]]))
    return moe


def _aux_value_and_grad(load_balance_ste_width, global_frac=None):
    moe = _build_moe(load_balance_ste_width)
    x = torch.ones(1, 1, 1)
    router_context = None
    if global_frac is not None:
        router_context = {
            'mode': 'use',
            'global_frac': global_frac.unsqueeze(0),
        }
    _, aux, *_ = moe(x, router_context=router_context)
    grad = torch.autograd.grad(aux, moe.router.weight)[0][:, 0].detach()
    return aux.detach(), grad


def main():
    aux_local_no_ste, grad_local_no_ste = _aux_value_and_grad(0.0)
    aux_local_ste, grad_local_ste = _aux_value_and_grad(0.5)

    assert torch.allclose(aux_local_no_ste, aux_local_ste), "LB-STE changed the local aux forward value"
    grad_local_diff = grad_local_ste - grad_local_no_ste
    assert grad_local_diff[2].abs() > 1e-4, "LB-STE should change the near-boundary excluded expert gradient"
    assert grad_local_diff[1].abs() > 1e-4, "LB-STE should change the threshold expert gradient"
    assert grad_local_diff[0].abs() < 1e-7, "LB-STE should not perturb experts outside the rectangle window"
    assert grad_local_diff[3].abs() < 1e-7, "LB-STE should not perturb experts outside the rectangle window"

    global_frac = torch.tensor([0.40, 0.30, 0.20, 0.10], dtype=torch.float32)
    aux_global_no_ste, grad_global_no_ste = _aux_value_and_grad(0.0, global_frac=global_frac)
    aux_global_ste, grad_global_ste = _aux_value_and_grad(0.5, global_frac=global_frac)

    assert torch.allclose(aux_global_no_ste, aux_global_ste), "LB-STE changed the global aux forward value"
    grad_global_diff = grad_global_ste - grad_global_no_ste
    assert grad_global_diff[2].abs() > 1e-4, "Hybrid global LB-STE should still change the local near-boundary gradient"
    assert grad_global_diff[1].abs() > 1e-4, "Hybrid global LB-STE should still change the local threshold gradient"
    assert grad_global_diff[0].abs() < 1e-7, "Hybrid global LB-STE should not perturb experts outside the rectangle window"
    assert grad_global_diff[3].abs() < 1e-7, "Hybrid global LB-STE should not perturb experts outside the rectangle window"

    print("Local aux forward:", aux_local_ste.item())
    print("Local grad without STE:", grad_local_no_ste)
    print("Local grad with STE:", grad_local_ste)
    print("Global aux forward:", aux_global_ste.item())
    print("Global grad without STE:", grad_global_no_ste)
    print("Global grad with STE:", grad_global_ste)
    print("Load-balance STE checks passed.")


if __name__ == "__main__":
    main()
