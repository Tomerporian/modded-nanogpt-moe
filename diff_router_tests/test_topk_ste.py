from pathlib import Path
import sys

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gpt_moe_model import GPTConfig, MoE, switch_topk


class ConstantExpert(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = float(value)

    def forward(self, x):
        return torch.full_like(x, self.value)


def _forward_outputs(ste_width):
    logits = torch.tensor([[3.0, 2.5, 2.4, 0.1]])
    topk_idx, routed_probs, gate = switch_topk(
        logits,
        k=2,
        activation="softmax",
        ste_width=ste_width,
    )

    # Toy expert outputs to verify the routed mixture itself is unchanged.
    expert_outputs = torch.tensor([[1.5, -0.5, 4.0, 2.0]])
    moe_out = (routed_probs * expert_outputs).sum(dim=-1)
    return topk_idx.detach(), routed_probs.detach(), gate.detach(), moe_out.detach()


def _router_grad(ste_width):
    logits = torch.tensor([[3.0, 2.5, 2.4, 0.1]], requires_grad=True)
    _, routed_probs, gate = switch_topk(
        logits,
        k=2,
        activation="softmax",
        ste_width=ste_width,
    )
    loss = gate[0, 1] + routed_probs[0, 2]
    loss.backward()
    return logits.grad.detach(), routed_probs.detach(), gate.detach()


def _selected_only_router_grad():
    logits = torch.tensor([[3.0, 2.5, 2.4, 0.1]], requires_grad=True)
    topk_idx, _, gate = switch_topk(logits, k=2, activation="softmax", ste_width=0.5)
    expert_values = torch.tensor([[1.5, -0.5, 4.0, 2.0]])
    selected_values = torch.gather(expert_values, dim=-1, index=topk_idx)
    loss = (gate * selected_values).sum()
    loss.backward()
    return logits.grad.detach()


def _support_router_grad():
    logits = torch.tensor([[3.0, 2.5, 2.4, 0.1]], requires_grad=True)
    topk_idx, routed_probs, gate = switch_topk(logits, k=2, activation="softmax", ste_width=0.5)
    expert_values = torch.tensor([[1.5, -0.5, 4.0, 2.0]])
    hard_mask = torch.zeros_like(logits)
    hard_mask.scatter_(dim=-1, index=topk_idx, value=1.0)
    threshold = torch.gather(logits, dim=-1, index=topk_idx).min(dim=-1, keepdim=True).values
    extra_mask = ((logits - threshold).abs() < 0.25) & ~hard_mask.bool()

    selected_values = torch.gather(expert_values, dim=-1, index=topk_idx)
    hard_loss = (gate * selected_values).sum()
    extra_loss = (routed_probs * extra_mask.to(dtype=routed_probs.dtype) * expert_values).sum()
    loss = hard_loss + extra_loss - extra_loss.detach()
    loss.backward()
    return logits.grad.detach()


def _moe_router_grad():
    cfg = GPTConfig(
        n_embd=1,
        hidden_dim_scale_factor=1.0,
        num_experts=4,
        top_k=2,
        router_type="switch",
        topk_activation="softmax",
        topk_ste_width=0.5,
    )
    moe = MoE(cfg)
    moe.experts = nn.ModuleList([
        ConstantExpert(1.5),
        ConstantExpert(-0.5),
        ConstantExpert(4.0),
        ConstantExpert(2.0),
    ])
    moe.router = nn.Linear(1, 4, bias=False)
    with torch.no_grad():
        moe.router.weight.copy_(torch.tensor([[3.0], [2.5], [2.4], [0.1]]))

    x = torch.ones(1, 1, 1)
    y, *_ = moe(x)
    y.sum().backward()
    return moe.router.weight.grad[:, 0].detach()


def main():
    idx_no_ste, routed_no_ste_forward, gate_no_ste_forward, out_no_ste = _forward_outputs(0.0)
    idx_ste, routed_ste_forward, gate_ste_forward, out_ste = _forward_outputs(0.5)
    grad_no_ste, routed_no_ste_backward, gate_no_ste_backward = _router_grad(0.0)
    grad_ste, routed_ste_backward, gate_ste_backward = _router_grad(0.5)

    assert torch.equal(idx_no_ste, idx_ste), "STE changed the selected top-k experts"
    assert torch.allclose(routed_no_ste_forward, routed_ste_forward), "STE changed the forward routed probabilities"
    assert torch.allclose(gate_no_ste_forward, gate_ste_forward), "STE changed the forward top-k gate"
    assert torch.allclose(out_no_ste, out_ste), "STE changed the MoE forward output"
    assert torch.allclose(routed_no_ste_backward, routed_ste_backward), "STE changed the forward routed probabilities during grad check"
    assert torch.allclose(gate_no_ste_backward, gate_ste_backward), "STE changed the forward top-k gate during grad check"

    near_boundary_expert = 2
    boundary_expert = 1
    assert grad_no_ste[0, near_boundary_expert].abs() < 1e-7, "Baseline top-k should not update the excluded expert"
    assert grad_ste[0, near_boundary_expert].abs() > 1e-4, "STE should update the near-boundary excluded expert"
    assert grad_ste[0, boundary_expert] < grad_no_ste[0, boundary_expert], "Boundary expert should absorb the threshold correction"

    selected_only_grad = _selected_only_router_grad()
    support_grad = _support_router_grad()
    moe_grad = _moe_router_grad()
    assert support_grad[0, near_boundary_expert].abs() > selected_only_grad[0, near_boundary_expert].abs(), "Support expert value should strengthen the excluded expert router gradient"
    assert torch.allclose(moe_grad, support_grad[0], atol=1e-6, rtol=1e-6), "MoE output should match the manual support-gradient construction"

    print("Forward identical:", True)
    print("Top-k indices:", idx_ste)
    print("Forward gate:", gate_ste_forward)
    print("Toy MoE output:", out_ste)
    print("No-STE grad:", grad_no_ste)
    print("STE grad:", grad_ste)
    print("Selected-only output grad:", selected_only_grad)
    print("Support output grad:", support_grad)
    print("MoE output grad:", moe_grad)
    print("Top-k STE checks passed.")


if __name__ == "__main__":
    main()
