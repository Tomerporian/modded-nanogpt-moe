from pathlib import Path
import contextlib
import io
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gpt_moe_model import GPTConfig, MoE, _rect_ste_threshold, switch_topk
from params import parse_args


LOSS_FLAG_FIELDS = {
    "maxvio": "maxvio_load_balance",
    "minmaxvio": "minmaxvio_load_balance",
    "totalvio": "totalvio_load_balance",
}


def _assert_parser_rejects(argv):
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            parse_args(argv)
    except SystemExit as exc:
        assert exc.code != 0
        return
    raise AssertionError(f"Expected parse_args({argv}) to fail")


def _router_grad(ste_width, threshold_mode):
    logits = torch.tensor([[3.0, 2.8, 2.4, 0.1]], requires_grad=True)
    topk_idx, routed_probs, gate = switch_topk(
        logits,
        k=2,
        activation="softmax",
        ste_width=ste_width,
        ste_threshold_mode=threshold_mode,
    )
    loss = gate[0, 1] + routed_probs[0, 2]
    loss.backward()
    threshold = _rect_ste_threshold(logits.detach(), topk_idx.detach(), threshold_mode=threshold_mode)
    return topk_idx.detach(), routed_probs.detach(), gate.detach(), threshold.detach(), logits.grad.detach()


def _violation_grad(loss_name, threshold_mode):
    loss_flags = {field: False for field in LOSS_FLAG_FIELDS.values()}
    loss_flags[LOSS_FLAG_FIELDS[loss_name]] = True
    cfg = GPTConfig(
        n_embd=1,
        hidden_dim_scale_factor=1.0,
        num_experts=4,
        top_k=2,
        router_type="switch",
        topk_activation="softmax",
        rect_ste_threshold=threshold_mode,
        topk_ste_width=0.0,
        load_balance_ste_width=0.5,
        **loss_flags,
    )
    moe = MoE(cfg)
    moe.router = torch.nn.Linear(1, 4, bias=False)
    with torch.no_grad():
        moe.router.weight.copy_(torch.tensor([[3.0], [2.8], [2.4], [0.1]]))
    x = torch.ones(1, 1, 1)
    _, aux, *_ = moe(x)
    grad = torch.autograd.grad(aux, moe.router.weight)[0][:, 0].detach()
    return aux.detach(), grad


def _all_selected_grad(ste_width):
    logits = torch.tensor([[3.0, 2.0, 1.0]], requires_grad=True)
    _, routed_probs, gate = switch_topk(
        logits,
        k=3,
        activation="softmax",
        ste_width=ste_width,
        ste_threshold_mode="topk_plus_one",
    )
    loss = gate.sum() + routed_probs[0, 0]
    loss.backward()
    return routed_probs.detach(), logits.grad.detach()


def main():
    args, _ = parse_args(["--rect-ste-threshold", "topk_plus_one"])
    assert args.rect_ste_threshold == "topk_plus_one"
    _assert_parser_rejects(["--rect-ste-threshold", "bad_mode"])

    idx_no_ste, routed_no_ste, gate_no_ste, threshold_no_ste, grad_no_ste = _router_grad(0.0, "topk")
    idx_topk, routed_topk, gate_topk, threshold_topk, grad_topk = _router_grad(0.5, "topk")
    idx_plus_one, routed_plus_one, gate_plus_one, threshold_plus_one, grad_plus_one = _router_grad(0.5, "topk_plus_one")

    assert torch.equal(idx_no_ste, idx_topk)
    assert torch.equal(idx_no_ste, idx_plus_one)
    assert torch.allclose(routed_no_ste, routed_topk)
    assert torch.allclose(routed_no_ste, routed_plus_one)
    assert torch.allclose(gate_no_ste, gate_topk)
    assert torch.allclose(gate_no_ste, gate_plus_one)
    assert torch.allclose(threshold_no_ste, torch.tensor([[2.8]]))
    assert torch.allclose(threshold_topk, torch.tensor([[2.8]]))
    assert torch.allclose(threshold_plus_one, torch.tensor([[2.4]]))

    assert not torch.allclose(grad_topk, grad_no_ste), "topk threshold mode should change the router gradient in this setup"
    assert torch.allclose(grad_plus_one, grad_no_ste, atol=1e-6, rtol=1e-6), "k+1 threshold mode should reduce to the baseline gradient when no selected expert falls inside its STE window"

    for loss_name in LOSS_FLAG_FIELDS:
        aux_topk, grad_loss_topk = _violation_grad(loss_name, "topk")
        aux_plus_one, grad_loss_plus_one = _violation_grad(loss_name, "topk_plus_one")
        assert torch.allclose(aux_topk, aux_plus_one), f"{loss_name} forward value changed across RectIndicatorSTE threshold modes"
        assert grad_loss_topk[0].abs() > 1e-4, f"{loss_name} topk-threshold STE should update the top expert in this setup"
        assert grad_loss_topk[1].abs() > 1e-4, f"{loss_name} topk-threshold STE should propagate through the selected-side boundary in this setup"
        assert grad_loss_topk[2].abs() < 1e-7 and grad_loss_topk[3].abs() < 1e-7, f"{loss_name} topk-threshold STE should not perturb out-of-window experts"
        assert torch.allclose(grad_loss_plus_one, torch.zeros_like(grad_loss_plus_one), atol=1e-7, rtol=1e-7), f"{loss_name} k+1-threshold STE should have no auxiliary correction when only the excluded boundary expert lies in the window"

    routed_all_no_ste, grad_all_no_ste = _all_selected_grad(0.0)
    routed_all_ste, grad_all_ste = _all_selected_grad(0.5)
    assert torch.allclose(routed_all_no_ste, routed_all_ste)
    assert torch.allclose(grad_all_no_ste, grad_all_ste, atol=1e-6, rtol=1e-6), "k+1 threshold mode should become a no-op when top_k == num_experts"

    print("topk threshold:", threshold_topk)
    print("topk+1 threshold:", threshold_plus_one)
    print("Baseline router grad:", grad_no_ste)
    print("topk router grad:", grad_topk)
    print("topk+1 router grad:", grad_plus_one)
    print("RectIndicatorSTE threshold mode checks passed.")


if __name__ == "__main__":
    main()
