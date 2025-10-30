"""
Manual analysis of routing logic to identify potential bugs.
This doesn't require torch - just traces through the logic.
"""

import numpy as np

def analyze_threshold_calculation():
    """Analyze the kthvalue threshold calculation."""
    print("="*80)
    print("ANALYZING THRESHOLD CALCULATION")
    print("="*80)

    num_experts = 8
    k = 2

    print(f"\nConfiguration: num_experts={num_experts}, k={k} (want top-{k})")
    print(f"\nCurrent implementation:")
    print(f"  kth_smallest_idx = num_experts - k = {num_experts} - {k} = {num_experts - k}")
    print(f"  threshold = kthvalue(exp_z, k={num_experts - k})")
    print(f"  This gets the {num_experts - k}-th smallest value")
    print(f"  Which is the {num_experts - (num_experts - k) + 1}-th largest = {k + 1}-th largest")

    # Test with concrete values
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    print(f"\nTest values (sorted ascending): {values}")

    kth_smallest_idx = num_experts - k  # 6
    threshold = values[kth_smallest_idx - 1]  # Python 0-indexed, so -1
    print(f"\nThreshold (kthvalue k={kth_smallest_idx}): {threshold}")
    print(f"This is value #{kth_smallest_idx} in sorted order = {threshold}")

    kept_values = np.maximum(values - threshold, 0)
    print(f"\nAfter relu(values - {threshold}): {kept_values}")
    print(f"Non-zero count: {np.count_nonzero(kept_values)}")
    print(f"Non-zero values: {values[kept_values > 0]}")

    if np.count_nonzero(kept_values) != k:
        print(f"\n⚠️  ERROR: Expected {k} non-zero values, got {np.count_nonzero(kept_values)}")
        print("This indicates a bug in the threshold calculation!")

        # What should it be?
        correct_kth_smallest_idx = num_experts - k + 1
        print(f"\n💡 POTENTIAL FIX:")
        print(f"  kth_smallest_idx should be: num_experts - k + 1 = {correct_kth_smallest_idx}")
        threshold_correct = values[correct_kth_smallest_idx - 1]
        print(f"  threshold = kthvalue(exp_z, k={correct_kth_smallest_idx}) = {threshold_correct}")
        kept_values_correct = np.maximum(values - threshold_correct, 0)
        print(f"  After relu: {kept_values_correct}")
        print(f"  Non-zero count: {np.count_nonzero(kept_values_correct)}")
    else:
        print(f"\n✓ Correct: Got exactly {k} non-zero values")

    # Edge case: what if values are equal?
    print("\n" + "-"*80)
    print("Edge case: Values with ties")
    print("-"*80)
    values_ties = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 6.0, 6.0])
    print(f"Test values with ties: {values_ties}")
    threshold_ties = values_ties[kth_smallest_idx - 1]
    print(f"Threshold (kthvalue k={kth_smallest_idx}): {threshold_ties}")
    kept_values_ties = np.maximum(values_ties - threshold_ties, 0)
    print(f"After relu: {kept_values_ties}")
    print(f"Non-zero count: {np.count_nonzero(kept_values_ties)}")

    if np.count_nonzero(kept_values_ties) != k:
        print(f"⚠️  With ties, got {np.count_nonzero(kept_values_ties)} non-zero instead of {k}")


def compare_routing_outputs():
    """Compare what switch vs diff routing would output."""
    print("\n" + "="*80)
    print("COMPARING SWITCH VS DIFF ROUTING OUTPUTS")
    print("="*80)

    # Simulate logits
    logits = np.array([0.5, 0.3, 0.1, 0.05, 1.5, 0.8, 1.0, 0.3])
    k = 2
    num_experts = len(logits)

    print(f"\nInput logits: {logits}")
    print(f"Top-k: {k}")

    # Switch routing
    print("\n--- SWITCH ROUTING ---")
    # softmax
    exp_logits = np.exp(logits - np.max(logits))  # numerically stable
    probs = exp_logits / exp_logits.sum()
    print(f"Probabilities (softmax): {probs}")

    # top-k
    topk_indices = np.argsort(probs)[-k:][::-1]  # descending
    topk_probs = probs[topk_indices]
    print(f"Top-{k} indices: {topk_indices}")
    print(f"Top-{k} probs: {topk_probs}")

    # renormalize
    switch_gate = topk_probs / topk_probs.sum()
    print(f"Renormalized gate weights: {switch_gate}")
    print(f"Sum: {switch_gate.sum()}")

    # Diff routing
    print("\n--- DIFFERENTIAL ROUTING ---")
    # exp(logits - max)
    z_max = np.max(logits)
    exp_z = np.exp(logits - z_max)
    print(f"exp(logits - max): {exp_z}")

    # threshold
    kth_smallest_idx = num_experts - k
    threshold = np.sort(exp_z)[kth_smallest_idx - 1]  # -1 for 0-indexing
    print(f"Threshold (kth_smallest_idx={kth_smallest_idx}): {threshold}")

    # relu and normalize
    topk_exp_z = np.maximum(exp_z - threshold, 0)
    print(f"After relu(exp_z - threshold): {topk_exp_z}")
    print(f"Non-zero count: {np.count_nonzero(topk_exp_z)}")

    topk_weights = topk_exp_z / (topk_exp_z.sum() + 1e-8)
    print(f"Normalized weights (all experts): {topk_weights}")

    # top-k indices
    diff_topk_indices = np.argsort(topk_weights)[-k:][::-1]
    diff_gate = topk_weights[diff_topk_indices]
    print(f"Top-{k} indices: {diff_topk_indices}")
    print(f"Gate weights (gathered): {diff_gate}")
    print(f"Sum: {diff_gate.sum()}")

    # Compare
    print("\n--- COMPARISON ---")
    print(f"Same indices? {np.array_equal(topk_indices, diff_topk_indices)}")
    print(f"Switch gate: {switch_gate}")
    print(f"Diff gate: {diff_gate}")
    print(f"Absolute difference: {np.abs(switch_gate - diff_gate)}")
    print(f"Mean abs diff: {np.abs(switch_gate - diff_gate).mean():.6f}")

    if np.abs(switch_gate - diff_gate).mean() < 0.01:
        print("\n⚠️  WARNING: Gate weights are very similar! (< 1% difference)")
        print("This might explain why results are similar.")
    else:
        print(f"\n✓ Gate weights are different (mean diff: {np.abs(switch_gate - diff_gate).mean():.2%})")


def check_implementation_details():
    """Check specific implementation details that might cause bugs."""
    print("\n" + "="*80)
    print("CHECKING IMPLEMENTATION DETAILS")
    print("="*80)

    print("\n1. Return values from routing functions:")
    print("-" * 40)
    print("switch_topk returns: (topk_idx, probs, gate)")
    print("  - topk_idx: shape [..., k]")
    print("  - probs: shape [..., num_experts] (full softmax)")
    print("  - gate: shape [..., k] (top-k weights, already gathered)")
    print()
    print("diff_routing returns: (topk_idx, probs, topk_weights)")
    print("  - topk_idx: shape [..., k]")
    print("  - probs: shape [..., num_experts] (full softmax, not used in routing)")
    print("  - topk_weights: shape [..., num_experts] (all weights, mostly zeros)")
    print()
    print("⚠️  NOTE: For diff routing, topk_weights has different shape than switch gate!")
    print("The MoE forward does: gate = torch.gather(gate, dim=-1, index=topk_idx)")
    print("This extracts the k values, making shapes consistent.")

    print("\n2. Gradient flow:")
    print("-" * 40)
    print("Switch: Uses discrete torch.topk on probabilities")
    print("  - torch.topk is not differentiable w.r.t. which indices are selected")
    print("  - Gradients flow through the selected probabilities")
    print()
    print("Diff: Uses relu(exp_z - threshold) for soft selection")
    print("  - relu is differentiable")
    print("  - But then uses torch.topk to get indices (also not differentiable)")
    print("  - So both methods have the same non-differentiable index selection!")
    print()
    print("⚠️  POTENTIAL ISSUE: Both methods use torch.topk for index selection,")
    print("    making them similarly non-differentiable for expert selection.")
    print("    The 'differential' aspect only affects the weight computation,")
    print("    not the expert selection process.")

    print("\n3. Weight computation:")
    print("-" * 40)
    print("Switch: Renormalizes top-k softmax probabilities")
    print("  gate = topk_probs / topk_probs.sum()")
    print()
    print("Diff: Thresholds and normalizes exp(logits)")
    print("  topk_weights = relu(exp_z - threshold) / sum(...)")
    print("  gate = gather(topk_weights, topk_idx)")
    print()
    print("✓ These should produce different weight values")


if __name__ == "__main__":
    analyze_threshold_calculation()
    compare_routing_outputs()
    check_implementation_details()

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nKey findings:")
    print("1. Check if threshold calculation has an off-by-one error")
    print("2. Both methods use torch.topk for index selection (not differentiable)")
    print("3. Weight values should differ between methods")
    print("4. If results are identical, check if diff router is being used at all")
