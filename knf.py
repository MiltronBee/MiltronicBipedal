# knf.py
import numpy as np
import torch
from mpmath import log, factorial

# Golden Ratio, the new harmonic target
PHI = (1 + 5**0.5) / 2

def compute_knf(n, f):
    """Computes the harmonic informational weight metric k(n, f)."""
    if f <= 1:
        return float('inf')
    return float(log(factorial(n)) / log(f))

# This function is no longer used by the policy but is kept for reference.
# The gating logic is now handled by modifying the log_std directly.
def w_lambda_gate_torch(action_logits):
    """
    Applies the volitional collapse gate for discrete actions.
    (This is now deprecated in favor of the continuous control method).
    """
    action_probs = torch.softmax(action_logits, dim=-1)
    num_actions = action_probs.shape[-1]
    num_to_mask = num_actions // 2
    
    _, mask_indices = torch.topk(action_probs, num_to_mask, largest=False, dim=-1)
    
    mask = torch.full_like(action_logits, 0)
    mask.scatter_(-1, mask_indices, -float('inf'))
    
    gated_logits = action_logits + mask
    return gated_logits