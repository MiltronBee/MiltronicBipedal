# miltronic_elements.py
import torch
import torch.nn as nn
import numpy as np
import wandb
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3 import PPO
from stable_baselines3.common.distributions import DiagGaussianDistribution
from stable_baselines3.common.callbacks import BaseCallback
from gym import spaces

from knf import compute_knf, PHI

# --- 1. Custom Policy for Continuous Actions (MiltronicMlpPolicy) ---
# This policy inherits from the standard ActorCriticPolicy but redefines the action distribution logic.
class MiltronicMlpPolicy(ActorCriticPolicy):
    def __init__(self, *args, harmonic_epsilon=0.1, expanded_phi_band=0.3, collapsed_log_std=-4.0, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.k_value = 0.0
        self.harmonic_epsilon = harmonic_epsilon
        self.expanded_phi_band = expanded_phi_band
        # Define the log_std value to use during collapse. A large negative value
        # corresponds to a very small standard deviation, yielding deterministic actions.
        self.collapsed_log_std = torch.tensor(collapsed_log_std, dtype=torch.float32)

        # These flags are controlled by the agent and callback
        self.is_policy_stable = False
        self.attempt_forced_collapse = False

    def _build(self, lr_schedule):
        # Build the MLP network from the parent class
        super()._build(lr_schedule)
        # Set the action distribution type for continuous actions
        self.action_dist = DiagGaussianDistribution(self.action_space.shape[0])

    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor) -> DiagGaussianDistribution:
        # The MLP outputs the mean of the Gaussian distribution
        mean_actions = self.action_net(latent_pi)
        
        # The core Miltronic gating logic remains the same
        current_epsilon = self.expanded_phi_band if self.attempt_forced_collapse else self.harmonic_epsilon
        is_harmonic = abs(self.k_value - PHI) <= current_epsilon
        should_collapse = self.training and is_harmonic and self.is_policy_stable
        
        if should_collapse:
            # REDEFINED COLLAPSE: Instead of masking logits, we override the learned
            # standard deviation with a very small, fixed value.
            log_std = torch.full_like(self.log_std, self.collapsed_log_std.item())
        else:
            # In a normal state, use the standard learned log_std
            log_std = self.log_std
            
        return self.action_dist.proba_distribution(mean_actions, log_std)

# --- 2. Modified PPO Agent (PPO_Miltronic) ---
# The agent is updated to handle the new k(n,f) calculation.
class PPO_Miltronic(PPO):
    def __init__(self, policy, env, action_discretization_bins=5, kl_stability_threshold=0.01, kl_ema_alpha=0.001, stability_eval_warmup=20, **kwargs):
        
        # Redefine 'n' and 'f' for the k calculation in a continuous context
        n_action_dims = env.action_space.shape[0]
        n_conceptual_actions = action_discretization_bins ** n_action_dims
        f_dim = kwargs.get("policy_kwargs", {}).get("net_arch", {}).get("pi", [64])[0]
        self.k = compute_knf(n_conceptual_actions, f_dim)
        
        self.stability_eval_warmup = stability_eval_warmup
        self.kl_update_counter = 0

        super().__init__(policy, env, **kwargs)
        
        self.kl_stability_threshold = kl_stability_threshold
        self.kl_ema_alpha = kl_ema_alpha
        self.kl_ema = None
        
        print(f"Miltronic Continuous Agent Initialized. n={n_conceptual_actions}, f={f_dim}, k={self.k:.4f}, φ≈{PHI:.3f}")

    def _setup_model(self) -> None:
        super()._setup_model()
        self.policy.k_value = self.k
        
    def train(self) -> None:
        super().train()
        self.kl_update_counter += 1
        approx_kl = self.logger.name_to_value.get('train/approx_kl')
        if approx_kl is not None and not np.isnan(approx_kl):
            if self.kl_ema is None: self.kl_ema = approx_kl
            else: self.kl_ema = ((1 - self.kl_ema_alpha) * self.kl_ema) + (self.kl_ema_alpha * approx_kl)
            if self.kl_update_counter < self.stability_eval_warmup:
                is_stable = False
            else:
                is_stable = self.kl_ema < self.kl_stability_threshold
            self.policy.is_policy_stable = is_stable
            self.logger.record("miltronic/kl_ema", self.kl_ema)
            self.logger.record("miltronic/kl_update_counter", self.kl_update_counter)
            self.logger.record("miltronic/is_policy_stable", float(is_stable))

# --- 3. Custom Logging Callback (MiltronicLoggingCallback) ---
# This component is highly modular and requires NO logical changes.
class MiltronicLoggingCallback(BaseCallback):
    def __init__(self, warmup_limit=500000, patience_limit=20000, trial_length=10000, verbose=0):
        super().__init__(verbose)
        self.warmup_limit = warmup_limit
        self.patience_limit = patience_limit
        self.trial_length = trial_length
        self.stable_step_counter = 0
        self.force_collapse_cooldown = 0
        self.collapse_events_rollout = 0

    def _on_rollout_start(self) -> None:
        self.collapse_events_rollout = 0

    def _on_step(self) -> bool:
        if self.num_timesteps < self.warmup_limit:
            self.model.policy.attempt_forced_collapse = False
            self.logger.record("miltronic_step/is_in_warmup", 1)
            return True
        
        self.logger.record("miltronic_step/is_in_warmup", 0)
        policy = self.model.policy

        if self.force_collapse_cooldown > 0: self.force_collapse_cooldown -= 1
        
        if policy.is_policy_stable: self.stable_step_counter += 1
        else: self.stable_step_counter = 0

        should_force = (self.stable_step_counter > self.patience_limit) and (self.force_collapse_cooldown == 0)
        policy.attempt_forced_collapse = should_force
        
        current_epsilon = policy.expanded_phi_band if should_force else policy.harmonic_epsilon
        is_harmonic_now = abs(policy.k_value - PHI) <= current_epsilon
        
        if policy.is_policy_stable and is_harmonic_now:
            self.collapse_events_rollout += 1
            if should_force:
                self.stable_step_counter = 0
                self.force_collapse_cooldown = self.trial_length
                self.logger.record("miltronic_event/forced_collapse_triggered", 1)

        self.logger.record("miltronic_step/stable_step_counter", self.stable_step_counter)
        self.logger.record("miltronic_step/is_attempting_force_collapse", float(should_force))
        return True

    def _on_rollout_end(self) -> None:
        wandb.log({"miltronic_rollout/total_collapse_events": self.collapse_events_rollout, "global_step": self.num_timesteps})