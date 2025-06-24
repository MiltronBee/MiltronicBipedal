# train.py
import gymnasium as gym
import torch
import wandb
import os

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from miltronic_elements import PPO_Miltronic, MiltronicMlpPolicy, MiltronicLoggingCallback

# --- Configuration for BipedalWalkerHardcore ---
CONFIG = {
    "total_timesteps": 3_000_000,
    "n_envs": 16,
    "policy_class": MiltronicMlpPolicy,
    "project_name": "miltronic-bipedal-hardcore",
    "env_name": "BipedalWalkerHardcore-v3",
    "seed": 2029,
    
    # --- Miltronic Hyperparameters ---
    "action_discretization_bins": 5, # Bins per action dimension for k(n,f)
    "features_dim": 64,             # Output size of the policy's MLP layers
    "harmonic_epsilon": 0.1,
    "kl_stability_threshold": 0.008,
    "kl_ema_alpha": 0.001,
    "expanded_phi_band": 1.5,
    "collapse_warmup_limit": 200000,
    "collapse_patience_limit": 50000,
    "collapse_trial_length": 20000,
    "stability_eval_warmup": 50,
    "collapsed_log_std": -5.0, # Large negative value = tiny std dev
    
    # --- PPO Hyperparameters (tuned for BipedalWalker) ---
    "n_steps": 2048,
    "batch_size": 1024,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "n_epochs": 10,
    "clip_range": 0.2,
    "ent_coef": 0.001,
    "learning_rate": 3e-4,
}

def train_miltronic():
    run_name = f"miltronic_bipedal_k_bins_{CONFIG['action_discretization_bins']}"
    
    run = wandb.init(
        project=CONFIG["project_name"],
        config=CONFIG,
        name=run_name,
        sync_tensorboard=True,
    )
    
    # Use standard vector environment creation for Box2D environments
    vec_env = make_vec_env(CONFIG["env_name"], n_envs=CONFIG["n_envs"], seed=CONFIG["seed"])
    
    # Define the MLP network architecture and pass other custom params to the policy
    policy_kwargs = dict(
        net_arch=dict(pi=[CONFIG["features_dim"], CONFIG["features_dim"]], vf=[CONFIG["features_dim"], CONFIG["features_dim"]]),
        harmonic_epsilon=CONFIG["harmonic_epsilon"],
        expanded_phi_band=CONFIG["expanded_phi_band"],
        collapsed_log_std=CONFIG["collapsed_log_std"],
    )
    
    model = PPO_Miltronic(
        policy=CONFIG["policy_class"],
        env=vec_env,
        action_discretization_bins=CONFIG["action_discretization_bins"],
        kl_stability_threshold=CONFIG["kl_stability_threshold"],
        kl_ema_alpha=CONFIG["kl_ema_alpha"],
        stability_eval_warmup=CONFIG["stability_eval_warmup"],
        verbose=1, n_steps=CONFIG["n_steps"], batch_size=CONFIG["batch_size"],
        n_epochs=CONFIG["n_epochs"], gamma=CONFIG["gamma"], gae_lambda=CONFIG["gae_lambda"],
        clip_range=CONFIG["clip_range"], ent_coef=CONFIG["ent_coef"],
        learning_rate=CONFIG["learning_rate"], tensorboard_log=f"runs/{run.id}",
        seed=CONFIG["seed"], device='cuda' if torch.cuda.is_available() else 'cpu',
        policy_kwargs=policy_kwargs
    )
    
    callback = MiltronicLoggingCallback(
        warmup_limit=CONFIG["collapse_warmup_limit"],
        patience_limit=CONFIG["collapse_patience_limit"],
        trial_length=CONFIG["collapse_trial_length"]
    )
    
    model.learn(total_timesteps=CONFIG["total_timesteps"], callback=callback, progress_bar=True)
    
    model_path = f"models/{run.name}.zip"
    model.save(model_path)
    print(f"Model saved to {model_path}")

    run.finish()
    vec_env.close()

def train_baseline_ppo():
    run_name = f"baseline_ppo_bipedal_seed_{CONFIG['seed']}"
    
    run = wandb.init(
        project=CONFIG["project_name"],
        config=CONFIG,
        name=run_name,
        sync_tensorboard=True,
    )
    
    # Use standard vector environment creation for Box2D environments
    vec_env = make_vec_env(CONFIG["env_name"], n_envs=CONFIG["n_envs"], seed=CONFIG["seed"])
    
    # Standard PPO policy network architecture
    policy_kwargs = dict(
        net_arch=dict(pi=[CONFIG["features_dim"], CONFIG["features_dim"]], vf=[CONFIG["features_dim"], CONFIG["features_dim"]])
    )
    
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1, n_steps=CONFIG["n_steps"], batch_size=CONFIG["batch_size"],
        n_epochs=CONFIG["n_epochs"], gamma=CONFIG["gamma"], gae_lambda=CONFIG["gae_lambda"],
        clip_range=CONFIG["clip_range"], ent_coef=CONFIG["ent_coef"],
        learning_rate=CONFIG["learning_rate"], tensorboard_log=f"runs/{run.id}",
        seed=CONFIG["seed"], device='cuda' if torch.cuda.is_available() else 'cpu',
        policy_kwargs=policy_kwargs
    )
    
    model.learn(total_timesteps=CONFIG["total_timesteps"], progress_bar=True)
    
    model_path = f"models/{run.name}.zip"
    model.save(model_path)
    print(f"Model saved to {model_path}")

    run.finish()
    vec_env.close()

if __name__ == '__main__':
    # freeze_support() may be needed for multiprocessing on Windows/macOS
    from multiprocessing import freeze_support
    freeze_support()
    os.makedirs("models", exist_ok=True)
    os.makedirs("runs", exist_ok=True)
    
    print("Starting Miltronic training...")
    train_miltronic()
    
    print("\nStarting baseline PPO training...")
    train_baseline_ppo()
    
    print("\nBoth training runs completed!")