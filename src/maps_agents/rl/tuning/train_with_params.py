"""
Training function that accepts hyperparameters from Optuna.
"""

from typing import Optional, Tuple
import os
from optuna.trial import Trial
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import EvalCallback

from maps_agents.rl.train_agent import make_env, get_layouts_for_variant
from maps_agents.rl import HierarchicalMultiDiscretePolicy, SimpleHierarchicalPolicy


class TrialEvalCallback(EvalCallback):
    """
    Callback for Optuna that reports intermediate values and prunes trials.
    """
    def __init__(
        self,
        eval_env,
        trial: Trial,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        deterministic: bool = False,
        verbose: int = 0,
    ):
        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose,
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        """
        Called on every environment step.
        Reports intermediate values to Optuna and handles pruning.
        """
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Trigger evaluation
            super()._on_step()

            # Report mean reward to Optuna
            if len(self.evaluations_results) > 0:
                mean_reward = float(np.mean(self.evaluations_results[-1]))
                self.trial.report(mean_reward, self.eval_idx)
                self.eval_idx += 1

                # Check if trial should be pruned
                if self.trial.should_prune():
                    self.is_pruned = True
                    return False  # Stop training

        return True
        

def train_with_hyperparams(
    agent_type: str,
    host: str,
    port: str,
    difficulty: str,
    mode: str,
    training_layouts: str,
    total_timesteps: int,
    n_envs: int,
    save_path: str,
    # Hyperparameters
    learning_rate: float,
    n_steps: int,
    batch_size: int,
    n_epochs: int,
    gamma: float,
    gae_lambda: float,
    clip_range: float,
    ent_coef: float,
    # Optional Optuna trial
    trial: Optional[Trial] = None,
    # Other parameters
    max_grad_norm: float = 0.5,
    vf_coef: float = 0.5,
    target_kl: Optional[float] = None,
    clip_range_vf: Optional[float] = None,
    n_eval_episodes: int = 10,
) -> Tuple[PPO, EvalCallback]:
    """
    Train PPO agent with specified hyperparameters.

    Returns:
        Tuple of (trained_model, eval_callback)
    """
    # Create save directory
    os.makedirs(save_path, exist_ok=True)

    # Set random seed
    set_random_seed(42)

    # Get layouts for variant
    train_layouts, test_layouts = get_layouts_for_variant(training_layouts)

    # Select policy
    policy_class = SimpleHierarchicalPolicy if mode == "simple" else HierarchicalMultiDiscretePolicy

    # Create training environments
    if n_envs == 1:
        env = DummyVecEnv([make_env(host, port, difficulty, mode, seed=42, layouts=train_layouts, env_idx=i) for i in range(len(train_layouts))])
        env = VecNormalize(env, norm_obs=False, norm_reward=True)
    else:
        env = SubprocVecEnv([
            make_env(host, port, difficulty, mode, seed=(42 + i * 100), layouts=train_layouts, env_idx=i)
            for i in range(n_envs)
        ])
        env = VecNormalize(env, norm_obs=False, norm_reward=True)

    # Create evaluation environment
    eval_env = DummyVecEnv([
        make_env(host, port, difficulty, mode, seed=100, eval_mode=True, layouts=test_layouts, env_idx=i) for i in range(len(test_layouts))
    ])
    eval_env = VecNormalize(eval_env, norm_obs=False, norm_reward=False)

    # Create callback with Optuna trial for pruning
    if trial is not None:
        eval_callback = TrialEvalCallback(
            eval_env=eval_env,
            trial=trial,
            n_eval_episodes=n_eval_episodes,
            eval_freq=max(25000 // n_envs, 1),
            deterministic=False,
            verbose=0
        )
    else:
        eval_callback = EvalCallback(
            eval_env=eval_env,
            best_model_save_path=f"{save_path}/best_model",
            log_path=f"{save_path}/logs",
            eval_freq=max(10000 // n_envs, 1),
            deterministic=False,
            n_eval_episodes=n_eval_episodes,
        )

    # Initialize PPO model with hyperparameters
    model = PPO(
        policy_class,
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        clip_range_vf=clip_range_vf,
        ent_coef=ent_coef,
        max_grad_norm=max_grad_norm,
        vf_coef=vf_coef,
        target_kl=target_kl,
        verbose=0,  # Quiet during tuning
        tensorboard_log=f"{save_path}/tensorboard",
        policy_kwargs={"difficulty": difficulty}
    )

    # Train
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback
    )

    # Save final model
    model.save(f"{save_path}/final_model")

    return model, eval_callback
