"""
Hyperparameter tuning script for PPO RL agent using Optuna.

Usage:
    python -m maps_agents.rl.tune_hyperparameters \
        --training-layouts all \
        --mode full \
        --difficulty easy \
        --n-trials 100 \
        --n-jobs 4

Features:
    - Bayesian optimization using Optuna TPE sampler
    - Median pruning to stop unpromising trials early
    - Parallel trial execution
    - SQLite database for persistence
    - TensorBoard logging
    - Multi-objective optimization (optional)
"""

import argparse
import os
from datetime import datetime
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import json

from maps_agents.rl.tuning.objective import objective


def create_study(
    study_name: str,
    storage: str,
    direction: str = "maximize",
    pruner: optuna.pruners.BasePruner = None,
    sampler: optuna.samplers.BaseSampler = None
) -> optuna.Study:
    """
    Create or load Optuna study.

    Args:
        study_name: Name of the study
        storage: SQLite database path
        direction: "maximize" or "minimize"
        pruner: Pruner for early stopping
        sampler: Sampler for hyperparameter selection

    Returns:
        Optuna Study object
    """
    if pruner is None:
        # Median pruner: stop if trial is worse than median at any step
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)

    if sampler is None:
        # Tree-structured Parzen Estimator (best for RL)
        # Note: multivariate=False used because objective has dynamic search space (constraint checks)
        sampler = TPESampler(n_startup_trials=10, multivariate=False)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction=direction,
        pruner=pruner,
        sampler=sampler,
        load_if_exists=True
    )

    return study


def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning for PPO RL agent"
    )

    # Training configuration
    parser.add_argument(
        "--training-layouts",
        type=str,
        choices=["all", "ribs", "the_islands", "zig_zag"],
        default="all",
        help="Training variant to optimize for"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["simple", "full"],
        default="full",
        help="Training mode"
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        choices=["easy", "medium", "hard"],
        default="easy",
        help="Game difficulty"
    )

    # Tuning configuration
    parser.add_argument(
        "--n-trials",
        type=int,
        default=10,
        help="Number of trials to run"
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel jobs (trials)"
    )
    parser.add_argument(
        "--n-timesteps",
        type=int,
        default=250000,
        help="Training timesteps per trial (reduced for faster tuning)"
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=25,
        help="Number of parallel environments per trial"
    )
    parser.add_argument(
        "--n-eval-episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes per trial"
    )

    # Storage
    parser.add_argument(
        "--study-name",
        type=str,
        default=None,
        help="Name of the Optuna study (default: auto-generated)"
    )
    parser.add_argument(
        "--storage",
        type=str,
        default="sqlite:///optuna_studies.db",
        help="Optuna storage (SQLite database)"
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="./tuning_runs",
        help="Directory to save trial models"
    )

    # Server
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Game server host"
    )
    parser.add_argument(
        "--port",
        type=str,
        default="3000",
        help="Game server port"
    )

    # Advanced
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Timeout in seconds for the study (optional)"
    )

    args = parser.parse_args()

    # Create study name if not provided
    if args.study_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.study_name = f"ppo_tuning_{args.training_layouts}_{args.mode}_{timestamp}"

    print(f"\n{'='*60}")
    print(f"Hyperparameter Tuning Configuration:")
    print(f"  Study Name: {args.study_name}")
    print(f"  Training Variant: {args.training_layouts}")
    print(f"  Mode: {args.mode}")
    print(f"  Difficulty: {args.difficulty}")
    print(f"  Number of Trials: {args.n_trials}")
    print(f"  Parallel Jobs: {args.n_jobs}")
    print(f"  Timesteps per Trial: {args.n_timesteps:,}")
    print(f"  Parallel Envs per Trial: {args.n_envs}")
    print(f"  Storage: {args.storage}")
    print(f"{'='*60}\n")

    # Check if game server is running
    try:
        import requests
        response = requests.get(f"http://{args.host}:{args.port}/health", timeout=5)
        if response.status_code != 200:
            print("WARNING: Game server might not be running properly")
    except:
        print("ERROR: Could not connect to game server")
        print("Please ensure the server is running: python launch_game.py")
        return

    # Create Optuna study
    study = create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="maximize"  # Maximize episode reward
    )

    # Define objective wrapper with fixed parameters
    def wrapped_objective(trial):
        return objective(
            trial=trial,
            training_layouts=args.training_layouts,
            mode=args.mode,
            difficulty=args.difficulty,
            n_eval_episodes=args.n_eval_episodes,
            n_timesteps=args.n_timesteps,
            n_envs=args.n_envs,
            host=args.host,
            port=args.port,
            save_path=args.save_path
        )

    # Run optimization
    print(f"Starting hyperparameter optimization...")
    print(f"Best results will be saved to: {args.save_path}")
    print(f"View progress: optuna-dashboard {args.storage}\n")

    study.optimize(
        wrapped_objective,
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        timeout=args.timeout,
        show_progress_bar=True
    )

    # Print results
    print(f"\n{'='*60}")
    print(f"Optimization Complete!")
    print(f"{'='*60}")
    print(f"Number of finished trials: {len(study.trials)}")
    print(f"Number of pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    print(f"Number of complete trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")

    print(f"\nBest trial:")
    trial = study.best_trial
    print(f"  Value (Mean Reward): {trial.value:.2f}")
    print(f"  Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Save best hyperparameters to JSON
    best_params_path = os.path.join(args.save_path, f"best_params_{args.study_name}.json")
    os.makedirs(args.save_path, exist_ok=True)
    with open(best_params_path, 'w') as f:
        json.dump({
            "study_name": args.study_name,
            "best_value": trial.value,
            "best_params": trial.params,
            "training_layouts": args.training_layouts,
            "mode": args.mode,
            "difficulty": args.difficulty
        }, f, indent=2)

    print(f"\nBest hyperparameters saved to: {best_params_path}")
    print(f"View study dashboard: optuna-dashboard {args.storage}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
