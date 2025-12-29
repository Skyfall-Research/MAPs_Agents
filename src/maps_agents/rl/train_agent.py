"""
Stable-Baselines3 training script for Theme Park Tycoon environment.
Uses hierarchical policy for efficient learning with sparse action spaces.

Supports two modes:
- simple: Uses SimpleParkActionSpace (5 actions) and MapsSimpleGymObservationSpace (vectors only)
- full: Uses ParkActionSpace (11 actions) and full observation space (grid + vectors)
"""

import os
import sys
import re
import argparse
import numpy as np
from stable_baselines3 import PPO, A2C, DQN, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from typing import Callable

# Add the parent directory to the path to import the environment
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from map_py.mini_amusement_park import MiniAmusementPark
from map_py.shared_constants import MAP_CONFIG
from maps_agents.rl import HierarchicalMultiDiscretePolicy, SimpleHierarchicalPolicy

TRAIN_LAYOUTS = MAP_CONFIG['train_layouts']
TEST_LAYOUTS = MAP_CONFIG['test_layouts']


def get_layouts_for_variant(variant: str):
    """
    Returns (train_layouts, test_layouts) based on the training variant.

    Args:
        variant: One of "all", "ribs", "the_islands", "zig_zag"

    Returns:
        Tuple of (list of training layouts, list of testing layouts)
    """
    if variant == "all":
        return TRAIN_LAYOUTS, TEST_LAYOUTS
    elif variant in TEST_LAYOUTS:
        # Layout-specific: train and test on the same single layout
        return [variant], [variant]
    else:
        raise ValueError(f"Unknown training variant: {variant}")


class ActionDistributionCallback(BaseCallback):
    """
    Callback to log action type distribution during training.
    Helps identify if the model is biased toward certain actions.
    """
    def __init__(self, log_freq: int = 1000, mode: str = "full", verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.mode = mode
        self.action_counts = {}
        self.invalid_count = {'valid': 0, 'invalid': 0}
        # Track validity per action type
        self.action_validity = {}  # {action_type: {'valid': count, 'invalid': count}}
        self.last_action_type = None
        self.error_type_counts = {}

        # Action names based on mode
        if mode == "simple":
            self.action_names = ["place", "move", "remove", "modify", "wait"]
        else:
            self.action_names = ["place", "move", "remove", "modify", "set_research",
                                 "survey_guests", "add_path", "remove_path", "add_water",
                                 "remove_water", "wait"]

    def _on_step(self) -> bool:
        # Get the action from the last step
        if len(self.locals.get('actions', [])) > 0:
            action = self.locals['actions'][0]

            # Track action type (dimension 0)
            action_type = int(action[0])
            self.action_counts[action_type] = self.action_counts.get(action_type, 0) + 1
            self.last_action_type = action_type

            # Initialize validity tracking for this action type if not present
            if action_type not in self.action_validity:
                self.action_validity[action_type] = {'valid': 0, 'invalid': 0}

        # Suppose your env sets info["invalid_action"] = True when it rejects/auto-corrects
        if self.locals.get('infos', None) is not None and len(self.locals['infos']) > 0:
            infos = self.locals['infos']
            info = infos[0]   # first env in VecEnv
            if 'error' in info:
                self.invalid_count['invalid'] += 1
                if self.last_action_type == 5:
                    error_message = info['error']['message']
                    # error_message = re.sub(r'\d+', 'x', error_message)
                    self.error_type_counts[error_message] = self.error_type_counts.get(error_message, 0) + 1
                # Track per action type
                if self.last_action_type is not None:
                    self.action_validity[self.last_action_type]['invalid'] += 1
            else:
                self.invalid_count['valid'] += 1
                # Track per action type
                if self.last_action_type is not None:
                    self.action_validity[self.last_action_type]['valid'] += 1

        # Log every log_freq steps
        if self.num_timesteps % self.log_freq == 0:
            print(f"\n=== Action Distribution at step {self.num_timesteps} ===")
            total_actions = sum(self.action_counts.values())
            if total_actions > 0:
                print("Action types:")
                for action_idx in sorted(self.action_counts.keys()):
                    count = self.action_counts[action_idx]
                    pct = 100.0 * count / total_actions
                    action_name = self.action_names[action_idx] if action_idx < len(self.action_names) else f"unknown_{action_idx}"
                    print(f"  {action_name}: {count} ({pct:.1f}%)")

            print(f"\n=== Action Validity at step {self.num_timesteps} ===")
            total_valid = self.invalid_count['valid']
            total_invalid = self.invalid_count['invalid']
            total = total_valid + total_invalid

            if total > 0:
                print(f"Overall: {total_valid} valid, {total_invalid} invalid ({100.0 * total_valid / total:.1f}% valid)")

                print("\nValidity rate per action type:")
                for action_idx in sorted(self.action_validity.keys()):
                    valid = self.action_validity[action_idx]['valid']
                    invalid = self.action_validity[action_idx]['invalid']
                    action_total = valid + invalid
                    if action_total > 0:
                        validity_rate = 100.0 * valid / action_total
                        action_name = self.action_names[action_idx] if action_idx < len(self.action_names) else f"unknown_{action_idx}"
                        print(f"  {action_name}: {valid}/{action_total} valid ({validity_rate:.1f}%)")

            print(f"\n=== Error Type Counts at step {self.num_timesteps} ===")
            for error_type, count in self.error_type_counts.items():
                print(f"  {error_type}: {count}")

            print("=" * 50 + "\n")

        return True

def make_env(host: str, port: str, difficulty: str = "easy", mode: str = "full",
             seed: int = 0, eval_mode: bool = False, layouts: list = None,
             env_idx: int = 0) -> Callable:
    """
    Create a function that returns the environment with hierarchical action sampling.

    Args:
        mode: "simple" for simplified action/obs spaces, "full" for complete spaces
        layouts: List of layouts to use. If multiple layouts, will be assigned round-robin by env_idx
        env_idx: Index of the environment (for parallel envs), used for layout assignment
    """
    if layouts is None or len(layouts) == 0:
        layouts = ["ribs"]  # fallback default

    # Assign layout using round-robin if multiple layouts provided
    selected_layout = layouts[env_idx % len(layouts)]

    def _init():
        env = MiniAmusementPark(
            host=host,
            port=port,
            difficulty=difficulty,
            observation_type="gym_simple" if mode == "simple" else "gym",  # Use simple or full gym format
            negative_reward_on_invalid_action=False if eval_mode else True,
            layout=selected_layout,  # Use selected layout instead of hardcoded "ribs"
            new_seed_on_reset=True if (not eval_mode) else False,
        )
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init

def train_agent(difficulty: str = "easy",
                mode: str = "full",
                total_timesteps: int = 100000,
                n_envs: int = 1,
                save_path: str = "./src/maps_agents/rl/trained_models",
                training_layouts: str = "all",
                host: str = "localhost",
                port: str = "3000"):
    """
    Train an RL agent on the Theme Park Tycoon environment.

    Args:
        mode: "simple" for simplified spaces (5 actions, vectors only),
              "full" for complete spaces (11 actions, grid + vectors)
        training_layouts: One of "all", "ribs", "the_islands", "zig_zag"
    """

    # Set random seed for reproducibility
    set_random_seed(42)

    # Get layouts based on training variant
    train_layouts, test_layouts = get_layouts_for_variant(training_layouts)
    print(f"Training layouts: {training_layouts}")
    print(f"Testing layouts: {test_layouts}")

    # Create save directory with variant-specific subdirectory
    # print(mode, difficulty, training_layouts)
    # print(save_path, f"/{mode}_{difficulty}_{training_layouts}")
    save_path = os.path.join(save_path, f"{mode}_{difficulty}_{training_layouts}")
    print("-", save_path)
    os.makedirs(save_path, exist_ok=True)

    # Select policy based on mode
    policy_class = SimpleHierarchicalPolicy if mode == "simple" else HierarchicalMultiDiscretePolicy

    # Create vectorized environment
    if n_envs == 1:
        env = DummyVecEnv([make_env(host, port, difficulty, mode, seed=42, layouts=train_layouts, env_idx=0)])
        env = VecNormalize(env, norm_obs=False, norm_reward=True)
    else:
        env = DummyVecEnv([make_env(host, port, difficulty, mode, seed=42 + i, layouts=train_layouts, env_idx=i)
                           for i in range(n_envs)])
        env = VecNormalize(env, norm_obs=False, norm_reward=True)

    # Create evaluation environment (use first test layout for periodic eval during training)
    eval_env = DummyVecEnv([make_env(host, port, difficulty, mode, seed=123, eval_mode=True,
                                     layouts=test_layouts, env_idx=0)])
    eval_env = VecNormalize(eval_env, norm_obs=False, norm_reward=False)

    # Create callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{save_path}/best_model",
        log_path=f"{save_path}/logs",
        eval_freq=max(50000 // n_envs, 1),
        deterministic=False,
        n_eval_episodes=5,
        render=False
    )

    # Initialize the agent
    print(f"Training with {mode} mode using {policy_class.__name__}")
    model = PPO(
        policy_class,  # Custom policy with hierarchical sampling
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=5000,
        batch_size=500,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        # target_kl=0.02,
        tensorboard_log=f"{save_path}/tensorboard_logs",
        policy_kwargs={
            "difficulty": difficulty,
        }
    )

    # Train the agent
    print(f"Starting training with PPO agent for {total_timesteps} timesteps")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback] #, checkpoint_callback, action_dist_callback]
    )

    # Save the final model
    final_model_path = os.path.join(save_path, "final_model.zip")
    model.save(final_model_path)
    print(f"Training completed. Model saved to {final_model_path}")

    # Test the trained model on all test layouts
    test_model(model, host, port, difficulty, mode, test_layouts=test_layouts)

    return final_model_path

def test_model(model, host: str, port: str, difficulty: str = "easy", mode: str = "full",
               n_episodes: int = 5, test_layouts: list = None):
    """
    Test a trained model on the environment across multiple layouts.

    Args:
        model: The trained model to test
        host: Host address
        port: Port number
        difficulty: Game difficulty level
        mode: "simple" or "full"
        n_episodes: Number of episodes to run per layout
        test_layouts: List of layouts to test on. If None, uses TEST_LAYOUTS
    """
    if test_layouts is None:
        test_layouts = TEST_LAYOUTS

    print(f"Testing model on {len(test_layouts)} layout(s) for {n_episodes} episodes each")

    # Test on each layout separately
    all_results = {}
    for layout in test_layouts:
        print(f"\n{'='*60}")
        print(f"Testing on layout: {layout}")
        print(f"{'='*60}")

        env = DummyVecEnv([make_env(host, port, difficulty, mode, seed=200, layouts=[layout], env_idx=0)])

        total_rewards = []
        episode_lengths = []

        for episode in range(n_episodes):
            obs = env.reset()
            episode_reward = 0
            episode_length = 0

            while True:
                action, _states = model.predict(obs, deterministic=True)
                obs, rewards, dones, infos = env.step(action)
                episode_reward += rewards[0]
                episode_length += 1

                if dones[0]:
                    break

            total_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            print(f"  Episode {episode + 1}: Reward = {episode_reward:.2f}, Length = {episode_length}")

        avg_reward = np.mean(total_rewards)
        avg_length = np.mean(episode_lengths)
        all_results[layout] = {"avg_reward": avg_reward, "avg_length": avg_length}
        print(f"  {layout} - Average reward: {avg_reward:.2f}, Average length: {avg_length:.2f}")

    # Print summary
    print(f"\n{'='*60}")
    print("Test Summary Across All Layouts:")
    print(f"{'='*60}")
    for layout, results in all_results.items():
        print(f"{layout:20s} - Avg Reward: {results['avg_reward']:8.2f}, Avg Length: {results['avg_length']:6.2f}")

def main():
    """
    Main function to run the training script.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train RL agent on Theme Park Tycoon")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["simple", "full"],
        default="full",
        help="Training mode: 'simple' for simplified action/obs spaces (5 actions, vectors only), 'full' for complete spaces (11 actions, grid + vectors)"
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        choices=["easy", "medium", "hard"],
        default="easy",
        help="Game difficulty level"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=2500000,
        help="Total timesteps for training"
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=1,
        help="Number of parallel environments"
    )
    parser.add_argument(
        "--training-layouts",
        type=str,
        choices=["all", "ribs", "the_islands", "zig_zag"],
        default="all",
        help="Training variant: 'all' trains on all training layouts and tests on all test layouts, "
             "layout-specific options train and test on that single layout only"
    )
    args = parser.parse_args()

    # Configuration
    config = {
        "host": "localhost",
        "port": "3000",
        "difficulty": args.difficulty,
        "mode": args.mode,
        "total_timesteps": args.timesteps,
        "n_envs": args.n_envs,
        "save_path": "./src/maps_agents/rl/trained_models",
        "training_layouts": args.training_layouts
    }

    print(f"\n{'='*60}")
    print(f"Training Configuration:")
    print(f"  Mode: {config['mode']}")
    print(f"  Difficulty: {config['difficulty']}")
    print(f"  Training variant: {config['training_layouts']}")
    print(f"  Total timesteps: {config['total_timesteps']:,}")
    print(f"  Parallel environments: {config['n_envs']}")
    print(f"{'='*60}\n")

    # Check if game server is running
    try:
        import requests
        response = requests.get(f"http://{config['host']}:{config['port']}/health", timeout=5)
        if response.status_code != 200:
            print("Game server might not be running properly")
    except:
        print("Could not connect to game server. Please ensure the server is running.")
        print("You can start the server using: python launch_game.py")
        return
    
    # Train the agent
    model = train_agent(
        host=config["host"],
        port=config["port"],
        difficulty=config["difficulty"],
        mode=config["mode"],
        total_timesteps=config["total_timesteps"],
        n_envs=config["n_envs"],
        save_path=config["save_path"],
        training_layouts=config["training_layouts"]
    )
    
    print("Training script completed successfully!")

if __name__ == "__main__":
    main()
