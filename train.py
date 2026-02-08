"""
train.py - PPO Training Script for Doom Agent

Trains a neural network to play Doom using Proximal Policy Optimization.
Supports:
- Multiple scenarios (basic, defend, corridor)
- Parallel environments for faster training
- Loading/saving models for incremental training
- Periodic evaluation to save the best model

Example usage:
    python train.py --scenario basic --steps 100000
    python train.py --scenario corridor --load models/basic --steps 500000 --num-envs 6
"""

import argparse
import os
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

from doom_env import DoomEnv


# Map simple names to scenario files
SCENARIOS = {
    "basic": "basic.cfg",
    "defend": "defend_the_center.cfg",
    "corridor": "deadly_corridor.cfg",
    "health": "health_gathering.cfg",
}


class RewardLoggingCallback(BaseCallback):
    """Prints rewards during training so you can see kills/deaths happening."""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        reward = self.locals.get("rewards", [0])[0]
        if reward != 0:
            sign = "+" if reward > 0 else ""
            print(f"  Reward: {sign}{reward:.1f}", flush=True)
        return True


def create_env(scenario_name, visible=False, n_envs=1):
    """
    Create the training environment(s).
    
    Args:
        scenario_name: Which scenario to use
        visible: Show game window (only works with n_envs=1)
        n_envs: Number of parallel environments
    """
    scenario_file = SCENARIOS.get(scenario_name, scenario_name)
    
    def make_env():
        env = DoomEnv(scenario_path=scenario_file, visible=visible)
        env = Monitor(env)  # Wraps env to track episode stats
        return env
    
    if n_envs == 1:
        return DummyVecEnv([make_env])
    else:
        # SubprocVecEnv runs each env in its own process
        return SubprocVecEnv([make_env for _ in range(n_envs)])


def main():
    parser = argparse.ArgumentParser(description="Train Doom RL Agent")
    parser.add_argument("--scenario", type=str, required=True,
                        help="Scenario: basic, defend, corridor, or path to .cfg")
    parser.add_argument("--steps", type=int, default=100000,
                        help="Total training timesteps")
    parser.add_argument("--load", type=str, default=None,
                        help="Path to model to continue training from")
    parser.add_argument("--save", type=str, default="models/doom_agent",
                        help="Where to save the trained model")
    parser.add_argument("--render", action="store_true",
                        help="Show game window during training")
    parser.add_argument("--num-envs", type=int, default=1,
                        help="Parallel environments (more = faster, use 6-8)")
    args = parser.parse_args()

    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    print(f"\n{'='*50}")
    print(f"DOOM RL TRAINING")
    print(f"{'='*50}")
    print(f"Scenario: {args.scenario}")
    print(f"Timesteps: {args.steps}")
    print(f"Parallel envs: {args.num_envs}")
    print(f"Load from: {args.load or 'New model'}")
    print(f"Save to: {args.save}")
    print(f"{'='*50}\n")

    # Can't render multiple environments at once
    if args.num_envs > 1 and args.render:
        print("Warning: --render disabled for parallel training")
        args.render = False

    env = create_env(args.scenario, visible=args.render, n_envs=args.num_envs)
    eval_env = create_env(args.scenario, visible=False, n_envs=1)

    # Load existing model or create new one
    if args.load and os.path.exists(args.load + ".zip"):
        print(f"Loading model from {args.load}...")
        model = PPO.load(args.load, env=env)
        # Bump up exploration when continuing training
        model.ent_coef = 0.1
        model.learning_rate = 0.0001
    else:
        print("Creating new PPO model...")
        model = PPO(
            policy="CnnPolicy",  # CNN to process game images
            env=env,
            verbose=1,
            learning_rate=0.00025,
            n_steps=2048,        # Steps per update
            batch_size=64,
            ent_coef=0.01,       # Entropy bonus for exploration
        )

    # Callbacks for logging and saving best model
    reward_callback = RewardLoggingCallback()
    
    best_model_dir = os.path.dirname(args.save) or "models"
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=best_model_dir + "/best/",
        eval_freq=5000,
        n_eval_episodes=10,
        deterministic=True,
    )

    # Train!
    print("\nStarting training...")
    model.learn(
        total_timesteps=args.steps,
        callback=[reward_callback, eval_callback],
    )

    model.save(args.save)
    print(f"\nModel saved to {args.save}.zip")

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
