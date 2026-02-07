"""
train.py - Training Script for Doom RL Agent

This script trains a PPO (Proximal Policy Optimization) agent on a specified
VizDoom scenario. It demonstrates:
1. Environment setup with Gymnasium wrappers
2. Model initialization/loading for transfer learning
3. Callback system for logging and evaluation
4. Model saving for later use

Usage:
    python train.py --scenario basic --steps 100000
    python train.py --scenario defend --load models/doom_agent --steps 200000

Author: Student Project
"""

import argparse
import os
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

from doom_env import DoomEnv


# =============================================================================
# SCENARIO CONFIGURATIONS
# =============================================================================
# Maps friendly names to scenario files for easy command-line usage

SCENARIOS = {
    "basic": "basic.cfg",
    "defend": "defend_the_center.cfg",
    "corridor": "deadly_corridor.cfg",
    "health": "health_gathering.cfg",
}


class RewardLoggingCallback(BaseCallback):
    """
    Prints each non-zero reward so you can see kills, deaths, etc.
    The PPO output already shows ep_rew_mean for average tracking.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        reward = self.locals.get("rewards", [0])[0]
        
        if reward != 0:
            sign = "+" if reward > 0 else ""
            print(f"  Reward: {sign}{reward:.1f}", flush=True)
        
        return True


def create_env(scenario_name, visible=False):
    """Create a wrapped DoomEnv for training."""
    scenario_file = SCENARIOS.get(scenario_name, scenario_name)
    
    def make_env():
        env = DoomEnv(scenario_path=scenario_file, visible=visible)
        env = Monitor(env)  # Wraps env to log episode stats
        return env
    
    return DummyVecEnv([make_env])


def main():
    parser = argparse.ArgumentParser(description="Train Doom RL Agent")
    parser.add_argument("--scenario", type=str, required=True,
                        help="Scenario name (basic, defend, corridor) or path to .cfg")
    parser.add_argument("--steps", type=int, default=100000,
                        help="Total training timesteps")
    parser.add_argument("--load", type=str, default=None,
                        help="Path to existing model to continue training")
    parser.add_argument("--save", type=str, default="models/doom_agent",
                        help="Path to save the trained model")
    parser.add_argument("--render", action="store_true",
                        help="Show game window during training")
    args = parser.parse_args()

    # Create directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    print(f"\n{'='*50}")
    print(f"DOOM RL TRAINING")
    print(f"{'='*50}")
    print(f"Scenario: {args.scenario}")
    print(f"Timesteps: {args.steps}")
    print(f"Load from: {args.load or 'New model'}")
    print(f"Save to: {args.save}")
    print(f"{'='*50}\n")

    # Create environment
    env = create_env(args.scenario, visible=args.render)
    
    # Create evaluation environment (for EvalCallback)
    eval_env = create_env(args.scenario, visible=False)

    # Initialize or load model
    if args.load and os.path.exists(args.load + ".zip"):
        print(f"Loading existing model from {args.load}...")
        model = PPO.load(args.load, env=env)
        # FORCE moderate entropy on loaded models to encourage exploration
        model.ent_coef = 0.1
        model.learning_rate = 0.0001
        print(f"Entropy coefficient set to {model.ent_coef} for exploration")
    else:
        print("Creating new PPO model...")
        model = PPO(
            policy="CnnPolicy",
            env=env,
            verbose=1,
            learning_rate=0.00003,  # Lower for stable overnight training
            n_steps=2048,
            batch_size=64,
            ent_coef=0.1,  # Moderate entropy for balanced exploration
        )

    # Setup callbacks
    reward_callback = RewardLoggingCallback()
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="models/best/",
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

    # Save final model
    model.save(args.save)
    print(f"\nModel saved to {args.save}.zip")

    # Cleanup
    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
