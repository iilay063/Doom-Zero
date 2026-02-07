"""
play.py - Watch a Trained Doom Agent Play

This script loads a trained model and runs it in a VizDoom scenario,
allowing you to observe the agent's behavior visually.

Usage:
    python play.py --model models/doom_agent --scenario basic
    python play.py --model models/doom_agent --scenario defend --episodes 10

Author: Student Project
"""

import argparse
import time
from stable_baselines3 import PPO

from doom_env import DoomEnv


# Scenario name mappings
SCENARIOS = {
    "basic": "basic.cfg",
    "defend": "defend_the_center.cfg",
    "corridor": "deadly_corridor.cfg",
    "health": "health_gathering.cfg",
}


def main():
    parser = argparse.ArgumentParser(description="Watch Doom Agent Play")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to trained model (without .zip)")
    parser.add_argument("--scenario", type=str, required=True,
                        help="Scenario name or path to .cfg")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of episodes to play")
    args = parser.parse_args()

    # Resolve scenario path
    scenario_file = SCENARIOS.get(args.scenario, args.scenario)

    print(f"\n{'='*50}")
    print(f"DOOM RL DEMO")
    print(f"{'='*50}")
    print(f"Model: {args.model}")
    print(f"Scenario: {args.scenario}")
    print(f"Episodes: {args.episodes}")
    print(f"{'='*50}\n")

    # Create environment with visible window
    env = DoomEnv(scenario_path=scenario_file, visible=True, render_mode="human")

    # Load trained model
    print(f"Loading model...")
    model = PPO.load(args.model)

    # Play loop
    for episode in range(args.episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0

        print(f"\n--- Episode {episode + 1} ---")

        while not done:
            # Get action from trained model
            action, _ = model.predict(obs, deterministic=True)
            
            # Execute action
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
            # Small delay for watchability
            time.sleep(0.02)

        print(f"Finished! Reward: {total_reward:.1f}, Steps: {steps}")
        time.sleep(1.0)  # Pause between episodes

    env.close()
    print("\nDemo complete!")


if __name__ == "__main__":
    main()
