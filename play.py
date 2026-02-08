"""
play.py - Demo Script for Trained Doom Agent

Loads a trained model and runs it visually so you can watch it play.
Useful for demos and evaluating how well the agent learned.

Usage:
    python play.py --model models/best/best_model --scenario basic
    python play.py --model models/best/best_model --scenario corridor --episodes 10
"""

import argparse
import time
from stable_baselines3 import PPO

from doom_env import DoomEnv


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
                        help="Scenario to play")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of episodes to watch")
    args = parser.parse_args()

    scenario_file = SCENARIOS.get(args.scenario, args.scenario)

    print(f"\n{'='*50}")
    print(f"DOOM RL DEMO")
    print(f"{'='*50}")
    print(f"Model: {args.model}")
    print(f"Scenario: {args.scenario}")
    print(f"Episodes: {args.episodes}")
    print(f"{'='*50}\n")

    # Create env with visible window
    env = DoomEnv(scenario_path=scenario_file, visible=True, render_mode="human")

    print(f"Loading model...")
    model = PPO.load(args.model)

    for episode in range(args.episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0

        print(f"\n--- Episode {episode + 1} ---")

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            time.sleep(0.02)  # Slow down for visibility

        print(f"Finished! Reward: {total_reward:.1f}, Steps: {steps}")
        time.sleep(1.0)

    env.close()
    print("\nDemo complete!")


if __name__ == "__main__":
    main()
