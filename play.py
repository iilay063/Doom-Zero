
import argparse
import os
import time
from stable_baselines3 import PPO
from doom_env import DoomEnv
import vizdoom.scenarios as scenarios

def main():
    parser = argparse.ArgumentParser(description="Play Doom with Trained Agent")
    parser.add_argument("--model_path", type=str, default="models/ppo_doom", help="Path to the trained model")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to play")
    args = parser.parse_args()

    if not os.path.exists(args.model_path + ".zip"):
        print(f"Error: Model not found at {args.model_path}.zip")
        return

    # Locate scenario
    scenario_path = os.path.join(scenarios.__path__[0], "basic.cfg")
    if not os.path.exists(scenario_path):
        scenario_path = "basic.cfg"

    # Create environment with rendering ENABLED
    env = DoomEnv(scenario_path=scenario_path, visible=True, render_mode="human")

    # Load Model
    model = PPO.load(args.model_path)

    for ep in range(args.episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _states = model.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            time.sleep(0.05) # Slow down slightly for viewing
            
        print(f"Episode {ep+1}: Total Reward: {total_reward}")
        time.sleep(1.0)

    env.close()

if __name__ == "__main__":
    main()
