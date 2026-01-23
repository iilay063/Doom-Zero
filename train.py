
import argparse
import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from doom_env import DoomEnv
import vizdoom.scenarios as scenarios
from stable_baselines3.common.callbacks import BaseCallback

class LoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_training_start(self) -> None:
        print("Callback initialized! Training started...", flush=True)

    def _on_step(self) -> bool:
        # Print every 100 steps (user had 10, let's do 100 to avoid complete spam but ensure visibility)
        if self.num_timesteps % 100 == 0:
            print(f"Step: {self.num_timesteps}", flush=True)
        return True

def main():
    parser = argparse.ArgumentParser(description="Train Doom RL Agent")
    parser.add_argument("--steps", type=int, default=1000, help="Total training timesteps")
    parser.add_argument("--render", action="store_true", help="Render the environment during training")
    parser.add_argument("--save_path", type=str, default="models/ppo_doom", help="Path to save the model")
    parser.add_argument("--load", action="store_true", help="Load existing model to continue training")
    args = parser.parse_args()

    # Create models directory
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    # Path to scenario - using basic.cfg from vizdoom package usually, 
    # but for robustness we might want to copy it or reference it directly.
    # For now, let's try to locate it dynamically or assume it's in the current dir if user explicitly wants that.
    # A robust way is to use vizdoom.scenarios.get_scenarios_path() if available, or just standard path.
    # Note: vizdoom.scenarios might not expose a direct path string easily in all versions.
    # Let's assume standard 'basic.cfg' is available in the library path, or we can use a hardcoded relative path 
    # if we copy the scenarios.
    
    # We will try to find the path from the installed vizdoom package for 'basic.cfg'
    # If not found, we fallback to a local file.
    scenario_path = os.path.join(scenarios.__path__[0], "basic.cfg")
    if not os.path.exists(scenario_path):
        print(f"Warning: Scenario not found at {scenario_path}. Looking in current directory.")
        scenario_path = "basic.cfg"

    print(f"Training with: Steps={args.steps}, Render={args.render}, Scenario={scenario_path}")

    # Create Environment
    env = DoomEnv(scenario_path=scenario_path, visible=args.render, render_mode="human" if args.render else "rgb_array")

    # Initialize or Load PPO Model
    if args.load and os.path.exists(args.save_path + ".zip"):
        print(f"Loading existing model from {args.save_path}...")
        model = PPO.load(args.save_path, env=env, verbose=1, learning_rate=0.0001, n_steps=1000)
    else:
        print("Creating new model...")
        model = PPO("CnnPolicy", env, verbose=1, learning_rate=0.0001, n_steps=1000)

    # Callback to save periodically
    # We save every 100,000 steps to avoid cluttering the folder with too many files.
    checkpoint_callback = CheckpointCallback(save_freq=100000, save_path='./models/', name_prefix='ppo_doom')
    logging_callback = LoggingCallback()

    # Train
    try:
        model.learn(total_timesteps=args.steps, callback=[checkpoint_callback, logging_callback])
    except KeyboardInterrupt:
        print("Training interrupted. Saving current model...")
    
    # Save final model
    model.save(args.save_path)
    print(f"Model saved to {args.save_path}")

    env.close()

if __name__ == "__main__":
    main()
