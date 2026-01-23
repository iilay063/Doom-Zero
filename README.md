# Doom-Zero RL Agent 🤖💥

A Reinforcement Learning agent capable of playing Doom (Classic), trained using **PPO** (Stable-Baselines3) and **VizDoom**.

## 🚀 Features
- **PPO Agent**: Learns to navigate and shoot monsters.
- **Hybrid Observation**: The agent sees a downscaled 160x120 Grayscale image for efficiency, while you can watch in 640x480 RGB.
- **Custom Environment**: Gymnasium wrapper for VizDoom manipulation.
- **Easy Deployment**: One-click install script for Windows.

## 📦 Installation
(Windows)

1. **Prerequisite**: Install [Anaconda](https://www.anaconda.com/) or Miniconda.
2. Clone this repository.
3. Run **`install_env.bat`**.
   *   This will creating a Conda environment (`doom-rl`) with Python 3.11 and install all dependencies.

## 🎮 Usage

### Training
To train the agent:
```powershell
run_train.bat
```
*   You will be asked how many steps to run (default: 100,000).
*   Use `run_train_render.bat` to **watch** the agent learn in real-time.

### Playing
To see the trained agent in action:
```powershell
run_play.bat
```
(Requires a trained model in `models/ppo_doom.zip`)

## 🧠 Model
The model is saved to `models/ppo_doom.zip`.
*   Checkpoints are saved every 100k steps to avoid data loss.
*   To resume training, use: `run_train.bat --load`.

## 🛠️ Requirements
*   Python 3.11
*   VizDoom
*   Gymnasium
*   Stable-Baselines3
