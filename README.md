# Doom-Zero: Reinforcement Learning Agent for VizDoom

A deep reinforcement learning project that trains an AI agent to play Doom scenarios using PPO (Proximal Policy Optimization).

## Project Overview

This project demonstrates:
- **Gymnasium Environment Wrapper** - Custom environment bridging VizDoom and Stable Baselines 3
- **Transfer Learning** - Universal action space allowing models to transfer between scenarios
- **Reward Shaping** - Custom rewards to encourage desired behavior (combat over rushing)
- **Parallel Training** - Multi-environment support for faster training

## Project Structure

```
Doom-Zero/
├── doom_env.py      # Gymnasium wrapper for VizDoom
├── train.py         # Training script with PPO
├── play.py          # Demo script to watch trained agent
├── scenarios/       # VizDoom scenario files (.cfg, .wad)
├── models/          # Trained model checkpoints
│   └── best/        # Best model from training
└── requirements.txt # Python dependencies
```

## Setup

```bash
# Create environment
conda create -n doom-rl python=3.10
conda activate doom-rl

# Install dependencies
pip install vizdoom stable-baselines3 gymnasium opencv-python
```

## Quick Start

### 1. Copy Scenario Files
```bash
python setup_scenarios.py
```

### 2. Train an Agent
```bash
# Train on basic scenario (shoot stationary target)
python train.py --scenario basic --steps 100000

# Train on corridor with parallel environments (faster)
python train.py --scenario corridor --steps 500000 --num-envs 6
```

### 3. Watch the Agent Play
```bash
python play.py --model models/best/best_model --scenario corridor
```

## Scenarios

| Scenario | Description | Difficulty |
|----------|-------------|------------|
| `basic` | Shoot a stationary target | Easy |
| `defend` | Survive waves of enemies | Medium |
| `home` | Navigate maze to find goal | Medium |
| `corridor` | Navigate corridor while fighting | Hard |

## Key Design Decisions

### Universal Action Space
The environment exposes 7 actions regardless of scenario:
- MOVE_LEFT, MOVE_RIGHT, ATTACK
- MOVE_FORWARD, MOVE_BACKWARD
- TURN_LEFT, TURN_RIGHT

This allows a model trained on `basic` to be fine-tuned on `corridor` even though they have different button configs.

### Reward Shaping
For combat scenarios, we add:
- **Kill bonus**: +150 per enemy killed
- **Step penalty**: -5 per alive enemy per step (discourages rushing)

This teaches the agent to prioritize combat over just running to the goal.

### Observation Preprocessing
- Original: 640x480 RGB
- Processed: 160x120 Grayscale
- Frame skip: 4 tics per action

## Training Results

| Scenario | Training Steps | Mean Reward |
|----------|---------------|-------------|
| Basic | 100k | ~90 |
| Defend | 800k | ~100 |
| Home | 2M | ~10 |
| Corridor | 8M | ~-200 |

## PPO Hyperparameters

```python
learning_rate = 0.00025 - 0.001 (depends on scenario)
n_steps = 2048
batch_size = 128
ent_coef = 0.01 - 0.15 (depends on scenario)
```

## References

- [VizDoom](https://github.com/Farama-Foundation/ViZDoom)
- [Stable Baselines 3](https://stable-baselines3.readthedocs.io/)
- [PPO Paper](https://arxiv.org/abs/1707.06347)