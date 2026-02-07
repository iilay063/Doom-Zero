# Doom-Zero: Reinforcement Learning in VizDoom

A deep learning project that trains an AI agent to play Doom using Curriculum Learning and Proximal Policy Optimization (PPO).

## Project Goal

Train an AI agent to progressively master increasingly difficult Doom scenarios:
1. **Basic**: Learn to shoot a stationary target
2. **Defend the Center**: Learn to turn and shoot approaching enemies
3. **Deadly Corridor**: Learn to navigate and survive
4. **Freedoom**: Apply learned skills to the full game

## Files

| File | Purpose |
|------|---------|
| `doom_env.py` | Gymnasium environment wrapper for VizDoom |
| `train.py` | Training script using PPO algorithm |
| `play.py` | Demo script to watch trained agent |
| `scenarios/` | VizDoom scenario configuration files |
| `models/` | Saved trained models |

## Setup

```bash
# Create conda environment
conda create -n doom-rl python=3.10
conda activate doom-rl

# Install dependencies
pip install vizdoom stable-baselines3 gymnasium opencv-python
```

## Usage

### Step 1: Copy Scenarios
Copy VizDoom scenarios to the `scenarios/` folder:
```python
import shutil, vizdoom.scenarios as s, os
for f in os.listdir(s.__path__[0]):
    if f.endswith(('.cfg', '.wad')):
        shutil.copy(os.path.join(s.__path__[0], f), 'scenarios/')
```

### Step 2: Train on Basic Scenario
```bash
python train.py --scenario basic --steps 100000
```

### Step 3: Watch the Agent
```bash
python play.py --model models/doom_agent --scenario basic
```

### Step 4: Continue Training on Harder Scenario
```bash
python train.py --scenario defend --load models/doom_agent --steps 200000
```

### Step 5: Repeat
Continue the cycle until the agent masters all scenarios.

## Technical Details

### Universal Action Space
The `doom_env.py` uses a fixed 7-action space across all scenarios. This enables transfer learning between scenarios with different button configurations.

### PPO Algorithm
We use Proximal Policy Optimization from Stable Baselines 3:
- Policy: CNN-based (CnnPolicy)
- Learning Rate: 0.0001
- Batch Size: 64

### Observation Preprocessing
- Resolution: 640x480 → 160x120
- Color: RGB → Grayscale
- Frame Skip: 4 tics per action

## Author
Student Deep Learning Project
