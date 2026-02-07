"""
doom_env.py - VizDoom Gymnasium Environment Wrapper

This file wraps VizDoom as a standard Gymnasium environment, enabling
Stable Baselines 3 algorithms to train on Doom scenarios.

Key Design Decisions:
1. UNIVERSAL ACTION SPACE: Always exposes 7 actions regardless of scenario.
   This allows transfer learning between scenarios with different button configs.
2. OBSERVATION PREPROCESSING: Converts screen to 120x160 grayscale for CNN input.
3. FRAME SKIP: Each action is repeated for 4 game tics for faster training.

Author: Student Project
"""

import gymnasium as gym
from gymnasium import spaces
import vizdoom as vzd
import numpy as np
import cv2
import os


class DoomEnv(gym.Env):
    """
    Custom Gymnasium environment for VizDoom.
    
    Attributes:
        scenario_path: Path to the VizDoom .cfg file
        visible: Whether to render the game window
        action_space: Discrete(7) - universal action space
        observation_space: Box(120, 160, 1) - grayscale image
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, scenario_path, render_mode=None, visible=False):
        super().__init__()
        self.scenario_path = scenario_path
        self.render_mode = render_mode
        self.visible = visible
        
        # Initialize VizDoom game instance
        self.game = vzd.DoomGame()
        
        # Handle relative paths (look in scenarios/ folder)
        if not os.path.exists(self.scenario_path):
            scenarios_path = os.path.join("scenarios", self.scenario_path)
            if os.path.exists(scenarios_path):
                self.scenario_path = scenarios_path
            else:
                raise FileNotFoundError(f"Scenario not found: {self.scenario_path}")
        
        self.game.load_config(self.scenario_path)
        
        # Window visibility
        self.game.set_window_visible(visible or render_mode == "human")
        
        # Screen settings
        self.game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
        self.game.set_screen_format(vzd.ScreenFormat.RGB24)
        
        # Minimal rendering for speed
        self.game.set_render_hud(False)
        self.game.set_render_crosshair(False)
        self.game.set_render_weapon(True)
        self.game.set_render_decals(False)
        self.game.set_render_particles(False)
        
        self.game.init()
        
        # Kill bonus to incentivize combat
        self.last_kills = 0
        self.kill_bonus = 20  # Bonus per kill
        
        # =================================================================
        # UNIVERSAL ACTION SPACE
        # =================================================================
        # We define 7 actions that cover all scenarios we use:
        # 0: MOVE_LEFT, 1: MOVE_RIGHT, 2: ATTACK
        # 3: MOVE_FORWARD, 4: MOVE_BACKWARD, 5: TURN_LEFT, 6: TURN_RIGHT
        #
        # In simpler scenarios (like "basic"), some buttons don't exist.
        # The agent will learn they have no effect there.
        # =================================================================
        
        self.universal_buttons = [
            vzd.Button.MOVE_LEFT,
            vzd.Button.MOVE_RIGHT,
            vzd.Button.ATTACK,
            vzd.Button.MOVE_FORWARD,
            vzd.Button.MOVE_BACKWARD,
            vzd.Button.TURN_LEFT,
            vzd.Button.TURN_RIGHT,
        ]
        
        self.action_space = spaces.Discrete(len(self.universal_buttons))  # Always 7
        
        # Map universal action index -> scenario's actual button index
        self.available_buttons = self.game.get_available_buttons()
        self.action_map = []
        for btn in self.universal_buttons:
            try:
                idx = self.available_buttons.index(btn)
                self.action_map.append(idx)
            except ValueError:
                self.action_map.append(None)  # Button not available
        
        # Observation space: 120x160 grayscale image
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(120, 160, 1), dtype=np.uint8
        )

    def _preprocess_observation(self, screen_buffer):
        """Convert raw screen to grayscale and resize."""
        obs = cv2.resize(screen_buffer, (160, 120))
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = np.expand_dims(obs, axis=-1)
        return obs

    def step(self, action):
        """Execute action and return (obs, reward, terminated, truncated, info)."""
        # Build action vector for VizDoom
        game_actions = [0.0] * len(self.available_buttons)
        
        mapped_idx = self.action_map[action]
        if mapped_idx is not None:
            game_actions[mapped_idx] = 1.0
        
        # Execute action for 4 tics (frame skip)
        reward = self.game.make_action(game_actions, 4)
        terminated = self.game.is_episode_finished()
        
        # Add kill bonus
        if not terminated:
            try:
                current_kills = self.game.get_game_variable(vzd.GameVariable.KILLCOUNT)
                kills_gained = current_kills - self.last_kills
                if kills_gained > 0:
                    reward += self.kill_bonus * kills_gained
                self.last_kills = current_kills
            except:
                pass
        
        if terminated:
            obs = np.zeros(self.observation_space.shape, dtype=np.uint8)
        else:
            obs = self._preprocess_observation(self.game.get_state().screen_buffer)
        
        return obs, reward, terminated, False, {}

    def reset(self, seed=None, options=None):
        """Reset the environment for a new episode."""
        super().reset(seed=seed)
        self.game.new_episode()
        
        # Reset kill tracking
        try:
            self.last_kills = self.game.get_game_variable(vzd.GameVariable.KILLCOUNT)
        except:
            self.last_kills = 0
        
        state = self.game.get_state()
        if state is None:
            return np.zeros(self.observation_space.shape, dtype=np.uint8), {}
        
        obs = self._preprocess_observation(state.screen_buffer)
        return obs, {}

    def render(self):
        """Return the current frame for recording."""
        if self.render_mode == "rgb_array":
            if self.game.is_episode_finished() or self.game.get_state() is None:
                return np.zeros((480, 640, 3), dtype=np.uint8)
            return self.game.get_state().screen_buffer
        return None

    def close(self):
        """Clean up VizDoom instance."""
        self.game.close()
