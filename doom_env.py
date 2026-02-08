"""
doom_env.py - VizDoom Gymnasium Environment Wrapper

Wraps VizDoom as a Gymnasium environment for use with Stable Baselines 3.
The key idea is a UNIVERSAL ACTION SPACE - we always expose the same 7 actions
regardless of which scenario is loaded. This enables transfer learning between
different scenarios.

Observation: 120x160 grayscale image (downscaled from 640x480)
Actions: 7 discrete (move left/right, attack, forward/back, turn left/right)
"""

import gymnasium as gym
from gymnasium import spaces
import vizdoom as vzd
import numpy as np
import cv2
import os


class DoomEnv(gym.Env):
    """
    Gymnasium environment for VizDoom scenarios.
    
    Uses a fixed 7-action space to allow training on one scenario and
    transferring to another. Actions that don't exist in simpler scenarios
    simply do nothing - the agent learns this automatically.
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, scenario_path, render_mode=None, visible=False, 
                 kill_bonus=150, step_penalty=5, total_enemies=6):
        """
        Args:
            scenario_path: Path to .cfg file (can be just filename if in scenarios/)
            render_mode: 'human' to show window
            visible: Alternative way to show window
            kill_bonus: Extra reward per kill (helps agent prioritize combat)
            step_penalty: Penalty per alive enemy per step (discourages rushing)
            total_enemies: Number of enemies in scenario (for step_penalty calc)
        """
        super().__init__()
        self.scenario_path = scenario_path
        self.render_mode = render_mode
        self.visible = visible
        self.kill_bonus = kill_bonus
        self.step_penalty = step_penalty
        self.total_enemies = total_enemies
        
        # Setup VizDoom
        self.game = vzd.DoomGame()
        
        # Try to find the scenario file
        if not os.path.exists(self.scenario_path):
            scenarios_path = os.path.join("scenarios", self.scenario_path)
            if os.path.exists(scenarios_path):
                self.scenario_path = scenarios_path
            else:
                raise FileNotFoundError(f"Scenario not found: {self.scenario_path}")
        
        self.game.load_config(self.scenario_path)
        self.game.set_window_visible(visible or render_mode == "human")
        
        # Screen settings - we render at higher res then downscale
        self.game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
        self.game.set_screen_format(vzd.ScreenFormat.RGB24)
        
        # Turn off extra rendering for speed
        self.game.set_render_hud(False)
        self.game.set_render_crosshair(False)
        self.game.set_render_weapon(True)
        self.game.set_render_decals(False)
        self.game.set_render_particles(False)
        
        self.game.init()
        
        # Track kills for reward shaping
        self.last_kills = 0
        
        # Universal action space - same 7 actions for all scenarios
        # This is the key to transfer learning between scenarios
        self.universal_buttons = [
            vzd.Button.MOVE_LEFT,
            vzd.Button.MOVE_RIGHT,
            vzd.Button.ATTACK,
            vzd.Button.MOVE_FORWARD,
            vzd.Button.MOVE_BACKWARD,
            vzd.Button.TURN_LEFT,
            vzd.Button.TURN_RIGHT,
        ]
        
        self.action_space = spaces.Discrete(7)
        
        # Figure out which of our 7 actions actually work in this scenario
        self.available_buttons = self.game.get_available_buttons()
        self.action_map = []
        for btn in self.universal_buttons:
            try:
                idx = self.available_buttons.index(btn)
                self.action_map.append(idx)
            except ValueError:
                self.action_map.append(None)  # Button doesn't exist here
        
        # Observation: downscaled grayscale image
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(120, 160, 1), dtype=np.uint8
        )

    def _preprocess_observation(self, screen_buffer):
        """Resize to 160x120 and convert to grayscale."""
        obs = cv2.resize(screen_buffer, (160, 120))
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = np.expand_dims(obs, axis=-1)
        return obs

    def step(self, action):
        """Run one step: execute action, get reward, check if done."""
        # Convert our universal action to VizDoom's button format
        game_actions = [0.0] * len(self.available_buttons)
        
        mapped_idx = self.action_map[action]
        if mapped_idx is not None:
            game_actions[mapped_idx] = 1.0
        
        # Run 4 game tics per action (frame skip for faster training)
        reward = self.game.make_action(game_actions, 4)
        terminated = self.game.is_episode_finished()
        
        # Reward shaping: bonus for kills, penalty for ignoring enemies
        if not terminated and self.kill_bonus > 0:
            try:
                current_kills = self.game.get_game_variable(vzd.GameVariable.KILLCOUNT)
                kills_gained = current_kills - self.last_kills
                if kills_gained > 0:
                    reward += self.kill_bonus * kills_gained
                self.last_kills = current_kills
                
                # Penalize each step where enemies are still alive
                # This discourages running past enemies without fighting
                if self.step_penalty > 0 and self.total_enemies > 0:
                    enemies_alive = self.total_enemies - current_kills
                    reward -= self.step_penalty * enemies_alive
            except:
                pass
        
        # Get observation
        if terminated:
            obs = np.zeros(self.observation_space.shape, dtype=np.uint8)
        else:
            obs = self._preprocess_observation(self.game.get_state().screen_buffer)
        
        return obs, reward, terminated, False, {}

    def reset(self, seed=None, options=None):
        """Start a new episode."""
        super().reset(seed=seed)
        self.game.new_episode()
        
        # Reset kill counter
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
        """Return current frame (for recording)."""
        if self.render_mode == "rgb_array":
            if self.game.is_episode_finished() or self.game.get_state() is None:
                return np.zeros((480, 640, 3), dtype=np.uint8)
            return self.game.get_state().screen_buffer
        return None

    def close(self):
        """Cleanup."""
        self.game.close()
