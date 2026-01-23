
import gymnasium as gym
from gymnasium import spaces
import vizdoom as vzd
import numpy as np
import cv2
import os

class DoomEnv(gym.Env):
    """
    Custom Gymnasium Environment for VizDoom.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, scenario_path, render_mode=None, visible=False):
        super().__init__()
        self.scenario_path = scenario_path
        self.render_mode = render_mode
        self.visible = visible
        
        # Initialize VizDoom Game
        self.game = vzd.DoomGame()
        
        if not os.path.exists(self.scenario_path):
             raise FileNotFoundError(f"Scenario not found: {self.scenario_path}")
             
        self.game.load_config(self.scenario_path)
        
        # Set window visibility logic
        # If 'human', we generally want to see it, OR if visible=True override is passed.
        should_be_visible = visible or (render_mode == "human")
        self.game.set_window_visible(should_be_visible)

        # Set screen resolution/format
        # Using a standard low res for training efficiency
        self.game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
        self.game.set_screen_format(vzd.ScreenFormat.RGB24)
        
        # Disable HUD/Crosshair for cleaner input, keep Weapon if desired
        self.game.set_render_hud(False)
        self.game.set_render_crosshair(False)
        self.game.set_render_weapon(True)
        self.game.set_render_decals(False)
        self.game.set_render_particles(False)
        
        # Initialize
        self.game.init()

        # Define Action Space
        actions_num = self.game.get_available_buttons_size()
        self.action_space = spaces.Discrete(actions_num)
        
        # Define Observation Space
        # Hybrid Approach:
        # Game runs at 640x480 (User sees this)
        # Agent sees 160x120 Grayscale (Model sees this)
        h, w = 120, 160
        self.observation_space = spaces.Box(low=0, high=255, shape=(h, w, 1), dtype=np.uint8)

    def step(self, action):
        actions = np.zeros(self.action_space.n)
        actions[action] = 1
        
        # 4 tics per step
        reward = self.game.make_action(list(actions), 4)

        # ---------------------------------------------------------------------
        # CUSTOM REWARD SHAPING
        # ---------------------------------------------------------------------
        # You can tweak the reward here. `reward` is what the game (WAD) gives.
        # Example 1: Add a small penalty for every step to encourage speed
        # reward -= 0.01 
        
        # Example 2: Penalize shooting (if action index 2 is shoot) to save ammo
        # if action == 2:
        #     reward -= 0.1
        
        # Example 3: Modify based on Game Variables (Health, Ammo)
        # state = self.game.get_state()
        # if state:
        #     health = state.game_variables[0] # Requires setting available vars in cfg
        # ---------------------------------------------------------------------
        
        done = self.game.is_episode_finished()
        
        if done:
            obs = np.zeros(self.observation_space.shape, dtype=np.uint8)
        else:
            state = self.game.get_state()
            # Custom Observation Processing
            # Resize to 160x120 and Convert to Grayscale
            obs = cv2.resize(state.screen_buffer, (160, 120))
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
            obs = np.expand_dims(obs, axis=-1) # Add channel dimension (120, 160, 1)
        
        info = {}
            
        truncated = False
        
        return obs, reward, done, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.new_episode()
        state = self.game.get_state()
        if state is None:
             return np.zeros(self.observation_space.shape, dtype=np.uint8), {}
             
        # Resize reset observation too
        obs = cv2.resize(state.screen_buffer, (160, 120))
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = np.expand_dims(obs, axis=-1)
        
        return obs, {}

    def render(self):
        # In 'human' mode, VizDoom handles the window itself.
        # We can still return the array if asked.
        if self.render_mode == "rgb_array":
             if self.game.is_episode_finished():
                 return np.zeros(self.observation_space.shape, dtype=np.uint8)
             return self.game.get_state().screen_buffer
        return None

    def close(self):
        self.game.close()
