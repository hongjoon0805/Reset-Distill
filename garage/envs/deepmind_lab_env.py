import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import akro

from garage import Environment, EnvSpec, EnvStep, StepType

def _to_gym_name(text):
    return "DeepmindLab" + ''.join(map(lambda x: x.capitalize(), text.split("_"))) + "-v0"

ACTION_LIST = [
    [-10,   0,   0,   0,   0,   0,   0], # LOOK LEFT
    [ 10,   0,   0,   0,   0,   0,   0], # LOOK RIGHT
    [  0, -10,   0,   0,   0,   0,   0], # LOOK DOWN
    [  0,  10,   0,   0,   0,   0,   0], # LOOK UP
    [  0,   0,  -1,   0,   0,   0,   0], # STRAFE LEFT
    [  0,   0,   1,   0,   0,   0,   0], # STRAFE RIGHT
    [  0,   0,   0,  -1,   0,   0,   0], # MOVE BACK
    [  0,   0,   0,   1,   0,   0,   0], # MOVE FOWARD
    [  0,   0,   0,   0,   1,   0,   0], # FIRE
    [  0,   0,   0,   0,   0,   1,   0], # JUMP
    [  0,   0,   0,   0,   0,   0,   1]  # CROUCH
]
ACTION_LIST = [np.array(action, dtype = np.intc) for action in ACTION_LIST]

class DeepmindLabEnv(gym.Env):

    metadata = {'render.modes': ['rgb_array']}

    def __init__(self, lab, name, colors, config, **kwargs):
        super().__init__(**kwargs)
        self._colors = colors
        self._lab = lab
        self._task_name = _to_gym_name(name)

        self.action_space = spaces.Discrete(len(ACTION_LIST))
        self.observation_space = spaces.Box(0, 255, shape = (int(config['width']), int(config['height']), 3), dtype = np.uint8)

        # self.action_space = akro.from_gym(self.action_space)
        # self.observation_space = akro.from_gym(self.observation_space)

        self._spec = EnvSpec(
            observation_space=self.observation_space,
            action_space=self.action_space,
            max_episode_length=int(1e3) # Sensitive to this value
        )

        self._last_observation = None

    def step(self, action: int, num_steps: int = 4):
        reward = self._lab.step(ACTION_LIST[action], num_steps = num_steps)
        terminal = not self._lab.is_running()
        obs = None if terminal else self._lab.observations()[self._colors]
        self._last_observation = obs if obs is not None else np.copy(self._last_observation)
        return self._last_observation, reward, terminal, dict()
    
    def reset(self):
        self._lab.reset()
        self._last_observation = self._lab.observations()[self._colors]
        return self._last_observation
    
    def seed(self, seed = None):
        self._lab.reset(seed=seed)

    def close(self):
        self._lab.close()

    def render(self, close=False):
        return self._lab.observations()[self._colors]
    
    def __str__(self):
        return self._task_name
    
    @property
    def spec(self):
        return self._spec