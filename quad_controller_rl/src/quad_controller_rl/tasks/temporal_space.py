import numpy as np
from gym import spaces

class TemporalState:
  def __init__(self, space_min, space_max):
    self.observation_space = spaces.Box(
        np.stack([space_min, space_min, space_min]),
        np.stack([space_max, space_max, space_max]))
    self.reset()
  
  def reset(self):
    self.state = np.zeros(self.observation_space.shape)
  
  def update(self, new_state):
    self.state = np.roll(self.state, 1, axis=0)
    self.state[0] = new_state