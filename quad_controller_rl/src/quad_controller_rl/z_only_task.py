from gym import spaces
import numpy as np

class ZOnlyTask:
    """Wraps a task to constrain the state and action spaces to only linear-z"""
    def __init__(self, task):
        self.task = task
        self.target_z = task.target_z
        o_s = self.task.observation_space
        a_s = self.task.action_space

        self.observation_space = spaces.Box(self.filter_state(o_s.low), self.filter_state(o_s.high))
        print("modified observation space: {} {}".format(self.observation_space.low, self.observation_space.high))
        self.action_space = spaces.Box(a_s.low[2:3], a_s.high[2:3])
        print("modified action space: {} {}".format(self.action_space.low, self.action_space.high))
    
    def filter_state(self, state):
        # return the z position, plus anything beyond the quaternion data (in my Hover task, this is the target z height and the velocity).
        return np.concatenate((state[2:3], state[7:]))

    def filter_action_response(self, action):
        return np.pad(action, (2,3), 'constant')
