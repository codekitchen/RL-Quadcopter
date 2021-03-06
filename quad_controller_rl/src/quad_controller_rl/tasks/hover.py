"""Takeoff task."""

import numpy as np
from gym import spaces
from geometry_msgs.msg import Vector3, Point, Quaternion, Pose, Twist, Wrench
from quad_controller_rl.tasks.base_task import BaseTask

class Hover(BaseTask):
    """Hover around a given target height, randomly generated for each episode."""

    def __init__(self):
        self.name = 'Hover'
        # State space: <position_x, .._y, .._z, orientation_x, .._y, .._z, .._w, target_z, velocity_z>
        cube_size = 300.0  # env is cube_size x cube_size x cube_size
        self.observation_space = spaces.Box(
            np.array([- cube_size / 2, - cube_size / 2,
                      0.0, -1.0, -1.0, -1.0, -1.0, 0.0, -200.0]),
            np.array([cube_size / 2,   cube_size / 2, cube_size,  1.0,  1.0,  1.0,  1.0, 100.0, 200.0]))

        # Action space: <force_x, .._y, .._z, torque_x, .._y, .._z>
        max_force = 25.0
        max_torque = 25.0
        self.action_space = spaces.Box(
            np.array([-max_force, -max_force, -max_force, -max_torque, -max_torque, -max_torque]),
            np.array([ max_force,  max_force,  max_force,  max_torque,  max_torque,  max_torque]))

        # Task-specific parameters
        self.max_duration = 30.0  # secs
        self.z_oob = 35.0
        self.reset()

    def reset(self):
        self.last_timestamp = 0
        self.last_pos = None
        self.last_action = np.zeros_like(self.action_space.shape)
        self.target_z = np.random.rand() * 15 + 5
        start_z = np.random.rand() * 25.0
        # Nothing to reset; just return initial condition
        return Pose(
                position=Point(0.0, 0.0, start_z),  # drop off from a slight random height
                orientation=Quaternion(0.0, 0.0, 0.0, 0.0),
            ), Twist(
                linear=Vector3(0.0, 0.0, 0.0),
                angular=Vector3(0.0, 0.0, 0.0)
            )

    def update(self, timestamp, pose, angular_velocity, linear_acceleration):
        # if np.random.rand() > 0.99:
        #     self.target_z = np.random.rand() * 15 + 5

        # Prepare state vector (pose only; ignore angular_velocity, linear_acceleration)
        if self.last_pos is None:
            vel = 0.0
        else:
            vel = (pose.position.z - self.last_pos) / max(timestamp - self.last_timestamp, 1e-4)
        self.last_pos = pose.position.z
        state = np.array([
                pose.position.x, pose.position.y, pose.position.z,
                pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w,
                self.target_z, vel])

        # Compute reward / penalty and check if this episode is complete
        done = False
        zdist = abs(self.target_z - pose.position.z)

        reward = -(zdist ** 2 + 0.01 * (np.sum(self.last_action) ** 2)) / 9000

        # reward = 10.0 - zdist
        if zdist < 0.5:
            reward += 0.5
        if timestamp > self.max_duration:
            done = True
        elif pose.position.z > self.z_oob:
            reward -= 1.0
            done = True
        elif pose.position.z < 0.2:
            reward -= 1.0
            done = True

        # Take one RL step, passing in current state and reward, and obtain action
        # Note: The reward passed in here is the result of past action(s)
        action = self.agent.step(state, reward, done)  # note: action = <force; torque> vector

        self.last_timestamp = timestamp
        self.last_action = action

        # Convert to proper force command (a Wrench object) and return it
        if action is not None:
            action = np.clip(action.flatten(), self.action_space.low, self.action_space.high)  # flatten, clamp to action space limits
            return Wrench(
                    force=Vector3(action[0], action[1], action[2]),
                    torque=Vector3(action[3], action[4], action[5])
                ), done
        else:
            return Wrench(), done
