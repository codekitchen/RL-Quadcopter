"""Land task."""

# pylint: disable=W0201

import numpy as np
from gym import spaces
from geometry_msgs.msg import Vector3, Point, Quaternion, Pose, Twist, Wrench
from quad_controller_rl.tasks.base_task import BaseTask
from quad_controller_rl.tasks.temporal_space import TemporalState

class Land(BaseTask):
    """Land gently."""

    def __init__(self):
        self.name = 'Land'
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
        self.max_duration = 8.0  # secs
        self.reset()

    def reset(self):
        self.last_timestamp = 0.0
        self.last_pos = None
        self.last_vel = 0.0
        self.target_z = 0.0
        self.last_action = np.zeros_like(self.action_space.shape)

        return Pose(
                position=Point(0.0, 0.0, np.random.normal(12, 0.2)),
                orientation=Quaternion(0.0, 0.0, 0.0, 0.0),
            ), Twist(
                linear=Vector3(0.0, 0.0, 0.0),
                angular=Vector3(0.0, 0.0, 0.0)
            )

    def update(self, timestamp, pose, angular_velocity, linear_acceleration):
        # Prepare state vector (pose only; ignore angular_velocity, linear_acceleration)
        if self.last_pos is None:
            vel = 0.0
        else:
            vel = (pose.position.z - self.last_pos) / max(timestamp - self.last_timestamp, 1e-4)
        state = np.array([
                pose.position.x, pose.position.y, pose.position.z,
                pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w,
                self.target_z, vel])

        # Compute reward / penalty and check if this episode is complete
        done = False

        reward = (15.0 - pose.position.z) / 15.0
        if pose.position.z > 3.0 and np.abs(vel) > pose.position.z:
            reward -= np.abs(vel) / 5.0

        if timestamp > self.max_duration:
            reward -= 1.0
            done = True
        elif pose.position.z > 20.0:
            reward -= 5.0
            done = True
        elif pose.position.z < 0.4 and vel > 0.6:
            reward -= 5.5
            done = True
        elif pose.position.z < 0.2 and vel < 0.1:
            reward += 1.0
            done = True

        # Take one RL step, passing in current state and reward, and obtain action
        # Note: The reward passed in here is the result of past action(s)
        action = self.agent.step(state, reward, done)  # note: action = <force; torque> vector

        self.last_timestamp = timestamp
        self.last_action = action
        self.last_pos = pose.position.z
        self.last_vel = vel

        # Convert to proper force command (a Wrench object) and return it
        if action is not None:
            action = np.clip(action.flatten(), self.action_space.low, self.action_space.high)  # flatten, clamp to action space limits
            return Wrench(
                    force=Vector3(action[0], action[1], action[2]),
                    torque=Vector3(action[3], action[4], action[5])
                ), done
        else:
            return Wrench(), done
