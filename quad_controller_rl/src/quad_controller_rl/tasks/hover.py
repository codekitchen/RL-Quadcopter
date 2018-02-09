"""Takeoff task."""

import numpy as np
from gym import spaces
from geometry_msgs.msg import Vector3, Point, Quaternion, Pose, Twist, Wrench
from quad_controller_rl.tasks.base_task import BaseTask
from quad_controller_rl.tasks.temporal_space import TemporalState

class Hover(BaseTask):
    """Simple task where the goal is to lift off the ground and reach a target height."""

    def __init__(self):
        # State space: <position_x, .._y, .._z, orientation_x, .._y, .._z, .._w>
        cube_size = 300.0  # env is cube_size x cube_size x cube_size
        space_min = np.array([0, - cube_size / 2, - cube_size / 2,       0.0, -1.0, -1.0, -1.0, -1.0])
        space_max = np.array([500,  cube_size / 2,   cube_size / 2, cube_size, 1.0,  1.0,  1.0,  1.0])
        self.state = TemporalState(space_min, space_max)
        self.observation_space = self.state.observation_space
        #print("Takeoff(): observation_space = {}".format(self.observation_space))  # [debug]

        # Action space: <force_x, .._y, .._z, torque_x, .._y, .._z>
        max_force = 25.0
        max_torque = 25.0
        self.action_space = spaces.Box(
            np.array([-max_force, -max_force, -max_force, -max_torque, -max_torque, -max_torque]),
            np.array([ max_force,  max_force,  max_force,  max_torque,  max_torque,  max_torque]))
        #print("Takeoff(): action_space = {}".format(self.action_space))  # [debug]

        # Task-specific parameters
        self.max_duration = 30.0  # secs
        self.target_z = 10.0  # target height (z position) to hover at
        self.hovertime = 0
        self.last_timestamp = 0
        self.z_oob = 40.0
        self.xy_oob = 80.0
        self.max_height = 0.0

    def reset(self):
        self.state.reset()
        self.hovertime = 0
        self.max_height = 0.0
        # Nothing to reset; just return initial condition
        return Pose(
                position=Point(0.0, 0.0, np.random.normal(0.5, 0.1)),  # drop off from a slight random height
                orientation=Quaternion(0.0, 0.0, 0.0, 0.0),
            ), Twist(
                linear=Vector3(0.0, 0.0, 0.0),
                angular=Vector3(0.0, 0.0, 0.0)
            )

    def update(self, timestamp, pose, angular_velocity, linear_acceleration):
        # Prepare state vector (pose only; ignore angular_velocity, linear_acceleration)
        cur_state = np.array([
                self.target_z,
                pose.position.x, pose.position.y, pose.position.z,
                pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
        self.state.update(cur_state)

        # Compute reward / penalty and check if this episode is complete
        done = False
        self.max_height = max(self.max_height, pose.position.z)
        zdist = abs(self.target_z - pose.position.z)
        # reward = 5 - min(zdist, 40.0)  # reward = zero for matching target z, -ve as you go farther, upto -40
        # hovering = zdist < 0.7
        # # punish for straying more than 5 units from orgin on the xy plane
        distance_from_origin = np.sqrt(pose.position.x ** 2 + pose.position.y ** 2)
        # reward -= min(distance_from_origin * 1.0, 40.0)
        # if timestamp > self.max_duration:  # agent has run out of time
        #     done = True
        # elif pose.position.z > 20.0 or distance_from_origin > 30.0:
        #     # punish and end the episode if it flies out of bounds
        #     reward -= 10
        #     done = True
        # elif hovering:
        #     reward += 2
        #     self.hovertime += timestamp - self.last_timestamp
        #     if self.hovertime > 10:
        #         # we've successfully hovered
        #         reward += 10
        #         done = True
        # else:
        #     self.hovertime = 0

        reward = (self.z_oob - zdist) / self.z_oob
        reward += ((self.xy_oob - distance_from_origin) / self.xy_oob) / 2.0
        if zdist < 1.5 and distance_from_origin < 5.0:
            reward += 1.0
        if timestamp > self.max_duration:
            done = True
        elif pose.position.z > self.z_oob or distance_from_origin > self.xy_oob:
            reward -= 1
            done = True
        elif self.max_height < 0.6 and timestamp > 3.0:
            reward -= 1
            done = True

        # Take one RL step, passing in current state and reward, and obtain action
        # Note: The reward passed in here is the result of past action(s)
        action = self.agent.step(self.state.state, reward, done)  # note: action = <force; torque> vector

        self.last_timestamp = timestamp

        # Convert to proper force command (a Wrench object) and return it
        if action is not None:
            action = np.clip(action.flatten(), self.action_space.low, self.action_space.high)  # flatten, clamp to action space limits
            return Wrench(
                    force=Vector3(action[0], action[1], action[2]),
                    torque=Vector3(action[3], action[4], action[5])
                ), done
        else:
            return Wrench(), done
