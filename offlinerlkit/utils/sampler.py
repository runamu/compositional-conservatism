import random
from typing import Any
from scipy.stats import truncnorm, uniform
import numpy as np
import matplotlib.pyplot as plt
import torch
import d4rl.pointmaze.maze_model as mm
import d4rl
from shapely.geometry import Polygon, Point
from shapely.affinity import translate
import math
import time
import ipdb
from einops import rearrange
from concurrent.futures import as_completed
from .time_decorator import timer_decorator, timer_decorator_cumulative

class Sampler():
    def __init__(self, dataset):
        self.datset = dataset # for the case of samplign data in dataset
        self.vel_x = None
        self.vel_y = None
        self.sample_mode = None

    def sample_xy(self, extract_from_dataset=False):
        if extract_from_dataset:
            return self.sample_xy_from_data()
        else:
            return self.randomly_sample_xy()

    def sample_va(self,  extract_from_dataset=False):
        if extract_from_dataset:
            return self.sample_va_from_data()
        else:
            return self.randomly_sample_va()

    def sample_only_actions(self):
        if self.task == 'maze2d':
            return self.sample_only_actions_maze2d()
        elif self.task == 'antmaze':
            return self.sample_only_actions_antmaze()

    @staticmethod
    def get_truncated_normal(mean=0, sd=1, low=-5, upp=5):
        return truncnorm((low - mean) / sd,  (upp - mean) / sd, loc=mean, scale=sd)

class MazeSampler(Sampler):
    def __init__(self, dataset):
        super().__init__( dataset)
        self._action_ctrl_min = -1
        self._action_ctrl_max = 1
        _, self.actions_dim = self.dataset['actions'].shape

    def sample_xy_from_data(self):
        idx = random.randint(0, len(self.dataset['observations']) - 1)
        x, y, self.vel_x, self.vel_y = self.dataset['observations'][idx].tolist()

        self.dataset['observations'] = np.delete(self.dataset['observations'], idx, axis=0)

        return np.array([x,y])

    def sample_va_from_data(self):
        velocities, actions = None, None
        if self.draw_type == 'best_transition': # cannot be interval mode
            a_rs = np.linspace(0, self._action_ctrl_max, num=self.action_count) # to linearspaxe
            thetas = np.linspace(0, math.pi * 2, num=self.action_count)
            actions =np.array([[r * math.cos(theta), r * math.sin(theta)] for r in a_rs for theta in thetas])

            velocities = np.array([[self.vel_x, self.vel_y]])

        return velocities, actions

    '''
    Sample x, y coordinate
    '''
    def randomly_sample_xy(self):
        shifted_wall = np.array(self.WALL) - self.shift
        # Create rectangles (as polygons) for each point and add them to a list
        rectangles = [Polygon([(x ,y), (x+1,y), (x+1,y+1), (x,y+1)]) for x, y in shifted_wall]

        # Create a polygon that represents the U-shaped maze
        shell = np.array([(0,0), (self.rows,0), (self.rows, self.cols), (0,self.cols)])
        shifted_shell = shell - self.shift
        maze = Polygon(shifted_shell)

        # Subtract the block boxes from the maze to get the final available area
        for rect in rectangles:
            maze = maze.difference(rect)

        # Create an inner polygon with offset from the wall
        inner_maze = maze.buffer(-1 * self.offset)

        def get_random_point_in_polygon(poly):
            minx, miny, maxx, maxy = poly.bounds
            while True:
                p = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
                if poly.contains(p):
                    return p

        return np.array(get_random_point_in_polygon(inner_maze).coords[0])



    '''
    Sample v, a coordinate

    '''

    def randomly_sample_va(self): # shape = (2, N, 2)
        v, a = None, None
        N = self.action_count

        if self.sample_mode == 'grid':
            v, a = [], []
            for i in range(5): # for 5 velocity section
                v_rs = np.linspace(i, i+1, num=int(N/5))
                a_rs = np.linspace(0, self._action_ctrl_max, num=int(N/5))
                thetas = np.linspace(0, math.pi*2, num=int(N/5))

                v.extend([[r * math.cos(theta), r * math.sin(theta)] for r in v_rs for theta in thetas]) # v num change # of actions => # of actions **2
                a.extend([[r * math.cos(theta), r * math.sin(theta)] for r in a_rs for theta in thetas])

        if self.sample_mode == 'uniform':
            v = np.random.uniform(-5, 5, (N, 2))
            a = np.random.uniform(self._action_ctrl_min, self._action_ctrl_max, (N, 2))

        elif self.sample_mode =='gaussian':
            v = self.get_truncated_normal(mean=0, sd=2, low=-5, upp=5).rvs((N, 2))
            a = self.get_truncated_normal(mean=0, sd=1, low=self._action_ctrl_min, upp=self._action_ctrl_max).rvs((N, 2))

        elif self.sample_mode == 'interval':
            if self.draw_type == 'best_transition':
                v, a=[], []
                for i in range(5): # for 5 velocity section
                    v_rs = np.linspace(i, i+1, num=int(N/5))
                    a_rs = np.linspace(0, self._action_ctrl_max, num=int(N/5))
                    thetas = np.linspace(0, math.pi*2, num=int(N/5))

                    v.append([[r * math.cos(theta), r * math.sin(theta)] for r in v_rs for theta in thetas]) # v num change # of actions => # of actions **2
                    a.append([[r * math.cos(theta), r * math.sin(theta)] for r in a_rs for theta in thetas])

            else:
                v, a=[], []
                for i in range(5): # for 5 velocity section
                    r = np.random.uniform(i,i+1, int(N/5))
                    theta = np.random.uniform(0, math.pi * 2, int(N/5))

                    v.append([[r * math.cos(theta), r * math.sin(theta)] for r, theta in zip(r, theta)])
                    a.append(np.random.uniform(0, self._action_ctrl_max, (int(N/5), 2)))

        return np.array(v), np.array(a) # v = (5, N, 2), a = (5, N, 2) if interval mode

    def sample_only_actions_maze2d(self):
        # uniform sample
        actions = self.uniform_sample(sample_num=self.action_count, dim=self.actions_dim, low=self._action_ctrl_min, high=self._action_ctrl_max)
        return actions



class AntSampler(Sampler):
    def __init__(self, dataset = None):
        super().__init__(dataset)

        self._action_ctrl_min = -1
        self._action_ctrl_max = 1
        _, self.actions_dim = self.dataset['actions'].shape

    def sample_only_actions_antmaze(self):
        # uniform sample
        actions = self.uniform_sample(sample_num=self.action_count, dim=self.actions_dim, low=self._action_ctrl_min, high=self._action_ctrl_max)
        return actions

    def uniform_sample(self, sample_num, dim, low, high):
        return uniform.rvs(size=(sample_num, dim)) * (high - low) + low

    @staticmethod
    def get_combinations(A, B):
        # Repeat A by the number of actions and tile actions by the number of A
        A_combinations = np.repeat(A, len(B), axis=0)
        B_combinations = np.tile(B, (len(A), 1))
        return A_combinations, B_combinations












"""
AntMaze
# | Num | Action                                                             | Control Min | Control Max | Name (in corresponding XML file) | Joint | Unit         |
# |-----|--------------------------------------------------------------------|-------------|-------------|----------------------------------|-------|--------------|
# | 0   | Torque applied on the rotor between the torso and front left hip   | -1          | 1           | hip_1 (front_left_leg)           | hinge | torque (N m) |
# | 1   | Torque applied on the rotor between the front left two links       | -1          | 1           | angle_1 (front_left_leg)         | hinge | torque (N m) |
# | 2   | Torque applied on the rotor between the torso and front right hip  | -1          | 1           | hip_2 (front_right_leg)          | hinge | torque (N m) |
# | 3   | Torque applied on the rotor between the front right two links      | -1          | 1           | angle_2 (front_right_leg)        | hinge | torque (N m) |
# | 4   | Torque applied on the rotor between the torso and back left hip    | -1          | 1           | hip_3 (back_leg)                 | hinge | torque (N m) |
# | 5   | Torque applied on the rotor between the back left two links        | -1          | 1           | angle_3 (back_leg)               | hinge | torque (N m) |
# | 6   | Torque applied on the rotor between the torso and back right hip   | -1          | 1           | hip_4 (right_back_leg)           | hinge | torque (N m) |
# | 7   | Torque applied on the rotor between the back right two links       | -1          | 1           | angle_4 (right_back_leg)         | hinge | torque (N m) |

# ### Observation Space

# The state space consists of positional values of different body parts of the ant,
# followed by the velocities of those individual parts (their derivatives) with all
# the positions ordered before all the velocities.

# The observation is a `ndarray` with shape `(111,)` where the elements correspond to the following:

# | Num | Observation                                                  | Min  | Max | Name (in corresponding XML file) | Joint | Unit                     |
# |-----|--------------------------------------------------------------|------|-----|----------------------------------|-------|--------------------------|
# | 0   | x-coordinate of the torso (centre)                           | -Inf | Inf | torso                            | free  | position (m)             |
# | 1   | y-coordinate of the torso (centre)                           | -Inf | Inf | torso                            | free  | position (m)             |
# | 2   | z-coordinate of the torso (centre)                           | -Inf | Inf | torso                            | free  | position (m)             |
# | 3   | x-orientation of the torso (centre)                          | -Inf | Inf | torso                            | free  | angle (rad)              |
# | 4   | y-orientation of the torso (centre)                          | -Inf | Inf | torso                            | free  | angle (rad)              |
# | 5   | z-orientation of the torso (centre)                          | -Inf | Inf | torso                            | free  | angle (rad)              |
# | 6   | w-orientation of the torso (centre)                          | -Inf | Inf | torso                            | free  | angle (rad)              |
# | 7   | angle between torso and first link on front left             | -Inf | Inf | hip_1 (front_left_leg)           | hinge | angle (rad)              |
# | 8   | angle between the two links on the front left                | -Inf | Inf | ankle_1 (front_left_leg)         | hinge | angle (rad)              |
# | 9   | angle between torso and first link on front right            | -Inf | Inf | hip_2 (front_right_leg)          | hinge | angle (rad)              |
# | 10  | angle between the two links on the front right               | -Inf | Inf | ankle_2 (front_right_leg)        | hinge | angle (rad)              |
# | 11  | angle between torso and first link on back left              | -Inf | Inf | hip_3 (back_leg)                 | hinge | angle (rad)              |
# | 12  | angle between the two links on the back left                 | -Inf | Inf | ankle_3 (back_leg)               | hinge | angle (rad)              |
# | 13  | angle between torso and first link on back right             | -Inf | Inf | hip_4 (right_back_leg)           | hinge | angle (rad)              |
# | 14  | angle between the two links on the back right                | -Inf | Inf | ankle_4 (right_back_leg)         | hinge | angle (rad)              |
# | 15  | x-coordinate velocity of the torso                           | -Inf | Inf | torso                            | free  | velocity (m/s)           |
# | 16  | y-coordinate velocity of the torso                           | -Inf | Inf | torso                            | free  | velocity (m/s)           |
# | 17  | z-coordinate velocity of the torso                           | -Inf | Inf | torso                            | free  | velocity (m/s)           |
# | 18  | x-coordinate angular velocity of the torso                   | -Inf | Inf | torso                            | free  | angular velocity (rad/s) |
# | 19  | y-coordinate angular velocity of the torso                   | -Inf | Inf | torso                            | free  | angular velocity (rad/s) |
# | 20  | z-coordinate angular velocity of the torso                   | -Inf | Inf | torso                            | free  | angular velocity (rad/s) |
# | 21  | angular velocity of angle between torso and front left link  | -Inf | Inf | hip_1 (front_left_leg)           | hinge | angle (rad)              |
# | 22  | angular velocity of the angle between front left links       | -Inf | Inf | ankle_1 (front_left_leg)         | hinge | angle (rad)              |
# | 23  | angular velocity of angle between torso and front right link | -Inf | Inf | hip_2 (front_right_leg)          | hinge | angle (rad)              |
# | 24  | angular velocity of the angle between front right links      | -Inf | Inf | ankle_2 (front_right_leg)        | hinge | angle (rad)              |
# | 25  | angular velocity of angle between torso and back left link   | -Inf | Inf | hip_3 (back_leg)                 | hinge | angle (rad)              |
# | 26  | angular velocity of the angle between back left links        | -Inf | Inf | ankle_3 (back_leg)               | hinge | angle (rad)              |
# | 27  | angular velocity of angle between torso and back right link  | -Inf | Inf | hip_4 (right_back_leg)           | hinge | angle (rad)              |
# | 28  | angular velocity of the angle between back right links       | -Inf | Inf | ankle_4 (right_back_leg)         | hinge | angle (rad)              |
"""
