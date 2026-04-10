# env.py

import numpy as np
from config import GRID_SIZE, REWARD_GOAL, REWARD_OBSTACLE, REWARD_STEP

class GridWorld:
    def __init__(self):
        self.size = GRID_SIZE
        self.reset()

        # 固定障碍
        self.obstacles = [(1,1), (2,2), (3,1)]

        self.goal = (4, 4)

    def reset(self):
        self.agent_pos = (0, 0)
        return self.state()

    def state(self):
        return self.agent_pos[0] * self.size + self.agent_pos[1]

    def step(self, action):
        x, y = self.agent_pos

        # 动作：0上 1下 2左 3右
        if action == 0:
            x -= 1
        elif action == 1:
            x += 1
        elif action == 2:
            y -= 1
        elif action == 3:
            y += 1

        # 边界检测
        if x < 0 or x >= self.size or y < 0 or y >= self.size:
            return self.state(), REWARD_OBSTACLE, False

        if (x, y) in self.obstacles:
            return self.state(), REWARD_OBSTACLE, False

        self.agent_pos = (x, y)

        if self.agent_pos == self.goal:
            return self.state(), REWARD_GOAL, True

        return self.state(), REWARD_STEP, False