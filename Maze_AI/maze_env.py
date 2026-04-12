#!/usr/bin/python3
# -*- coding: utf-8 -*-
# 迷宫环境类 - 用于强化学习

import numpy as np
import mapp1

class MazeEnv:
    """基于mapp1.py的迷宫环境"""
    
    def __init__(self):
        self.map_list = [row[:] for row in mapp1.map_list]  # 深拷贝地图
        self.maze_height = len(self.map_list)  # 6
        self.maze_width = len(self.map_list[0])  # 6
        
        # 起点：(5, 2) - 右边的出口
        self.start_pos = (5, 2)
        # 终点：(0, 2) - 左边的入口
        self.goal_pos = (0, 2)
        
        self.reset()
        
        # 动作：0=上 1=下 2=左 3=右
        self.action_size = 4
    
    def reset(self):
        """重置环境"""
        self.agent_pos = self.start_pos
        return self._get_state()
    
    def _get_state(self):
        """将位置转换为状态编号"""
        x, y = self.agent_pos
        return x * self.maze_width + y
    
    def _is_walkable(self, x, y):
        """检查位置是否可以行走"""
        # 检查边界
        if x < 0 or x >= self.maze_width or y < 0 or y >= self.maze_height:
            return False
        # 检查是否是墙（值为1）
        return self.map_list[y][x] != 1
    
    def step(self, action):
        """
        执行动作
        action: 0=上 1=下 2=左 3=右
        返回: (next_state, reward, done)
        """
        x, y = self.agent_pos
        
        # 根据动作计算新位置
        if action == 0:  # 上
            new_y = y - 1
            new_x = x
        elif action == 1:  # 下
            new_y = y + 1
            new_x = x
        elif action == 2:  # 左
            new_x = x - 1
            new_y = y
        elif action == 3:  # 右
            new_x = x + 1
            new_y = y
        else:
            new_x, new_y = x, y
        
        # 检查新位置是否可行走
        if not self._is_walkable(new_x, new_y):
            # 撞墙，不移动
            return self._get_state(), -1.0, False
        
        # 更新位置
        self.agent_pos = (new_x, new_y)
        next_state = self._get_state()
        
        # 检查是否到达目标
        if self.agent_pos == self.goal_pos:
            return next_state, 100.0, True  # 到达目标，获得高奖励
        
        # 每走一步扣一点分（鼓励找最短路径）
        return next_state, -0.1, False
