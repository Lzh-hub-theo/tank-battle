# agent.py

import numpy as np
from config import *

class QLearningAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.q_table = np.zeros((state_size, action_size))

        self.epsilon = EPSILON

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)
        return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state):
        best_next = np.max(self.q_table[next_state])

        self.q_table[state][action] += ALPHA * (
            reward + GAMMA * best_next - self.q_table[state][action]
        )

    def decay_epsilon(self):
        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DECAY