# visualize.py

import matplotlib.pyplot as plt
import numpy as np

def plot_path(env, agent):
    grid = np.zeros((env.size, env.size))

    for obs in env.obstacles:
        grid[obs] = -1

    grid[env.goal] = 2

    state = env.reset()
    path = [env.agent_pos]

    for _ in range(50):
        action = np.argmax(agent.q_table[state])
        state, _, done = env.step(action)
        path.append(env.agent_pos)

        if done:
            break

    for (x, y) in path:
        grid[x][y] = 1

    plt.imshow(grid)
    plt.title("Path Planning Result")
    plt.colorbar()
    plt.show()