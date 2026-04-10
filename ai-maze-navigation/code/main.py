# main.py

from train import train
from env import GridWorld
from visualize import plot_path

def main():
    agent = train()

    env = GridWorld()
    plot_path(env, agent)

if __name__ == "__main__":
    main()