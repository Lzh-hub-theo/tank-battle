# Maze_AI + 强化学习 —— 让ai学会走迷宫
## 开发声明
本项目基于github上原有的**迷宫游戏**进行开发，迷宫游戏原项目地址：[https://github.com/Wonz5130/Maze_AI.git](https://github.com/Wonz5130/Maze_AI.git)

衷心感谢 Wonz5130 提供的迷宫游戏源代码

## 项目描述
本项目是通过pytorch + DQN 网络，让ai学会走迷宫游戏

## 项目进度
* 2026/4/12 完成了ai对地图基于Q-learning的强化学习与小地图游戏的对接
* 2026/4/13 实现了自动绘制reward收敛曲线图、保存ai已经训练好的模型、初步实现了DQN学习
* 2026/4/14 实现了大地图的展示，不过昨天的DQN网络无法正确学习
* 2026/4/22 通过BFS给迷宫网络展示了每一个点到终点的最短路径分数的二维矩阵。这个矩阵用于指导ai走一步哪一步是正确的
* 20206/4/24 通过合理的调整参数+运行，ai在训练过程中开始到达终点。但是目前的ai训练不稳定。


### （九）项目地址

[GitHub](https://github.com/Wonz5130/Maze_AI)

### （十）License

[MIT](https://github.com/Wonz5130/Maze_AI/blob/master/LICENSE)