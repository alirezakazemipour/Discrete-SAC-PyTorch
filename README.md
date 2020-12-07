# Discrete-SAC-PyTorch
Implementation of Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor    
Added another branch for [Soft Actor-Critic Algorithms and Applications ](https://arxiv.org/pdf/1812.05905.pdf) -> [SAC_V1](https://github.com/alirezakazemipour/SAC/tree/SAC_V1).

## Demo
Non-Greedy Stochastic Action Selection| Greedy Action Selection
:-----------------------:|:-----------------------:|:-----------------------:|
![](demo/non-greedy.gif)| ![](demo/greedy.gif)

## Results
> x-axis: episode number.

![](results/running_reward.png)| ![](results/max_episode_reward.png)| ![](results/alpha.png)
:-----------------------:|:-----------------------:|:-----------------------:|
![](results/policy_loss.png)| ![](results/q_loss.png)| ![](results/alpha_loss.png)

## Dependencies
- gym == 0.17.3
- numpy == 1.19.2
- opencv_contrib_python == 4.4.0.44
- psutil == 5.5.1
- torch == 1.6.0

## Reference
1. [_Soft Actor-Critic For Discrete Action Settings_, Christodoulou, 2019](https://arxiv.org/abs/1910.07207)
2. [_Soft Actor-Critic Algorithms and Applications_, Haarnoja et al., 2018](https://arxiv.org/abs/1812.05905)