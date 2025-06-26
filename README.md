# SAC Path Planning for Differential Drive Robot (MATLAB)

This project implements Soft Actor-Critic (SAC) for path planning and obstacle avoidance of a differential drive robot in a 2D environment using MATLAB. The robot learns to navigate from a start point to a goal while avoiding circular obstacles using lidar-like sensors.

## 🚀 Features

- Soft Actor-Critic (SAC) from scratch using MATLAB
- Custom environment and agent implementation
- Simulated 2D map with adjustable goal and obstacles
- Lidar-like perception with configurable ray count
- Neural network-based Actor-Critic model (no toolbox required)
- Support for training and testing with visualization
- Automatic model saving/loading (`Trained.mat`)
- Reward plot and performance logging

## 📁 Project Structure

```
├── Agent.m # SAC Agent (Actor, Critic, Alpha update)
├── Environment.m # Environment (Obstacle map, Lidar, Dynamics)
├── Train.m # Training script
├── Test.m # Evaluation/Testing script
├── Trained.mat # Saved model (automatically created)
├── Reward *.png # Training reward graph (saved image)
├── Map *.png # Environment visualization (saved image)
└── Path *.png # Robot trajectory visualization (saved image)
```

## 🧠 Algorithm Overview

This project uses **Soft Actor-Critic (SAC)**, an off-policy reinforcement learning algorithm, which combines:

- **Stochastic Actor-Critic architecture**
- **Entropy regularization** for exploration
- **Twin Q-networks** for reducing overestimation
- **Target networks** for stable updates
- **Replay buffer** and mini-batch training

## 📊 Inputs & State Representation

The input state includes:

- Robot position and heading (x, y, θ)
- Relative goal position
- Lidar scan distances (normalized)

The action space is **2D continuous**, corresponding to left and right wheel speeds.

## 🏃‍♂️ How to Run

### 1. Train the Agent

Run the training script:

```matlab
Train.m
```

- The model will be trained using SAC.
- The weights will be saved in Trained.mat.
- Rewards and performance metrics are printed per episode.
- You can enable visualization by uncommenting the env.plot() line.

### 2. Test the Trained Agent

Run the training script:

```matlab
Test.m
```

- Evaluates the agent using the trained model.
- Outputs reward, Q-values, and average speed.
- Generates path and environment plots (*.png).
- Optionally saves video (uncomment related lines in test.m).

## 📦 Requirements

- MATLAB R2021a or later recommended
- No external toolbox needed (Deep Learning toolbox not required)
- Should work with all OS (Windows/Linux/Mac) running MATLAB

## 📈 Training Output Example
Episode: 567   Steps: 98   Total Reward: 920.50   Entropy: 2.35   Alpha: 0.0561   Q1: 930.2   Q2: 928.9   TargetQ1: 920.1   TargetQ2: 919.5

## 🏆 Evaluation

- Trained agent shows smooth and successful navigation
- Avoids obstacles reliably
- Performance improves significantly over episodes
- Can handle randomized goal/obstacle layouts

## 📎 Citation / Attribution

Developed by Ngo Gia Bao as part of Bachelor Graduation Thesis
Department of Automatic Control – Hanoi University of Science and Technology (HUST), 2024

## 📫 Contact
If you have any questions or suggestions, feel free to reach out:
- 📧 Email: ngobao.contact@gmail.com
- 🌐 GitHub: https://github.com/brnbq22
