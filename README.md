# Quantum-Control-Residual-SAC ⚛️

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Framework: PyTorch](https://img.shields.io/badge/PyTorch-Stable_Baselines3-EE4C2C.svg)](https://pytorch.org/)

> **面向开放量子系统高保真度控制的残差强化学习与动态课程学习框架**  
> A Residual-SAC framework for high-fidelity quantum control in open quantum systems, achieving 93.87% fidelity under strong dissipation.

## 📖 Project Overview (项目简介)
本项目针对中性原子里德堡态（Rydberg atoms）量子计算平台中面临的强马尔可夫耗散（自发辐射与退相干）及随机高斯失谐噪声难题，提出了一种基于深度强化学习（Deep Reinforcement Learning）的量子逻辑门脉冲控制流形重构算法。

基于最大熵连续控制基底（Soft Actor-Critic, SAC），本项目重构了多项核心控制机制，突破了传统数值最优控制（如 GRAPE）在面临高维非凸地形时易陷入局部最优的瓶颈。

## ✨ Core Features (核心工作)
*   **Residual-RL Architecture (残差强化学习):** 融合量子物理先验底色，将全局黑盒探索转化为局部流形微扰，实现 $0$ 步跨越收敛门槛。
*   **Curriculum Learning (动态课程调度):** 实现智能体从纯幺正相干演化空间向强耗散非平稳环境的无缝知识迁移，彻底规避高方差环境下的早期崩溃。
*   **Log-Reward Shaping (对数奖励塑形):** 引入量子最优控制中的对数失真度，有效打破逼近物理极限时面临的线性梯度消失困境。
*   **Hardware-in-the-loop (硬件在环低通滤波):** 动作降维拟设与二阶巴特沃斯滤波器结合，确保输出的“平顶梯形波”满足实验室 AWG 的真实带宽约束。

## 📂 Repository Structure (仓库结构)
*   `quantum_pulse_final.py`: 核心强化学习环境与 SAC 智能体训练、验证推理主程序。
*   `Master_Pulse_Baseline.csv`: 用于残差网络初始化的大宗师（基准）物理底色流形数据。
*   `SAC_MaxNoise_Final_20260527_1248.zip`: 完整版残差 SAC 框架预训练权重（极限保真度：**93.87%**）。
*   `SAC_Ablation_Linear_20260528_1731.zip`: 移除对数奖励塑形后的消融实验模型权重（遭遇梯度天花板，保真度：**92.04%**）。
*   `Ultimate_Pulse_MaxNoise_20260527_1248.png` / `SAC_MaxNoise_20260527_1248_1.png`: 最终收敛波形拓扑与训练过程记录图表。

## 🚀 Quick Start (快速运行)

**1. 依赖安装 (Dependencies)**
建议在 Python 3.10+ 环境下运行，核心依赖包括：
```bash
pip install torch qutip stable-baselines3 gymnasium scipy numpy matplotlib
