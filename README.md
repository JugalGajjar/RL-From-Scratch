# Reinforcement Learning From Scratch
*A complete hands-on roadmap to learn Reinforcement Learning (RL) — from first principles to state-of-the-art.*

[![Website](https://img.shields.io/badge/website-live-success?style=for-the-badge&logo=github)](https://jugalgajjar.github.io/RL-From-Scratch)
[![Python](https://img.shields.io/badge/python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3110/)

---

## 📘 Overview

This repository is a **step-by-step learning journey** through the foundations and frontiers of Reinforcement Learning (RL).  
It is designed for learners, researchers, and educators who want to:

- Understand **core RL theory** — MDPs, Bellman equations, policies, and value functions  
- Build **hands-on implementations** of algorithms from scratch  
- Explore **modern research directions** like GRPO, DPO, DreamerV3, Decision Transformers, and GFlowNets  
- Apply RL to **real-world problems** in robotics, games, finance, and AI agents  
- Create an **open-source educational resource** for the RL community  

Each phase includes:
- 📄 Theory & Math Intuition  
- 🧩 Hands-on Code (Jupyter Notebooks & PyTorch Implementations)  
- 📊 Visualizations & Learning Curves  
- 🔍 References to Papers, Lectures, and Courses  

---

## 🌐 Website & Docs

A companion **website** (hosted via **GitHub Pages**) mirrors this repo with clean tutorials, diagrams, and interactive visuals.

- Live site: [jugalgajjar.github.io/RL-From-Scratch/](https://jugalgajjar.github.io/RL-From-Scratch/)
- Source: [`docs/`](./docs)
- Built with **Jekyll** and auto-deployed via **GitHub Actions**

> You can learn entirely from the website or browse the code in this repo—both are kept in sync.

---

## 🗺️ Learning Roadmap

| Phase | Title | Description |
|:------|:-------|:-------------|
| **0** | [Prerequisites](./00-Prerequisites) | Probability, Linear Algebra, Calculus, Python, ML & DL basics, Markov Chains |
| **1** | [Fundamentals](./01-Fundamentals) | MDPs, Bellman equations, dynamic programming, Monte Carlo, TD learning |
| **2** | [Value-Based Methods](./02-Value-Based) | SARSA, Q-Learning, exploration, and function approximation |
| **3** | [Policy-Based Methods](./03-Policy-Based) | Policy Gradients, REINFORCE, Actor-Critic, A2C |
| **4** | [Deep Reinforcement Learning](./04-DeepRL) | DQN, DDQN, Dueling DQN, DDPG, TD3, PPO, SAC |
| **5** | [Advanced Topics](./05-Advanced) | Model-Based RL, Hierarchical RL, Multi-Agent RL, Meta-RL, Offline RL, RLHF |
| **6** | [SOTA RL & Research Trends](./06-SOTA-RL) | GRPO, DPO, DreamerV3, Decision Transformer, GFlowNets, Diffusion Policies |
| **7** | [Applications](./07-Applications) | Robotics, Games, Recommenders, Finance, Code Agents |

---

## 🧩 Phase-by-Phase Breakdown

### **Phase 0 — Prerequisites**
> Build mathematical and programming foundations.

**Topics**
- Probability, Statistics, Linear Algebra, Calculus  
- Gradient Descent and Optimization  
- Python Essentials (scikit-learn, PyTorch, Gymnasium)  
- Basic Machine Learning (Regression, Classification, Evaluation Metrics)
- Basic Deep Learning (Neuron, MLP, CNN)
- Introduction to Markov Chains

**Mini Projects**
- End-to-End Linear Regression 
- Image Classification using CNN 

📁 Folder: [`00-Prerequisites`](./00-Prerequisites)

---

### **Phase 1 — Reinforcement Learning Fundamentals**
> Learn how agents interact with environments to maximize reward.

**Topics**
- Agent–Environment Interface  
- MDPs & Bellman Equations  
- Dynamic Programming (Policy/Value Iteration)  
- Monte Carlo & Temporal-Difference Methods  

**Projects**
- GridWorld (Policy Iteration)  
- Blackjack (Monte Carlo Estimation)  

📁 Folder: [`01-Fundamentals`](./01-Fundamentals)

---

### **Phase 2 — Value-Based Methods**
> Learn how to estimate optimal action values.

**Topics**
- ε-Greedy Exploration  
- SARSA (On-Policy TD Control)  
- Q-Learning (Off-Policy TD Control)  
- Experience Replay  
- Function Approximation (Linear / NN)  

**Projects**
- FrozenLake-v1 (Tabular Q-Learning)  
- CartPole-v1 (NN-based Q-Learning)

📁 Folder: [`02-Value-Based`](./02-Value-Based)

---

### **Phase 3 — Policy-Based Methods**
> Directly learn parameterized policies without value tables.

**Topics**
- Policy Gradients  
- REINFORCE  
- Variance Reduction (Baselines)  
- Actor-Critic (A2C)

**Projects**
- MountainCarContinuous-v0 (REINFORCE)  
- CartPole (A2C)

📁 Folder: [`03-Policy-Based`](./03-Policy-Based)

---

### **Phase 4 — Deep Reinforcement Learning**
> Integrate neural networks for complex, high-dimensional control.

**Topics**
- Deep Q-Networks (DQN) + Stability Tricks  
- Double DQN, Dueling DQN, Prioritized Replay  
- Continuous Control: DDPG, TD3  
- PPO (Proximal Policy Optimization)  
- SAC (Soft Actor-Critic)

**Projects**
- Atari Pong (DQN)  
- LunarLanderContinuous-v2 (DDPG/PPO)

📁 Folder: [`04-DeepRL`](./04-DeepRL)

---

### **Phase 5 — Advanced Topics**
> Modern RL variants and complex systems.

**Topics**
- Model-Based RL (Dyna-Q, World Models)  
- Hierarchical RL (Options)  
- Multi-Agent RL (Cooperative/Competitive)  
- Meta-RL & Few-Shot  
- Offline RL / Imitation Learning  
- RLHF (Reinforcement Learning from Human Feedback)

**Projects**
- MuJoCo Hopper-v4 (Model-Based RL)  
- Cooperative MARL Simulation  
- Mini RLHF Text Summarizer  

📁 Folder: [`05-Advanced`](./05-Advanced)

---

### **Phase 6 — SOTA RL & Research Trends**
> Cutting-edge algorithms in modern AI systems.

- **GRPO** — Generalized Reinforcement Policy Optimization (adaptive KL-regularized policy optimization)  
- **DPO** — Direct Preference Optimization (supervised alternative to RLHF using pairwise preferences)  
- **DreamerV3** — Latent world models & imagination-based training for continuous control  
- **Decision Transformer** — Treat RL as sequence modeling over trajectories  
- **GFlowNets** — Sample trajectories proportional to reward for structured exploration  
- **Diffusion Policies** — Diffusion-based action generation for robust control  
- **RLHF** — Reward modeling + policy optimization with human feedback

📁 Folder: [`06-SOTA-RL`](./06-SOTA-RL)

---

### **Phase 7 — Applications**
> Real-world and cross-domain problems.

**Domains**
- Robotics (control & navigation)  
- Game AI (Unity ML-Agents, Chess, Go)  
- Recommenders  
- Finance (Trading & Portfolio)  
- Code Agents & Autonomous Systems  

**Projects**
- Robotic Arm (PyBullet)  
- Stock Trading Bot (PPO)  
- Game Agent with Custom Rewards  

📁 Folder: [`07-Applications`](./07-Applications)

---

## 🧱 Repository Structure

```
RL-From-Scratch/
│
├── 00-Prerequisites/
├── 01-Fundamentals/
├── 02-Value-Based/
├── 03-Policy-Based/
├── 04-DeepRL/
├── 05-Advanced/
├── 06-SOTA-RL/
├── 07-Applications/
├── docs/
├── LICENSE
└── README.md
```

---

## 🧭 How to Use This Repository

1. Clone the repo:
   ```bash
   git clone https://github.com/JugalGajjar/RL-From-Scratch.git
   cd RL-From-Scratch
   ```
2. Learn sequentially from `00-Prerequisites` → `07-Applications`.
3. You can use the Jupyter notebooks or use the (https://jugalgajjar.github.io/RL-From-Scratch/)[Website].

---

## 🤝 Contributing

Contributions are welcome!
If you’d like to add an implementation, visualization, or paper reproduction:
1. Fork the repository
2. Create a feature branch
3. Submit a PR with a short explanation and example outputs

---

## 🧠 License & Citation

This repository is open-sourced under the **MIT License**.  
If you use this project in your research, coursework, or publications, please cite it as:

> **Jugal Gajjar. (2025). JugalGajjar/RL-From-Scratch: v0.7 (v0.7). Zenodo.**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17693816.svg)](https://doi.org/10.5281/zenodo.17693816)

---

## 📬 Contact

Feel free to reach out for questions, collaboration, or feedback:
- **Email:** [812jugalgajjar@gmail.com](mailto:812jugalgajjar@gmail.com)  
- **LinkedIn:** [linkedin.com/in/jugalgajjar](https://www.linkedin.com/in/jugalgajjar)  
- **Portfolio:** [jugalgajjar.github.io](https://jugalgajjar.github.io)
