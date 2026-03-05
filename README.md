# 6G-DRL-SFC-Placement
## Deep Reinforcement Learning for Context-Aware Online Service Function Chain Deployment and Migration

6G-DRL-SFC-Placement is a research framework for **context-aware Service Function Chain (SFC) deployment and stateful VNF migration** using **Deep Reinforcement Learning (DRL)**.

The framework enables intelligent network-state–adaptive orchestration of Virtual Network Functions (VNFs) across the Cloud Continuum Framework (CCF), which integrates distributed compute resources.

Learning is performed using **Proximal Policy Optimization (PPO)** with policy and value networks interacting with a network environment.

---

# Core Features

- Deep Reinforcement Learning for **SFC deployment and VNF migration**
- Context-aware **network-state adaptive orchestration**
- Multi-objective optimization considering:
  - End-to-End latency
  - Migration overhead
  - SLA constraints
- **VNF migration modeling**
- Dynamic traffic generation
- Heterogeneous compute and communication resources

Implemented using:

- Python  
- OpenAI Gymnasium  
- Stable-Baselines3  
- NetworkX  

---

# System Overview

The CCF provides a unified resource pool that orchestrates distributed computing and networking resources across the continuum from Edge to Central Cloud.
Each node provides heterogeneous computational and communication resources that must be efficiently allocated to support service requests.

---

## Service Function Chains (SFCs)

**Network service** are modeled as Service Function Chains, consisting of ordered Virtual Network Functions such as:

- Network Address Translation (NAT)
- Firewall (FW)
- Traffic Monitor (TM)
- Video Optimization Controller (VOC)
- Intrusion Detection / Prevention Systems (IDPS)

Each SFC request specifies:

- Minimum bandwidth requirement  
- Maximum end-to-end delay  
- Computational resource demand  

The objective is to determine optimal deployment locations for VNFs.

---

## DRL-Based Service Orchestrator

A reinforcement learning agent operates as a **network-aware service orchestrator** that observes the current system state and decides **where VNFs should be deployed or migrated** across the Network.

The DRL agent continuously adapts to changing network conditions to minimize service latency while accounting for migration costs.

---

# Learning Architecture

The deployment problem is formulated as a **Markov Decision Process (MDP)**.

The PPO agent learns an optimal policy using:

### Policy Network
Selects the optimal physical nodes for VNF placement.

### Value Network
Estimates the expected long-term return of network states.

### Advantage-Based Policy Updates
Stabilize learning and ensure reliable policy convergence.

The agent observes the following system information:

- Current VNF deployment locations
- Link latency and congestion
- CPU, bandwidth, and memory utilization
- SLA constraints
- Traffic demand patterns

Based on these observations, the agent predicts optimal **deployment and migration actions**.

---

# Dynamic & Scalable Environment

The simulation environment models realistic network behavior including:

- Time-varying traffic modeled using a **Poisson arrival process**
- Heterogeneous computational resources across nodes
- Variable link capacities reflecting dynamic network conditions
- Network topologies ranging from **10 to 60 nodes**
- Service Function Chains with up to **13 VNFs**

---


# Repository Structure
```
DRL_SFC_PLACEMNT/
│
├── envs/
│   ├── enviroment.py
│   └── network_topology.py
│
├── agents/
│   ├── A2C_agent.py
│   ├── callback.py
│   └── PPO_agent.py
│
├── Greedy_Baseline.py
├── migrationcost.py
├── number_of_migration.py
├── scalability.py
│
├── training_A2C.py
├── training_ppo.py
├── training_trpo
│
├── results/
│
└── README.md
```

# Running the Framework

Install dependencies:
```bash
pip install -r requirements.txt
```

Run training:
```bash
python main.py
```
**Citation**
If you use this framework in your research, please cite:

**Deep Reinforcement Learning for Context-Aware Online Service Function Chain Deployment and Migration over 6G Networks**
ACM/SIGAPP Symposium on Applied Computing (SAC), 2025.
URL: https://dl.acm.org/doi/abs/10.1145/3672608.3707975


@inproceedings{wassie2025deep,
  title={Deep reinforcement learning for context-aware online service function chain deployment and migration over 6g networks},
  author={Wassie, Solomon Fikadie and Di Maio, Antonio and Braun, Torsten},
  booktitle={Proceedings of the 40th ACM/SIGAPP Symposium on Applied Computing},
  pages={1361--1370},
  year={2025}
}