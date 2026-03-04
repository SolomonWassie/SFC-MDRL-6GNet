# 6G-DRL-SFC-Placement
## Deep Reinforcement Learning for Context-Aware Online Service Function Chain Deployment and Migration over 6G Networks

6G-DRL-SFC-Placement is a research framework for **context-aware Service Function Chain (SFC) deployment and stateful VNF migration** using **Deep Reinforcement Learning (DRL)**.

The framework enables intelligent network-state–adaptive orchestration of Virtual Network Functions (VNFs) across the Cloud Continuum Framework (CCF), which integrates distributed compute resources from:

- Extreme Edge  
- Edge / MEC nodes  
- Central Cloud data centers  

Unlike traditional heuristic or greedy placement approaches, the framework continuously observes **time-varying network conditions** and dynamically learns optimal deployment policies that minimize service delay while accounting for migration overhead.

Learning is performed using **Proximal Policy Optimization (PPO)** with policy and value networks interacting with a simulated network environment.

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

The CCF provides a unified resource pool that orchestrates distributed computing and networking resources across the continuum from Extreme Edge to Central Cloud.

Nodes within the CCF include:

- Extreme-edge devices (e.g., smartphones, IoT devices, AR/VR devices)
- Edge / MEC servers located near base stations
- Central cloud data centers with large-scale computational capacity

Each node provides heterogeneous computational and communication resources that must be efficiently allocated to support service requests.

---

## Service Function Chains (SFCs)

Network services are modeled as **Service Function Chains**, consisting of ordered Virtual Network Functions such as:

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



---

# Running the Framework

Install dependencies:
```bash
pip install -r requirements.txt
```

Run training:
```bash
python main.py
```
---

# Publication

If you use this framework in your research, please cite:

Deep Reinforcement Learning for Context-Aware Online Service Function Chain Deployment and Migration over 6G Networks

ACM/SIGAPP Symposium on Applied Computing (SAC), 2025.