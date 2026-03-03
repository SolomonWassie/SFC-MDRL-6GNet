**SFC-MDRL-6GNet**

SFC-MDRL-6GNet is a deep reinforcement learning–based framework for context-aware online Service Function Chain (SFC) deployment and stateful Virtual Network Function (VNF) migration across the Cloud Continuum in 6G network architectures.

The framework enables intelligent, network-state–adaptive VNF placement and migration across Extreme Edge, Edge, and Central Cloud environments within the Cloud Continuum Framework (CCF).

Unlike traditional heuristic or greedy methods, SFC-MDRL-6GNet continuously monitors time-varying user traffic, infrastructure resource availability, and link conditions to dynamically optimize VNF deployment decisions.

**Optimization Objectives**
- **End-to-End (E2E) SFC delay**

- Stateful VNF migration cost
- SLA violation reduction
- VNF request acceptance ratio
- Resource utilization efficiency
- Processing and communication latency
- Migration overhead (context transfer cost, bandwidth usage, energy consumption, SLA penalties)
- Multi-objective optimization formulation modeling the latency–migration trade-off

**Core System Model**

SFC-MDRL-6GNet operates within a 6G architecture composed of:

Integrates Extreme-edge, Far-edge, Near-edge, and Cloud nodes into a unified infrastructure.
Hosts the Service Orchestrator responsible for VNF lifecycle management.



Supports training, deployment, and continuous refinement of DRL models.



Stateful VNFs requiring context-aware migration across distributed nodes

**Learning Architecture**

SFC-MDRL-6GNet formulates VNF deployment as a Markov Decision Process (MDP) and employs:

Proximal Policy Optimization (PPO)

The learning framework includes:

Policy Network → selects optimal physical nodes for VNF placement

Value Network → estimates long-term expected return

Advantage-based policy updates for stable and reliable convergence

RL Agent Capabilities

The agent:

Observes current VNF placements

Monitors link latency and congestion

Tracks CPU, bandwidth, and memory usage

Evaluates SLA constraints

Predicts optimal reconfiguration and migration actions

**Context-Aware Stateful Migration**

Each VNF is modeled as stateful, requiring migration of context information when relocated.

Migration cost considers:

Context size

Path bandwidth

Link congestion

SLA violation impact

Deployment cost

Total migration time

This enables:

Long-term cost-aware relocation decisions

Reduced unnecessary migrations

Adaptive scaling under fluctuating traffic demands

**Dynamic & Scalable Environment**

SFC-MDRL-6GNet simulates:

Time-varying Poisson traffic arrivals

Heterogeneous compute resources

Variable link capacities

Large-scale network topologies (10–60 nodes)

SFC lengths up to 13 VNFs

📊 **Performance Gains**

Compared to greedy baseline approaches, the framework achieves:

28.8% lower end-to-end delay

34% lower migration overhead


