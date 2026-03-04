import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import networkx as nx
import numpy as np
import random
from envs.network_topology import create_network_graph
class communication_overEnv(gym.Env):
    """
    Custom Gym environment for VNF migration.
    """
    def __init__(self, num_nodes, num_vnfs):
        super(communication_overEnv, self).__init__()
        self.network_graph = create_network_graph(num_nodes)
        self.sfc_graph = self.create_sfc_graph(num_vnfs)
        self.nodes = list(self.network_graph.nodes())
        self.vnf_nodes = list(self.sfc_graph.nodes())
        self.action_space = MultiDiscrete([len(self.nodes) - 1] * len(self.vnf_nodes))
        self.observation_space = MultiDiscrete([len(self.nodes)] * len(self.vnf_nodes))
        self.initial_observation = [0] * len(self.vnf_nodes)
        self.time_step = 0

    def create_sfc_graph(self, num_vnfs):
        """
        Create a graph representing a Service Function Chain (SFC) with a specified number of VNFs and edges,
        and set attributes for nodes and edges.
        """
        sfc_graph = nx.MultiGraph()
        # Add nodes with attributes
        for i in range(num_vnfs):
            sfc_graph.add_node(i, 
                               cpu=f'{1.5 + 0.5 * i} GHz', 
                               ram=f'{2 + 2 * i} GB', 
                               storage=f'{5 + 5 * i} GB', 
                               processing_delay=0.5 + 0.42 * i)
        # Add edges to connect each node to the next one in series
        edges = [(i, i + 1) for i in range(num_vnfs - 1)]
        sfc_graph.add_edges_from(edges)

        return sfc_graph

    def reset(self, seed=300, options=None):
        super().reset(seed=seed)
        random.seed(seed)
        np.random.seed(seed)
        self.current_observation = [0] * len(self.vnf_nodes)
        self.reward = 0
        self.time_step = 0  # Reset the time step
        return np.array(self.current_observation), {}  # Return observation and info

    def step(self, action):
        # Update latencies before each step
        for u, v in self.network_graph.edges():
            self.network_graph.edges[u, v]['latency'] = random.randint(5, 20)
        
        self.initial_observation = action  # Assuming action is correctly formatted
        latency = self.calculate_latency(self.initial_observation)
        delay = self.calculate_processing_latency(self.initial_observation)
        total_latency = latency + delay
        mig_cost=self.calculate_migration_cost(self.initial_observation)
        #obj=-((0.2*mig_cost)+ 0.7*latency)
        self.reward = -total_latency
        observation = np.array(self.initial_observation)
        done = False
        truncated = False
        
        # Include updated latencies and total latency in info
        info = {
            'mig_cost': mig_cost,
        }

        return observation, self.reward, done, truncated, info

    def calculate_latency(self, obs):
        """
        Calculate the minimum latency for paths between nodes given in the observation (obs).
        """
        all_latencies = []
        node_labels = list(obs)

        # Find the minimum latency
        for i in range(len(node_labels) - 1):
            source = self.nodes[node_labels[i]]
            target = self.nodes[node_labels[i + 1]]
            # Check if there is a path between the source and target nodes
            if nx.has_path(self.network_graph, source, target):
                shortest_path = nx.shortest_path(self.network_graph, source, target)
                path_latency = sum(float(self.network_graph[u][v]['latency']) for u, v in zip(shortest_path[:-1], shortest_path[1:]))
                all_latencies.append(path_latency)
        if all_latencies:
            min_latency = min(all_latencies)
            return min_latency
        return 0

    def calculate_processing_latency(self, obs):
        total_processing_latency = 0
        # Set of valid node IDs as strings
        vnf_indices = random.sample(list(self.sfc_graph.nodes()), len(obs))
        # Assuming obs contains the indices of VNFs as they appear in the service function chain
        for vnf_id in vnf_indices:
            # Check if the VNF ID exists in the SFC graph before accessing its attributes
            if vnf_id in self.sfc_graph.nodes:
                vnf_attributes = self.sfc_graph.nodes[vnf_id]
                processing_delay = vnf_attributes.get('processing_delay', 0.0)
                total_processing_latency += processing_delay
            else:
                print(f"VNF ID {vnf_id} does not exist in the graph.")

        return total_processing_latency

    def calculate_migration_cost(self, selected_node):
        total_ram_bits = 0
        total_transfer_time = 0
        for i in range(len(selected_node) - 1):
            source_node = self.nodes[selected_node[i]]
            destination_node = self.nodes[selected_node[i + 1]]

            # Check if source node exists and retrieve RAM, converting GB to bits
            if source_node in self.network_graph.nodes:
                ram_str = self.network_graph.nodes[source_node].get('ram', '0 GB')
                # Convert GB to bits for calculation: 1 GB = 8 * 10^9 bits
                ram_value_bits = int(ram_str.replace(' GB', '')) * 8 
                total_ram_bits += ram_value_bits  # Accumulate total RAM in bits

                # Ensure there is a path and calculate bandwidth in bps (bits per second)
                if nx.has_path(self.network_graph, source_node, destination_node):
                    shortest_path = nx.shortest_path(self.network_graph, source_node, destination_node)
                    path_bandwidths = [
                        float(self.network_graph[u][v].get('bandwidth', '0.1 Gbps')) 
                        for u, v in zip(shortest_path[:-1], shortest_path[1:])
                    ]
                    path_bandwidth = sum(path_bandwidths)

                    # Calculate transfer time using bits and bps
                    if path_bandwidth > 0:
                            transfer_time = total_ram_bits / path_bandwidth
                            total_transfer_time += transfer_time
                    else:
                     continue
                    
                else:
                   continue
            else:
                continue

        if total_transfer_time == 0:
            return 0

        return total_transfer_time
   
   
# Example usage
env = communication_overEnv(num_nodes=60, num_vnfs=5)
obs, _ = env.reset()
