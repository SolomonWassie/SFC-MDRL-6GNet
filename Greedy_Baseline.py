import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd
import os
from matplotlib.ticker import ScalarFormatter

def create_network_graph(num_nodes):
    g = nx.Graph()
    nodes = list(range(num_nodes))
    g.add_nodes_from(nodes)
    
    # Create a circular connected structure
    edges = [(i, (i + 1) % num_nodes) for i in range(num_nodes)]
    
    # Add some additional edges to make the graph more complex but structured
    additional_edges = [(i, (i + random.randint(2, 10)) % num_nodes) for i in range(num_nodes)]
    edges.extend(additional_edges)
    
    g.add_edges_from(edges)

    # Generate node attributes
    node_attributes = {
        i: {'cpu': random.uniform(1.5, 7.0), 'ram': random.uniform(2, 22), 'storage': random.uniform(5, 60)}
        for i in range(num_nodes)
    }
    nx.set_node_attributes(g, node_attributes)
    
    # Generate edge attributes
    edge_attributes = {
        (u, v): {'bandwidth': f'{random.randint(60, 100)} Gbps', 'latency': random.randint(3, 10)}
        for u, v in g.edges
    }
    nx.set_edge_attributes(g, edge_attributes)
    
    return g

def calculate_latency(network, path):
    return sum(network[u][v]['latency'] for u, v in zip(path[:-1], path[1:]))

def greedy_allocation(network, time_steps):
    allocation = {}
    latencies = []
    best_nodes = []
    
    # Precompute shortest paths for all pairs of nodes
    shortest_paths = dict(nx.all_pairs_dijkstra_path(network, weight='latency'))

    for t in range(time_steps):
        best_path = None
        best_latency = None

        src, dst = random.sample(list(network.nodes), 2)
        
        while src in allocation.values() or dst in allocation.values():
            src, dst = random.sample(list(network.nodes), 2)

        try:
            path = shortest_paths[src][dst]
            path_latency = calculate_latency(network, path)

            reference_latency_to_node = float('inf')
            for alloc_src in allocation.values():
                for node in [src, dst]:
                    if alloc_src != node:
                        try:
                            alloc_path = shortest_paths[alloc_src][node]
                            alloc_path_latency = calculate_latency(network, alloc_path)
                            reference_latency_to_node = min(reference_latency_to_node, alloc_path_latency)
                        except KeyError:
                            continue
            
            total_latency = min(reference_latency_to_node, path_latency)

            if best_latency is None or total_latency < best_latency:
                best_latency = total_latency
                best_path = (src, dst)
            
            if best_path:
                allocation[t] = best_path
                latencies.append(best_latency)
                best_nodes.append(best_path)
            else:
                latencies.append(float('inf'))
        except KeyError:
            latencies.append(latencies if latencies else float('inf'))

    return allocation, latencies, best_nodes

def plot_latency(latencies, window_size=10, save_path=None):
     # Calculate the moving average
    latencies_series = pd.Series(latencies)
    moving_avg = latencies_series.rolling(window=window_size).mean()  

def plot_latency_and_reward(latencies, ppo_data, window_size=10, save_path=None):
    # Filter out infinite latencies and negative values
    filtered_latencies = [latency for latency in latencies if not np.isinf(latency)]
    
    # Calculate the moving average to smooth the data
    smoothed_latencies = pd.Series(filtered_latencies).rolling(window=window_size).mean()

    fig, ax1 = plt.subplots(figsize=(16, 10))

    # Plot smoothed latencies
    ax1.plot(range(1, len(smoothed_latencies) + 1), smoothed_latencies, linestyle='-', color='r', linewidth=3, label='Greedy allocation')
    ax1.set_xlabel('Time Step', fontsize=28)
    ax1.set_ylabel('Reward', fontsize=28)

    # Normalize the PPO rewards to the same scale as latencies
    for label, data in ppo_data.items():
        normalized_rewards = (data["Average Reward"] - data["Average Reward"].min()) / (data["Average Reward"].max() - data["Average Reward"].min()) * (max(filtered_latencies) - min(filtered_latencies)) + min(filtered_latencies)
        # Plot PPO rewards over time for each dataset
        ax1.plot(data.index, normalized_rewards, label=f'PPO reward with lr {label}', linestyle='-', linewidth=4)

    ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax1.ticklabel_format(style='sci', axis='y', scilimits=(3, 3))
    ax1.yaxis.get_offset_text().set_fontsize(24)
    
    # Explicitly set the font size for x and y tick labels
    ax1.tick_params(axis='x', labelsize=24)
    ax1.tick_params(axis='y', labelsize=24)
    
    fig.tight_layout()  # Adjust layout to prevent overlap
    
    # Center the legend within the plot and push it slightly down
    ax1.legend(loc='upper center', fontsize=24, bbox_to_anchor=(0.5, 0.95), ncol=2)
    
    plt.grid(True)  # Enable grid for better readability
    
    if save_path:
        plt.savefig(save_path, format='pdf')  # Save as PDF if path is provided
    plt.show()

if __name__ == "__main__":
    network = create_network_graph(num_nodes=160)  # Create a network graph with 160 nodes
    allocation, latencies, best_nodes = greedy_allocation(network, time_steps=1000)  # Perform greedy allocation
    # Plot latency and save to PDF
    plot_latency(latencies, save_path='latency_over_time.pdf')

    # Define the path to the results folder
    results_folder = 'results'

    # Define the CSV file names containing PPO training rewards
    csv_files = [
        'PPO_vnf_migration_training_rewards_lr_0.0001.csv', 
        'PPO_vnf_migration_training_rewards_lr_0.003.csv', 
        'PPO_vnf_migration_training_rewards_lr_0.0005.csv'
    ]

    # Initialize a dictionary to hold dataframes
    dataframes = {}

    # Load each CSV file into a dataframe
    for file in csv_files:
        file_path = os.path.join(results_folder, file)
        df = pd.read_csv(file_path)  # Read CSV file into a DataFrame
        learning_rate = file.split('_')[-1].replace('.csv', '')  # Extract learning rate from the file name
        dataframes[learning_rate] = df  # Store DataFrame in the dictionary with learning rate as key

    # Plot the combined latency and PPO rewards and save to PDF
    plot_latency_and_reward(latencies, dataframes, save_path='latency_and_ppo_rewards.pdf')
