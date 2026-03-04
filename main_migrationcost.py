
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from gymnasium.envs.registration import register
import gymnasium as gym
import numpy as np
from envs.enviroment import communication_overEnv
import os

# Register the environment
register(
    id='communication_overEnv-v2',
    entry_point='main_enviroment:communication_overEnv',
    max_episode_steps=10
)# Initialize and train the model using PPO for different configurations
node_configs = [30, 40, 50, 60]  # Different numbers of nodes to test
length_sfc = [5, 7, 9, 11, 13]   # Different SFC lengths to test
seeds = [10, 20, 30, 100] 
all_actions = []

for num_nodes in node_configs:
    for sfc_len in length_sfc:
        for seed in seeds: 
            # Initialize the environment
            env = gym.make("communication_overEnv-v2", num_nodes=num_nodes, num_vnfs=sfc_len)
            check_env(env)
            # Initialize and train the model using PPO
            model = PPO('MlpPolicy', env, verbose=1, learning_rate=0.0005, gamma=0.99, seed=seed)
            model.learn(total_timesteps=1000)
            
            # Run simulation to gather data on actions sampled
            obs, _ = env.reset()
            for episode in range(500):
                action, _states = model.predict(obs, deterministic=False)
                obs, rewards, dones, truncated, info = env.step(action)
                mig_cost = info.get("mig_cost", 0)
                sampled_action = list(action)
                for i in range(len(sampled_action) - 1):
                    if sampled_action[i] == sampled_action[i + 1]: 
                        continue
                all_actions.append((num_nodes, sfc_len, sampled_action[i], mig_cost))
            if dones or truncated:
                obs, _ = env.reset()
            env.close()

# Convert the data to a DataFrame
df = pd.DataFrame(all_actions, columns=["Num Nodes", "SFC Length", "Sampled Action Length", "Mig Cost"])
df.dropna(inplace=True)
df["Mig Cost"] = df["Mig Cost"].abs()

# Group data to calculate mean migration costs
grouped_df = df.groupby(["Num Nodes", "SFC Length"]).agg({"Mig Cost": "mean"}).reset_index()
grouped_df.columns = ["Num Nodes", "SFC Length", "Mean Mig Cost"]

# Save results to a CSV file (append if file exists)
file_path = "PPO_vnf_migration_costs2.csv"
grouped_df.to_csv(file_path, mode='a' if os.path.exists(file_path) else 'w', header=not os.path.exists(file_path), index=False)

# Define the specific SFC lengths to display
sfc_display_lengths = [5, 7, 9, 11, 13]

# Plotting
plt.figure(figsize=(12, 8), dpi=300)
colors = ['#FF5733', '#33FF57', '#3357FF', '#FF33A1', '#FFA533']
bar_width = 0.15

# Create bar plots
for i, sfc_len in enumerate(sfc_display_lengths):
    subset = grouped_df[grouped_df["SFC Length"] == sfc_len]
    bar_positions = np.arange(len(node_configs)) + i * bar_width
    plt.bar(bar_positions, subset["Mean Mig Cost"], 
            width=bar_width, color=colors[i], 
            edgecolor='black', label=f'SFC Length {sfc_len}')

plt.xticks(np.arange(len(node_configs)) + bar_width * (len(sfc_display_lengths) - 1) / 2, 
           node_configs, fontsize=40)
plt.yticks(fontsize=37)
plt.xlabel('Number of Nodes N', fontsize=37)
plt.ylabel('Service Relocation Cost', fontsize=35)

plt.legend(loc='upper center', bbox_to_anchor=(0.64, 1.02), fontsize=21,
           ncol=2, labelspacing=0.8, columnspacing=1.0, handletextpad=0.8)
plt.grid(True, axis='y')

pdf_file_path = "Migration_Cost.pdf"
plt.tight_layout()
plt.savefig(pdf_file_path, format='pdf', bbox_inches='tight', dpi=300)
plt.show()
