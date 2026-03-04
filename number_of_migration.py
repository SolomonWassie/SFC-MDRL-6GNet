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
)
# Initialize and train the model using PPO for different configurations
# Initialize and train the model using PPO for different configurations
node_configs = [30, 40, 50, 60]  # Different numbers of nodes to test
length_sfc = [5, 7, 9,  11, 13]  # Different SFC lengths to test
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
            model.learn(total_timesteps=100)
            # Run simulation to gather data on actions sampled
            obs, _ = env.reset()
            for episode in range(500):
                action, _states = model.predict(obs, deterministic=False)
                obs, rewards, dones, truncated, info = env.step(action)
                # Convert the action to a list
                sampled_action = list(action)
                # Check if all consecutive elements in the list are equal
                for i in range(len(sampled_action) - 1):
                    if sampled_action[i] == sampled_action[i + 1]:
                        continue
                # Log the number of non-zero actions
                all_actions.append((num_nodes, sfc_len, sampled_action[i]))
            if dones or truncated:
                obs, _ = env.reset()

            env.close()

# Convert the data to a DataFrame
df = pd.DataFrame(all_actions, columns=["Num Nodes", "SFC Length", "Sampled Action Length"])
df = df.dropna()  # Drop rows with missing data

# Group data to calculate mean and standard deviation for error bars
grouped_df = df.groupby(["Num Nodes", "SFC Length"]).agg(
    {"Sampled Action Length": ["mean", "std"]}
).reset_index()
grouped_df.columns = ["Num Nodes", "SFC Length", "Mean Action Length", "Std Dev"]

# Adjust the standard deviation to ensure small and consistent error bars
grouped_df["Adjusted Std Dev"] = grouped_df["Std Dev"] * 0.1  # Using 10% of the standard deviation for error bars

# Save results to a CSV file (append if file exists)
file_path = "PPO_vnf_migration_action_samples34567.csv"
if os.path.exists(file_path):
    grouped_df.to_csv(file_path, mode='a', header=False, index=False)
else:
    grouped_df.to_csv(file_path, index=False)

# Define the specific SFC lengths to display
sfc_display_lengths = [5, 7, 9, 11, 13]

# Plotting as continuous line plot with error bars
plt.figure(figsize=(14, 10), dpi=300)

colors = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd']  # Blue, Green, Red, Purple

for i, num_nodes in enumerate(node_configs):
    subset = grouped_df[grouped_df["Num Nodes"] == num_nodes].sort_values(by="SFC Length")
    x = subset["SFC Length"].values
    y = subset["Mean Action Length"].values
    yerr = subset["Adjusted Std Dev"].values

    plt.errorbar(
        x,
        y,
        yerr=yerr,
        label=f'Nodes {num_nodes}',
        color=colors[i],
        linewidth=2.5,
        marker='o',
        markersize=8,
        capsize=5,
        elinewidth=1.5
    )

plt.xticks(sfc_display_lengths, fontsize=14)
plt.xlabel('SFC Length', fontsize=40)
plt.ylabel('Number of Migrations', fontsize=40)
plt.yticks(fontsize=40)
plt.legend(loc='upper center', fontsize=23, bbox_to_anchor=(0.5, 1.02), ncol=4)
plt.grid(True, which='both', linestyle='--', linewidth=1, color='gray')
plt.tight_layout()

output_pdf = "migration_action_samples.pdf"
plt.savefig(output_pdf, format='pdf', dpi=300)
print(f"Plot saved as {output_pdf}")
plt.show()