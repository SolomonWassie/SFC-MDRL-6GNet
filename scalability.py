import os
import logging
import gymnasium as gym
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from gymnasium.envs.registration import register
from envs.enviroment import communication_overEnv  # Ensure you have your custom env in a file named main_environment.py

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

register(
    id='communication_overEnv-v2',
    entry_point='main_enviroment:communication_overEnv',
    max_episode_steps=1000
)

# Initialize and train the model using PPO for different configurations
node_configs = [10, 25, 35, 45]  # Different numbers of nodes to test
length_sfc = [3, 5, 7, 9, 12]  # Different SFC lengths to test
all_actions = []

for num_nodes in node_configs:
    for sfc_len in length_sfc:
        class RewardAndActionCallback(BaseCallback):
            def __init__(self, env, num_nodes, sfc_len, verbose=0):
                super(RewardAndActionCallback, self).__init__(verbose)
                self.env = env
                self.num_nodes = num_nodes
                self.sfc_len = sfc_len
                self.episode_rewards = []
                self.total_actions_sampled = 0
                self.actions_sampled = 0

            def _on_step(self) -> bool:
                # Initialize episode_reward with a default value
                episode_reward = None

                if 'episode' in self.locals['infos'][0]:  # Check if episode data is available
                    episode_reward = self.locals['infos'][0]['episode']['r']
                    self.episode_rewards.append(episode_reward)

                # Log the actions sampled at this step
                if isinstance(self.env.action_space, gym.spaces.MultiDiscrete):
                    self.actions_sampled = np.prod(self.env.action_space.nvec)
                    self.total_actions_sampled += self.actions_sampled

                    # Ensure episode_reward has a value before appending to all_actions
                    all_actions.append((self.num_nodes, self.sfc_len, self.total_actions_sampled, episode_reward))

                return True

        # Training loop for different configurations
        env = gym.make("communication_overEnv-v2", num_nodes=num_nodes, num_vnfs=sfc_len)
        env = Monitor(env)

        model = PPO(
            'MlpPolicy', env, verbose=1, batch_size=128, n_epochs=100,
            learning_rate=0.0005, gamma=0.99, seed=300
        )

        # Callback to record rewards and actions sampled
        callback = RewardAndActionCallback(env, num_nodes, sfc_len)

        # Train the model
        model.learn(total_timesteps=100, callback=callback)


        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100, deterministic=True)
        logger.info(f'PPO Validation for {num_nodes} nodes, SFC {sfc_len}: mean_reward = {mean_reward:.2f} +/- {std_reward:.2f}')

# Convert actions and rewards data to DataFrame
df = pd.DataFrame(all_actions, columns=["Num Nodes", "SFC Length", "Total Actions Sampled", "Episode Reward"])

# Save results to a CSV file
file_path = "PPO_vnf_migration_action_samples22.csv"
df.to_csv(file_path, index=False)
logger.info(f'Training data saved to {file_path}')

# Define the window size for the moving average
window_size = 500  # Adjust this based on the length of your data

# Calculate moving averages for rewards
df['Episode Reward MA'] = df.groupby(["Num Nodes", "SFC Length"])["Episode Reward"].transform(
    lambda x: x.rolling(window=window_size, min_periods=1).mean())

# Plotting the moving average of episode rewards
plt.figure(figsize=(12, 6))
for num_nodes in node_configs:
    for sfc_len in length_sfc:
        subset = df[(df["Num Nodes"] == num_nodes) & (df["SFC Length"] == sfc_len)]
        plt.plot(subset["Total Actions Sampled"], subset["Episode Reward MA"], label=f'{num_nodes} Nodes, SFC {sfc_len}', marker='o')

plt.title('Moving Average of Episode Rewards for Different Node Configurations and SFC Lengths', fontsize=16)
plt.xlabel('Total Actions Sampled', fontsize=14)
plt.ylabel('Episode Reward (Moving Average)', fontsize=14)
plt.legend()
plt.grid(True)
plt.show()
