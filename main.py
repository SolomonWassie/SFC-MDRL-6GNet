import logging
import time
import random
import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from gymnasium.envs.registration import register

# Import your environment class (make sure this file name is correct)
from envs.enviroment import communication_overEnv  # change if your filename is different


# =========================
# Logging
# =========================
logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
logger = logging.getLogger(__name__)


# =========================
# Register the environment
# =========================
register(
    id="CommunicationOver-v2",
    entry_point="main_enviroment:communication_overEnv",  # filename:ClassName
)


# =========================
# Configuration
# =========================
seed_value = 300
learning_rates = [0.0001, 0.0005, 0.001]

num_nodes = 60
num_vnfs = 5

total_timesteps = 1000000
batch_size = 256
n_epochs = 100
gamma = 0.99

# Output CSV filenames (no log_dir folder)
combined_csv = "PPO_training_results_all_lrs.csv"

results = []


# =========================
# Custom callback to track episode rewards + extra info
# =========================
class RewardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_mig_costs = []
        self.episode_energies = []

    def _on_step(self) -> bool:
        if self.locals["dones"][0]:
            info = self.locals["infos"][0]

            # Monitor adds: info["episode"] = {"r","l","t"}
            ep_info = info.get("episode", {})
            reward = ep_info.get("r", None)

            if reward is not None:
                self.episode_rewards.append(reward)

            # Your env adds these
            self.episode_mig_costs.append(info.get("mig_cost", np.nan))
            self.episode_energies.append(info.get("energy", np.nan))

            logger.info(
                f"Episode finished | Reward={reward} | mig_cost={info.get('mig_cost')} | energy={info.get('energy')}"
            )
        return True


# =========================
# Train & evaluate function
# =========================
def train_and_evaluate(learning_rate: float):
    logger.info(f"\n=== Training PPO with learning_rate = {learning_rate} ===")

    # Reproducibility
    np.random.seed(seed_value)
    random.seed(seed_value)

    # Fresh environment per LR
    env = gym.make("CommunicationOver-v2", num_nodes=num_nodes, num_vnfs=num_vnfs)
    env = Monitor(env)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        batch_size=batch_size,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        gamma=gamma,
        seed=seed_value,
        tensorboard_log=None,  # <-- removed log directory
    )

    callback = RewardCallback()

    start_time = time.time()
    model.learn(total_timesteps=total_timesteps, callback=callback)
    training_time = time.time() - start_time

    # Save per-learning-rate CSV (in current folder)
    per_lr_csv = f"PPO_training_rewards_lr_{str(learning_rate).replace('.', '')}.csv"

    df_train = pd.DataFrame({
        "Learning Rate": [learning_rate] * len(callback.episode_rewards),
        "Episode": list(range(len(callback.episode_rewards))),
        "Reward": callback.episode_rewards,
        "Migration Cost": callback.episode_mig_costs[:len(callback.episode_rewards)],
        "Energy": callback.episode_energies[:len(callback.episode_rewards)],
        "Training Time (seconds)": [training_time] * len(callback.episode_rewards),
    })

    window_size = 10
    df_train["Average Reward"] = df_train["Reward"].rolling(window=window_size).mean()

    df_train.to_csv(per_lr_csv, index=False)
    logger.info(f"Saved: {per_lr_csv} | Training Time: {training_time:.2f} sec")

    results.append(df_train)

    env.close()


# =========================
# Train for each learning rate
# =========================
for lr in learning_rates:
    train_and_evaluate(lr)


# =========================
# Save all results together (combined CSV)
# =========================
df_results = pd.concat(results, ignore_index=True)
df_results.to_csv(combined_csv, index=False)
logger.info(f"\nSaved combined results CSV: {combined_csv}")


# =========================
# Plotting (Average Reward)
# =========================
plt.figure(figsize=(12, 6))
for lr in learning_rates:
    subset = df_results[df_results["Learning Rate"] == lr]
    plt.plot(subset["Episode"], subset["Average Reward"], label=f"LR={lr}")

plt.title("Average Training Reward Over Episodes (Different Learning Rates)", fontsize=16)
plt.xlabel("Episode", fontsize=14)
plt.ylabel("Average Reward", fontsize=14)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
