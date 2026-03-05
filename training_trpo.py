import logging
import random

import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from stable_baselines3.common.monitor import Monitor
from gymnasium.envs.registration import register

# Ensure env module is importable so Gym can locate the class by entry_point
from envs.enviroment import communication_overEnv  # noqa: F401

from Agent.callback import RewardCallback
from Agent.trpo_agent import build_trpo_agent


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
    entry_point="envs.enviroment:communication_overEnv",
)


# =========================
# Configuration
# =========================
seed_value = 300
learning_rates = [0.0001, 0.0005, 0.001]

num_nodes = 60
num_vnfs = 5

total_timesteps = 1_000_000
gamma = 0.99

# TRPO hyperparameters
n_steps = 1024
batch_size = 128
gae_lambda = 0.95

combined_csv = "TRPO_training_results_all_lrs.csv"


def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)


def make_env():
    env = gym.make("CommunicationOver-v2", num_nodes=num_nodes, num_vnfs=num_vnfs)
    env = Monitor(env)

    # Your original code attempts to override a bad default env seed
    # Keep it safe across different env reset signatures
    try:
        env.reset(seed=None)
    except TypeError:
        try:
            env.reset()
        except Exception:
            pass

    return env


def train_and_evaluate(learning_rate: float) -> pd.DataFrame:
    logger.info(f"\n=== Training TRPO with learning_rate = {learning_rate} ===")

    set_seeds(seed_value)
    env = make_env()

    model = build_trpo_agent(
        env,
        seed_value=seed_value,
        learning_rate=learning_rate,
        gamma=gamma,
        n_steps=n_steps,
        batch_size=batch_size,
        gae_lambda=gae_lambda,
        verbose=1,
    )

    callback = RewardCallback()
    model.learn(total_timesteps=total_timesteps, callback=callback)

    per_lr_csv = f"TRPO_training_rewards_lr_{str(learning_rate).replace('.', '')}.csv"

    df_train = pd.DataFrame({
        "Learning Rate": [learning_rate] * len(callback.episode_rewards),
        "Episode": list(range(len(callback.episode_rewards))),
        "Reward": callback.episode_rewards,
        "Migration Cost": callback.episode_mig_costs[:len(callback.episode_rewards)],
    })

    window_size = 10
    df_train["Average Reward"] = df_train["Reward"].rolling(window=window_size).mean()

    df_train.to_csv(per_lr_csv, index=False)
    logger.info(f"Saved: {per_lr_csv}")

    env.close()
    return df_train


def plot_results(df_results: pd.DataFrame) -> None:
    plt.figure(figsize=(12, 6))
    for lr in sorted(df_results["Learning Rate"].unique()):
        subset = df_results[df_results["Learning Rate"] == lr]
        plt.plot(subset["Episode"], subset["Average Reward"], label=f"LR={lr}")

    plt.title("Average Training Reward Over Episodes (Different Learning Rates)", fontsize=16)
    plt.xlabel("Episode", fontsize=14)
    plt.ylabel("Average Reward", fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    results = []
    for lr in learning_rates:
        results.append(train_and_evaluate(lr))

    df_results = pd.concat(results, ignore_index=True)
    df_results.to_csv(combined_csv, index=False)
    logger.info(f"\nSaved combined results CSV: {combined_csv}")

    plot_results(df_results)


if __name__ == "__main__":
    main()