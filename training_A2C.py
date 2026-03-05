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
from Agent.A2C_agent import build_a2c_agent


# =========================
# Logging
# =========================
logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
logger = logging.getLogger(__name__)


# =========================
# Register the environment
# NOTE: must match the real module path where your class exists.
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

# A2C-specific hyperparameters
n_steps = 5
gae_lambda = 1.0
ent_coef = 0.0
vf_coef = 0.5
max_grad_norm = 0.5

combined_csv = "A2C_training_results_all_lrs.csv"


def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)


def make_env(seed: int):
    env = gym.make("CommunicationOver-v2", num_nodes=num_nodes, num_vnfs=num_vnfs)
    env = Monitor(env)

    # Gymnasium-style seeding (safe-guarded)
    try:
        env.reset(seed=seed)
    except TypeError:
        pass

    return env


def train_and_evaluate(learning_rate: float) -> pd.DataFrame:
    logger.info(f"\n=== Training A2C with learning_rate = {learning_rate} ===")

    set_seeds(seed_value)
    env = make_env(seed_value)

    model = build_a2c_agent(
        env,
        seed_value=seed_value,
        learning_rate=learning_rate,
        gamma=gamma,
        n_steps=n_steps,
        gae_lambda=gae_lambda,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        verbose=1,
    )

    callback = RewardCallback()
    model.learn(total_timesteps=total_timesteps, callback=callback)

    per_lr_csv = f"A2C_training_rewards_lr_{str(learning_rate).replace('.', '')}.csv"

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