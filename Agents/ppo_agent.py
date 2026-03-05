from stable_baselines3 import PPO


def build_ppo_agent(
    env,
    *,
    seed_value: int,
    batch_size: int,
    n_epochs: int,
    learning_rate: float,
    gamma: float,
    verbose: int = 1,
):
    """
    Creates and returns a PPO model configured for your environment.
    """
    return PPO(
        policy="MlpPolicy",
        env=env,
        verbose=verbose,
        batch_size=batch_size,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        gamma=gamma,
        seed=seed_value,
        tensorboard_log=None,
    )