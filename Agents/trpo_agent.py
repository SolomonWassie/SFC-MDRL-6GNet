from sb3_contrib import TRPO


def build_trpo_agent(
    env,
    *,
    seed_value: int,
    learning_rate: float,
    gamma: float,
    n_steps: int,
    batch_size: int,
    gae_lambda: float,
    verbose: int = 1,
):
    """
    Creates and returns a TRPO model configured for your environment.
    """
    return TRPO(
        policy="MlpPolicy",
        env=env,
        verbose=verbose,
        learning_rate=learning_rate,
        gamma=gamma,
        n_steps=n_steps,
        batch_size=batch_size,
        gae_lambda=gae_lambda,
        seed=seed_value,
        tensorboard_log=None,
    )