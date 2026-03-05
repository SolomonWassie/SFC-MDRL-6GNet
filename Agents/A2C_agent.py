from stable_baselines3 import A2C


def build_a2c_agent(
    env,
    *,
    seed_value: int,
    learning_rate: float,
    gamma: float,
    n_steps: int,
    gae_lambda: float,
    ent_coef: float,
    vf_coef: float,
    max_grad_norm: float,
    verbose: int = 1,
):
    """
    Creates and returns an A2C model configured for your environment.
    """
    return A2C(
        policy="MlpPolicy",
        env=env,
        verbose=verbose,
        learning_rate=learning_rate,
        gamma=gamma,
        n_steps=n_steps,
        gae_lambda=gae_lambda,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        seed=seed_value,
        tensorboard_log=None,
    )