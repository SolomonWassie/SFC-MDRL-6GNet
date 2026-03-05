import logging
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

logger = logging.getLogger(__name__)


class RewardCallback(BaseCallback):
    """
    Collects per-episode:
      - reward (from Monitor: info["episode"]["r"])
      - mig_cost (from your env info dict)
    """
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_mig_costs = []

    def _on_step(self) -> bool:
        # SB3 uses VecEnv internally -> dones/infos are arrays
        if self.locals["dones"][0]:
            info = self.locals["infos"][0]

            ep_info = info.get("episode", {})
            reward = ep_info.get("r", None)
            if reward is not None:
                self.episode_rewards.append(reward)

            self.episode_mig_costs.append(info.get("mig_cost", np.nan))

            logger.info(
                f"Episode finished | Reward={reward} | mig_cost={info.get('mig_cost')}"
            )
        return True