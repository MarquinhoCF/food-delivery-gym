import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback

class StepRewardLogger(BaseCallback):
    def __init__(self, log_path, verbose=0):
        super().__init__(verbose)
        self.log_path = log_path
        self.step_rewards = []

    def _on_step(self) -> bool:
        reward = self.locals["rewards"][0]  # Pega a recompensa do passo atual
        self.step_rewards.append(reward)
        return True

    def _on_training_end(self):
        # Salvar as recompensas por passo em um CSV
        df = pd.DataFrame(self.step_rewards, columns=["reward"])
        df.to_csv(self.log_path, index=False)
