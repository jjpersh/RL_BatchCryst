import gymnasium as gym
from gymnasium import spaces
import numpy as np

class CrystallizationEnv(gym.Env):
    def __init__(self, profile_length=100, runtime_bounds=(10.0, 100.0)):
        super(CrystallizationEnv, self).__init__()

        self.profile_length = profile_length
        self.temp_min = 10.0
        self.temp_max = 90.0
        self.runtime_min, self.runtime_max = runtime_bounds

        # Action: [temperature profile..., runtime]
        self.action_space = spaces.Box(
            low=np.array([self.temp_min] * self.profile_length + [self.runtime_min]),
            high=np.array([self.temp_max] * self.profile_length + [self.runtime_max]),
            dtype=np.float32
        )

        # Observation can remain static for now
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(1,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        obs = np.zeros((1,), dtype=np.float32)
        return obs, {}

    def step(self, action):
        # Split action into temperature profile and runtime
        temperature_profile = np.clip(action[:-1], self.temp_min, self.temp_max)
        runtime = np.clip(action[-1], self.runtime_min, self.runtime_max)

        # Simulate crystallization
        d50, span = self._simulate_crystallization(temperature_profile, runtime)

        # Reward: higher D50 and lower span
        reward = d50 - span  # You can adjust: e.g., reward = 2 * d50 - 1.5 * span

        obs = np.zeros((1,), dtype=np.float32)
        terminated = True  # One-shot environment

        return obs, reward, terminated, False, {
            "D50": d50,
            "span": span,
            "runtime": runtime
        }

    def _simulate_crystallization(self, temperature_profile, runtime):
        # === Replace this with your actual simulation ===
        mean_temp = np.mean(temperature_profile)

        # Example behavior: better D50 when temp is steady and runtime is optimal
        d50 = 2.0 + 0.01 * (mean_temp - 50) + 0.05 * np.exp(-((runtime - 60) ** 2) / 1000)
        span = 1.0 + 0.1 * np.std(temperature_profile) + 0.01 * abs(runtime - 60)

        return max(d50, 0.1), max(span, 0.1)
