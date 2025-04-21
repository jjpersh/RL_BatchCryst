import gymnasium as gym
from gymnasium import spaces
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import sys

# Add PharmaPy to path
sys.path.append(r"C:\Users\jjper\Documents\RESEARCH\takeda\PharmaPy")

from PharmaPy.Phases import LiquidPhase, SolidPhase
from PharmaPy.Kinetics import CrystKinetics
from PharmaPy.Crystallizers import BatchCryst
from PharmaPy.Interpolation import PiecewiseLagrange

class CrystallizationEnv(gym.Env):
    def __init__(self):
        super(CrystallizationEnv, self).__init__()

        # --- Constants ---
        self.prim = (3e8, 0, 3)
        self.sec = (4.46e10, 0, 2, 1)
        self.growth = (5, 0, 1.32)
        self.solub_cts = [1.45752618e+01, -9.98982300e-02, 1.72100000e-04]
        self.kinetics = CrystKinetics(self.solub_cts, nucl_prim=self.prim, nucl_sec=self.sec, growth=self.growth)

        self.path = 'compounds_mom.json'
        self.temp_init = 312.3
        self.x_distrib = np.geomspace(1, 1500, 35)
        self.n_steps = 100
        self.dt = 7200 / self.n_steps

        self.action_space = spaces.Box(low=np.array([-5.0]), high=np.array([5.0]), dtype=np.float64)
        self.observation_space = spaces.Box(low=np.array([0.0]), high=np.array([100.0]), dtype=np.float64)

        # Placeholders for reset
        conc_init = self.kinetics.get_solubility(self.temp_init)
        conc_init = (conc_init, 0)
        self.current_step = 0
        self.current_temp = self.temp_init
        self.liquid = LiquidPhase(self.path, temp=self.temp_init, vol=0.1, mass_conc=conc_init)
        distrib = np.zeros_like(self.x_distrib)
        self.solid = SolidPhase(self.path, temp=self.temp_init, mass_frac=(1, 0), distrib=distrib, x_distrib=self.x_distrib)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        conc_init = self.kinetics.get_solubility(self.temp_init)
        conc_init = (conc_init, 0)

        self.liquid = LiquidPhase(self.path, temp=self.temp_init, vol=0.1, mass_conc=conc_init)
        distrib = np.zeros_like(self.x_distrib)
        self.solid = SolidPhase(self.path, temp=self.temp_init, mass_frac=(1, 0), distrib=distrib, x_distrib=self.x_distrib)

        self.current_temp = self.temp_init
        self.current_step = 0

        return np.array([self.current_temp], dtype=np.float64), {}

    def step(self, action):
        if isinstance(action, np.ndarray):
            action = np.asarray(action).item()
        delta = np.clip(action, -5.0, 5.0)
        self.current_temp += delta
        self.current_temp = np.clip(self.current_temp, 273.15, 335.15)

        interpolator = PiecewiseLagrange(self.dt, [self.current_temp], order=1)
        controls = {'temp': interpolator.evaluate_poly}

        # Run simulation for one step
        print(self.current_temp)
        self.CR01 = BatchCryst(target_comp='solute', method='1D-FVM', controls=controls)
        self.CR01.Kinetics = self.kinetics
        self.CR01.Phases = (self.liquid, self.solid)
        results = self.CR01.solve_unit(self.dt, verbose=False) 
        D50, span = self.compute_d50_span(self.CR01.result)
        reward = D50 - span

        # Update phases for next step
        self.liquid = copy.deepcopy(self.CR01.Phases[0])
        self.solid = copy.deepcopy(self.CR01.Phases[1])

        self.current_step += 1
        done = self.current_step >= self.n_steps

        return np.array([self.current_temp], dtype=np.float64), reward, done, False, {"D50": D50, "span": span}

    def compute_d50_span(self, results):
        final_distrib = results.distrib[-1, :]
        if np.sum(final_distrib) == 0:
            return 0.0, 0.0
        x_sizes = results.x_cryst
        pdf = final_distrib / np.sum(final_distrib)
        cdf = np.cumsum(pdf)
        D10 = np.interp(0.10, cdf, x_sizes)
        D50 = np.interp(0.50, cdf, x_sizes)
        D90 = np.interp(0.90, cdf, x_sizes)
        span = (D90 - D10) / D50
        return D50, span


class Actor(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.mu_head = nn.Linear(hidden_dim, 1)     # Mean of the distribution
        self.log_std = nn.Parameter(torch.zeros(1)) # Learnable log-std

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = self.mu_head(x)
        std = torch.exp(self.log_std)  # ensure std > 0
        return mu, std

class Critic(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize env, actor, critic
env = CrystallizationEnv()
actor = Actor(input_dim=1, hidden_dim=64).to(device)
critic = Critic(input_dim=1, hidden_dim=64).to(device)

actor_optimizer = optim.Adam(actor.parameters(), lr=1e-3)
critic_optimizer = optim.Adam(critic.parameters(), lr=1e-3)

gamma = 0.99
num_episodes = 500

for episode in range(num_episodes):
    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

    total_reward = 0
    done = False

    while not done:
        mu, std = actor(state)
        dist = torch.distributions.Normal(mu, std)
        action = dist.sample()
        action_clipped = torch.clamp(action, -5.0, 5.0)

        next_state, reward, terminated, truncated, info = env.step(action_clipped.cpu().numpy())
        done = terminated or truncated

        next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(device)
        reward_tensor = torch.tensor([reward], dtype=torch.float32).to(device)

        # Critic estimate
        value = critic(state)
        next_value = critic(next_state)
        td_target = reward_tensor + gamma * next_value * (1 - int(done))
        td_error = td_target - value

        # Critic loss
        critic_loss = td_error.pow(2)
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        # Actor loss
        log_prob = dist.log_prob(action)
        actor_loss = -log_prob * td_error.detach()
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        state = next_state
        total_reward += reward

    print(f"Episode {episode+1}: Total Reward = {total_reward:.2f}")


