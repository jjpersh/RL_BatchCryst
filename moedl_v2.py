import gymnasium as gym
from gymnasium import spaces
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import sys
import matplotlib.pyplot as plt
from IPython.display import clear_output

# Add PharmaPy to path
sys.path.append(r"C:\Users\jjper\Documents\RESEARCH\takeda\PharmaPy")

from PharmaPy.Phases import LiquidPhase, SolidPhase
from PharmaPy.Kinetics import CrystKinetics
from PharmaPy.Crystallizers import BatchCryst
from PharmaPy.Interpolation import PiecewiseLagrange

class CrystallizationEnv(gym.Env):
    def __init__(self):
        super(CrystallizationEnv, self).__init__()

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

        self.CR01 = BatchCryst(target_comp='solute', method='1D-FVM', controls=controls)
        self.CR01.Kinetics = self.kinetics
        self.CR01.Phases = (self.liquid, self.solid)
        results = self.CR01.solve_unit(self.dt, verbose=False)
        D50, span = self.compute_d50_span(self.CR01.result)
        reward = D50 - span

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
        self.mu_head = nn.Linear(hidden_dim, 1)
        self.log_std = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = self.mu_head(x)
        std = torch.exp(self.log_std)
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

env = CrystallizationEnv()
actor = Actor(input_dim=1, hidden_dim=64).to(device)
critic = Critic(input_dim=1, hidden_dim=64).to(device)

actor_optimizer = optim.Adam(actor.parameters(), lr=1e-3)
critic_optimizer = optim.Adam(critic.parameters(), lr=1e-3)

gamma = 0.99
num_episodes = 500

# ------------------ Added Visualization Code ------------------
reward_history = []
temperature_history = []
d50_history = []
span_history = []

plt.ion()
fig, axs = plt.subplots(3, 1, figsize=(10, 8))
fig.suptitle("RL Progress (Real-Time)")
# --------------------------------------------------------------

for episode in range(num_episodes):
    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

    total_reward = 0
    done = False
    ep_temps = []
    ep_d50s = []
    ep_spans = []

    while not done:
        mu, std = actor(state)
        dist = torch.distributions.Normal(mu, std)
        action = dist.sample()
        action_clipped = torch.clamp(action, -5.0, 5.0)

        next_state, reward, terminated, truncated, info = env.step(action_clipped.cpu().numpy())
        done = terminated or truncated

        next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(device)
        reward_tensor = torch.tensor([reward], dtype=torch.float32).to(device)

        value = critic(state)
        next_value = critic(next_state)
        td_target = reward_tensor + gamma * next_value * (1 - int(done))
        td_error = td_target - value

        critic_loss = td_error.pow(2)
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        log_prob = dist.log_prob(action)
        actor_loss = -log_prob * td_error.detach()
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        state = next_state
        total_reward += reward

        ep_temps.append(float(state.cpu().numpy()[0][0]))
        ep_d50s.append(info["D50"])
        ep_spans.append(info["span"])

    reward_history.append(total_reward)
    temperature_history.append(ep_temps)
    d50_history.append(ep_d50s)
    span_history.append(ep_spans)

    # ------------------ Update Plots ------------------
    clear_output(wait=True)
    axs[0].cla()
    axs[0].plot(reward_history, label="Total Reward")
    axs[0].set_title("Episode Reward")
    axs[0].legend()

    axs[1].cla()
    axs[1].plot(ep_temps, label="Temperature (K)")
    axs[1].set_title("Temperature Profile (Last Episode)")
    axs[1].legend()

    axs[2].cla()
    axs[2].plot(ep_d50s, label="D50")
    axs[2].plot(ep_spans, label="Span")
    axs[2].set_title("Crystal Metrics (Last Episode)")
    axs[2].legend()

    plt.pause(0.01)
    # --------------------------------------------------

    print(f"Episode {episode+1}: Total Reward = {total_reward:.2f}")

plt.ioff()
plt.show()

# ------------------ Final Policy Plot ------------------
temps = torch.linspace(273.15, 335.15, 100).unsqueeze(1).to(device)
with torch.no_grad():
    mus, stds = actor(temps)

mus = mus.cpu().numpy()
stds = stds.cpu().numpy()
temps = temps.cpu().numpy()

plt.figure(figsize=(8, 5))
plt.plot(temps, mus, label="Mean Action")
plt.fill_between(temps.flatten(), mus.flatten() - stds.flatten(), mus.flatten() + stds.flatten(), alpha=0.3, label="Std Dev")
plt.title("Final Policy: Action vs Temperature")
plt.xlabel("Temperature (K)")
plt.ylabel("Action (Delta T)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
# -------------------------------------------------------
