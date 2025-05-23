import gymnasium as gym
from gymnasium import spaces
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import sys
import os
from torch.utils.tensorboard import SummaryWriter

# Add PharmaPy to path
sys.path.append(r"C:\Users\jjper\Documents\RESEARCH\takeda\PharmaPy")

from PharmaPy.Phases import LiquidPhase, SolidPhase
from PharmaPy.Kinetics import CrystKinetics
from PharmaPy.Crystallizers import BatchCryst
from PharmaPy.Interpolation import PiecewiseLagrange

trialname = input("Enter Trial Name:")
writer = SummaryWriter(log_dir=f"runs/{trialname}")

class CrystallizationEnv(gym.Env):
    def __init__(self):
        super(CrystallizationEnv, self).__init__()
        self.prim = (3e8, 0, 3)
        self.sec = (4.46e10, 0, 2, 1)
        self.growth = (5, 0, 1.32)
        self.solub_cts = [1.45752618e+01, -9.98982300e-02, 1.72100000e-04]
        self.kinetics = CrystKinetics(self.solub_cts, nucl_prim=self.prim, nucl_sec=self.sec, growth=self.growth)

        self.path = 'compounds_mom.json'
        self.temp_init = 323.15
        self.x_distrib = np.geomspace(1, 1500, 35)
        self.n_steps = 4000  # Increased cap
        self.dt = 7200 / 500

        self.action_space = spaces.Box(low=np.array([-0.5]), high=np.array([0.5]), dtype=np.float64)
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
        delta = np.clip(action, -0.5, 0.5)
        self.current_temp += delta
        self.current_temp = np.clip(self.current_temp, 285.15, 335.15)

        interpolator = PiecewiseLagrange(self.dt, [self.current_temp], order=1)
        controls = {'temp': interpolator.evaluate_poly}

        self.CR01 = BatchCryst(target_comp='solute', method='1D-FVM', controls=controls)
        self.CR01.Kinetics = self.kinetics
        self.CR01.Phases = (self.liquid, self.solid)
        results = self.CR01.solve_unit(self.dt, verbose=False) 
        D50, span = self.compute_d50_span(self.CR01.result)

        if span == 0:
            span = 2

        # Encourage smooth cooling
        time_penalty = self.current_step * 0.001

        if delta > 0:
            cooling_penalty = 5.0 * delta
            gentle_bonus = 0.0
        elif -0.02 <= delta <= 0:
            cooling_penalty = 0.0
            gentle_bonus = 100
        else:
            cooling_penalty = abs(delta)*.1
            gentle_bonus = 0.0

        reward = D50 * 5 + (1 / span) - time_penalty - cooling_penalty + gentle_bonus


        self.liquid = copy.deepcopy(self.CR01.Phases[0])
        self.solid = copy.deepcopy(self.CR01.Phases[1])

        self.current_step += 1
        done = self.current_temp <= 290.0 or self.current_step >= self.n_steps
        
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
        self.log_std = nn.Parameter(torch.tensor([0.5]))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = self.mu_head(x)
        std = torch.exp(self.log_std)
        mu_squashed = torch.tanh(mu) * 0.5
        return mu_squashed, std

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

episode_log = []
top_profiles = []
all_actions = []
all_entropies = []

try:
    for episode in range(num_episodes):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

        total_reward = 0
        done = False
        temp_profile = []

        actions_taken = []
        episode_entropies = []

        while not done:
            mu, std = actor(state)
            dist = torch.distributions.Normal(mu, std)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()

            action_np = action.cpu().detach().numpy()
            actions_taken.append(action_np.item())
            episode_entropies.append(entropy.item())
            all_actions.append(action_np.item())

            next_state, reward, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated

            temp_profile.append(env.current_temp)

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

            entropy_reg = 0.01  # exploration boost
            actor_loss = -log_prob * td_error.detach() - entropy_reg * entropy
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            state = next_state
            total_reward += reward

        writer.add_histogram("Cumulative/Actions_Sampled", np.array(all_actions), episode)
        writer.add_scalar("Episode/Mean_Entropy", np.mean(episode_entropies), episode)
        writer.add_scalar("Episode/Total_Reward", total_reward, episode)

        D50 = info.get("D50", 0)
        span = info.get("span", 0)

        writer.add_scalar("Episode/D50", D50, episode)
        writer.add_scalar("Episode/Span", span, episode)
        writer.add_scalar("Loss/Actor", actor_loss.item(), episode)
        writer.add_scalar("Loss/Critic", critic_loss.item(), episode)

        episode_log.append({
            "episode": episode + 1,
            "total_reward": total_reward,
            "D50": D50,
            "span": span
        })

        if len(top_profiles) < 5:
            top_profiles.append((total_reward, episode + 1, temp_profile))
            top_profiles.sort(key=lambda x: x[0], reverse=True)
        elif total_reward > top_profiles[-1][0]:
            top_profiles[-1] = (total_reward, episode + 1, temp_profile)
            top_profiles.sort(key=lambda x: x[0], reverse=True)

        print(f"Episode {episode+1}: Reward = {total_reward:.2f}, D50 = {D50:.2f}, Span = {span:.2f}")

finally:
    writer.close()

    os.makedirs("data", exist_ok=True)
    log_df = pd.DataFrame(episode_log)
    log_df.to_csv(os.path.join("data", trialname + "_episode_summary.csv"), index=False)

    records = []
    for reward, ep, profile in top_profiles:
        for step, temp in enumerate(profile):
            records.append({
                "episode": ep,
                "reward": reward,
                "step": step,
                "time": step * env.dt,
                "temperature": temp
            })
    
    top_df = pd.DataFrame(records)
    top_df.to_csv(os.path.join("data", trialname + "_top_5.csv"), index=False)

    print("Logs and top profiles saved to data directory.")
