# PPO con Pareto Tracer para Blackjack Multiobjetivo
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import imageio
import matplotlib.pyplot as plt

# ==================== Actor-Critic ====================
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim * 4, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh()
        )
        self.policy_head = nn.Linear(64, act_dim)
        self.value_head_1 = nn.Linear(64, 1)
        self.value_head_2 = nn.Linear(64, 1)

    def forward(self, x):
        base = self.shared(x)
        logits = self.policy_head(base)
        value1 = self.value_head_1(base)
        value2 = self.value_head_2(base)
        return logits, value1.squeeze(-1), value2.squeeze(-1)

    def get_action(self, x):
        if isinstance(x, tuple):
            x = torch.tensor(list(x), dtype=torch.float32)
        logits, _, _ = self.forward(x)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), dist

# ==================== PPO Agent ====================
class PPOAgent:
    def __init__(self, model, lr=2.5e-4, gamma=0.99, eps_clip=0.1, entropy_coef=0.01):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.entropy_coef = entropy_coef

    def compute_returns_and_advantages(self, rewards, values, dones):
        returns = []
        G = 0
        for r, v, done in zip(reversed(rewards), reversed(values), reversed(dones)):
            G = r + self.gamma * G * (1 - done)
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)
        values = torch.cat(values).detach()
        advs = returns - values
        return returns.detach(), advs.detach()

    def ppo_update(self, obs, actions, old_log_probs, returns1, advs1, returns2, advs2, weights, epochs=4, batch_size=64):
        dataset_size = len(obs)
        for epoch in range(epochs):
            idxs = np.arange(dataset_size)
            np.random.shuffle(idxs)
            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                batch_idx = idxs[start:end]

                obs_b = torch.stack([obs[i] for i in batch_idx])
                act_b = torch.tensor([actions[i] for i in batch_idx])
                old_log_b = torch.stack([old_log_probs[i] for i in batch_idx])
                adv1_b = advs1[batch_idx]
                adv2_b = advs2[batch_idx]
                returns1_b = returns1[batch_idx]
                returns2_b = returns2[batch_idx]

                logits, val1, val2 = self.model(obs_b)
                dist = torch.distributions.Categorical(logits=logits)
                log_probs = dist.log_prob(act_b)
                entropy = dist.entropy().mean()

                ratio = torch.exp(log_probs - old_log_b)

                adv_b = weights[0] * adv1_b + weights[1] * adv2_b
                if len(adv_b) < 2 or torch.allclose(adv_b, adv_b[0]):
                    adv_b = torch.zeros_like(adv_b)
                else:
                    adv_b = (adv_b - adv_b.mean()) / (adv_b.std(unbiased=False) + 1e-8)

                clip_adv = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * adv_b
                loss_clip = -torch.min(ratio * adv_b, clip_adv).mean()
                val_loss = F.mse_loss(val1, returns1_b) + F.mse_loss(val2, returns2_b)

                loss = loss_clip + 0.5 * val_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward(retain_graph=False)
                self.optimizer.step()

    def collect_trajectory(self, env, max_steps=200):
        obs_list, actions, log_probs = [], [], []
        rewards1, rewards2, values1, values2, dones = [], [], [], [], []

        obs = env.reset()[0]
        frame = torch.tensor(list(obs), dtype=torch.float32)
        state_stack = frame.repeat(4, 1)

        for _ in range(max_steps):
            input_tensor = state_stack.flatten().unsqueeze(0)
            logits, val1, val2 = self.model(input_tensor)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            obs_list.append(input_tensor.squeeze(0))
            actions.append(action.item())
            log_probs.append(log_prob.detach())
            values1.append(val1.detach())
            values2.append(val2.detach())

            obs, reward, done, truncated, info = env.step(action.item())

            rewards1.append(reward)
            rewards2.append(-int(action.item() == 1))
            dones.append(float(done or truncated))

            next_frame = torch.tensor(list(obs), dtype=torch.float32)
            state_stack = torch.cat([state_stack[1:], next_frame.unsqueeze(0)], dim=0)

            if done or truncated:
                break

        return obs_list, actions, log_probs, rewards1, rewards2, values1, values2, dones

# ==================== Experimento ====================
if __name__ == "__main__":
    env = gym.make("Blackjack-v1", sab=True, render_mode="human")
    obs_dim = 3
    act_dim = env.action_space.n

    n = 11
    weights_list = [[i / (n - 1), 1 - i / (n - 1)] for i in range(n)]
    reward_points = []

    for weights in weights_list:
        model = ActorCritic(obs_dim, act_dim)
        agent = PPOAgent(model)

        for episode in range(100):
            data = agent.collect_trajectory(env)
            obs, acts, logs, r1, r2, v1, v2, dones = data
            ret1, adv1 = agent.compute_returns_and_advantages(r1, v1, dones)
            ret2, adv2 = agent.compute_returns_and_advantages(r2, v2, dones)
            agent.ppo_update(obs, acts, logs, ret1, adv1, ret2, adv2, weights)

        # Evaluación
        total_r1, total_r2 = 0, 0
        episodios_eval = 50
        puntos = []
        for _ in range(episodios_eval):
            obs = env.reset()[0]
            done = False
            r1, r2 = 0, 0
            frame = torch.tensor(list(obs), dtype=torch.float32)
            state_stack = frame.repeat(4, 1)
            while not done:
                env.render()  # Mostrar estado actual del entorno
                time.sleep(0.5)  # Esperar medio segundo para observar
                input_tensor = state_stack.flatten().unsqueeze(0)
                action, _, _ = model.get_action(input_tensor)
                obs, reward, done, truncated, _ = env.step(action.item())
                r1 += reward
                r2 += -int(action.item() == 1)
                next_frame = torch.tensor(list(obs), dtype=torch.float32)
                state_stack = torch.cat([state_stack[1:], next_frame.unsqueeze(0)], dim=0)
                done = done or truncated
            total_r1 += r1
            total_r2 += r2
            puntos.append((r1, r2))

        reward_points.append([total_r1 / episodios_eval, total_r2 / episodios_eval])

        # Graficar conjunto aproximado
        puntos = np.array(puntos)
        plt.scatter(puntos[:, 0], puntos[:, 1], alpha=0.3, label=f"w={weights}")

    # Visualización del frente de Pareto
    reward_points = np.array(reward_points)
    plt.plot(reward_points[:, 0], reward_points[:, 1], 'ko-', label="Frente de Pareto")
    plt.xlabel("Ganancia promedio")
    plt.ylabel("Penalización por hits")
    plt.title("Frente y conjunto de Pareto - Blackjack")
    plt.legend()
    plt.grid(True)
    plt.show()