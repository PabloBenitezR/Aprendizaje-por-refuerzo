# ppo_agent.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ppo_agent.py
class PPOAgent:
    def __init__(self, model, lr=3e-4, gamma=0.99, eps_clip=0.2):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.gamma = gamma
        self.eps_clip = eps_clip

    def compute_returns_and_advantages(self, rewards, values, dones):
        returns = []
        G = 0
        for r, v, done in zip(reversed(rewards), reversed(values), reversed(dones)):
            G = r + self.gamma * G * (1 - done)
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)
        values = torch.tensor(values, dtype=torch.float32)
        advs = returns - values
        return returns, advs

    def ppo_update(self, obs, actions, old_log_probs, returns1, advs1, returns2, advs2, weights, epochs=4, batch_size=64):
        dataset_size = len(obs)
        for epoch in range(epochs):
            idxs = np.arange(dataset_size)
            np.random.shuffle(idxs)
            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                batch_idx = idxs[start:end]

                obs_b = torch.stack([obs[i] for i in batch_idx])
                if torch.any(torch.isnan(obs_b)) or torch.any(torch.isinf(obs_b)):
                    raise ValueError("NaNs or Infs in input observations during PPO update")

                act_b = torch.tensor([actions[i] for i in batch_idx])
                old_log_b = torch.stack([old_log_probs[i] for i in batch_idx])
                adv1_b = advs1[batch_idx]
                adv2_b = advs2[batch_idx]
                returns1_b = returns1[batch_idx]
                returns2_b = returns2[batch_idx]

                logits, val1, val2 = self.model(obs_b)
                if torch.any(torch.isnan(logits)) or torch.any(torch.isinf(logits)):
                    print(f"Epoch {epoch}, batch {start}-{end}: logits = {logits}")
                    raise ValueError("NaNs or Infs in logits during PPO update")

                dist = torch.distributions.Categorical(logits=logits)
                log_probs = dist.log_prob(act_b)

                ratio = torch.exp(log_probs - old_log_b)

                adv_b = weights[0] * adv1_b + weights[1] * adv2_b
                if len(adv_b) < 2 or torch.allclose(adv_b, adv_b[0]):
                    adv_b = torch.zeros_like(adv_b)
                else:
                    adv_b = (adv_b - adv_b.mean()) / (adv_b.std(unbiased=False) + 1e-8)

                clip_adv = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * adv_b
                loss_clip = -torch.min(ratio * adv_b, clip_adv).mean()

                val_loss = F.mse_loss(val1, returns1_b) + F.mse_loss(val2, returns2_b)

                loss = loss_clip + 0.5 * val_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def collect_trajectory(self, env, max_steps=200):
        obs_list, actions, log_probs = [], [], []
        rewards1, rewards2, values1, values2, dones = [], [], [], [], []

        obs = env.reset()[0]
        for _ in range(max_steps):
            obs_tensor = torch.tensor(list(obs), dtype=torch.float32)
            logits, val1, val2 = self.model(obs_tensor)
            if torch.any(torch.isnan(logits)) or torch.any(torch.isinf(logits)):
                raise ValueError(f"NaNs or Infs in logits during trajectory collection: {logits}")
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            obs_list.append(obs_tensor)
            actions.append(action.item())
            log_probs.append(log_prob)
            values1.append(val1.item())
            values2.append(val2.item())

            obs, reward, done, truncated, info = env.step(action.item())

            # Multiobjetivo: reward1 = ganancia, reward2 = penalizar hits
            rewards1.append(reward)
            rewards2.append(-int(action.item() == 1))
            dones.append(float(done or truncated))

            if done or truncated:
                break

        return obs_list, actions, log_probs, rewards1, rewards2, values1, values2, dones
