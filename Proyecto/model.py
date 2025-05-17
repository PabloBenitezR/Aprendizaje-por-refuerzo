# model.py
import torch
import torch.nn as nn

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 64),
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
        if torch.any(torch.isnan(logits)) or torch.any(torch.isinf(logits)):
            raise ValueError(f"NaNs or Infs detected in logits during get_action: {logits}")
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), dist
