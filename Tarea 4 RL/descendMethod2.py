import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import pandas as pd

# Wrapper para convertir el entorno en multiobjetivo

class MultiObjectiveLunarLander(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        
        # Dividimos la recompensa en dos objetivos:
        # - Objetivo 1: reward original.
        # - Objetivo 2: penalización por usar motores (acción != 0).
        fuel_penalty = -0.3 if action != 0 else 0.0
        multi_reward = np.array([reward, fuel_penalty], dtype=np.float32)
        return obs, multi_reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, info

# Red de política (MLP) para acciones discretas

class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, act_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return F.softmax(logits, dim=-1)


# Función para calcular la dirección de descenso de Lara 

def compute_lara_direction(g1, g2):
    """
    Dados dos gradientes g1 y g2,
    calcula la dirección según el método de Lara para maximizar la recompensa:
        d = (g1 / ||g1|| + g2 / ||g2||).
    Se incluye un epsilon para evitar división por cero.
    """
    eps = 1e-8
    norm_g1 = np.linalg.norm(g1) + eps
    norm_g2 = np.linalg.norm(g2) + eps
    direction = -(g1 / norm_g1 + g2 / norm_g2)
    return direction

# Función para calcular los retornos descontados
def compute_discounted_returns(rewards, gamma):
    returns = np.zeros_like(rewards, dtype=np.float32)
    if rewards.ndim == 1:
        running_return = 0.0
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + gamma * running_return
            returns[t] = running_return
    else:
        T, num_obj = rewards.shape
        for obj in range(num_obj):
            running_return = 0.0
            for t in reversed(range(T)):
                running_return = rewards[t, obj] + gamma * running_return
                returns[t, obj] = running_return
    return returns


# Función para ejecutar un episodio y recolectar datos

def run_episode(env, policy, device):
    obs, _ = env.reset()
    obs = torch.from_numpy(obs).float().to(device)
    done = False
    
    log_probs = []
    rewards = []  # Lista de vectores de recompensas (dim = num_objetivos)
    
    while not done:
        probs = policy(obs)
        m = Categorical(probs)
        action = m.sample()
        log_prob = m.log_prob(action)
        log_probs.append(log_prob)
        
        action_item = action.item()
        next_obs, reward_vec, terminated, truncated, _ = env.step(action_item)
        rewards.append(reward_vec)
        done = terminated or truncated
        
        obs = torch.from_numpy(next_obs).float().to(device)
    
    rewards = np.array(rewards, dtype=np.float32)
    return log_probs, rewards


# Función principal de entrenamiento

def main():
    num_episodios = 5000
    gamma = 0.99
    lr = 1e-2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    env_id = "LunarLander-v3"
    base_env = gym.make(env_id)
    env = MultiObjectiveLunarLander(base_env)
    
    obs_dim = base_env.observation_space.shape[0]
    act_dim = base_env.action_space.n
    
    policy = PolicyNetwork(obs_dim, act_dim).to(device)
    
    episode_rewards = []
    
    for ep in range(num_episodios):
        log_probs, rewards = run_episode(env, policy, device)
        T = len(rewards)
        num_obj = rewards.shape[1]
        total_reward = np.sum(rewards, axis=0)
        episode_rewards.append(total_reward)
        returns = compute_discounted_returns(rewards, gamma)
        
        losses = []
        for obj in range(num_obj):
            ret_obj = torch.tensor(returns[:, obj], dtype=torch.float32).to(device)
            loss_obj = sum(-lp * r for lp, r in zip(log_probs, ret_obj))
            losses.append(loss_obj)
        
        grads = []
        for loss in losses:
            grad_list = torch.autograd.grad(loss, policy.parameters(), retain_graph=True)
            grad_vector = torch.nn.utils.parameters_to_vector(grad_list)
            grads.append(grad_vector.detach().cpu().numpy())
        
        # Usar la dirección de Lara 
        g1, g2 = grads[0], grads[1]
        d = compute_lara_direction(g1, g2)
        
        # Actualizar parámetros con gradient descent
        with torch.no_grad():
            theta = torch.nn.utils.parameters_to_vector(policy.parameters())
            theta = theta + lr * torch.tensor(d, dtype=theta.dtype)
            torch.nn.utils.vector_to_parameters(theta, policy.parameters())
        
        avg_returns = np.mean(returns, axis=0)
        dir_norm = np.linalg.norm(d)
        print(f"Ep {ep+1:04d}: Retorno prom. Objetivo1 = {avg_returns[0]:.2f}, "
              f"Objetivo2 = {avg_returns[1]:.2f}, ||dirección|| = {dir_norm:.3f}")
    
    env.close()
    
    episode_rewards = np.array(episode_rewards)
    plt.figure(figsize=(10,6))
    episodes = np.arange(1, num_episodios + 1)
    for obj in range(episode_rewards.shape[1]):
        plt.plot(episodes, episode_rewards[:, obj], label=f"Objetivo {obj+1}")
    plt.xlabel("Episodios")
    plt.ylabel("Recompensa Total (sin descuento)")
    plt.title("Evolución de la recompensa total por episodio (Método de Lara - Ascenso)")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    means = np.mean(episode_rewards, axis=0)
    stds = np.std(episode_rewards, axis=0)
    data = {
        "Objetivo": [f"Objetivo {i+1}" for i in range(episode_rewards.shape[1])],
        "Promedio": means,
        "Desviación Estándar": stds
    }
    df = pd.DataFrame(data)
    print("\nEstadísticas de la recompensa acumulada por episodio:")
    print(df)

if __name__ == '__main__':
    main()
