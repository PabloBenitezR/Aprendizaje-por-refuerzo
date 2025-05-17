# main_pareto_blackjack.py
import torch
import numpy as np
import gym
import matplotlib.pyplot as plt
from model import ActorCritic
from ppo_agent import PPOAgent

if __name__ == "__main__":
    env = gym.make("Blackjack-v1", sab=True)
    obs_dim = 3
    act_dim = env.action_space.n

    weights_list = [
        [1.0, 0.0],
        [0.75, 0.25],
        [0.5, 0.5],
        [0.25, 0.75],
        [0.0, 1.0]
    ]

    reward_points = []

    for weights in weights_list:
        model = ActorCritic(obs_dim, act_dim)
        agent = PPOAgent(model)

        for episode in range(100):
            traj = agent.collect_trajectory(env)
            obs, actions, log_probs, r1, r2, v1, v2, dones = traj
            ret1, adv1 = agent.compute_returns_and_advantages(r1, v1, dones)
            ret2, adv2 = agent.compute_returns_and_advantages(r2, v2, dones)
            agent.ppo_update(obs, actions, log_probs, ret1, adv1, ret2, adv2, weights)

        # Evaluar política entrenada
        total_r1, total_r2 = 0, 0
        for _ in range(50):
            obs = env.reset()[0]
            done = False
            r1, r2 = 0, 0
            while not done:
                obs_tensor = torch.tensor(list(obs), dtype=torch.float32)
                action, _, _ = model.get_action(obs_tensor)
                obs, reward, done, truncated, _ = env.step(action.item())
                r1 += reward
                r2 += -int(action.item() == 1)
                done = done or truncated
            total_r1 += r1
            total_r2 += r2
        reward_points.append([total_r1 / 50, total_r2 / 50])

    reward_points = np.array(reward_points)
    plt.plot(reward_points[:, 0], reward_points[:, 1], 'o-')
    plt.xlabel("Recompensa promedio")
    plt.ylabel("Penalizaci\u00f3n por hits")
    plt.title("Frente de Pareto aproximado (Blackjack)")
    plt.grid(True)
    plt.show()