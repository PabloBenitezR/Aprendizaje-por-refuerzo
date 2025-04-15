
import numpy as np
import gymnasium as gym
import mo_gymnasium as mo_gym
from collections import defaultdict
import matplotlib.pyplot as plt

# ------------------------------
# Configuración del entorno
# ------------------------------
env = mo_gym.make("mo-lunar-lander-v3")

n_actions = env.action_space.n
n_states = 1000  # se usará discretización aproximada

# ------------------------------
# Parámetros de entrenamiento
# ------------------------------
alpha = 0.1
gamma = 0.99
epsilon = 0.1
episodes = 500
objectives = len(env.reward_range)

# ------------------------------
# Inicialización de Q-table de vectores
# ------------------------------
Q = defaultdict(lambda: np.zeros((n_actions, objectives)))
pareto_front = []

# ------------------------------
# Función de discretización
# ------------------------------
def discretize(state):
    return tuple(np.round(state, decimals=1))

# ------------------------------
# Función de acción epsilon-greedy
# ------------------------------
def select_action(state):
    if np.random.rand() < epsilon:
        return np.random.randint(n_actions)
    else:
        values = Q[state]
        scalarized = values @ np.ones(objectives)
        return np.argmax(scalarized)

# ------------------------------
# Función para almacenar soluciones pareto-óptimas (2D)
# ------------------------------
def update_pareto(front, new_vec):
    non_dominated = []
    for vec in front:
        if np.all(vec >= new_vec) and np.any(vec > new_vec):
            return front
        elif not (np.all(new_vec >= vec) and np.any(new_vec > vec)):
            non_dominated.append(vec)
    non_dominated.append(new_vec)
    return non_dominated

# ------------------------------
# Entrenamiento
# ------------------------------
reward_records = []

for ep in range(episodes):
    state, _ = env.reset()
    state = discretize(state)
    done = False
    total_reward = np.zeros(objectives)

    while not done:
        action = select_action(state)
        next_state, reward_vector, terminated, truncated, _ = env.step(action)
        next_state = discretize(next_state)
        done = terminated or truncated

        best_next = np.max(Q[next_state], axis=0)
        Q[state][action] = (1 - alpha) * Q[state][action] + alpha * (reward_vector + gamma * best_next)

        state = next_state
        total_reward += reward_vector

    reward_records.append(total_reward)
    pareto_front = update_pareto(pareto_front, total_reward)

    if ep % 50 == 0:
        print(f"Episodio {ep}, Recompensa acumulada: {total_reward}")

env.close()

# ------------------------------
# Visualización
# ------------------------------
reward_array = np.array(reward_records)
plt.plot(reward_array[:, 0], label='Recompensa 1')
plt.plot(reward_array[:, 1], label='Recompensa 2')
plt.title("Recompensas por episodio")
plt.xlabel("Episodio")
plt.ylabel("Recompensas")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

pareto_front = np.array(pareto_front)
plt.scatter(pareto_front[:, 0], pareto_front[:, 1], color='red')
plt.title("Frontera de Pareto aproximada")
plt.xlabel("Objetivo 1")
plt.ylabel("Objetivo 2")
plt.grid(True)
plt.tight_layout()
plt.show()
