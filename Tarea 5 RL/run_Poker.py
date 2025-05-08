import numpy as np
from Poker_env import Poker

env = Poker()
obs, _ = env.reset()
step = 0
match_done = False

while not match_done:
    print(f"\n--- ronda {step} ---")
    env.render()  
    action = env.action_space.sample()
    print("Acción del oponente:", action)

    obs, reward, done_hand, truncated, info = env.step(action)
    print("Recompensa recibida:", reward)

    step +=1

    if np.isscalar(reward):        
        gain = reward
    else:                          
        gain = reward[0]

    if done_hand:
        
        if gain > 0:
            print("Ganador de la mano: Agente")
        elif gain < 0:
            print("Ganador de la mano: Oponente")
        else:
            print("Empate en la mano")
        print(f"Fin de mano Stacks: {env.stacks}\n")

        
        if env.stacks[1] == 0:
            print("\nOponente sin fichas")
            match_done = True
        
        elif env.stacks[0] >= env.small_blind and env.stacks[1] >= env.big_blind:
            obs, _ = env._reset_hand()

        else:
            print("\nfin de la partida")
            print("¡Se acabaron las fichas!")
            print("Stacks finales:", env.stacks)
            match_done = True