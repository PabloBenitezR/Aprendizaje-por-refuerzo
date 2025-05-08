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

    step += 1

   
    if done_hand:
        print(f"\n>>> Fin de mano, Stacks: {env.stacks}\n")

        
        if env.stacks[1] <= 0:
            print("\nOponente sin fichas, river directo")
            hole_agent   = [env._int2card(c) for c in env.hands[0]]
            hole_opponent= [env._int2card(c) for c in env.hands[1]]
            community    = [env._int2card(c) for c in env.community]
            print("Hole Agent    :", hole_agent)
            print("Hole Opponent :", hole_opponent)
            print("Community     :", community)
            match_done = True
        
        elif env.stacks[0] >= env.small_blind and env.stacks[1] >= env.big_blind:
            obs, _ = env._reset_hand()

        else:
            print("\nfinde la partida")
            print("¡Se acabaron las fichas!")
            print("Stacks finales:", env.stacks)
            match_over = True
