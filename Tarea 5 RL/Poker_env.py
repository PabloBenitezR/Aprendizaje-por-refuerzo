import numpy as np
import gymnasium as gym
import random
from collections import Counter

# las cartas son enteros de 0-51 ya que hay 52 cartas
# las cartas no reveladas son -1

card_l = -1 
card_h = 51

obs_space = gym.spaces.Dict({
    # el conjunto de cartas del mazo
    "hole": gym.spaces.Box(low=card_l, high=card_h, shape=(2,), dtype=np.int8),
    # las cartas de la mesa
    "community": gym.spaces.Box(low=card_l, high=card_h, shape=(5,), dtype=np.int8),
    # el pot actual es la cantidad de fichas en el centro
    "pot": gym.spaces.Box(low=0,high=np.inf, shape=(), dtype=np.float32),
    # el stack es la cantidad de fichas que tiene el jugador
    "stacks": gym.spaces.Box(low=0, high=np.inf, shape=(2,), dtype=np.float32),
    # bets es la cantidad de apuestas que puede hacer cada jugador,
    "bets": gym.spaces.MultiDiscrete([4, 4]), 
})

# 0=fold, 1=call/check, 2=raise, 3=big raise
action_space = gym.spaces.Discrete(4)

class Poker(gym.Env):
    metadata = {"render_modes":["human"]}

    SUITS = ["♣","♦","♥","♠"]
    RANKS = ["A","2","3","4","5","6","7","8","9","10","J","Q","K"]

    def _int2card(self, code):
        suit = self.SUITS[ code // 13 ]
        rank = self.RANKS[ code % 13 ]
        return f"{rank}{suit}"

    def __init__(self):
        super().__init__()
        self.observation_space = obs_space
        self.action_space      = action_space

        self.reward_space = gym.spaces.Box(low=np.inf, high=np.inf, shape=(2,), dtype=np.float32
                                           )

        self.bet_unit    = 1.0 # unidad de apuesta inicial

        # apuestas iniciales de cada jugador, mismas que se van intercalando
        self.small_blind = 1.0
        self.big_blind   = 2.0

        # stacks iniciales
        self.start_stack = 100.0
        self.stacks      = [self.start_stack, self.start_stack]

        # el pot inicial es 0
        self.pot = 0
        
        # al inicio no hay apuestas
        self.bets = [0, 0]

        self.stage = 0
        self.folded = False      # ver si el jugador fold 
        self.fold_player = None  # quién fold

        # el deck (mazo), lo que cada jugador tiene en la mano 
        # y lo que hay en la mesa no existen al principio
        self.deck = None
        self.hands = None
        self.community = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # sólo al principio: inicializar stacks
        self.stacks = [self.start_stack, self.start_stack]
        return self._reset_hand()
    
    def _reset_hand(self):
        # reinicia la mano, baraja y reparte
        self.deck      = list(range(52)); random.shuffle(self.deck)
        self.hands     = [self._draw(2), self._draw(2)]
        self.community = [-1]*5
        self.stage     = 0
        self.folded    = False
        self.fold_player = None

        # blinds
        b0 = min(self.small_blind, self.stacks[0])
        b1 = min(self.big_blind,   self.stacks[1])

        self.stacks[0] -= b0
        self.stacks[1] -= b1
        self.pot        = b0 + b1
        self.bets       = [b0, b1]
        return self._get_obs(), {}
    
    def eval_strength(self, hand, community):

        cards = hand + [c for c in community if c>=0]
        ranks = [c % 13 for c in cards]

        # pre‐flop 
        if len(cards) == 2:
            return sum(ranks) / (2*12) 
        
        # flop
        cnt = Counter(ranks)
        if any(v >= 3 for v in cnt.values()):
            return 0.7   # tercia
        if any(v == 2 for v in cnt.values()):
            return 0.6   # par
        
        hole_strength = sum(hand_rank % 13 for hand_rank in hand) / (2*12)
        return 0.8 * hole_strength
    
    # se define el paso de la partida
    def step(self, action):

        # acción del agente
        self._apply_bet(0, action)

        str_agent = self.eval_strength(self.hands[0], self.community)
        str_opp   = self.eval_strength(self.hands[1], self.community)

        # acción del oponente
        if random.random() < (1.0 - str_opp):
            opp_action = 0    
        else:
            opp_action = 2 if random.random() < str_opp else 1  

        self._apply_bet(1, opp_action)

        # para el fold o all-in, verifica si el jugador se ha retirado o
        # si se ha llegado al final de la partida
        if self.folded or (self.stacks[0] == 0 and self.stacks[1] == 0):
            done = True
            # asigna pot al ganador
            if self.folded:
                winner = 1 - self.fold_player
            else:
                winner = self._determine_winner()
            self.stacks[winner] += self.pot

            # calcula reward para el agente
            reward = self.pot if winner == 0 else -self.bets[0]
            # actualiza stacks finales
            self.pot = 0.0
            # limpia fold para la próxima mano
            return self._get_obs(), reward, True, False, {}

        # si no termina la partida, avanza a la siguiente etapa
        done = self._advance_stage()

        reward = 0.0
        if done: # final de la partida
            # hace el cálculo del ganador
            winner = self._determine_winner()
            reward = self.pot if winner == 0 else -self.bets[0]
            self.stacks[0] += reward

        round_penalty = float(self.stage)
        reward_v = np.array([reward, round_penalty], dtype=np.float32)

        return self._get_obs(), reward_v, done, False, {}
    
    def _draw(self, n):
        return [self.deck.pop() for _ in range(n)]
    
    def _apply_bet(self, player, action):
        if action == 0:      # fold
            self.folded = True
            self.fold_player = player
            amount = 0.0

        elif action == 1:    # call/check
            max_bet = max(self.bets)
            amount  = max_bet - self.bets[player]

        elif action == 2:    # raise1
            amount = self.bet_unit

        elif action == 3:    # raise2
            amount = 2 * self.bet_unit

        amount = min(amount, self.stacks[player])
        # actalizar stack, bets y pot
        self.stacks[player] -= amount
        self.bets[player]   += amount
        self.pot            += amount
     

    def _advance_stage(self):
        # revela flop, turn, river o devuelve True si llega al final
        self.stage += 1
        if self.stage == 1:
            # flop: revela 3 cartas
            self.community[:3] = self._draw(3)
        elif self.stage == 2:
            # turn: revela carta 4
            self.community[3] = self._draw(1)[0]
        elif self.stage == 3:
            # river: revela carta 5
            self.community[4] = self._draw(1)[0]
        elif self.stage >= 4:
            return True
        return False

    def _determine_winner(self):
        # compara hands (cartas del jugador) + community(cartas de la mesa),
        #  devuelve 0 (perder), 1 (ganar) o None (empate)
        pass

    def _get_obs(self):
        return {
            "hole":      np.array(self.hands[0], dtype=np.int8),
            "community": np.array(self.community, dtype=np.int8),
            "pot":       np.array(self.pot, dtype=np.float32),
            "stacks":    np.array(self.stacks, dtype=np.float32),
            "bets":      np.array(self.bets, dtype=np.int8),
        }
    
    def render(self, mode="human"):
        hole_syms = [self._int2card(c) for c in self.hands[0]]
        opp_syms  = [self._int2card(c) for c in self.hands[1]]
        comm_syms = [self._int2card(c) if c>=0 else "__" for c in self.community]

        
        str_agent = self.eval_strength(self.hands[0], self.community)
        str_opp   = self.eval_strength(self.hands[1], self.community)

        print(f"Hole Agent : {hole_syms} | str: {str_agent:.2f}")
        print(f"Hole Opponent: {opp_syms} | str: {str_opp:.2f}")
        print(f"Community  : {comm_syms}")
        print(f"Pot: {self.pot}, Stacks: {self.stacks}\n")
