import asyncio
import tensorflow as tf
from keras import layers, models
import numpy as np
import random
from collections import deque
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.player import (
    Gen4EnvSinglePlayer,
    MaxBasePowerPlayer,
    ObsType,
    RandomPlayer,
    SimpleHeuristicsPlayer,
    background_cross_evaluate,
    background_evaluate_player,
)
from keras.models import load_model
from tabulate import tabulate
from poke_env import AccountConfiguration
from gymnasium.spaces import Space, Box
from gymnasium.utils.env_checker import check_env
import time as t
class SimpleRLPlayer(Gen4EnvSinglePlayer):
    def calc_reward(self, last_battle, current_battle) -> float:
        return self.reward_computing_helper(
            current_battle, fainted_value=2.0, hp_value=1.0, victory_value=30.0
        )

    def embed_battle(self, battle: AbstractBattle) -> ObsType:
        # -1 indicates that the move does not have a base power
        # or is not available
        moves_base_power = -np.ones(4)
        moves_dmg_multiplier = np.ones(4)
        for i, move in enumerate(battle.available_moves):
            moves_base_power[i] = (
                    move.base_power / 100
            )  # Simple rescaling to facilitate learning
            if move.type:
                moves_dmg_multiplier[i] = battle.opponent_active_pokemon.damage_multiplier(move)

        # We count how many pokemons have fainted in each team
        fainted_mon_team = len([mon for mon in battle.team.values() if mon.fainted]) / 6
        fainted_mon_opponent = (
                len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6
        )

        # Final vector with 10 components
        final_vector = np.concatenate(
            [
                moves_base_power,
                moves_dmg_multiplier,
                [fainted_mon_team, fainted_mon_opponent],
            ]
        )
        return np.float32(final_vector)

    def describe_embedding(self) -> Space:
        low = [-1, -1, -1, -1, 0, 0, 0, 0, 0, 0]
        high = [3, 3, 3, 3, 4, 4, 4, 4, 1, 1]
        return Box(
            np.array(low, dtype=np.float32),
            np.array(high, dtype=np.float32),
            dtype=np.float32,
        )
# Create the Q-network
def create_q_network(input_shape, num_actions):
    model = models.Sequential()
    model.add(layers.Input(shape=(input_shape,)))  # Use Input layer for input shape
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(num_actions, activation='linear'))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
    return model
# Define the DQN agent
class DQNAgent:
    def __init__(self, input_shape, num_actions):
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = create_q_network(input_shape, num_actions)
        self.target_model = create_q_network(input_shape, num_actions)
        self.update_target_model()

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.num_actions)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.target_model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


config1 = AccountConfiguration("Opp1", None)
config2 = AccountConfiguration("Opp2", None)
config3 = AccountConfiguration("Opp3", None)
config4 = AccountConfiguration("GG1", None)
config5 = AccountConfiguration("GG2", None)
config6 = AccountConfiguration("GG3", None)

opponent = RandomPlayer(battle_format="gen4randombattle",
                        account_configuration=config2)
train_env = SimpleRLPlayer(
    battle_format="gen4randombattle", opponent=opponent, start_challenging=False,
    account_configuration=config5
)

# Compute dimensions
n_action = train_env.action_space_size()
input_shape = np.array(train_env.observation_space.shape).prod()

# Training loop
num_episodes = 6
batch_size = 2
num_actions = n_action
agent = DQNAgent(input_shape=input_shape, num_actions=num_actions)
train_env.start_challenging(n_challenges=num_episodes)
for e in range(1, num_episodes + 1):
    train_env.reset()
    initial_state = train_env.embed_battle(train_env.current_battle)  # Embed the initial state
    state = np.reshape(initial_state, [1, agent.input_shape])  # Reshape to fit the neural network input
    done = False
    time = 0
    while not done:
        action = agent.act(state)
        next_state, reward, done, _, info = train_env.step(action)  # Use train_env here
        next_state = np.reshape(next_state, [1, agent.input_shape])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        time += 1
        if done:
            agent.update_target_model()
            print(f"episode: {e}/{num_episodes}, score: {time}, e: {agent.epsilon:.2}")
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
train_env.reset_env(restart=False)
train_env.close()

# Test Function
def test(agent, environment, nb_episodes=100):

    victories = 0
    for e in range(nb_episodes):
        environment.reset()
        s = np.reshape(environment.embed_battle(environment.current_battle), [1, agent.input_shape])
        done = False
        while not done:
            action = agent.act(s)
            next_state, reward, done, _, info = environment.step(action)
            next_state = np.reshape(environment.embed_battle(environment.current_battle), [1, agent.input_shape])
            s = next_state
            if done:
                if reward > 0:  # Assuming a positive reward indicates a win
                    victories += 1
                print(f"Episode {e + 1}/{nb_episodes} finished. Reward: {reward}")
    print(f"Test completed: {victories}/{nb_episodes} victories")
    environment.close()

opponent = RandomPlayer(battle_format="gen4randombattle",
                        account_configuration=config3)
eval_env = SimpleRLPlayer(
    battle_format="gen4randombattle", opponent=opponent, start_challenging=True,
    account_configuration=config6
)

opp_maxi = AccountConfiguration("max", None)
opp_heur = AccountConfiguration("heur", None)

maxi = MaxBasePowerPlayer(battle_format="gen4randombattle",
                          account_configuration=opp_maxi)
heur = SimpleHeuristicsPlayer(battle_format="gen4randombattle",
                              account_configuration=opp_heur)

eval2 = AccountConfiguration("trained_vs_maxi", None)
eval3 = AccountConfiguration("trained_vs_heur", None)

eval_env2 = SimpleRLPlayer(battle_format="gen4randombattle", opponent=maxi, start_challenging=True,
                           account_configuration=eval2)
eval_env3 = SimpleRLPlayer(battle_format="gen4randombattle", opponent=heur, start_challenging=True,
                           account_configuration=eval3)

test(agent, eval_env)
test(agent, eval_env2)
test(agent, eval_env3)


agent.model.save("dqn_model.h5")