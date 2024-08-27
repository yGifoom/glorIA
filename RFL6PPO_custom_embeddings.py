from typing import Union, Awaitable

import tensorflow as tf
from keras import layers, models
import numpy as np
import random
from collections import deque

from keras.src.layers import layer
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.player import (
    Gen4EnvSinglePlayer,
    MaxBasePowerPlayer,
    ObsType,
    RandomPlayer,
    SimpleHeuristicsPlayer,
    background_cross_evaluate,
    background_evaluate_player, BattleOrder,
)
from poke_env.player.player import Player
import src.gloria.embedding.get_embeddings as get_embeddings
from tabulate import tabulate
from poke_env import AccountConfiguration
from gymnasium.spaces import Space, Box

class SimpleRLPlayer(Gen4EnvSinglePlayer):
    def calc_reward(self, last_battle, current_battle) -> float:
        return self.reward_computing_helper(
            current_battle, fainted_value=2.0, hp_value=1.0, victory_value=30.0
        )

    def embed_battle(self, battle: AbstractBattle) -> ObsType:
        return get_embeddings.GlorIA().embed_battle(battle)

    def describe_embedding(self) -> Space: #Using Oscars method
        return get_embeddings.GlorIA().describe_embedding()

class Opponent(Player):
    def embed_battle(self, battle: AbstractBattle) -> ObsType:
        return get_embeddings.GlorIA().embed_battle(battle)
    def choose_move(
        self, battle: AbstractBattle
    ) -> Union[BattleOrder, Awaitable[BattleOrder]]:
        st = self.embed_battle(battle)
        st = np.reshape(st, [1, agent.input_shape])
        return train_env.action_to_move(action=agent.act(st), battle=battle)

# Create the PPO-network
def create_policy_network(input_shape, num_actions):
    # Maybe add embedding layer after input layer (output size of Emb. Layer to be determined)
    # Possibliy add dropout layers in between Dense layers to prevent overfitting by adding noise
    # Layers and thickness will change due to input layer/ embedding layer size
    model = models.Sequential()
    model.add(layers.Input(shape=(input_shape,)))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(num_actions, activation='softmax'))  # Output probabilities
    return model


def create_value_network(input_shape):
    model = models.Sequential()
    model.add(layers.Input(shape=(input_shape,)))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(1, activation='linear'))  # Output a single value
    return model


class PPOAgent:
    def __init__(self, input_shape, num_actions, gamma=0.95, epsilon=0.2, actor_lr=0.0003, critic_lr=0.001):
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon  # Clipping value for PPO (uses the Clipped Surrogate Function)
        self.actor = create_policy_network(input_shape, num_actions)
        self.critic = create_value_network(input_shape)
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)
        self.memory = deque(maxlen=2000)

    def act(self, state):
        state = np.reshape(state, [1, self.input_shape])
        action_probs = self.actor.predict(state)[0]
        action = np.random.choice(self.num_actions, p=action_probs)
        return action

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in minibatch:
            state = np.reshape(state, [1, self.input_shape])
            next_state = np.reshape(next_state, [1, self.input_shape])

            with tf.GradientTape() as tape:
                value = self.critic(state)
                next_value = self.critic(next_state)
                target = reward + (1 - done) * self.gamma * next_value
                advantage = target - value

                # Instantiate the loss function and compute the loss
                mse_loss_fn = tf.keras.losses.MeanSquaredError()
                critic_loss = mse_loss_fn(target, value)

            critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)

            # Check if gradients are None
            if any(grad is None for grad in critic_grads):
                raise ValueError("Gradients are None for some variables")

            self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

        with tf.GradientTape() as tape:
            # Calculate the PPO loss for the actor network
            action_probs = self.actor(state)
            action_prob = action_probs[0][action]
            old_action_prob = action_prob  # This should be stored separately if using multiple steps
            ratio = action_prob / old_action_prob
            surrogate1 = ratio * advantage
            surrogate2 = tf.clip_by_value(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantage
            actor_loss = -tf.minimum(surrogate1, surrogate2)

        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

    def load(self, name):
        self.actor.load_weights(name + "_actor")
        self.critic.load_weights(name + "_critic")

    def save(self, name):
        self.actor.save_weights(name + "_actor")
        self.critic.save_weights(name + "_critic")

config1, config2 = AccountConfiguration("GG", None), AccountConfiguration("Player2", None)

# Instantiate two SimpleRLPlayer agents
randy = RandomPlayer(battle_format="gen4randombattle", account_configuration=AccountConfiguration("rnady", None))
opp = Opponent(battle_format="gen4randombattle", account_configuration=config1)
train_env = SimpleRLPlayer(battle_format="gen4randombattle", account_configuration=config2, opponent=randy, start_challenging=False)
# Compute dimensions
n_action = train_env.action_space_size()
input_shape = np.array(train_env.observation_space.shape).prod()

# Training loop
num_episodes = 6
batch_size = 2
num_actions = n_action
agent = PPOAgent(input_shape=input_shape, num_actions=num_actions)


train_env.start_challenging(n_challenges=num_episodes)

# Start the battles
for e in range(1, num_episodes + 1):
    train_env.reset()
    initial_state = train_env.embed_battle(train_env.current_battle)

    state = np.reshape(initial_state, [1, agent.input_shape])

    done = False
    time = 0
    while not done:

        action = agent.act(state)
        next_state, reward, done, _, info = train_env.step(action)
        next_state = np.reshape(train_env.embed_battle(train_env.current_battle), [1, agent.input_shape])
        agent.remember(state, action, reward, next_state, done)
        state = next_state

        if done:
            print(f"episode: {e}/{num_episodes}, score: {time}")

        time += 1

    # Perform PPO optimization at the end of the episode
    if len(agent.memory) > batch_size:
        agent.replay(batch_size)

# Reset environment
train_env.reset_env(restart=False)
train_env.close()


# Test Function
def test(agent, environments, nb_episodes=100):

    for environment in environments:
        victories = 0
        for e in range(nb_episodes):
            environment.reset()
            state = np.reshape(environment.embed_battle(environment.current_battle), [1, agent.input_shape])
            done = False
            while not done:

                action = agent.act(state)
                next_state, reward, done, _, info = environment.step(action)
                next_state = np.reshape(environment.embed_battle(environment.current_battle), [1, agent.input_shape])

                state = next_state

                if done:
                    if reward > 0:  # Assuming a positive reward indicates a win
                        victories += 1
                    print(f"Episode {e + 1}/{nb_episodes} finished. Reward: {reward}")
        print(f"Test completed: {victories}/{nb_episodes} victories")
        environment.reset_env(restart=False)
        environment.close()


# Players and Environments setup (ignore) --------------------------------------------------------------
opponent = RandomPlayer(battle_format="gen4randombattle",
                        account_configuration=AccountConfiguration("rand", None))
eval_env = SimpleRLPlayer(
    battle_format="gen4randombattle", opponent=opponent, start_challenging=True,
    account_configuration=AccountConfiguration("trained_vs_rand", None)
)

maxi = MaxBasePowerPlayer(battle_format="gen4randombattle",
                          account_configuration=AccountConfiguration("max", None))

heur = SimpleHeuristicsPlayer(battle_format="gen4randombattle",
                              account_configuration=AccountConfiguration("heur", None))

eval2 = AccountConfiguration("trained_vs_maxi", None)
eval3 = AccountConfiguration("trained_vs_heur", None)
eval_env2 = SimpleRLPlayer(battle_format="gen4randombattle", opponent=maxi, start_challenging=True,
                           account_configuration=eval2)
eval_env3 = SimpleRLPlayer(battle_format="gen4randombattle", opponent=heur, start_challenging=True,
                           account_configuration=eval3)
#--------------------------------------------------------------------------------------------------------

test(agent, [eval_env, eval_env2, eval_env3], nb_episodes=2)

# agent.model.save("dqn_model.h5")
