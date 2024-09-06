from typing import List, Union, Awaitable

import tensorflow as tf
from keras import layers, models
import numpy as np
import random
import time as t
from collections import deque

# from keras.layers import layer
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.player.battle_order import ForfeitBattleOrder
from poke_env.player import (
    Gen4EnvSinglePlayer,
    MaxBasePowerPlayer,
    ObsType,
    RandomPlayer,
    SimpleHeuristicsPlayer,
    Player,
    background_cross_evaluate,
    background_evaluate_player, BattleOrder,
)
from poke_env.player.player import Player
from src.gloria.embedding.get_embeddings import GlorIA # POKEMONS, MOVES, ABILITIES, ITEMS, UNKNOWN_POKEMON, EFFECTS
from tabulate import tabulate
from poke_env import AccountConfiguration
from gymnasium.spaces import Space, Box


class Opponent(Player):
    def __init__(self, battle_format, account_configuration):
        super().__init__(battle_format=battle_format, account_configuration=account_configuration)
        self.gloria_instance = GlorIA(opponent=self, battle_format=battle_format, start_challenging=False)
        self.model: PPOAgent = None
        self.previous_state = None
        self.previous_action = None
        self.current_battle_tag = None


    def action_to_move(self, action: int, battle: AbstractBattle) -> BattleOrder:
        """Converts actions to move orders.

        The conversion is done as follows:

        action = -1:
            The battle will be forfeited.
        0 <= action < 4:
            The actionth available move in battle.available_moves is executed.
        4 <= action < 10
            The action - 4th available switch in battle.available_switches is executed.

        If the proposed action is illegal, a random legal move is performed.

        :param action: The action to convert.
        :type action: int
        :param battle: The battle in which to act.
        :type battle: Battle
        :return: the order to send to the server.
        :rtype: str
        """
        if action == -1:
            return ForfeitBattleOrder()
        elif (
            action < 4
            and action < len(battle.available_moves)
            and not battle.force_switch
        ):
            return self.create_order(battle.available_moves[action])
        elif 0 <= action - 4 < len(battle.available_switches):
            return self.create_order(battle.available_switches[action - 4])
        else:
            return self.choose_random_move(battle)

    def choose_move(self, battle: AbstractBattle):
        if self.current_battle_tag is None:
            self.current_battle_tag = battle.battle_tag
        st = self.gloria_instance.embed_battle(battle)
        st = np.reshape(st, [1, self.agent.input_shape])
        action = self.agent.act(st)
        finished = battle.finished
        if self.previous_state is not None:  # also action and reward are None
            self.agent.remember(self.previous_state, self.previous_action, 0, st, finished)
        self.previous_state = st
        self.previous_action = action

        return self.action_to_move(action=action, battle=battle)

def create_embedding_layers(input_layer):
    
    # max number in vocab, dimentions to be mapped to, slice of input layer to embed
    to_embed = {
        "species" : {
           "size" :  296, "dims": 158, "in": [106 + i*(233+8) for i in range(12)],
        },
        "abilities": {
           "size" :  103, "dims": 52, "in": [107 + i*(233+8) for i in range(12)],
        },
        "items" : {
           "size" :  39, "dims": 19, "in": [108 + i*(233+8) for i in range(12)],  # 38 + no_item which is len ITEMS + 1
        },
        "moves" : {
           "size" :  188, "dims": 94, "in": [[109 + j + i*(233+8) for j in range(5)] for i in range(12)],
        },
    }
    
    embeds = list()
    
        # Iterate over the features to be embedded
    for feature, params in to_embed.items():
        vocab_size = params['size']
        embed_dim = params['dims']
        input_indices = params['in']

        # Check if input indices is a list of lists (for moves) or a flat list (for species, abilities, items)
        if isinstance(input_indices[0], list):
            # For features with multiple inputs (like moves)
            for indices in input_indices:
                input_slice = tf.keras.layers.Lambda(lambda x: tf.gather(x, indices, axis=1))(input_layer)
                embedding = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)(input_slice)
                embeds.append(layers.Flatten()(embedding))
        else:
            # For features with single input per item in list
            for index in input_indices:
                input_slice = tf.keras.layers.Lambda(lambda x: x[:, index:index + 1])(input_layer)
                embedding = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)(input_slice)
                embeds.append(layers.Flatten()(embedding))

    # Concatenate all embedding layers
    if embeds:
        concatenated_embeds = layers.Concatenate()(embeds)
    else:
        concatenated_embeds = None

    # Returning the final model or layer output containing the embeddings
    return concatenated_embeds

# Create the PPO-network
def create_policy_network(input_shape, num_actions):
    # Maybe add embedding layer after input layer (output size of Emb. Layer to be determined)
    # Possibliy add dropout layers in between Dense layers to prevent overfitting by adding noise
    # Layers and thickness will change due to input layer/ embedding layer size
    input_all = layers.Input(shape=(input_shape,), dtype='int32', name='input_all')
    
    embedding = create_embedding_layers(input_all)
    dense_1 = layers.Dense(128, activation='relu')(embedding)
    dense_2 = layers.Dense(128, activation='relu')(dense_1)
    dense_3 = layers.Dense(128, activation='relu')(dense_2)
    output = layers.Dense(num_actions, activation='softmax')(dense_3)  # Output probabilities
    model = models.Model(inputs = input_all, outputs = output)
    
    return model 


def create_value_network(input_shape):
    input_all = layers.Input(shape=(input_shape,), dtype='int32', name='input_all')
    
    embedding = create_embedding_layers(input_all)
    dense_1 = layers.Dense(128, activation='relu')(embedding)
    dense_2 = layers.Dense(128, activation='relu')(dense_1)
    dense_3 = layers.Dense(128, activation='relu')(dense_2)
    output = layers.Dense(1, activation='linear')(dense_3)  # Output probabilities
    model = models.Model(inputs = input_all, outputs = output)
    
    return model

class PPOAgent:
    def __init__(self, input_shape, num_actions, gamma=0.9999, epsilon=0.2, actor_lr=0.000058, critic_lr=0.000058):
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

config1, config2 = AccountConfiguration("opp", None), AccountConfiguration("training", None)

# Instantiate two GlorIA agents
randy = RandomPlayer(battle_format="gen4randombattle", account_configuration=AccountConfiguration("rnady", None))
opp = Opponent(battle_format="gen4randombattle", account_configuration=config1)
train_env = GlorIA(battle_format="gen4randombattle", account_configuration=config2, opponent=opp, start_challenging=False)
# Compute dimensions
n_action = train_env.action_space_size()
input_shape = np.array(train_env.observation_space.shape).prod()

# Training loop
num_episodes = 6
batch_size = 2
num_actions = n_action
agent = PPOAgent(input_shape=input_shape, num_actions=num_actions)
opp.agent = agent


train_env.start_challenging(n_challenges=num_episodes)

# Start the battles

for e in range(1, num_episodes + 1):
    train_env.reset()
    initial_state = train_env.embed_battle(train_env.current_battle)
    state = np.reshape(initial_state, [1, agent.input_shape])
    done = False
    time = 0
    while not done:
        try:
            action = agent.act(state)
        except Exception as e:
            print(e, "AAAAAAAAAAAAAAAAAA")
            print(state)
            exit()
        next_state, reward, done, _, info = train_env.step(action)
        next_state = np.reshape(train_env.embed_battle(train_env.current_battle), [1, agent.input_shape])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done:
            print(f"episode: {e}/{num_episodes}, score: {time}")
        time += 1
    opponent_last_state = opp.gloria_instance.embed_battle(opp.battles[opp.current_battle_tag])
    agent.remember(opp.previous_state, opp.previous_action, -reward, opponent_last_state, done)
    opp.current_battle_tag = None  # Reset the battle tag for the opponent
    # Perform PPO optimization at the end of the episode
    if len(agent.memory) > batch_size:
        agent.replay(batch_size)

# Reset environment
# train_env.reset()

# train_env.reset_env(restart=False)
train_env.close()


# Test Function
def test(agent, environments, nb_episodes=100):

    for environment in environments:
        victories = 0
        environment.start_challenging(n_challenges=nb_episodes)
        for e in range(1, nb_episodes + 1):
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
                    print(f"Episode {e}/{nb_episodes} finished. Reward: {reward}")
        print(f"Test completed: {victories}/{nb_episodes} victories")

        while not environment.current_battle._finished:
            pass
        environment.reset_env(restart=False)
        environment.close()


# Players and Environments setup (ignore) --------------------------------------------------------------
opponent = RandomPlayer(battle_format="gen4randombattle",
                        account_configuration=AccountConfiguration("rand", None))
eval_env = GlorIA(
    battle_format="gen4randombattle", opponent=opponent, start_challenging=False,
    account_configuration=AccountConfiguration("trained_vs_rand", None)
)

maxi = MaxBasePowerPlayer(battle_format="gen4randombattle",
                          account_configuration=AccountConfiguration("max", None))

heur = SimpleHeuristicsPlayer(battle_format="gen4randombattle",
                              account_configuration=AccountConfiguration("heur", None))

eval2 = AccountConfiguration("trained_vs_maxi", None)
eval3 = AccountConfiguration("trained_vs_heur", None)
eval_env2 = GlorIA(battle_format="gen4randombattle", opponent=maxi, start_challenging=False,
                           account_configuration=eval2)
eval_env3 = GlorIA(battle_format="gen4randombattle", opponent=heur, start_challenging=False,
                           account_configuration=eval3)
#--------------------------------------------------------------------------------------------------------

# test(agent, [eval_env, eval_env2, eval_env3], nb_episodes=2)

# agent.model.save("dqn_model.h5")
