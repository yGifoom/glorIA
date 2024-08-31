import numpy as np
import json
from poke_env.environment import Battle, Move, Pokemon
from poke_env.data.gen_data import GenData
from gloria.embedding.get_embeddings import MOVES, POKEMONS

import sys
from subprocess import Popen, PIPE, TimeoutExpired, run

SIMULATOR_BUFFER = "./pokemon-showdown/simulator_buffer/"

class Simulator():
    def __init__(self):
        '''
        self.moves = dict(m for m in data.moves.items() if m[0] in MOVES.keys())
        self.pokemon = dict(m for m in data.pokedex.items() if m[0] in POKEMONS.keys())
        '''
        self.simulator_on_server = Popen(["node", "./pokemon-showdown/sim/file_buffer_bitch.js"], stdout = PIPE, stdin = PIPE, stderr=PIPE, text=True)
        self.rd_moves = list(MOVES.keys()) # we use the index as the key from now on
        self.rd_pokemon = list(POKEMONS.keys())

    def sim(self, battle: Battle, sim_battle: dict, actions: tuple[int] = None, decoded_actions: str = None) -> tuple[Battle, dict]:
        def decode_action(actions_idx: tuple[int]) -> str:
            '''
            decode mask of the available action to a string 
            to be written in the simulation buffer
            '''
            res = list()
            for i in range(len(actions_idx)):
                if not actions_idx[i]: res.append("") # case only one move

                if actions_idx[i] >= len(self.rd_moves): # it is a switch
                    actions_idx[i] -= len(self.rd_moves) # IF WE STORE THE MASK AS MOVES CONCATENATE POKEMON  
                    res.append("switch " + self.rd_pokemon[actions_idx[i]])
                else:
                    res.append("move " + self.rd_moves[actions_idx[i]])

            return ",".join(res)
        
        if actions:
            moves = decode_action(actions)
        elif decoded_actions:
            moves = decoded_actions
        else:
            print("no actions??")


            with open('./pokemon-showdown/simulator_buffer/battle.json', "w") as f:
                json.dump(sim_battle, f)

            run(["node", "./pokemon-showdown/sim/file_buffer_bitch.js"])
                
                
            with open('./pokemon-showdown/simulator_buffer/new_battle.json', "r") as f:
                new_sim_battle = json.load(f)

            len_old_obs = len(new_sim_battle["log"])
            new_obs = new_sim_battle["log"][len_old_obs:] # all the new observations to update battle

            if new_obs:
                self.update_battle(battle, new_obs)

            return battle, new_sim_battle

        to_simulate = f"{moves}\n{json.dumps(sim_battle)}"

        try:
            #send to the javascript simulator the moves + the state
            json_str, errs = self.simulator_on_server.communicate(to_simulate, timeout=2)  # will not go on until child is finished
            print(errs)
            new_sim_battle = json.loads(json_str)
        except TimeoutExpired:
            print("there was a timeout error in simulating a battle " + TimeoutExpired)
            self.simulator_on_server.kill()
            errs = self.simulator_on_server.communicate(to_simulate)

        len_old_obs = len(sim_battle["log"])
        new_obs = new_sim_battle["log"][len_old_obs:] # all the new observations to update battle

        print(sim_battle["log"][-2:], "req:", sim_battle["requestState"])
        print(new_sim_battle["log"][-2:], "req:", new_sim_battle["requestState"])

        if new_obs:
            self.update_battle(battle, new_obs)

        return battle, new_sim_battle
    
    def update_battle(old_battle: Battle, obs: list[str]):
        for ob in obs:
            old_battle.parse_message(ob.split("|"))

        '''
        # write moves to be done in buffer
        with open(SIMULATOR_BUFFER + "battle.json", "w") as f:
            json.dump(to_simulate, f)

        with open(SIMULATOR_BUFFER + "flag.txt", "w") as f:
            f.write("1")
        #TODO implement communication between typescript and python
        # assume result of newsimul state is outputted in next_sim_battle
        '''

if __name__ == "__main__":
    ## take the data for test ##
    from poke_env.player import RandomPlayer
    from time import time
    import asyncio
    from copy import deepcopy
    BATTLE = list()

    class DebugRandomPlayer(RandomPlayer):

        def choose_move(self, battle):
            if battle.turn == 10: 
                BATTLE.append(deepcopy(battle))
           
            return self.choose_random_move(battle)
    
    random_player = DebugRandomPlayer(battle_format="gen4randombattle")
    second_random_player = RandomPlayer(battle_format="gen4randombattle")
    
    async def test():   
        await random_player.battle_against(second_random_player, n_battles=1)

    asyncio.run(test())
    BATTLE = BATTLE[0]
    print(f"Battle gathered by simulation is of type:{type(BATTLE)}")
    print("done gathering for battle")

    with open('./pokemon-showdown/simulator_buffer/battle.json') as f:
        ps_battle = json.load(f)
    
    print('test commencing') ##
    S = Simulator()
    start = time()
    new_battle, new_sim_battle = S.sim(BATTLE, ps_battle)
    tempo = time() - start
    print(f"{BATTLE.turn} switched to {new_battle.turn}")
    print(f"{BATTLE.opponent_active_pokemon} switched to {new_battle.opponent_active_pokemon}")
    print(f"ps battle was at turn { ps_battle['turn'] } , new battle is at turn { new_sim_battle['turn'] }")
    print(f"time needed was {tempo} s")