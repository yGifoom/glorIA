# glorIA
A pokemon showdown AI using reinforcement learning

## EMBEDDING
- for deployment, there may be discrepancies between current embedding and actual embedding
- it's the showdown server that provides the team and the rosters. Therefore, when starting training,
  let's fix a showdown version and use only that one.
- state space for volatile effects is total-removed
- toxic counter has max 15 ticks (after 15 ticks always 93% damage)
- sleep 	counter has max 5 ticks
- first turn is first turn on the field
- we decided to add uknown value for every filed of a pokemon, as we may get partial information about them,
also the unknown flag will work correctly threrafter
- check if TYPECHANGE or TYPE_CHANGE (TYPECHANGE WON)
- UNKNOWN pokemon counter is 6 items in one hot
- part of the input is going to be in one hot encoding (the dimentionality is very low, a.k.a < 100) whereas some of the input will pass through an embedding layer before getting to the hidden layers.

      numerical-input   one-hot-encoded-input 
            |                   |
            |                   |
            |                   |
        embedding_layer         |
            |                   |
            |                   | 
             \                 /        
               \              / 
              dense_hidden_layer
                     | 
                     | 
                  output_layer 
                  
this differs from the paper which embedded the entire input layer into 13*892 vectors (1 per the match
and one per pokemon)
(btw the 892 was just a magic number, the creator of the paper confirmed)
- last used move is incorporate it into the species, moves, ability, type
item embedding because it still says something about the pokemon.
- types are encoded with two variables, one of max value 18, the other 19; we are using the encoding of the 
library adjusted for fairy
- unknown elements will be represented with all zeros, except in some variables like:
weather, stage hazards, volatile effects..... which can never be unknown __THIS IS SOME RISKY BUSINESS, IF ANYTHInG BRAKES, CHECK THIS SHIT FIRST__
- for hp we use all zeros for 0hp
- no weather flag for input is no more
- 8hp bins are now 17 with a special bin for 100% hp
- the paper we were following maybe brakes MARKOV's hypothesis because of the move disable: we need to include in the state of the match the disabled move,
but it is sufficient to just add a masking of illegal moves before the softmax, the model will learn to discourage moves which can be disabled by taunt
- we added a bit for trapped for every pokemon
(maybe turns of outrage could have been better)

good_volatile:
ATTRACT= 7
CUSTAP_BERRY= 23 
DESTINY_BOND = 25
FEINT= 43
LEECH_SEED= 79
REFLECT= 135
TYPECHANGE= 184
CONFUSION (18)
CURSE (22)
DISABLE (26)
ENCORE (34)
FLASH_FIRE (46)
PERISH0 (106)
PERISH1 (107)
PERISH2 (108)
PERISH3 (109)
SAFEGUARD (138)
SLOW_START (148)
SUBSTITUTE (162)
TAUNT (170)
YAWN (195)
TRAPPED (181)
UNKNOWN (1)

bibliography:

### DEV NOTES
It is required that you install the [Pokemon showdown](https://github.com/smogon/pokemon-showdown)local server for testing,
repo is included in submodules by default, a small script runs the server.
(note that repo requires a version of nodejs, install it globally i can't be bothered!)




