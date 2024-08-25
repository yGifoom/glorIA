import numpy as np
from poke_env.environment import Battle
from gloria.embedding.get_embeddings import MOVES, POKEMONS

# MISSING FUNCTION AND IMPORTS:
def model(state, action): 
    '''
    probability of the model choosing an action (everything is masked except that action)
    in a state
    '''
    return 0.1

def can_moves(state):
    '''
    calculate all the available actions from a given state
    '''
    n_options = len(POKEMONS) + len(MOVES)

    if np.random.randint(0, 100) == 0:
        res =  np.zeros(n_options)
    else:
        res = np.ndarray(0,2, (n_options))
    
    return res

# ASSUMPTION: available actions should be a np array of 1s and 0s masking all possible moves 
# and pokemon to switch out to:
# [pokemon switch, moves]
# since POKEMON is 296 long and MOVES is 188, available actions are 484


class Node():
    def __init__(self, s: np.array):
        '''
        Nodes should initialize all their values and of their corresponding
        actions and then be updated at every rollout
        '''
        self.state: np.array = s # encoded state
        self.visited = 1 # times this node was visited during rollouts
        available_actions = can_moves(s)
        self.isterminal = True if np.count_nonzero(available_actions) > 0 else False # is leaf or terminal?
        self.actions: np.array = self._encode_actions(available_actions) # actions available from state s

        self.ucb: np.array = self._upper_confidence_bound(a)

    def update(self, v):
        '''
        update value of the node during backpropagation
        '''
        raise NotImplementedError
    
    def simulate(self):
        '''
        use the learned model to simulate opponent's moves
        '''
        raise NotImplementedError
        
    def _upper_confidence_bound(self, a):
        ucb = a.P**self.beta * (self.visited**1/2) / ((a.N) + 1)
    
    def _encode_actions(self, available_actions: np.array):
        for i in range(len(available_actions)):
            if available_actions[i] == 1:
                available_actions[i] = action(i) 


    
    def __eq__(node1, node2):
        res = np.array_equal(node1.state, node2.state)
        return res

    def __hash__(self):
        raise NotImplementedError
         

class action:
    def __init__(self, available_action: int):
        # available_action is just the index of the only 1
        self.a = self._decode_action(available_action)

        self.Q = 0 # reward for action
        # probability of taking this action, 
        # available action is all the actions masked minus one
        self.P = model(available_action) 
        self.N = 1 # times action was taken
        self.U = 0 # upper confidence bound for this action

    def _decode_action(self, action):
        raise NotImplementedError

    def __hash__(self):
        raise NotImplementedError

class MCTS():
    def __init__(self, state: np.array, available_actions: np.array, alpha=0.1, beta=0.5):
        # HYPERPARAMETERS
        self.beta = beta # between 0-1 
        self.alpha = alpha # between 0-1 
        #

        # TODO: translate available_actions so that it is a dictionary? 
        # parsing trough the array each time might be too time consuming
        self.root = Node(state)
        self.children: dict = {state: None}


    def rollout(self):
        path = self._choose(self.root) # find path to a leaf/terminal node
        leaf = path[-1] 
        self._backpropagate(path, self._evaluate(leaf.s))

    def new_root(self, state):
        '''
        once rollouts are done pick the new starting point, pruning all unnecessary
        nodes
        '''
        raise NotImplementedError
    
    def _backpropagate(self, path: list[Node], v):
        for n in path:
            n.update(v)

    def _choose(self, node: Node):
        '''
        find the path from root to leaf/terminal node
        return the list of the path
        '''
        path = list()
        while True:
            expand_action = np.argmax(map(lambda x: 0 if x == 0 else x.U, node.actions)) # array is either 0 or action object
            path.append(node, expand_action)
            
            next_state = node.simulate(expand_action)

            if next_state not in map(lambda x: x.state, self.children[node]): 
                # means that we never visited this state
                self.children[node] = {expand_action: next_state}
                node = Node(next_state)
                break
            node = self._ucb_select(node)
        
    def _ucb_select(self, node):
        '''
        select next node by their reward
        '''
        raise NotImplementedError
    
    def _evaluate(self, state: np.array):
        '''
        evaluate the reward for having reached a state
        '''
        raise NotImplementedError




    

    