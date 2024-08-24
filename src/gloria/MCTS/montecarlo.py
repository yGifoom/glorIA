import numpy as np

class Node():
    def __init__(self, s: np.array, available_actions: np.array):
        '''
        Nodes should initialize all their values and of their corresponding
        actions and then be updated at every rollout
        '''
        self.children = list() # all the children nodes 
        self.state: np.array = s # encoded state
        self.visited = 1 # times this node was visited during rollouts
        
        self.actions: list[action] = [action(a) for a in available_actions] # actions available from state s

    def update(self):
        '''
        update value of the node during backpropagation
        '''
        raise NotImplementedError
        
    def _upper_confidence_bound(self, a):
        ucb = a.P**self.beta * (self.visited**1/2) / ((a.N) + 1)
    
    def __eq__(node1, node2):
        res = np.array_equal(node1.state, node2.state)
        return res
         

class action:
        def __init__(self, a):
            self.Q = None # reward for action
            self.P = None # probability of taking this action
            self.N = None # times action was taken 


    

class MCTS():
    def __init__(self, alpha=0.1, beta=0.5):
        # HYPERPARAMETERS
        self.beta = beta # between 0-1 
        self.alpha = alpha # between 0-1 
        #

        self.Nodes: dict[Node] = dict()

    def rollout(self):
        path = self._choose() # find path to a leaf/terminal node
        leaf = path[-1] 
        self._backpropagate(path, self._evaluate(leaf.s))
    
    def _backpropagate(self, path, v):
        for n in path:
            n.update(v)

    def _choose(self):
        '''
        find the path from root to leaf/terminal node
        return the list of the path
        '''
        raise NotImplementedError
    
    def _evaluate(self, state: np.array):
        '''
        evaluate the reward for having reached a state
        '''
        raise NotImplementedError




    

    