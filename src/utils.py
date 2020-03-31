import torch.nn.functional as F
import torch as T


def f_one_hot_state(depth, min_depth):
    '''
        Functor for one_hot_state
    '''
    return lambda state: one_hot_state(state, depth, min_depth)

def one_hot_state(state, depth, min_depth):
    '''
        Returns the one hot encoded state (tensor of type float32)
    - state : LongTensor
    - depth : Number of categories, ie 3 for Tic Tac Toe (X, O and empty)
    - min_depth : Minimum value associated to depth, -1 for Tic Tac Toe since O = -1
    '''
    return F.one_hot(state - min_depth, depth).to(T.float32)
