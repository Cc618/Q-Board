# All DQNs

from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    '''
        Multi Layer Perceptron, only fully connected layers and activations
    '''
    def __init__(self, n_state, n_action, hidden, flatten=False):
        '''
        - n_state : Dimension of the state
        - hidden : Int list, describes hidden layers
        - n_action : Number of possible actions (or rewards)
        - flatten : If true, flattens the input to one dimension

        * Can be represented as :
        ReLU(Linear(n_state, hidden[0])) -> ReLU(Linear(hidden[i - 1], hidden[i])) -> Linear(hidden[-1], n_action)
        '''
        super().__init__()

        self.n_state = n_state
        self.flatten = flatten

        if hidden == []:
            layers = [nn.Linear(n_state, n_action)]
        else:
            layers = [nn.Linear(n_state, hidden[0]), nn.ReLU()]

            # ReLU activated dense layers
            for i in range(0, len(hidden) - 1):
                layers.append(nn.Linear(hidden[i], hidden[i + 1]))
                layers.append(nn.ReLU())

            # Last layer, no activation
            layers.append(nn.Linear(hidden[-1], n_action))

        self.net = nn.Sequential(*layers)

    def forward(self, state):
        if self.flatten:
            state = state.view(-1, self.n_state)

        return self.net(state)


# class CNN(nn.Module):
#     '''
#         Convolutional Neural Network
#     '''
#     def __init__(self, depth, n_action, conv=[], mlp=[]):
#         '''
#         - n_state : Dimension of the state
#         - n_action : Number of possible actions (or rewards)
#         - conv / mlp : Int list, describes hidden layers
#         - flatten : If true, flattens the input to one dimension
#         '''
#         # TODO : Conv
#         super().__init__()

#         self.n_state = n_state
#         self.flatten_size = 

#         if mlp == []:
#             layers = [nn.Linear(self.flatten_size, n_action)]
#         else:
#             layers = [nn.Linear(self.flatten_size, mlp[0]), nn.ReLU()]

#             # ReLU activated dense layers
#             for i in range(0, len(mlp) - 1):
#                 layers.append(nn.Linear(mlp[i], mlp[i + 1]))
#                 layers.append(nn.ReLU())

#             # Last layer, no activation
#             layers.append(nn.Linear(mlp[-1], n_action))

#         self.net = nn.Sequential(*layers)

#     def forward(self, state):
#         if self.flatten:
#             state = state.view(-1, self.n_state)

#         return self.net(state)

class Conn(nn.Module):
    '''
        Convolutional Neural Network
    '''
    def __init__(self, depth, n_action, conv=[], mlp=[]):
        '''
        - n_state : Dimension of the state
        - n_action : Number of possible actions (or rewards)
        - conv / mlp : Int list, describes hidden layers
        - flatten : If true, flattens the input to one dimension
        '''
        # TODO : Conv / MLP
        super().__init__()

        self.flatten_size = 32 * 4 * 3

        self.convolve = nn.Sequential(
            nn.Conv2d(depth, 32, 4),
            nn.ReLU(),
        )

        self.connect = nn.Sequential(
            nn.Linear(self.flatten_size, 256),
            nn.ReLU(),
            nn.Linear(256, n_action),
        )

    def forward(self, state):
        state = self.convolve(state)
        state = state.view(-1, self.flatten_size)
        state = self.connect(state)

        return state

