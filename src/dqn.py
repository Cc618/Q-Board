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


# class MLP(nn.Module):
#     '''
#         Multi Layer Perceptron, only fully connected layers and activations
#     '''
#     def __init__(self, n_state, n_action, hidden, flatten=False):
#         '''
#         - n_state : Dimension of the state
#         - hidden : Int list, describes hidden layers
#         - n_action : Number of possible actions (or rewards)
#         - flatten : If true, flattens the input to one dimension

#         * Can be represented as :
#         ReLU(Linear(n_state, hidden[0])) -> ReLU(Linear(hidden[i - 1], hidden[i])) -> Linear(hidden[-1], n_action)
#         '''
#         super().__init__()

#         self.n_state = n_state
#         self.flatten = flatten

#         if hidden == []:
#             layers = [nn.Linear(n_state, n_action)]
#         else:
#             layers = [nn.Linear(n_state, hidden[0]), nn.ReLU()]

#             # ReLU activated dense layers
#             for i in range(0, len(hidden) - 1):
#                 layers.append(nn.Linear(hidden[i], hidden[i + 1]))
#                 layers.append(nn.ReLU())

#             # Last layer, no activation
#             layers.append(nn.Linear(hidden[-1], n_action))

#         self.net = nn.Sequential(*layers)

#     def forward(self, state):
#         if self.flatten:
#             state = state.view(-1, self.n_state)

#         return self.net(state)

