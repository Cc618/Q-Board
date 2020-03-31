from random import random, randint
import torch as T
from torch import optim, nn
import torch.nn.functional as F

class QAgent:
    '''
        Base (abstract) class for agents
    '''
    def __init__(self, n_state, n_action, dqn, lr=1e-3, discount_factor=.98, exploration_decay=.99, exploration_min=.05):
        self.n_state = n_state
        self.n_action = n_action
        self.dqn = dqn
        self.discount_factor = discount_factor
        self.exploration_decay = exploration_decay
        self.exploration_min = exploration_min

        self.exploration_rate = 0
        self.opti = optim.Adam(self.dqn.parameters(), lr=lr)

    def __get_loss(self, actions, states, next_states, rewards, dones):
        '''
            Computes the loss, doesn't back prop
        '''
        raise NotImplementedError()

    def get_rewards(self, state):
        return self.dqn(state)

    def act(self, state):
        if random() < self.exploration_rate:
            return randint(0, self.n_action - 1)
        
        return T.argmax(self.get_rewards(state)).detach().item()

    def learn(self, actions, states, next_states, rewards, dones):
        '''
            Learns from trajectories
        '''
        # TODO : Exploration rate change here ?
        self.exploration_rate = min(self.exploration_rate * self.exploration_decay, self.exploration_min)

        loss = self.__get_loss(actions, states, next_states, rewards, dones)

        # TODO : Logger add loss
        self.opti.zero_grad()
        loss.backward()
        self.opti.step()
        

class DQNAgent(QAgent):
    '''
        Simple Deep Q Network
    '''
    def __init__(self, n_state, n_action, dqn, lr=1e-3, discount_factor=.98, exploration_decay=.99, exploration_min=.05):
        super().__init__(n_state, n_action, dqn, lr, discount_factor, exploration_decay, exploration_min)

    def __get_loss(self, actions, states, next_states, rewards, dones):
        # Predicted Q values
        q = (self.dqn(states) * F.one_hot(actions, self.n_action)).sum(1)

        # The target Q values
        q_target = rewards + self.discount_factor * (1 - dones) * T.max(next_states, 1)[0]

        loss = F.mse_loss(q, q_target.detach()).mean()
        # loss = F.smooth_l1_loss(q, q_target.detach()).mean()

        return loss
