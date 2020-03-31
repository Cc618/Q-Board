from random import random, randint
import torch as T
from torch import optim, nn
import torch.nn.functional as F


def make(agent, dqn, env, dqn_args=[], agent_args=[]):
    '''
        Shorthand to make an agent
    - agent : Agent's class
    - dqn : Agent's class
    '''
    return agent(env.n_state, env.n_action, dqn(env.n_state, env.n_action, *dqn_args), *agent_args)


class QAgent:
    '''
        Base (abstract) class for agents
    '''
    def __init__(self, n_state, n_action, dqn, logger=None, lr=1e-3,
                discount_factor=.98, exploration_decay=.99,
                exploration_min=.05, state_preprocessor=lambda x: x):
        self.n_state = n_state
        self.n_action = n_action
        self.dqn = dqn
        self.logger = logger
        self.discount_factor = discount_factor
        self.exploration_decay = exploration_decay
        self.exploration_min = exploration_min
        self.state_preprocessor = state_preprocessor

        self.exploration_rate = 1
        self.opti = optim.Adam(self.dqn.parameters(), lr=lr)

    def __get_loss(self, actions, states, next_states, rewards, dones):
        '''
            Computes the loss, doesn't back prop
        '''
        raise NotImplementedError()

    def get_rewards(self, state):
        state = self.state_preprocessor(state)

        return self.dqn(state)

    def act(self, state):
        if random() < self.exploration_rate:
            return randint(0, self.n_action - 1)
        
        return T.argmax(self.get_rewards(state)).detach().item()

    def learn(self, actions, states, next_states, rewards, dones):
        '''
            Learns from trajectories
        * states and next_states are already preprocessed
        '''
        # TODO : Exploration rate change here ?
        self.exploration_rate = min(self.exploration_rate * self.exploration_decay, self.exploration_min)

        loss = self.__get_loss(actions, states, next_states, rewards, dones)

        if self.logger:
            self.logger.losses.append(loss)

        self.opti.zero_grad()
        loss.backward()
        self.opti.step()
        

class DQNAgent(QAgent):
    '''
        Simple Deep Q Network
    '''
    def __init__(self, n_state, n_action, dqn, *args, **kwargs):
        '''
            Args are super args
        '''
        super().__init__(n_state, n_action, dqn, *args, **kwargs)

    def __get_loss(self, actions, states, next_states, rewards, dones):
        # Predicted Q values
        q = (self.dqn(states) * F.one_hot(actions, self.n_action)).sum(1)

        # The target Q values
        q_target = rewards + self.discount_factor * (1 - dones) * T.max(next_states, 1)[0]

        loss = F.mse_loss(q, q_target.detach()).mean()
        # loss = F.smooth_l1_loss(q, q_target.detach()).mean()

        return loss
