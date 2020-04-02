# Environments

import random as rand
import torch as T
from log import as_red, as_blue, as_green


def random_act(n_action):
    '''
        Returns a functor which takes random actions
    - Returns f(state) -> action
    !!! Can take invalid actions
    '''
    def f(_):
        return rand.randint(0, n_action - 1)

    return f


def user_act(n_action):
    '''
        Returns a functor which takes actions from user
    - Returns f(state) -> action
    '''
    def f(_):
        action = None
        while not action:
            try:
                action = int(input(f'Action from 0 to {n_action} > '))
            except:
                pass

        while action < 0 or action >= n_action:
            print('Invalid action')
            action = int(input(f'Action from 0 to {n_action} > '))

        return action

    return f


def play(p1_act, p2_act, env, render=True, state_preprocessor=lambda x: x):
    '''
        Plays a game on env
    - p1_act / p2_act : Functor f(state) -> action
    - Returns total reward for p1 and p2 (in a tuple)
    '''
    state, p1 = env.reset()
    if render:
        env.render()

    done = False
    total_p1_reward, total_p2_reward = 0, 0
    while not done:
        s = state_preprocessor(state)
        action = (p1_act if p1 else p2_act)(s)
        state, reward, done, new_p1 = env.step(action)

        if p1:
            total_p1_reward += reward
        else:
            total_p2_reward += reward

        if render:
            env.render()

        p1 = new_p1

    return total_p1_reward, total_p2_reward


def test(p1_act, p2_act, env, games=100, state_preprocessor=lambda x: x):
    '''
        Tests p1 on several games on env
    - p1_act / p2_act : Functor f(state) -> action
    - Returns (victories, draws)
    !!! Set exploration rate to 0 for accurate test
    '''
    victories = 0
    draws = 0
    for _ in range(games):
        state, p1 = env.reset()
        done = False
        while not done:
            action = (p1_act if p1 else p2_act)(state_preprocessor(state))
            state, reward, done, new_p1 = env.step(action)

            if done:
                if env.was_draw:
                    draws += 1
                elif (p1 and reward > 0) or (not p1 and reward < 0):
                    victories += 1

            p1 = new_p1

    return victories, draws


def train(p1, p2_act, mem, env, epochs, logger, train_p2=True):
    '''
        Trains p1 on several games on env
    - p1 : Agent
    - p2_act : Functor f(state) -> action
    - mem : Memory
    - logger : Used to display stats
    - train_p2 : If True, adds also p2's trajectories
    * The state is preprocessed by p1
    '''
    # TODO : Save
    for e in range(1, epochs + 1):
        total_reward = 0
        state, p1_turn = env.reset()
        state = p1.state_preprocessor(state)
        done = False
        old_p1_state = None
        old_p2_state = None
        while not done:
            act = p1.act if p1_turn else p2_act
            action = act(state)

            new_state, reward, done, new_p1_turn = env.step(action)
            new_state = p1.state_preprocessor(new_state)

            # Add trajectories in parallel
            if p1_turn:
                old_p1_state = state
                total_reward += reward
            else:
                old_p2_state = state

            # TODO : Train p2
            if new_p1_turn:
                if old_p1_state is not None:
                    mem.add(action, old_p1_state, new_state, reward, done)
            elif old_p2_state is not None:
                    mem.add(action, old_p2_state, new_state, reward, done)

            state = new_state
            p1_turn = new_p1_turn

        victory = int(not env.was_draw and ((p1 and reward > 0) or (not p1 and reward < 0)))
        logger.update(e, total_reward, victory, int(env.was_draw))


class BoardEnv:
    '''
        Abstract class for all environments
    * Each environment has two players
    '''
    def __init__(self, n_state, n_action):
        '''
        - n_state : Dimension of the observation space (the state is a tensor of shape [n_state])
        - n_action : Dimension of the action space (the action is a long within [0, n_action))
        '''
        self.n_state = n_state
        self.n_action = n_action

        self.reset()

    def __repr__(self):
        s = as_green('P1\n' if self.p1_turn else 'P2\n')

        return s + self.to_str()

    def reset(self):
        '''
            Resets the game
        - Returns state, p1_turn (in children)
        !!! Must be called by children
        '''
        self.p1_turn = rand.randint(0, 1) == 0
        self.was_draw = False

    def to_str(self):
        '''
            Returns the string representation of the environment
        '''
        raise NotImplementedError()

    def render(self):
        print(str(self))


class TicTacToe(BoardEnv):
    REWARD_WIN = 1
    REWARD_LOOSE = -1
    REWARD_DRAW = -.1
    REWARD_NONE = 0
    REWARD_INVALID_ACTION = -1

    def __init__(self):
        super().__init__(n_state=9, n_action=9)

    def __p2_state(self):
        '''
            State for the second player
        '''
        return -self.state

    def __play_turn(self, action):
        '''
        - Returns reward, done
        * action is supposed within 0 and 8 included
        '''
        # Invalid action
        if self.state[action] != 0:
            return TicTacToe.REWARD_INVALID_ACTION, True

        # The value which represents current player's chip
        player_id = 1 if self.p1_turn else -1

        # Update state
        self.state[action] = player_id

        # Check diagonals
        if self.state[4] != 0:
            if (self.state[0] == self.state[4] and self.state[4] == self.state[8]) or \
                    (self.state[2] == self.state[4] and self.state[4] == self.state[6]):
                return TicTacToe.REWARD_WIN if self.state[4] == player_id else TicTacToe.REWARD_LOOSE, True

        # Check columns
        for x in range(3):
            if self.state[x] != 0:
                if self.state[x] == self.state[x + 3] and self.state[x + 3] == self.state[x + 6]:
                    return TicTacToe.REWARD_WIN if self.state[x] == player_id else TicTacToe.REWARD_LOOSE, True

        # Check rows
        for y in range(3):
            if self.state[y * 3] != 0:
                if self.state[y * 3] == self.state[y * 3 + 1] and self.state[y * 3 + 1] == self.state[y * 3 + 2]:
                    return TicTacToe.REWARD_WIN if self.state[y * 3] == player_id else TicTacToe.REWARD_LOOSE, True

        # Check draw
        if self.turns >= 9:
            self.was_draw = True
            return TicTacToe.REWARD_DRAW, True

        # Not a game end
        return TicTacToe.REWARD_NONE, False

    def reset(self):
        '''
            Resets the game
        - Returns state, p1_turn
        '''
        super().reset()

        # State for player 1
        self.state = T.zeros([9], dtype=T.long)
        self.turns = 0

        return self.state, self.p1_turn

    def to_str(self):
        s = '-------\n'

        def digit_symbol(c, i):
            '''
                Returns the string representation of the state
            '''
            if c == -1:
                return as_blue('O')
            if c == 1:
                return as_red('X')
            return str(i)

        for y in range(3):
            # Column
            s += '|'
            for x in range(3):
                # Symbol
                s += digit_symbol(self.state[x + y * 3], x + y * 3)
                # Column
                s += '|'
            s += '\n'
        s += '-------'

        return s

    def step(self, action):
        '''
            Action is either an integer or a long tensor
        - Returns state, reward, done, p1_turn
        * The returned state and reward are only for the player that just played
        '''
        # Current turn
        turn = self.p1_turn
        self.turns += 1

        # Play turn
        reward, done = self.__play_turn(action)

        # Change turn
        self.p1_turn = not self.p1_turn

        return (self.state if turn else self.__p2_state()), reward, done, self.p1_turn

    @classmethod
    def random_act(cls):
        '''
            Creates a functor that takes valid random actions for this env
        '''
        def act(state):
            a = rand.randint(0, 8)
            while state[a] != 0:
                a = rand.randint(0, 8)

            return a

        return act

    # TODO :
    # @classmethod
    # def minimax_act(cls, depth):
    #     '''
    #         Creates a functor that takes best actions using minimax algorithm
    #     !!! The state must be unpreprocessed
    #     '''
    #     def score(state):
    #         '''
    #             Score in this state, 0 if draw / nothing, 1 if win
    #         '''
    #         # Check diagonals
    #         if state[4] != 0:
    #             if (state[0] == state[4] and state[4] == state[8]) or \
    #                     (state[2] == state[4] and state[4] == state[6]):
    #                 return 1

    #         # Check columns
    #         for x in range(3):
    #             if state[x] != 0:
    #                 if state[x] == state[x + 3] and state[x + 3] == state[x + 6]:
    #                     return 1

    #         # Check rows
    #         for y in range(3):
    #             if state[y * 3] != 0:
    #                 if state[y * 3] == state[y * 3 + 1] and state[y * 3 + 1] == state[y * 3 + 2]:
    #                     return 1

    #         # Draw
    #         return 0

    #     def minimax(state, depth):
    #         for action in range(9):
    #             # If empty
    #             if state[action] == 0:





    #     def act(state):
    #         scores = [minimax(action, depth) for action in range(9)]

    #         # Arg max
    #         amax = 0
    #         for i in range(1, 9):
    #             if scores[i] > scores[amax]:
    #                 amax = i

    #         return amax

    #     return act
