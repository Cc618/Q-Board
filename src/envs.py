# Environments

import random as rand
import torch as T


def seed(seed):
    '''
        Sets the seed of all environments (python + pytorch seed)
    '''
    rand.seed(seed)
    T.manual_seed(seed)


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
        s = 'P1\n' if self.p1_turn else 'P2\n'

        return s + self.to_str()

    def reset(self):
        '''
            Resets the game
        - Returns state, p1_turn (in children)
        !!! Must be called by children
        '''
        self.p1_turn = rand.randint(0, 1) == 0

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
        self.state = T.zeros([9], dtype=T.float32)
        self.turns = 0

        return self.state, self.p1_turn

    def to_str(self):
        s = '-------\n'

        def digit_symbol(c, i):
            '''
                Returns the string representation of the state
            '''
            if c == -1:
                return 'O'
            if c == 1:
                return 'X'
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
