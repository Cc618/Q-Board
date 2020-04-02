# Environments

import random as rand
import torch as T
from log import as_red, as_blue, as_green


class BoardEnv:
    '''
        Abstract class for all environments
    * Each environment has two players
    '''
    def __init__(self, n_state, n_action):
        '''
        - n_state : Dimension of the observation space
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
        self.turns = 0
        self.p1_turn = rand.randint(0, 1) == 0
        self.was_draw = False

    def to_str(self):
        '''
            Returns the string representation of the environment
        '''
        raise NotImplementedError()

    def p2_state(self):
        '''
            State for the second player
        '''
        return -self.state

    def step(self, action):
        '''
            Action is an integer or a tensor (depends on the env)
        - Returns state, reward, done, p1_turn
        * The returned state and reward are only for the player that just played
        '''
        # Current turn
        turn = self.p1_turn
        self.turns += 1

        # Play turn
        reward, done = self.play_turn(action)

        # Change turn
        self.p1_turn = not self.p1_turn

        return (self.state if turn else self.p2_state()), reward, done, self.p1_turn

    def render(self):
        print(str(self))


class TicTacToe(BoardEnv):
    REWARD_WIN = 1
    REWARD_LOOSE = -1
    REWARD_DRAW = -.1
    REWARD_NONE = 0
    REWARD_INVALID_ACTION = -10

    def __init__(self):
        super().__init__(n_state=9, n_action=9)

    def play_turn(self, action):
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


class Connect4(BoardEnv):
    REWARD_WIN = 1
    REWARD_LOOSE = -1
    REWARD_DRAW = -.1
    REWARD_NONE = 0
    REWARD_INVALID_ACTION = -10
    WIDTH = 7
    HEIGHT = 6

    def __init__(self):
        super().__init__(n_state=[Connect4.HEIGHT, Connect4.WIDTH], n_action=Connect4.WIDTH)

    def __winner(self):
        '''
            Wether there is a winner
        '''
        # Diags
        for x in range(Connect4.WIDTH - 3):
            for y in range(Connect4.HEIGHT - 3):
                if self.state[x, y] == self.state[x + 1, y + 1] == self.state[x + 2, y + 2] == self.state[x + 3, y + 3] != 0:
                    return True

                if self.state[x + 3, y] == self.state[x + 2, y + 1] == self.state[x + 1, y + 2] == self.state[x, y + 3] != 0:
                    return True

        # Rows
        for x in range(Connect4.WIDTH - 3):
            for y in range(Connect4.HEIGHT):
                if self.state[x, y] == self.state[x + 1, y] == self.state[x + 2, y] == self.state[x + 3, y] != 0:
                    return True

        # Cols
        for x in range(Connect4.WIDTH):
            for y in range(Connect4.HEIGHT - 3):
                if self.state[x, y] == self.state[x, y + 1] == self.state[x, y + 2] == self.state[x, y + 3] != 0:
                    return True

        return False

    def play_turn(self, action):
        '''
        - Returns reward, done
        * action is supposed within 0 and 8 included
        '''
        pos = -1

        # Get the position of the chip
        for y in range(Connect4.HEIGHT - 1, -1, -1):
            if self.state[action, y] == 0:
                pos = y
                break

        # Invalid action
        if pos == -1:
            return Connect4.REWARD_INVALID_ACTION, True

        # The value which represents current player's chip
        player_id = 1 if self.p1_turn else -1

        # Update state
        self.state[action, pos] = player_id

        if self.__winner():
            return Connect4.REWARD_WIN, True

        if self.turns >= Connect4.WIDTH * Connect4.HEIGHT:
            return Connect4.REWARD_DRAW, True

        # Not a game end
        return Connect4.REWARD_NONE, False

    def reset(self):
        '''
            Resets the game
        - Returns state, p1_turn
        '''
        super().reset()

        # State for player 1
        # [x, y]
        self.state = T.zeros([Connect4.WIDTH, Connect4.HEIGHT], dtype=T.long)
        self.turns = 0

        return self.state, self.p1_turn

    def to_str(self):
        sep = '--' + '---' * Connect4.WIDTH
        s = sep + '\n'

        def symbol(c, x, y):
            '''
                Returns the string representation of the state
            '''
            if c == -1:
                return as_blue(' O')
            if c == 1:
                return as_red(' X')
            return ' ' + str(x)

        for y in range(Connect4.HEIGHT):
            # Column
            s += '|'
            for x in range(Connect4.WIDTH):
                # Symbol
                s += symbol(self.state[x][y], x, y)
                # Column
                s += '|'
            s += '\n'

        s += sep

        return s

    @classmethod
    def random_act(cls):
        '''
            Creates a functor that takes valid random actions for this env
        '''
        def act(state):
            a = rand.randint(0, 6)
            while state[a, 0] != 0:
                a = rand.randint(0, 6)

            return a

        return act

    @classmethod
    def towers_act(cls):
        '''
            Random actions but when there is a stack of 3
        chips, plays this position
        '''
        def act(state):                
            for x in range(Connect4.WIDTH):
                # Get the position of the chip
                for y in range(Connect4.HEIGHT):
                    if state[x, y] == 0:
                        # y is the top of the stack
                        if y >= 3 and state[x, y - 1] == state[x, y - 2] == state[x, y - 3]:
                            return x

            # Random move
            a = rand.randint(0, 6)
            while state[a, 0] != 0:
                a = rand.randint(0, 6)

            return a

        return act

