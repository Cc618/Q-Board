from random import shuffle
import torch as T


class LinearMemory:
    '''
        A basic memory, when 'size' steps are memorized,
    the functor on_learn is called with a batch as parameter.
    All trajectories are learned and have same weights
    '''
    def __init__(self, state_dim, size, on_learn):
        '''
        - state_dim : Int list or int, dimension of the state
        '''
        super().__init__()

        # Make compatible int and int list
        if isinstance(state_dim, int):
            state_dim = [state_dim] 

        self.size = size
        self.on_learn = on_learn
        self.sample_i = 0

        self.actions = T.empty([size], dtype=T.long)
        self.states = T.empty([size, *state_dim])
        self.next_states = T.empty([size, *state_dim])
        self.rewards = T.empty([size])
        self.dones = T.empty([size])

    def add(self, action, state, next_state, reward, done):
        self.actions[self.sample_i] = action
        self.states[self.sample_i] = state
        self.next_states[self.sample_i] = next_state
        self.rewards[self.sample_i] = reward
        self.dones[self.sample_i] = done

        self.sample_i += 1
        if self.sample_i >= self.size:
            # Shuffle data
            idx = [i for i in range(self.size)]
            shuffle(idx)

            self.on_learn(self.actions[idx], self.states[idx], self.next_states[idx], self.rewards[idx], self.dones[idx])

            # 'Clear' data
            self.sample_i = 0

