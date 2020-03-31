from random import shuffle
import torch as T


class LinearMemory:
    '''
        A basic memory, when 'size' steps are memorized,
    the functor on_learn is called with a batch as parameter.
    All trajectories are learned and have same weights
    '''
    def __init__(self, n_state, size, on_learn):
        super().__init__()

        self.size = size
        self.on_learn = on_learn
        self.sample_i = 0

        self.states = T.empty([size, n_state])
        self.next_states = T.empty([size, n_state])
        self.actions = T.empty([size], dtype=T.long)
        self.rewards = T.empty([size])
        self.dones = T.empty([size])

    def add(self, state, next_state, action, reward, done):
        self.states[self.sample_i] = state
        self.next_states[self.sample_i] = next_state
        self.actions[self.sample_i] = action
        self.rewards[self.sample_i] = reward
        self.dones[self.sample_i] = done

        self.sample_i += 1
        if self.sample_i >= self.size:
            # Shuffle data
            idx = [i for i in range(self.size)]
            shuffle(idx)

            self.on_learn(self.states[idx], self.next_states[idx], self.actions[idx], self.rewards[idx], self.dones[idx])

            # 'Clear' data
            self.sample_i = 0

