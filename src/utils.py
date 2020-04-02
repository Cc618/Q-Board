import random as rand
import torch.nn.functional as F
import torch as T


def f_one_hot_state(depth, min_depth, flatten=False):
    '''
        Functor for one_hot_state
    '''
    if flatten:
        return lambda state: one_hot_state(state, depth, min_depth).view(-1)
    else:
        return lambda state: one_hot_state(state, depth, min_depth)


def one_hot_state(state, depth, min_depth):
    '''
        Returns the one hot encoded state (tensor of type float32)
    - state : LongTensor
    - depth : Number of categories, ie 3 for Tic Tac Toe (X, O and empty)
    - min_depth : Minimum value associated to depth, -1 for Tic Tac Toe since O = -1
    '''
    return F.one_hot(state - min_depth, depth).to(T.float32)


def seed(seed):
    '''
        Sets the seed of all environments (python + pytorch seed)
    '''
    rand.seed(seed)
    T.manual_seed(seed)


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
