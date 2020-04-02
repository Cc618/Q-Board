# Gathers AIs for environments

import envs
import agents
import dqn
from log import Logger
from mem import LinearMemory
from utils import f_one_hot_state


def tic_tac_toe(path='data/tic_tac_toe', seed=161831415):
    env = envs.TicTacToe()

    rand_epochs = 5000
    ai_epochs = 0

    test_games = 100
    mem_size = 200
    log_freq = 500

    # 3 states per position
    depth = 3
    # The state is preprocessed and has this shape now
    n_state = env.n_state * depth
    log = Logger(log_freq)
    # Simple dqn
    net = dqn.MLP(n_state, env.n_action, [256], flatten=True)
    ai = agents.DQNAgent(env.n_state, env.n_action, net,
                        logger=log, lr=5e-4, discount_factor=.92,
                        exploration_decay=.98, exploration_min=.1,
                        state_preprocessor=f_one_hot_state(depth, -1, flatten=True))
    mem = LinearMemory(n_state, mem_size, ai.learn)
    # Train first against random agent
    rand_act = envs.TicTacToe.random_act()

    # Loading
    ai.load(path)

    # Training
    print('Training vs random')
    envs.train(ai, rand_act, mem, env, rand_epochs, log, False)
    print('Training vs ai')
    envs.train(ai, ai.act, mem, env, ai_epochs, log, True)

    # Saving
    ai.save(path)

    # Testing
    ai.exploration_rate = 0
    win, draw = envs.test(ai.act, rand_act, env, games=test_games, state_preprocessor=ai.state_preprocessor)

    print(f'Test on {test_games} games : Victories : {win} Draws : {draw}')
    print(f'Win or draw rate : {(win + draw) / test_games * 100:.1f} %')

    # Playing
    while 1:
        print('New Game')
        p1, p2 = envs.play(ai.act, envs.user_act(env.n_action), env, state_preprocessor=ai.state_preprocessor)
        if p1 > 0:
            print('AI won')
        elif p2 > 0:
            print('You won')
        else:
            print('Error / Draw')

