import envs
import agents
import dqn
from log import Logger
from mem import LinearMemory
from utils import f_one_hot_state
import solvers








solvers.connect4()







# env = envs.TicTacToe()


# # test_games = 100
# # win, draw = envs.test(envs.TicTacToe.random_act(), envs.TicTacToe.random_act(), env, games=test_games)

# # print(f'Test on {test_games} games, victories : {win} draws : {draw}')
# # print(f'Win or draw rate : {(win + draw) / test_games * 100:.1f} %')



# epochs = 5000
# test_games = 100
# mem_size = 100
# log_freq = 500

# n_state = env.n_state * env.n_action
# log = Logger(log_freq)
# net = dqn.MLP(n_state, env.n_action, [256], flatten=True)
# ai = agents.DQNAgent(env.n_state, env.n_action, net, logger=log, lr=1e-3, state_preprocessor=f_one_hot_state(env.n_action, -1, flatten=True))
# rand_act = envs.TicTacToe.random_act()

# # Training
# envs.train(ai, rand_act, LinearMemory(n_state, mem_size, ai.learn), env, epochs, log, False)

# # Testing
# win, draw = envs.test(ai.act, rand_act, env, state_preprocessor=ai.state_preprocessor)

# print(f'Test on {test_games} games, victories : {win} draws : {draw}')
# print(f'Win or draw rate : {(win + draw) / test_games * 100:.1f} %')

# # Playing
# while 1:
#     print('GAME')
#     envs.play(ai.act, envs.user_act(env.n_action), env, state_preprocessor=ai.state_preprocessor)


