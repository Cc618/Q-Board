import envs
import agents
import dqn
from log import Logger
from mem import LinearMemory
from utils import f_one_hot_state


env = envs.TicTacToe()


# test_games = 100
# win, draw = envs.test(envs.TicTacToe.random_act(), envs.TicTacToe.random_act(), env, games=test_games)

# print(f'Test on {test_games} games, victories : {win} draws : {draw}')
# print(f'Win or draw rate : {(win + draw) / test_games * 100:.1f} %')


# TODO : Process state


epochs = 5000
test_games = 100
mem_size = 200
log_freq = 500

log = Logger(log_freq)
net = dqn.MLP(env.n_state * env.n_action, env.n_action, [256], flatten=True)
ai = agents.DQNAgent(env.n_state, env.n_action, net, logger=log, lr=2e-4, state_preprocessor=f_one_hot_state(env.n_action, -1))
rand_act = envs.TicTacToe.random_act()

# Training
envs.train(ai.act, rand_act, LinearMemory(env.n_state, mem_size, ai.learn), env, epochs, log, False)

# Testing
win, draw = envs.test(ai.act, rand_act, env)

print(f'Test on {test_games} games, victories : {win} draws : {draw}')
print(f'Win or draw rate : {(win + draw) / test_games * 100:.1f} %')
