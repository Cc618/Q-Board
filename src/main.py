import envs
import agents
import dqn
from log import Logger
from mem import LinearMemory
from utils import one_hot_state


env = envs.TicTacToe()


# test_games = 100
# win, draw = envs.test(envs.TicTacToe.random_act(), envs.TicTacToe.random_act(), env, games=test_games)

# print(f'Test on {test_games} games, victories : {win} draws : {draw}')
# print(f'Win or draw rate : {(win + draw) / test_games * 100:.1f} %')


# TODO : Process state


ai = agents.make(agents.DQNAgent, dqn.MLP, env, [[128]], [])

log = Logger(20)
mem_size = 100
envs.train(ai.act, ai.act, LinearMemory(env.n_state, mem_size, ai.learn), env, 100, log)


