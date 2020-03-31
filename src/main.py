import envs
from utils import one_hot_state

import torch.nn.functional as F
import torch as T


env = envs.TicTacToe()


# s, _ = env.reset()
# s, _, _, _ = env.step(0)
# s, _, _, _ = env.step(1)


# # s = F.one_hot((s + 1).to(T.long), 3)
# s = one_hot_state(s, 3, -1)

# print(s[0])
# print(s[1])
# print(s[2])

# exit()



test_games = 100
win, draw = envs.test(envs.TicTacToe.random_act(), envs.TicTacToe.random_act(), env, games=test_games)

print(f'Test on {test_games} games, victories : {win} draws : {draw}')
print(f'Win or draw rate : {(win + draw) / test_games * 100:.1f} %')
