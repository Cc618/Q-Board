import envs
from utils import one_hot_state

import torch.nn.functional as F
import torch as T


env = envs.TicTacToe()


s, _ = env.reset()
s, _, _, _ = env.step(0)
s, _, _, _ = env.step(1)


# s = F.one_hot((s + 1).to(T.long), 3)
s = one_hot_state(s, 3, -1)

print(s[0])
print(s[1])
print(s[2])

exit()

# print(envs.play(envs.user_act(env.n_action), envs.TicTacToe.random_act(), env))

