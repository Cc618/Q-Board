import envs


env = envs.TicTacToe()
env.reset()

env.step(0)
env.step(1)
env.step(2)
env.step(4)
env.step(5)
env.step(3)
env.step(6)
env.step(8)

a = env.step(7)
print(*a)

env.render()



