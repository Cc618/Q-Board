import envs


env = envs.TicTacToe()


print(envs.play(envs.user_act(env.n_action), envs.TicTacToe.random_act(), env))

