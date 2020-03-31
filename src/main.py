import env


ev = env.TicTacToe()
ev.reset()

ev.step(0)
ev.step(1)
ev.step(2)
ev.step(4)
ev.step(5)
ev.step(3)
ev.step(6)
ev.step(8)

a = ev.step(7)
print(*a)

ev.render()



