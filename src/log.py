

def as_red(s):
    '''
        As red text
    '''
    return '\033[31m' + s + '\033[0m'


def as_blue(s):
    '''
        As blue text
    '''
    return '\033[34m' + s + '\033[0m'


def as_green(s):
    '''
        As green text
    '''
    return '\033[32m' + s + '\033[0m'


def as_yellow(s):
    '''
        As yellow text
    '''
    return '\033[33m' + s + '\033[0m'


class Logger:
    '''
        Used to display stats
    '''
    def __init__(self, log_freq):
        self.log_freq = log_freq

        self.losses = []
        self.rewards = []
        self.victories = []
        self.draws = []

    def update(self, epoch, reward, victories, draws):
        '''
            Appends data on a game and display it if necessary
        '''
        self.rewards.append(reward)
        self.victories.append(victories)
        self.draws.append(draws)

        if epoch % self.log_freq == 0:
            self.show(epoch)
    
    def show(self, epoch):
        '''
            Displays all informations gathered during training / testing
        * Flushes all data
        '''
        avg_loss = 0 if len(self.losses) == 0 else sum(self.losses) / len(self.losses)
        avg_reward = sum(self.rewards) / len(self.rewards)
        min_reward = min(self.rewards)
        max_reward = max(self.rewards)
        victories = sum(self.victories)

        self.losses.clear()
        self.rewards.clear()
        self.draws.clear()
        self.victories.clear()

        s = '| '
        s += f'Epoch : {epoch:5d} | '
        s += f'Average loss : {avg_loss:<7.4f} | '
        s += f'Average reward : {avg_reward:<5.2f} | '
        s += f'Min reward : {min_reward:<5.2f} | '
        s += f'Max reward : {max_reward:<5.2f} | '
        s += f'Victories : {victories:4d} | '

        print(s)
