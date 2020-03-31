

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
    def __init__(self):
        self.losses = []
        self.rewards = []
    
    def show(self, epoch):
        '''
            Displays all informations gathered during training / testing
        * Flushes all data
        '''
        avg_loss = 0 if len(self.losses) == 0 else sum(self.losses) / len(self.losses)
        avg_reward = sum(self.rewards) / len(self.rewards)
        min_reward = min(self.rewards)
        max_reward = max(self.rewards)

        self.losses.clear()
        self.rewards.clear()

        s = '| '
        s += f'Epoch : {epoch:5d} | '
        s += f'Average loss : {avg_loss:<7.4f} | '
        s += f'Average reward : {avg_reward:<4.2f} | '
        s += f'Min reward : {min_reward:<4.2f} | '
        s += f'Max reward : {max_reward:<4.2f} | '
