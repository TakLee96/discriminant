from time import time


stringify = lambda a: map(lambda x: str(x), a)


class Timer:

    def __init__(self):
        self.time = time()

    def start(self, *args):
        self.time = time()
        print ' '.join(stringify(args))

    def end(self, *args):
        diff = time() - self.time
        print '....', ' '.join(stringify(args)), '[' + str(diff) + ' sec]', '\n'


timer = Timer()
__all__ = [ timer ]
