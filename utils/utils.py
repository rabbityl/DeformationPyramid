import os
class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0.0
        self.sq_sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.sq_sum += val ** 2 * n

class Logger:
    def __init__(self, log_path):
        # self.path = path
        # log_path = self.path + '/log'
        if os.path.exists(log_path):
            os.remove(log_path)
        self.fw = open(log_path,'a')

    def write(self, text):
        self.fw.write(text)
        self.fw.flush()

    def close(self):
        self.fw.close()
