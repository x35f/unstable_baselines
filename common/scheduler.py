
class Scheduler(object):
    def __init__(self, start_value, end_value, duration: int):
        self.start_value = start_value
        self.end_value = end_value
        self.curr = 0
        self.duration = max(1, duration)
    
    def next(self):
        frac = min(self.curr, self.duration) / self.duration
        self.curr = min(self.curr + 1, self.duration)
        return (self.end_value - self.start_value) * frac + self.start_value
            
    def reset(self, idx = 0):
        self.curr = idx