class Scheduler():
    def __init__(self, initial_val, final_val=None, num_iterations=None, schedule_type = "linear", value_type = float):
        assert schedule_type in ['linear', 'identical']
        self.initial_val = initial_val
        self.final_val=  final_val
        self.tot_iterations = num_iterations
        self.curr_iteration = -1
        self.schedule_type = schedule_type
        self.value_type = value_type
    
    def next(self):
        self.curr_iteration += 1
        
        if self.schedule_type == "linear":
            assert(self.final_val != None)
            if self.curr_iteration >= self.tot_iterations:
                return self.final_val
            return self.initial_val + (self.final_val - self.initial_val) * (self.curr_iteration / self.tot_iterations)
        elif self.schedule_type == 'identical':
            return self.initial_val
        else:
            raise NotImplementedError