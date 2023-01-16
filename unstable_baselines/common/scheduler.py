class Scheduler():
    def __init__(self, initial_val, start_timestep=None , end_timestep=None, target_val=None, schedule_type="linear", value_type=float):
        assert schedule_type in ['linear', 'identical']
        if schedule_type == 'linear':
            assert(target_val != None and start_timestep != None and end_timestep != None)
        self.initial_val = initial_val
        self.target_val=  target_val
        self.start_timestep = start_timestep
        self.end_timestep = end_timestep
        self.curr_timestep = -1
        self.schedule_type = schedule_type
        self.value_type = value_type

    
    def next(self):
        self.curr_timestep += 1
        
        if self.schedule_type == "linear":
            if self.curr_timestep >= self.end_timestep:
                return self.target_val
            elif self.curr_timestep <= self.start_timestep:
                return self.initial_val
            else:
                return self.initial_val + (self.target_val - self.initial_val) * ((self.curr_timestep - self.start_timestep) * 1.0 / (self.end_timestep - self.start_timestep))
        elif self.schedule_type == 'identical':
            return self.initial_val
        else:
            raise NotImplementedError


if __name__ == "__main__":
    scheduler = Scheduler(initial_val=1,
      target_val=15,
      start_timestep=20,
      end_timestep=100,
      schedule_type="linear")
    for i in range(120):
        print(i,scheduler.next())
