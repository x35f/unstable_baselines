from abc import abstractmethod
class BaseAgent(object):
    def __init__(self,**kwargs):
        super(BaseAgent,self).__init__(kwargs)
    
    @abstractmethod
    def update(self,data_batch):
        pass

    @abstractmethod
    def select_action(self, state):
        pass

    @abstractmethod
    def load_model(self, dir):
        pass
    
    @abstractmethod
    def save_model(self, target_dir, ite):
        pass