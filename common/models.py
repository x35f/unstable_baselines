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