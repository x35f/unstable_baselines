import numpy as np
from abc import ABC, abstractmethod
class BaseTrainer():
    def __init__(self, agent, logger, args, **kwargs):
        pass

    @abstractmethod
    def train(self):
        #do training 
        pass

    @abstractmethod
    def test(self):
        #do testing
        pass
        
        