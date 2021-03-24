import numpy as np
from abc import ABC, abstractmethod
class BaseTrainer():
    def __init__(self, agent, args, **kwargs):
        pass

    @abstractmethod
    def train(self):
        #do training 
        pass

    @abstractmethod
    def test(self):
        #do testing
        pass

    @abstractmethod
    def save_video_demo(self, ite, width=128, height=128, fps=30):
         pass
        
        