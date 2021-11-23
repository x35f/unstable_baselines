import numpy as np
from abc import ABC, abstractmethod
import os
import cv2
class BaseTrainer():
    def __init__(self, agent, logger, train_env, eval_env,args, **kwargs):
        self.agent = agent
        self.logger = logger
        self.train_env = train_env
        self.eval_env = eval_env
        pass

    @abstractmethod
    def train(self):
        #do training 
        pass

    @abstractmethod
    def test(self):
        #do testing
        pass
        
        
    def save_video_demo(self, ite, width=128, height=128, fps=30):
        video_demo_dir = os.path.join(self.logger.log_dir,"demos")
        if not os.path.exists(video_demo_dir):
            os.makedirs(video_demo_dir)
        video_size = (height, width)
        video_save_path = os.path.join(video_demo_dir, "ite_{}.avi".format(ite))

        #initilialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter(video_save_path, fourcc, fps, video_size)

        #rollout to generate pictures and write video
        state = self.eval_env.reset()
        img = self.eval_env.render(mode="rgb_array", width=width, height=height)
        for step in range(self.max_trajectory_length):
            action, _ = self.agent.select_action(state, evaluate=True)
            next_state, reward, done, _ = self.eval_env.step(action)
            state = next_state
            img = self.eval_env.render(mode="rgb_array", width=width, height=height)
            video_writer.write(img)
            if done:
                break
                
        video_writer.release()