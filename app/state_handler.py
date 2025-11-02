import numpy as np
import torch
import threading
from nca.utils.config import Config

state_lock = threading.Lock()

class StateHandler:
    def __init__(self, config: Config, img_size, color_dim, cond_dim):
        self.config = config
        self.color_dim = color_dim
        self.img_canvas = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        self.img_tensor = torch.zeros((1, color_dim, img_size, img_size))
        self.seed_tensor = torch.zeros((1, color_dim, img_size, img_size))
        self.condition_tensor = torch.zeros((1, cond_dim))
        self.speed = 10


    def get_speed(self):
        return float(1.0 / self.speed)
    
    def set_speed(self, speed):
        self.speed = speed

    def get_color_dim(self):
        return self.color_dim

    def get_config(self) -> Config:
        return self.config

    def get_img_canvas(self):
        return self.img_canvas
    
    def set_img_canvas(self, img_canvas):
        if img_canvas.shape[2] != self.img_canvas.shape[2]:
            raise ValueError(f"New image canvas has different dimensions than the old one, expected {self.img_canvas.shape[2]} channels, got {img_canvas.shape[2]} channels")
        self.img_canvas = img_canvas[:, :, :3]
        return self.img_canvas
    
    def get_img_tensor(self):
        return self.img_tensor
    
    def set_img_tensor(self, img_tensor):
        self.img_tensor = img_tensor
        return self.img_tensor
    
    def get_seed_tensor(self):
        return self.seed_tensor
    
    def set_seed_tensor(self, seed_tensor):
        self.seed_tensor = seed_tensor
        return self.seed_tensor
    
    def get_condition_tensor(self):
        return self.condition_tensor
    
    def set_condition_tensor(self, condition_tensor):
        self.condition_tensor = condition_tensor
        return self.condition_tensor
    
    def set_state(self, img_canvas, img_tensor, condition_tensor):
        self.img_canvas = img_canvas
        self.img_tensor = img_tensor
        self.condition_tensor = condition_tensor
        return self.img_canvas, self.img_tensor, self.condition_tensor

    def get_state(self):
        return self.img_canvas, self.img_tensor, self.condition_tensor
    
    def lock_state(self):
        state_lock.acquire()        

    def release_state(self):
        state_lock.release()

    def get_lock(self):
        return state_lock