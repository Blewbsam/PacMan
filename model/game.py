import sys
import os
import numpy
import matplotlib.pyplot as plt
from PIL import Image
import torch
from utils import *
from utils import transform_image_with_thresholding
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'build')) # relative path to build folder used for import
import pacman

DIRECTIONS = ["Up","Down","Left", "Right"]

class Game:
    def __init__(self):
        self.gs = pacman.GameState()
        self.display = pacman.Display(self.gs)
    def update(self):
        self.display.update()
        self.display.render()
    def get_state(self):
        if self.gs.game_over():
            return None
        image = self.display.get_screenshot()
        image = image.transpose(2,0,1)
        return torch.from_numpy(image).float() #of shape (3,630,570)
    def get_reduced_state(self):  
        if self.gs.game_over():
            return None
        image = transform_image_with_thresholding(self.display.get_screenshot())
        image = image.transpose(2,0,1)
        return torch.from_numpy(image).float() #of shape (3,21,19)
    def get_score(self):
        return self.gs.get_score()
    def running(self):
        return not self.gs.game_over()
    def decision_available(self):
        ''' returns true is pacman is contained in a cell. i.e able to switch direction'''
        return self.display.pacman_contained()
    def step(self,action):
        ''' makes action in environment '''
        assert action >= 0 and action <= 3
        self.display.step(action)
        self.update()
    def is_game_lost(self):
        assert self.gs.game_over()
        return self.gs.is_game_lost()

if __name__ == "__main__":
    game = Game()
    while game.running():
        game.update()
 