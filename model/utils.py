import numpy as np
import matplotlib.pyplot as plt
import torch
import random
from collections import deque, namedtuple


def display_image(img_array):
    print("image shape: ", img_array.shape)
    plt.imshow(img_array)
    plt.axis("off")
    plt.show()


small_orange = (12,32,2)
large_orange = (12,21,43)
# gotten image has dimension (630,570,4)
# takes image of shape (630,570,4), returns (21,19,3)
def transform_image(img_array):
    # Initialize the result array with shape (21, 19, 3)
    result = np.zeros((21, 19, 3))
    # Iterate through 30x30 blocks
    for i in range(0, img_array.shape[0], 30):
        for j in range(0, img_array.shape[1], 30):
            block = img_array[i:i+30, j:j+30, :3]  # Ignore the alpha channel

            mean_rgb = np.mean(block,axis=(0,1))
            result[i//30, j//30, :] = mean_rgb

    return result / 255


def transform_image_with_thresholding(img_array, threshold=100):
    result = np.zeros((21, 19, 3))
    for i in range(0, img_array.shape[0], 30):
        for j in range(0, img_array.shape[1], 30):
            block = img_array[i:i+30, j:j+30, :3]
            bright_pixels = block > threshold            
            block_weights = bright_pixels.astype(float) * 5 + 1  # Bright pixels weighted more
            weighted_sum = np.sum(block * block_weights, axis=(0, 1))
            weights_sum = np.sum(block_weights, axis=(0, 1))
            avg_rgb = weighted_sum / weights_sum if weights_sum.sum() > 0 else np.zeros(3)
            result[i//30, j//30, :] = avg_rgb
    return result / 255



def plot_steps(steps):
    plt.plot(steps)
    plt.show()



Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory():
    def __init__(self,capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)