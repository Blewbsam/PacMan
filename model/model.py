import torch
from PIL import Image
import torch.nn as nn
import numpy as np

class DQNAgent(nn.Module):
    def __init__(self,num_actions):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=1,kernel_size=4,stride=1),
            nn.ReLU(),
            # nn.Conv2d(in_channels=3, out_channels=3, kernel_size=8, stride=2),
            # nn.ReLU(),
            # nn.Conv2d(in_channels=3,out_channels=3,kernel_size=32,stride=1),
            # nn.ReLU(),
            nn.Flatten(),            
            nn.Linear(288, 512),  # Fully connected layer (dimensions must be fixed)
            nn.ReLU(),
            nn.Linear(512, num_actions),  # Output layer for Q-values
        )
    def forward(self,x):
        return self.model(x)



# img = Image.open("test.png")
# data = np.asarray(img)
# data = data[:,:,:3]
# data = data.transpose(2, 0, 1)

# print(data.shape)

# # Convert data to PyTorch tensor
# data = torch.from_numpy(data).float()  # Convert to float32

# # Add a batch dimension (batch_size, channels, height, width)
# data = data.unsqueeze(0)  # Adds a batch dimension, making the shape (1, channels, height, width)
# print(data.shape)

# shape = data.shape

# # Initialize the agent
# agent = DQNAgent(shape, 4)
# print(agent)

# # Forward pass through the model
# res = agent.forward(data)
# print(res)