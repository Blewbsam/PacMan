import torch
from PIL import Image
import torch.nn as nn
import numpy as np

class DQNAgent(nn.Module):
    '''
    Pure Convolutional Network
    '''
    def __init__(self,num_actions):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1),  # Reduce spatial dimensions
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(195, 512),  # Adjust based on input size and strides
            nn.ReLU(),
            nn.Linear(512, num_actions),  # Output layer for Q-values
        )
    def forward(self,x):
        return self.model(x)

    def get_name(self):
        return "DQN"
    


class DQRNAgent(nn.Module):
    '''
    Convolutional with a lstm appended
    '''
    def __init__(self,num_actions):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=16,kernel_size=3,stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32,out_channels=1,kernel_size=3,stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.lstm = nn.LSTM(input_size=195, hidden_size=512, batch_first=True)
        self.fc = nn.Linear(512, num_actions)  # Output 4 actions 
    def forward(self, x):
        batch_size, seq_len, channels, height, width = x.size()

        x = x.view(batch_size * seq_len, channels, height, width)  # Shape: [batch_size * seq_len, channels, height, width]
        
        x = self.cnn(x)  # Shape: [batch_size * seq_len, feature_dim]
        
        feature_dim = x.size(-1)
        x = x.view(batch_size, seq_len, feature_dim)  # Shape: [batch_size, seq_len, feature_dim]
        
        # Process with LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)  # LSTM output: [batch_size, seq_len, hidden_size]
        
        x = h_n[-1]  # Final hidden state: [batch_size, hidden_size]
        
        x = self.fc(x)  # Output shape: [batch_size, num_actions]
        return x
    def get_name(self):
        return "DQRN"

class DQRNSAgent(nn.Module):
    '''
    Convolutional with a lstm appended
    '''
    def __init__(self,num_actions):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=16,kernel_size=3,stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32,out_channels=1,kernel_size=3,stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.lstm = nn.LSTM(input_size=195, hidden_size=512, batch_first=True)
        self.lin = nn.Linear(512, num_actions)  # Output 4 actions 
        self.softmax = nn.Softmax()
    def forward(self, x):
        batch_size, seq_len, channels, height, width = x.size()

        x = x.view(batch_size * seq_len, channels, height, width)  # Shape: [batch_size * seq_len, channels, height, width]
        
        x = self.cnn(x)  # Shape: [batch_size * seq_len, feature_dim]
        
        feature_dim = x.size(-1)
        x = x.view(batch_size, seq_len, feature_dim)  # Shape: [batch_size, seq_len, feature_dim]
        
        # Process with LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)  # LSTM output: [batch_size, seq_len, hidden_size]
        
        x = h_n[-1]  # Final hidden state: [batch_size, hidden_size]
        x = self.lin(x)  # Output shape: [batch_size, num_actions]
        x = self.softmax(x)
        return x
    def get_name():
        return "DQRNS"
    


class DQRNDeepAgent(nn.Module):
    '''
    Convolutional with two layered lstm appended
    '''
    def __init__(self,num_actions):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=16,kernel_size=3,stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32,out_channels=1,kernel_size=3,stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.lstm = nn.LSTM(input_size=195, hidden_size=512, num_layers=3,batch_first=True, dropout=0.2)
        self.fc = nn.Linear(512, num_actions)  
    def forward(self, x):
        batch_size, seq_len, channels, height, width = x.size()
        x = x.view(batch_size * seq_len, channels, height, width)  # Shape: [batch_size * seq_len, channels, height, width]
        
        x = self.cnn(x)  # Shape: [batch_size * seq_len, feature_dim]
        
        # Restore sequence dimension
        feature_dim = x.size(-1)
        x = x.view(batch_size, seq_len, feature_dim)  # Shape: [batch_size, seq_len, feature_dim]
        
        lstm_out, (h_n, c_n) = self.lstm(x)  # LSTM output: [batch_size, seq_len, hidden_size]
        

        x = h_n[-1]  # Final hidden state: [batch_size, hidden_size]
        
        # Fully connected layer for output
        x = self.fc(x)  # Output shape: [batch_size, num_actions]
        return x
    def get_name(self):
        return "DQRNDeep"
    


class FullDQRNAgent(nn.Module):
    '''
    
    '''
    def __init__(self,num_actions):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.MaxPool2d(5,1),
            nn.Conv2d(in_channels=3,out_channels=16,kernel_size=3,stride=1),
            nn.ReLU(),
            nn.AvgPool2d(5,1),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32,out_channels=1,kernel_size=3,stride=1), 
            nn.ReLU(),
            nn.Flatten()
        )

        self.lstm = nn.LSTM(input_size=200,hidden_size=512, num_layers=3, batch_first=True,dropout=0)
        self.fc = nn.Linear(512,num_actions)
    
    def forward(self,x):
        raise NotImplementedError()
    def get_name(self):
        return "FullDQRN"