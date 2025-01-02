import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import time
import random
import matplotlib.pyplot as plt
from utils import ReplayMemory, plot_steps, Transition, plot_epsilons,plot_scores,print_verbose
from model import DQRNAgent
from game import Game, DIRECTIONS
from collections import deque


BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 10000
TAU = 0.005
LR = 1e-4
N_ACTIONS = 4
EPISODES = 4
SAVE_PATH = "weights"
STEP_FIG_PATH = "plots"
SEQUENCE_LENGTH = 4
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(device)






steps_done = 0
def select_action(policy_net,buffer):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    # get states from buffer
    state = torch.stack(list(buffer)).unsqueeze(0)

    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return torch.tensor(policy_net(state).argmax(1),device=device,dtype=torch.long),eps_threshold
    else:
        return torch.tensor([random.randint(0,N_ACTIONS-1)], device=device, dtype=torch.long),eps_threshold

    

def optimize_model(policy_net,target_net,optimizer,memory):

    





    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()
    return loss



null_state = torch.zeros(3,21,19).to(device)


def train(policy_net,target_net,optimizer,memory,num_episodes,verbose=False):
    episode_steps = []
    scores = []
    sequence_buffer = deque([null_state for i in range(SEQUENCE_LENGTH)],maxlen=SEQUENCE_LENGTH) # circular buffer.


    for i_episode in range(num_episodes):
        game = Game()
        prevState = None
        prevScore = 0
        action_count = 0

        while game.running():
            if not game.decision_available():
                game.update()

                if not game.running():
                    state = game.get_state()
                    state_tensor = state.unsqueeze(0).to(device) if state is not None else null_state
                    scores.append(game.get_score())
                    reward = -100 if game.is_game_lost() else 100
                    reward_tensor = torch.tensor(reward).unsqueeze(0).to(device)
                    memory.push(prevState.unsqueeze(0),action.unsqueeze(0),state_tensor,reward_tensor)
                continue    

            action_count += 1

            state = game.get_state().to(device)
            sequence_buffer.append(state)
            score = game.get_score()
            reward_tensor = torch.tensor(score-prevScore).unsqueeze(0).to(device)

            if (prevState != None):
                memory.push(prevState.unsqueeze(0),action.unsqueeze(0),state.unsqueeze(0),reward_tensor)

            action,epsilon = select_action(policy_net,sequence_buffer)
            game.step(action.item())
            prevState = state
            prevScore = score


            if action_count % (BATCH_SIZE/4) == 0:
                loss = optimize_model(policy_net,target_net,optimizer,memory)
            
                if verbose:
                    print_verbose(i_episode,score,reward_tensor,item(),loss,epsilon)




def main(load_path=None):
    policy_net = DQRNAgent(N_ACTIONS).to(device)
    if load_path != None:
        policy_net.load_state_dict(torch.load(load_path,weights_only=True))
    target_net = DQRNAgent(N_ACTIONS).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(),lr=LR,amsgrad=True)
    memory = ReplayMemory(10000)
    train(policy_net,target_net,optimizer,memory,EPISODES,True)

    torch.save(policy_net.state_dict(),f"{SAVE_PATH}/weights_{EPISODES}_episodes.pth") 

if __name__ == "__main__":
    main()
