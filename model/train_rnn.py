import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import time
import random
import matplotlib.pyplot as plt
from utils import ReplayMemory, plot_steps, Transition, plot_epsilons,plot_scores,print_verbose, plot_loss
from model import DQRNAgent
from game import Game, DIRECTIONS
from collections import deque


BATCH_SIZE = 128
GAMMA = 0.95
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
LR = 1e-4
N_ACTIONS = 4
EPISODES = 5000
SAVE_PATH = "weights"
STEP_FIG_PATH = "plots/RNN"
SEQUENCE_LENGTH = 4
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(device)

NULL_STATE = torch.zeros(3,21,19).to(device)

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
    if len(memory) < BATCH_SIZE:
        print("Inadequate memory")
        return

    ## sample SEQUENCE_LEGNTH items of batch_size 
    transitions = memory.sequence_sample(SEQUENCE_LENGTH,BATCH_SIZE)
    # move conversions to memory itself
    state_batch = torch.stack(
        [torch.stack([step.state for step in sequence], dim=0)
        for sequence in transitions], dim=0)

    is_not_terminal = lambda x: x is not None
    non_terminal_mask = torch.cat(
        [torch.tensor([is_not_terminal(sequence[-1].next_state)],device=device,dtype=torch.bool)
        for sequence in transitions], dim=0
    )

    get_non_terminal = lambda x: x if x is not None else NULL_STATE
    non_terminal_next_states = torch.stack(
        [torch.stack([get_non_terminal(step.next_state) for step in sequence],dim=0)
        for sequence in transitions], dim=0
    )


    action_batch = torch.stack(
        [torch.stack([step.action for step in sequence],dim=0)
        for sequence in transitions], dim = 0)

    selected_actions = action_batch[:, -1,0].unsqueeze(1)

    reward_batch = torch.stack(
        [torch.stack([step.reward for step in sequence],dim=0)
        for sequence in transitions], dim = 0)
    
    selected_rewards = reward_batch[:,-1,0]

    state_action_values = policy_net(state_batch).gather(1,selected_actions)

    next_state_values = torch.zeros(BATCH_SIZE,device=device)

    with torch.no_grad():
        target_Q_values = target_net(non_terminal_next_states).max(1).values
        next_state_values[non_terminal_mask] = target_Q_values[non_terminal_mask]
    
    expected_state_action_values = (next_state_values * GAMMA) + selected_rewards 

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()
    return loss.item()

def train(policy_net,target_net,optimizer,memory,num_episodes,verbose=False):
    action_counts = []
    scores = []
    losses = []
    sequence_buffer = deque([NULL_STATE for i in range(SEQUENCE_LENGTH)],maxlen=SEQUENCE_LENGTH) # circular buffer
    
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
                    state_tensor = state.to(device) if state is not None else None 
                    scores.append(game.get_score())
                    reward = -100 if game.is_game_lost() else 100
                    reward_tensor = torch.tensor(reward).unsqueeze(0).to(device)
                    memory.push(prevState,action,state_tensor,reward_tensor)
                    action_counts.append(action_count)
                continue    

            action_count += 1

            state = game.get_state().to(device)
            sequence_buffer.append(state)
            score = game.get_score()
            reward_tensor = torch.tensor(score-prevScore).unsqueeze(0).to(device)

            if (prevState != None):
                memory.push(prevState,action,state,reward_tensor)

            action,epsilon = select_action(policy_net,sequence_buffer)
            game.step(action.item())
            prevState = state
            prevScore = score


            if action_count % (BATCH_SIZE/4) == 0:
                loss = optimize_model(policy_net,target_net,optimizer,memory)  
                losses.append(loss)
                if verbose:
                    print_verbose(i_episode,score,reward_tensor.item(),loss,epsilon)

    losses = [l for l in losses if l is not None]
    plot_loss(losses,f"{STEP_FIG_PATH}/loss_{EPISODES}.png")
    plot_scores(scores,f"{STEP_FIG_PATH}/scores_{EPISODES}.png")
    plot_steps(action_counts,f"{STEP_FIG_PATH}/steps_{EPISODES}.png") 




def main(load_path=None):
    policy_net = DQRNAgent(N_ACTIONS).to(device)
    if load_path != None:
        policy_net.load_state_dict(torch.load(load_path,weights_only=True))
    target_net = DQRNAgent(N_ACTIONS).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(),lr=LR,amsgrad=True)
    memory = ReplayMemory(10000)
    train(policy_net,target_net,optimizer,memory,EPISODES,True)

    torch.save(policy_net.state_dict(),f"{SAVE_PATH}/rnn/rnn_{EPISODES}_episodes.pth") 

if __name__ == "__main__":
    main()
