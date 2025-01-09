
import torch
import torch.nn as nn
import torch.optim as optim
import math
import random
import matplotlib.pyplot as plt
from utils import ReplayMemory, plot_steps, Transition, plot_epsilons,plot_scores,print_verbose,transform_image_with_thresholding
from model import DQNAgent
from game import Game
import time


BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 10000
TAU = 0.005
LR = 1e-4
N_ACTIONS = 4
EPISODES = 1000
SAVE_PATH = "weights/cnn"
STEP_FIG_PATH = "plots"
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(device)



steps_done = 0
def select_action(policy_net,state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            q_values = policy_net(state)
            index = q_values.argmax(1)
            return torch.tensor(policy_net(state).argmax(1),device=device,dtype=torch.long),eps_threshold, q_values[0,index].item() 
    else:
        return torch.tensor([random.randint(0,N_ACTIONS-1)], device=device, dtype=torch.long),eps_threshold,None



def compute_next_state_values(target_net,non_terminal_mask,non_terminal_next_states):
    next_state_values = torch.zeros(BATCH_SIZE,device=device)
    with torch.no_grad():
        next_state_values[non_terminal_mask] = target_net(non_terminal_next_states).max(1).values
    return next_state_values


def prepare_batch_data(transitions):
    batch = Transition(*zip(*transitions))
    non_terminal_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    non_termianl_next_states = torch.cat([s for s in batch.next_state if s is not None]) # push empty arry when None
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    return non_terminal_mask,non_termianl_next_states,state_batch,action_batch,reward_batch

def optimize_model(policy_net,target_net,optimizer,memory):
    if len(memory) < BATCH_SIZE:
        return

    transition_sample = memory.sample(BATCH_SIZE)
    non_terminal_mask, non_terminal_next_states,state_batch, action_batch,reward_batch = prepare_batch_data(transition_sample)

    # Q(s_t, a)
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for next state
    next_state_values = compute_next_state_values(target_net,non_terminal_mask,non_terminal_next_states)


    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    #measure loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # backprop and descent
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()
    return loss

def handle_game_end(game,prevState,action,memory,scores):
    """Handles end of game and stores relavent data that led to games end"""
    state = game.get_reduced_state()
    state_tensor = state.unsqueeze(0).to(device) if state is not None else None
    scores.append(game.get_score())
    reward = -100 if game.is_game_lost() else 100 # penalty for losing the game and extreme reward for winning
    reward_tensor = torch.tensor(reward).unsqueeze(0).to(device)
    memory.push(prevState.unsqueeze(0),action.unsqueeze(0),state_tensor,reward_tensor)

def train(policy_net,target_net,optimizer,memory,num_episodes,verbose=False):
    episode_steps = []
    scores = []
    for i_episode in range(num_episodes):
        game = Game()
        prevState = None
        prevScore = 0
        action_count = 0

        while game.running():
            # select and make action given 
            if not game.decision_available():
                game.update()
                if not game.running():
                    handle_game_end(game,prevState,action,memory,scores)
                continue

            action_count += 1
            state = game.get_reduced_state().to(device)
            score = game.get_score()
            reward_tensor = torch.tensor(score - prevScore).unsqueeze(0).to(device)
            
            if prevState is not None:
                memory.push(prevState.unsqueeze(0),action.unsqueeze(0),state.unsqueeze(0), reward_tensor)
            action,epsilon, action_qval = select_action(policy_net,state)
            game.step(action.item())
            prevState = state
            prevScore = score
            loss = None
            if action_count % (BATCH_SIZE/4) == 0:
                # gradient calculations and step
                loss = optimize_model(policy_net,target_net,optimizer,memory)

                # soft update of model parameters
                target_net_state_dict = target_net.state_dict()
                policy_net_state_dict = policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
                target_net.load_state_dict(target_net_state_dict)

            if verbose:
                print_verbose(i_episode,score,reward_tensor.item(),loss,epsilon,action_qval,action.item())

        episode_steps.append(action_count)
    plot_scores(scores,f"{STEP_FIG_PATH}/scores_{EPISODES}.png")
    plot_steps(episode_steps,f"{STEP_FIG_PATH}/steps_{EPISODES}.png")
    
def main(load_path=None):
    policy_net = DQNAgent(N_ACTIONS).to(device)
    if load_path != None:
        policy_net.load_state_dict(torch.load(load_path,weights_only=True))
    target_net = DQNAgent(N_ACTIONS).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(),lr=LR,amsgrad=True)
    memory = ReplayMemory(10000)
    train(policy_net,target_net,optimizer,memory,EPISODES,True)
    torch.save(policy_net.state_dict(),f"{SAVE_PATH}/cnn_{EPISODES}_episodes.pth") 

if __name__ == "__main__":
    main()