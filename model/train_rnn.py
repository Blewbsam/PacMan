import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
import numpy as np
from collections import deque


from utils import ReplayMemory, plot_steps, plot_scores, print_verbose, plot_loss
from model import DQRNDeepAgent
from game import Game


# Configurations & Hyperparameters
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 100
LR = 1e-4
N_ACTIONS = 4
EPISODES = 3000
SAVE_PATH = "weights"
STEP_FIG_PATH = "plots/RNN"
SEQUENCE_LENGTH = 8
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
NULL_STATE = torch.zeros(3, 21, 19).to(device)  # Default "empty" state tensor


steps_done = 0 # total actions generated in all episodes
def select_action(policy_net, buffer):
    global steps_done
    """Select an action based on epsilon-greedy policy."""
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-steps_done / EPS_DECAY)
    state = torch.stack(list(buffer)).unsqueeze(0)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            q_values = policy_net(state)
            index = q_values.argmax(1)
            return torch.tensor(index, device=device, dtype=torch.long), eps_threshold, q_values[0,index].item()
    else:
        return torch.tensor([random.randint(0, N_ACTIONS - 1)], device=device, dtype=torch.long), eps_threshold, 0


def optimize_model(policy_net, target_net, optimizer, memory,scheduler=None):
    """Optimize the model using a batch from the replay buffer."""
    if len(memory) < BATCH_SIZE:
        print("Inadequate memory")
        return

    # Sample a batch from memory
    transitions = memory.sequence_sample(SEQUENCE_LENGTH, BATCH_SIZE)
    state_batch,non_terminal_mask, non_terminal_next_states, action_batch, reward_batch = prepare_batch_data(transitions)

    selected_actions = action_batch[:, -1, 0].unsqueeze(1)
    selected_rewards = reward_batch[:, -1, 0]

    # Get Q-values from policy network and target network
    state_action_values = policy_net(state_batch).gather(1, selected_actions)
    next_state_values = compute_next_state_values(target_net,non_terminal_mask,non_terminal_next_states)

    expected_state_action_values = (next_state_values * GAMMA) + selected_rewards

    # Compute loss and update model
    loss = nn.SmoothL1Loss()(state_action_values, expected_state_action_values.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()
    if scheduler is not None:
        scheduler.step()

    return loss.item()


def prepare_batch_data(transitions):
    """Extract and prepare data for model optimization from the transitions."""
    get_non_terminal = lambda x: x if x is not None else NULL_STATE
    state_batch = torch.stack([torch.stack([step.state for step in sequence], dim=0) for sequence in transitions], dim=0)
    non_terminal_mask = torch.cat([torch.tensor([step[-1].next_state is not None], device=device, dtype=torch.bool) for step in transitions], dim=0)
    non_terminal_next_states = torch.stack([torch.stack([get_non_terminal(step.next_state) for step in sequence], dim=0) for sequence in transitions], dim=0)
    action_batch = torch.stack([torch.stack([step.action for step in sequence], dim=0) for sequence in transitions], dim=0)
    reward_batch = torch.stack([torch.stack([step.reward for step in sequence], dim=0) for sequence in transitions], dim=0)
    
    return state_batch, non_terminal_mask,non_terminal_next_states, action_batch, reward_batch


def compute_next_state_values(target_net, non_terminal_mask,non_terminal_next_states):
    """Compute the next state's Q-values using the target network."""
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        target_Q_values = target_net(non_terminal_next_states).max(1).values
        next_state_values[non_terminal_mask] = target_Q_values[non_terminal_mask]
    return next_state_values


def train(policy_net, target_net, optimizer, memory, num_episodes,scheduler=None,verbose=False):
    """Train the agent over a number of episodes."""
    action_counts = []
    scores = []
    losses = []

    for i_episode in range(num_episodes):
        game = Game()
        sequence_buffer = deque([NULL_STATE] * SEQUENCE_LENGTH, maxlen=SEQUENCE_LENGTH)
        prevState, prevScore, action_count = None, 0, 0

        while game.running():
            if not game.decision_available():
                game.update()
                if not game.running():
                    handle_game_end(game, prevState,action,memory, scores)
                continue

            action_count += 1
            state = game.get_reduced_state().to(device)
            sequence_buffer.append(state)
            score = game.get_score()

            reward_tensor = torch.tensor((score - prevScore) / 10).unsqueeze(0).to(device)
            loss = None
            if prevState is not None:
                memory.push(prevState, action, state, reward_tensor)

            action, epsilon, q_selected = select_action(policy_net, sequence_buffer)
            game.step(action.item())
            prevState, prevScore = state, score

            if action_count % (BATCH_SIZE / 4) == 0:
                loss = optimize_model(policy_net, target_net, optimizer, memory,scheduler)
                losses.append(loss)




            if verbose:
                print_verbose(i_episode,score,reward_tensor.item(), loss,epsilon,q_selected,action.item(),optimizer.param_groups[0]["lr"])

    # Plot results after training
    losses = [l for l in losses if l is not None]
    plot_loss(losses, f"{STEP_FIG_PATH}/loss_{EPISODES}.png")
    plot_scores(scores, f"{STEP_FIG_PATH}/scores_{EPISODES}.png")
    plot_steps(action_counts, f"{STEP_FIG_PATH}/steps_{EPISODES}.png")

def handle_game_end(game, prevState, action, memory, scores):
    """Handle the end of the game and store relevant data."""
    state = game.get_reduced_state()
    state_tensor = state.to(device) if state is not None else None
    scores.append(game.get_score())
    reward = -100 if game.is_game_lost() else 100
    reward_tensor = torch.tensor(reward).unsqueeze(0).to(device)
    memory.push(prevState, action, state_tensor, reward_tensor)


def main(model, load_path=None):
    """Initialize networks, optimizer, memory, and start training."""
    policy_net = model(N_ACTIONS).to(device)
    if load_path:
        policy_net.load_state_dict(torch.load(load_path, weights_only=True))

    target_net = model(N_ACTIONS).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    # optimizer = optim.SGD(policy_net.parameters(), lr=LR)
    optimizer = optim.AdamW(params=policy_net.parameters(),lr=LR,amsgrad=True)
    # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,1e-5,1e-2,step_size_up=1000,verbose=True)
    scheduler = None


    memory = ReplayMemory(10000)
    train(policy_net, target_net, optimizer, memory, EPISODES,scheduler,verbose=True)

    torch.save(policy_net.state_dict(), f"{SAVE_PATH}/RNN/{policy_net.get_name()}_{EPISODES}_{SEQUENCE_LENGTH}.pth")


if __name__ == "__main__":
    main(DQRNDeepAgent,"./weights/RNN/DQRNDeep_2000_8.pth")