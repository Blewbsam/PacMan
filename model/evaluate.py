import sys
import os
import torch
from game import Game 
from train_rnn import NULL_STATE, SEQUENCE_LENGTH
from train_rnn import select_action as select_rnn_action
from train_cnn import select_action as select_cnn_action
from utils import plot_scores
from model import getModel
from collections import deque




N_ACTIONS = 4

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def evaluate_rnn(model,episodes):
    scores = []

    for i in range(episodes):
        print(i)
        sequence_buffer = deque([NULL_STATE for i in range(SEQUENCE_LENGTH)],maxlen=SEQUENCE_LENGTH)
        game = Game()
        while game.running():
            if not game.decision_available():
                game.update()
                if not game.running():
                    scores.append(game.get_score())
                continue

            state = game.get_reduced_state().to(device)
            sequence_buffer.append(state)   
            action, _ = select_rnn_action(model,sequence_buffer)
            game.step(action.item())
            if not game.running():
                scores.append(game.get_score()) 

    return scores


def evaluate_cnn(model,episodes):
    scores = []
    for i in range(episodes):
        print(i)
        game = Game()
        while game.running():
            if not game.decision_available():
                game.update()
                if not game.running():
                    scores.append(game.get_score())
                continue
            state = game.get_reduced_state().to(device)
            action, _ = select_cnn_action(model,state)
            game.step(action.item())
            if not game.running():
                scores.append(game.get_score())
    return scores
                





def main(nn_model,weigth_path,isRNN,episodes):
    model = nn_model(N_ACTIONS).to(device)
    model.load_state_dict(torch.load(weigth_path,weights_only=True))
    scores = evaluate_rnn(model,episodes) if isRNN else evaluate_cnn(model,episodes)
    print("Average score:", sum(scores) / len(scores))
    plot_scores(scores)

if __name__ == "__main__": # expecting input to be python evaluate.py [agent-type] [path] [model-type] [episodes]
    args = sys.argv
    path = args[1]
    isRNN = bool()
    episodes = 0

    try:
        model = getModel(args[1])
    except:
        print(f"Unabled to find model with given name: {args[1]}")
        quit()

    if os.path.exists(args[2]):
        path = args[2]
    else:
        print("Invalid Path given for model weights")
        quit()
    if args[3].lower() == "rnn":
        isRNN = True
    elif args[3].lower() == "cnn":
        isRNN = False
    else:
        print("Impropoer model description, should be either rnn or cnn.")
        quit()
    try:
        episodes = int(args[4])
    except:
        print("Invalid number of episodes to run given.")
        quit()


    main(model,path,isRNN,episodes) 