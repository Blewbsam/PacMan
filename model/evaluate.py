import torch
from game import Game 
from train_rnn import NULL_STATE, SEQUENCE_LENGTH, select_action
from utils import plot_scores
from model import DQRNAgent
from collections import deque



N_ACTIONS = 4

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def evaluate(model,episodes):
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

            state = game.get_state().to(device)
            sequence_buffer.append(state)   
            action, _ = select_action(model,sequence_buffer)
            game.step(action.item())
            if not game.running():
                print("Game stopped here")
                scores.append(game.get_score()) 
            # game.update()

    return scores





def main(nn_model):
    model = nn_model(N_ACTIONS).to(device)
    model.load_state_dict(torch.load("weights/rnn/rnn_5000_episodes.pth",weights_only=True))
    scores = evaluate(model,50)
    print("Average score:", sum(scores) / len(scores))
    plot_scores(scores)


if __name__ == "__main__":
    main(DQRNAgent)

