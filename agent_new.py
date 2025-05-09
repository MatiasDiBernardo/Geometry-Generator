import torch
import random 
import numpy as np
from collections import deque

from geometry import WallsGenerator, WallGeneratorGUI
from model import Linear_QNet, QTrainer

MAX_MEMORY = 500_000
BATCH_SIZE = 5000
LR = 0.001

class Agent():
    def __init__(self, game):
        # Objects
        self.game = game

        # Model parameters
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft() when memory reaches max

        # Models
        inital_layer = 6   # State (Lx, Ly, Dx, Dy, IdxIW, IdxFw)
        hidden_layer = 64  # Hidden State
        output_layer = game.num_walls   # Action
        self.model = Linear_QNet(inital_layer, hidden_layer, output_layer)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_action(self, state):
        """The epsilon criteria and the game of explorations fix the
        ammount of exporation vs explotation present in the model.

        Args:
            state (np.array): Array with state of the game

        Returns:
            np.array[4]: Index values of walls
        """
        games_of_exporation = 400
        self.epsilon = games_of_exporation - self.game.level_iterations
        final_move = np.zeros(self.game.num_walls, dtype=int)

        if random.randint(0, 200) < self.epsilon:
            for i in range(len(final_move)):
                move = int(np.random.uniform(0, self.game.N))
                final_move[i] = move
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            for i in range(len(final_move)):
                move = int(prediction[i] * (self.game.N - 1))
                final_move[i] = move

        return final_move

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

def train():
    total_score = 0
    total_reward = 0
    record = 0
    score = 0
    game_count = 1
    
    n_grid = 250
    game = WallsGenerator(2, 3, 0.8, 0.5, n_grid, 4)
    gui = WallGeneratorGUI()
    agent = Agent(game)

    # Start from pretrained
    # checkpoint = torch.load("models/Record_1158.pth")
    # agent.model.load_state_dict(checkpoint)

    while True:
        # Get a numerical representation of the state of the game
        state_old = game.get_state()
        old_score = score

        # Predict next move
        pred_move = agent.get_action(state_old)

        # Update the game based on the prediction
        score, reward, game_over = game.play_step(pred_move)

        # Get the state of the updated game
        state_new = game.get_state()  # State of the game after prediction
        walls = game.walls_position()

        # Train short memory
        agent.train_short_memory(state_old, pred_move, reward, state_new, game_over)

        # remember
        agent.remember(state_old, pred_move, reward, state_new, game_over)

        total_reward += reward

        # Visualize the game progress
        gui.render(state_new, walls, score)

        if game_over:
            # train long memory, plot result
            agent.train_long_memory()

            # Save the model when reaches a new record
            if old_score > record:
                record = old_score
                agent.model.save()

            mean_score = total_score / game.level_iterations

            # if game.iterations % 100 == 0:
            print('Game: ', game_count, '| Score: ', int(old_score), '| Record:', int(record), '| Mean score: ', int(mean_score),  '| Reward: ', total_reward)

            total_score += old_score
            total_reward = 0
            game_count += 1

if __name__ == '__main__':
    train()
