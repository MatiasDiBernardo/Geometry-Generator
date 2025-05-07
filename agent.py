import torch
import random 
import numpy as np
from collections import deque

from game_agent import Game_2048, Game_GUI
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

    def game_state(self, state):
        # Encode all the values in binary up to 2^11 == 2048. O usar 2^16 == 65536 for 256 input vector.

        # Eso lo puedo pasar todo junto que sería 11*4*4 o paso a la red 16 arrays de dimension 11 (valores de 2 a 1024)
        # Aca puedo agregar info en función de como este el estado para pasar a la red
        # Por ahora solo paso matriz de 4x4 del tablero a vector de 11x4x4 = 176.

        encoded_vals = np.array([])
        for i in range(state.shape[0]):
            for j in range(state.shape[1]):
                val_binary = np.zeros(16)
                if state[i, j] != 0:
                    index = np.log2(state[i, j]) - 1
                    val_binary[int(index)] = 1
                encoded_vals = np.concatenate([encoded_vals, val_binary])
            
        return encoded_vals # astype('int32')
    
    def get_action(self, state):
        """The epsilon criteria and the game of explorations fix the
        ammount of exporation vs explotation present in the model.

        Args:
            state (_type_): _description_

        Returns:
            _type_: _description_
        """
        games_of_exporation = 150
        self.epsilon = games_of_exporation - self.game.iterations
        final_move = np.zeros(4)

        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 3)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

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
    
    game = Game_2048()
    gui = Game_GUI()
    agent = Agent(game)

    # Start from pretrained
    # checkpoint = torch.load("models/Record_1158.pth")
    # agent.model.load_state_dict(checkpoint)

    while True:
        # Get a numerical representation of the state of the game
        state_old = agent.game_state(game.state)
        old_score = score

        # Predict next move
        pred_move = agent.get_action(state_old)

        # Update the game based on the prediction
        score, reward, game_over = game.play_step(pred_move)

        # Get the state of the updated game
        state_new = agent.game_state(game.state)  # State of the game after prediction

        # Train short memory
        agent.train_short_memory(state_old, pred_move, reward, state_new, game_over)

        # remember
        agent.remember(state_old, pred_move, reward, state_new, game_over)

        total_reward += reward

        # Visualize the game progress
        gui.render(game.state, game.score)

        if game_over:
            # train long memory, plot result
            agent.train_long_memory()

            # Save the model when reaches a new record
            if old_score > record:
                record = old_score
                agent.model.save()

            mean_score = total_score / game.iterations

            # if game.iterations % 100 == 0:
            print('Game: ', game.iterations, '| Score: ', int(old_score), '| Record:', int(record), '| Mean score: ', int(mean_score),  '| Reward: ', total_reward)

            total_score += old_score
            total_reward = 0

if __name__ == '__main__':
    train()
