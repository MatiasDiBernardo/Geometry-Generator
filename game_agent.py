import pygame
import numpy as np
import sys
import random

class Game_2048():
    """
    Logic of the game. For the agent.
    """
    def __init__(self):
        self.state = self.set_initial_state()
        self.score = 2
        self.max_val = 2
        self.max_score = 50  # Set to 50 to avoid early reward.
        self.full_board_movements = 0
        self.iterations = 0
    
    def reset(self):
        self.state = self.set_initial_state()
        self.score = 2
        self.max_val = 50
        self.iterations += 1
    
    def set_initial_state(self):
        state = np.zeros((4, 4))
        state[random.randint(0, 3), random.randint(0, 3)] = 2

        return state

    def dir_dicc(self, dir):
        if dir[0] == 1:
            return "LEFT"
        if dir[1] == 1:
            return "RIGHT"
        if dir[2] == 1:
            return "UP"
        if dir[3] == 1:
            return "DOWN"

    def update_squares(self, x_pos, y_pos, dir, value):
        move = True
        while(move):
            # I check for the cases where the block needs to move
            move = False 

            if dir == "LEFT" and x_pos != 0:
                x_pos -= 1
                # Check for emtpy space
                if self.state[y_pos, x_pos] == 0:
                    self.state[y_pos, x_pos] = value
                    self.state[y_pos, x_pos + 1] = 0
                    move = True
                    continue

                # Check for similar value to update
                if self.state[y_pos, x_pos] == value:
                    self.state[y_pos, x_pos] = value * 2
                    self.state[y_pos, x_pos + 1] = 0
                    break

            if dir == "RIGHT" and x_pos != self.state.shape[0] - 1:
                x_pos += 1
                # Check for emtpy space
                if self.state[y_pos, x_pos] == 0:
                    self.state[y_pos, x_pos] = value
                    self.state[y_pos, x_pos - 1] = 0
                    move = True
                    continue

                # Check for similar value to update
                if self.state[y_pos, x_pos] == value:
                    self.state[y_pos, x_pos] = value * 2
                    self.state[y_pos, x_pos - 1] = 0
                    break

            if dir == "DOWN" and y_pos != self.state.shape[0] - 1:
                y_pos += 1
                # Check for emtpy space
                if self.state[y_pos, x_pos] == 0:
                    self.state[y_pos, x_pos] = value
                    self.state[y_pos - 1, x_pos] = 0
                    move = True
                    continue

                # Check for similar value to update
                if self.state[y_pos, x_pos] == value:
                    self.state[y_pos, x_pos] = value * 2
                    self.state[y_pos - 1, x_pos] = 0
                    break
                
            if dir == "UP" and y_pos != 0:
                y_pos -= 1
                # Check for emtpy space
                if self.state[y_pos, x_pos] == 0:
                    self.state[y_pos, x_pos] = value
                    self.state[y_pos + 1, x_pos] = 0
                    move = True
                    continue

                # Check for similar value to update
                if self.state[y_pos, x_pos] == value:
                    self.state[y_pos, x_pos] = value * 2
                    self.state[y_pos + 1, x_pos] = 0
                    break

    def update_state(self, move):

        direction = self.dir_dicc(move)

        # Establish direction of the loop 
        n = self.state.shape[0]  #Only supports square tables
        if direction == "LEFT":
            iter = range(n)
        if direction == "RIGHT":
            iter = range(n - 1, -1, -1)
        if direction == "UP":
            iter = range(n)
        if direction == "DOWN":
            iter = range(n - 1, -1, -1)

        for i in iter:
            for j in iter:
                value = self.state[i, j]
                if value != 0:
                    self.update_squares(j, i, direction, value)

    def create_random_block(self):
        avaliable_indices = np.argwhere(self.state == 0)

        # If there are avaliable indices
        if avaliable_indices.shape[0] != 0:
            index_crate_random = avaliable_indices[random.randint(0, len(avaliable_indices) - 1)]
            self.state[index_crate_random[0], index_crate_random[1]] = 2
    
    def loss_game_condition(self):
        game_over = False

        # If the table is full 
        if np.all(self.state != 0):
            game_over = True
            # Check for possible moves
            n =  self.state.shape[0]
            for i in range(n - 1):
                for j in range(n - 1):
                    value = self.state[i, j]
                    value_right = self.state[i, j + 1]
                    value_down = self.state[i + 1, j]

                    # If a possible move exist then keeps playing
                    if value == value_right or value == value_down:
                        game_over = False
        
        return game_over
    
    def ammount_of_blocks_increse(self, old_state):
        # Compares the ammount of blocks between two states
        num_old_state = np.count_nonzero(old_state)
        num_new_state = np.count_nonzero(self.state)

        return num_new_state > num_old_state
    
    def calculate_reward(self, game_over, old_state):
        """It's rewarded when reaches a new max value
        on the table.

        Args:
            game_over (boolean): If the games ended is true.

        Returns:
            int: Reward value. (+10 if good, -10 if bad, 0 if neutral)
        """
        # Penalty for ended game
        if game_over and self.score < 100:
            return -15
        
        if game_over:
            return -10

        # Penalty for poblated top row
        # if  (np.sum(self.state[0, :]) >= 6):
        #     return -1

        top_row_empty = (np.sum(self.state[0, :]) == 0) or (np.sum(self.state[0, :]) == 2)

        index_max = np.unravel_index(self.state.argmax(), self.state.shape)
        max_val_actual = self.state[index_max] 

        positive_reward = 0

        if max_val_actual > self.max_val: 
            self.max_val = max_val_actual
            positive_reward += 10

            if self.max_val == self.state[3, 0]:
                positive_reward += 50

        return positive_reward
    
    def avoid_getting_stuck(self, game_over):
        """The agent was getting stuck when reaches the full board cover
        and not doing the move that allowes the game to keep playing.
        This function solution this issue setting game over to true if
        this situation happens.

        Args:
            game_over (boolean): If the game ended or not.

        Returns:
            boolean: game_over
        """

        treshold_movements = 5  # Ammount of stuck movements that are allowed
        if (np.count_nonzero(self.state) == 16):
            self.full_board_movements += 1
        else:
            self.full_board_movements = 0
        
        if self.full_board_movements > treshold_movements:
            return True
        
        return game_over
    
    def play_step(self, move):

        old_state = np.copy(self.state)
        self.update_state(move)
        self.create_random_block()
        self.score = np.sum(self.state)
        game_over = self.loss_game_condition()
        game_over = self.avoid_getting_stuck(game_over)

        reward = self.calculate_reward(game_over, old_state)

        # If the agent loss the game resets
        if game_over:
            self.reset()
        
        # If the agent wins the game resets
        if self.max_val == 2048:
            self.reset()

        return self.score, reward, game_over

class Game_GUI:
    """
    Visual implementation of the game.
    """
    def __init__(self):
        WIDTH = 310
        HEIGTH = 400

        pygame.init()

        self.win = pygame.display.set_mode((WIDTH, HEIGTH))
        self.font = pygame.font.SysFont('arial', 25)
        self.clock = pygame.time.Clock()
        self.speed_index = 2  # Three postions of speed (slow, medium, fast)
        self.block_width = 60
        self.dim_table = 4
        self.positions_in_pixels = self.index_to_pixel()
    
    def index_to_pixel(self):
        x_offset = 110
        y_offset = 20
        inline_space = 10

        pixel_postions = np.zeros((4, 4, 2))
        for i in range(self.dim_table):
            for j in range(self.dim_table):
                x = x_offset + i * (self.block_width + inline_space) 
                y = y_offset + j * (self.block_width + inline_space)
                pixel_postions[i, j] = np.array([x, y])
        
        return pixel_postions

    def draw_score(self, score):
        label_score = self.font.render(f"Score: {score}", True, (255, 255, 255))
        self.win.blit(label_score, (20, 20))
    
    def draw_blocks(self, state):
        val_to_color = {0: (20, 120, 20),       # Grey
                        2: (254, 18, 18),       # Red
                        4: (78, 254, 18),       # Green
                        8: (254, 163, 18),      # Orange
                        16: (18, 252, 254),     # Light blue
                        32: (223, 254, 18),     # Yellow
                        64: (18, 52, 254),      # Blue 
                        128: (133, 18, 254),    # Violet 
                        256: (254, 18, 239),    # Pink
                        512: (58, 132, 2),      # Dark green
                        1024: (132, 2, 8),      # Dark red
                        2048: (255, 255, 255)}  # Black
        
        for i in range(self.dim_table):
            for j in range(self.dim_table):
                val = state[i, j]
                color = val_to_color[int(val)]
                x, y = self.positions_in_pixels[i, j]

                pygame.draw.rect(self.win, color, (y, x, self.block_width, self.block_width))

                if int(val) != 0:
                    label_surface = self.font.render(str(int(val)), True, (255, 255, 255))
                    self.win.blit(label_surface, (y + 5, x + 15))

    def change_speed(self):
        speed_values = [5, 10, 300]
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.speed_index += 1

        return speed_values[self.speed_index % 3]
        
    def render(self, state, score):
        self.win.fill((0, 0, 0))
        self.draw_score(score)
        self.draw_blocks(state)
        pygame.display.update()
        self.clock.tick(self.change_speed())
