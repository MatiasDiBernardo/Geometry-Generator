import pygame
import sys
import numpy as np

class WallsGenerator():
    def __init__(self, Lx, Ly, Dx, Dy, N_grid, num_walls):
        self.Lx = Lx
        self.Ly = Ly
        self.Dx = Dx
        self.Dy = Dy
        self.N = N_grid   # Puntos en la grilla
        self.num_walls = num_walls   # Cantidad de paredes
        self.points_initial_wall = 10  # Cantidad de puntos en la pared inicial
        self.calculate_interior_rect()
        self.update_initial_walls()
        
        # Condiciones de control
        self.score = 0
        self.valid_geometries = []
        self.level_iterations = 0
        self.max_level_iterations = 1000
        self.max_score_condition = 20
    
    def reset(self):
        # Define dimensiones aleatorias
        min_x = 200
        max_x = 400

        min_y = 100
        max_y = 600

        self.Lx = np.random.uniform(min_x, max_x)
        self.Ly = np.random.uniform(min_y, max_y)

        if self.Ly > self.x:
            self.Lx, self.Ly = self.Ly, self.Lx

        self.Dx = np.random.uniform(30, 100)
        self.Dy = np.random.uniform(30, 100)
        self.calculate_interior_rect()
    
    def update_state(self, action):
        self.update_initial_walls()
        self.intial_wall_positions()
        self.calculalte_walls_positions(action)

        # Checks if the walls generated are valid
        self.level_iterations += 1
        if self.is_valid():
            self.score += 1

    def update_initial_walls(self):
        arr1 = np.zeros(self.points_initial_wall)
        arr2 = np.zeros(self.points_initial_wall)

        self.index_initial_wall = np.random.randint(0, self.points_initial_wall)
        self.index_finish_wall = np.random.randint(0, self.points_initial_wall)

        self.initial_wall = arr1[self.index_initial_wall]
        self.finish_wall = arr2[self.index_finish_wall]
    
    def calculate_interior_rect(self):
        # Posiciones borde interior
        top_left = (self.Dx, self.Dy)
        rect_width = self.Lx - 2 * self.Dx
        rect_height = self.Ly - 2 * self.Dy
        
        # Borde interior
        self.interior_rect = pygame.Rect(
            top_left[0],
            top_left[1],
            rect_width,
            rect_height
        )

    def intial_wall_positions(self):
        M = len(self.initial_wall)
        dx = (self.Lx//2) / M
        self.pos_wall_initial = []
        self.pos_wall_final = []
        
        # Grid pared inicial (esta la posibilidad de hacer len(initial) - 1 para dejar el nodo libre)
        for i in range(len(self.initial_wall)):
            pos = (dx * i, 0)
            if self.initial_wall[i] == 1:
                self.pos_wall_initial.append(pos)

        # Grid pared final
        for i in range(len(self.finish_wall)):
            pos = (dx * i, self.Ly)
            if self.finish_wall[i] == 1:
                self.pos_wall_final.append(pos)

    def calculalte_walls_positions(self, grid_points):
        # Area calc
        top_area =self.Lx//2 * self.Dy
        middle_area = self.Dx * (self.Ly - 2 * self.Dy)
        bottom_area =self.Lx//2 * self.Dy
        
        area_total = top_area + middle_area + bottom_area

        # Define setp: Si propongo que dx == dy
        dx = 1
        for i in range(1,self.Lx//2):
            area_dots = self.N * dx**2
            if area_dots > area_total:
                break
            dx = i 
        
        # Iterar 
        row = 0
        col = 0

        # Intial position and conditions
        spx = 0  # Start position X
        spy = dx #  Start position Y
        
        # Margenes
        x_margin1 = self.Lx//2
        y_margin1 = self.Dy

        x_margin2 = self.Dx
        y_margin2 = self.Ly - self.Dy
        
        x_margin3 = self.Lx//2
        y_margin3 = self.Ly

        # Crea la grilla
        self.pos_wall_middle = []
        for i in range(self.N):
            # Primera área
            if spx + dx * row >= x_margin1:
                col += 1
                row = 0
                pos = (spx + dx * row, spy + dx * col)
                row += 0

                if grid_points[i] == 1:
                    self.pos_wall_middle.append(pos)
                continue

            if spx + dx * row < x_margin1 and spy + dx * col < y_margin1:  
                pos = (spx + dx * row, spy + dx * col)
                row += 1

                if grid_points[i] == 1:
                    self.pos_wall_middle.append(pos)
                continue
            
            # Segunda área
            if spx + dx * row >= x_margin2 and spy + dx * col < y_margin2:
                col += 1
                row = 0
                pos = (spx + dx * row, spy + dx * col)
                row += 1

                if grid_points[i] == 1:
                    self.pos_wall_middle.append(pos)
                continue

            if spx + dx * row < x_margin2 and spy + dx * col < y_margin2:  
                pos = (spx + dx * row, spy + dx * col)
                row += 1

                if grid_points[i] == 1:
                    self.pos_wall_middle.append(pos)
                continue

            # Tercera área
            if spx + dx * row >= x_margin3:
                col += 1
                row = 0
                pos = (spx + dx * row, spy + dx * col)
                row += 1

                if grid_points[i] == 1:
                    self.pos_wall_middle.append(pos)
                continue

            if spx + dx * row < x_margin3 and spy + dx * col < y_margin3:  
                pos = (spx + dx * row, spy + dx * col)
                row += 1

                if grid_points[i] == 1:
                    self.pos_wall_middle.append(pos)
                continue
    
    def walls_position(self):
        return self.pos_wall_initial + self.pos_wall_middle + self.pos_wall_final

    def normalize_coordinates(self):
        """
        Esta función mapea las posiciones en el plano de PyGame a un escenario real.
        Toma como eje de coordenadas el punto arriba a la izquierda. La conversión es 100 px = 1 mts
        """
        walls = self.pos_wall_initial + self.pos_wall_middle + self.pos_wall_final
        norm_walls = []
        for w in walls:
            x = (w[0]) / 100
            y = (w[1]) / 100
            norm_walls.append((x, y))
        
        return np.array(norm_walls)
    
    def is_valid(self):
        walls = self.pos_wall_initial + self.pos_wall_middle + self.pos_wall_final
        valid_geometry = True
        for i in range(len(walls) - 1):
            if self.interior_rect.clipline(walls[i], walls[i + 1]):
                valid_geometry = False
                self.score = 0

        return valid_geometry
    
    def calculate_reward(self, game_over):
        # Penalty for too many iteration on the same room
        if game_over:
            return -15
        
        # Penaliza si es un camino no valido
        if self.score == 0:
            return -10
        
        # Si el el score es mayor a 0
        if self.score > 0:
            return 10

    def loss_game_condition(self):
        return self.level_iterations < self.max_level_iterations

    def play_step(self, move):
        self.update_state(move)
        game_over = self.loss_game_condition()

        reward = self.calculate_reward(game_over)

        # If the agent loss the game resets
        if game_over:
            self.reset()
        
        # If the agent wins the game resets
        if self.score == self.max_score_condition:
            self.reset()

        return self.score, reward, game_over
    
class WallGeneratorGUI():
    """
    Visual implementation of the game.
    """
    def __init__(self):
        pygame.init()

        # Screen settings
        self.WIDTH = 600
        self.HEIGHT = 800

        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        self.font = pygame.font.SysFont(None, 36)  # None = default font, 36 = font size
        self.clock = pygame.time.Clock()
        self.speed_index = 2  # Three postions of speed (slow, medium, fast)
        self.font = pygame.font.SysFont(None, 36)  # None = default font, 36 = font size
        pygame.display.set_caption("Walls Generator")

        # General Design
        self.PadX = 100
        self.PadY = 100
        self.points_initial_wall = 10
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.line_width = 2

    def draw_room_layout(self, state):
        # Pasar de state a variables
        Lx, Ly, Dx, Dy, _, _ = state

        # Posiciones borde interior
        top_left = (Dx, self.PadY + Dy)
        rect_width = Lx - 2 * Dx
        rect_height = Ly - 2 * Dy
        
        # Borde interior
        self.interior_rect = pygame.Rect(
            top_left[0],
            top_left[1],
            rect_width,
            rect_height
        )

        # Posiciones borde exterior
        top_left = (self.PadX, self.PadY)
        top_right = (self.PadX + Lx, self.PadY)
        bottom_left = (self.PadX, self.PadY + Ly)
        bottom_rignt = (self.PadX + Lx, self.PadY + Ly)
        
        # Borde exterior
        pygame.draw.line(self.screen, self.WHITE, top_left, top_right, self.line_width)
        pygame.draw.line(self.screen, self.WHITE, top_right, bottom_rignt, self.line_width)
        pygame.draw.line(self.screen, self.WHITE, bottom_rignt, bottom_left, self.line_width)
        pygame.draw.line(self.screen, self.WHITE, bottom_left, top_left, self.line_width)
        
        # Border Interior 
        pygame.draw.rect(self.screen, self.WHITE, self.interior_rect, self.line_width)

    def draw_walls(self, walls):
        color = (255, 255, 0)
        self.line_width = 4
        for i in range(len(walls) - 1):
            pygame.draw.line(self.screen, color, walls[i], walls[i + 1], self.line_width)
    
    def draw_score(self, score):
        if score != 0:
            label_valid = self.font.render("VALID :)", True, (0, 0, 255))
            self.screen.blit(label_valid, (self.WIDTH//2 - self.WIDTH//8, self.HEIGHT//8))
            label_points = self.font.render(f"Score: {score}", True, (255, 255, 255))
            self.screen.blit(label_points, (self.WIDTH//2 - self.WIDTH//6, self.HEIGHT//8))
        else:
            label_invalid = self.font.render("INVALID :(", True, (255, 0, 0))
            self.screen.blit(label_invalid, (self.WIDTH//2 + self.WIDTH//8, self.HEIGHT//8))
            label_points = self.font.render(f"Score: {score}", True, (255, 255, 255))
            self.screen.blit(label_points, (self.WIDTH//2 - self.WIDTH//6, self.HEIGHT//8))

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

    def render(self, state, wall_pos, score):
        self.screen.fill(self.BLACK)

        self.draw_room_layout(state)
        self.draw_walls(wall_pos)
        self.draw_score(score)

        pygame.display.update()
        self.clock.tick(self.change_speed())
