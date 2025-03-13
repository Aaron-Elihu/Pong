import pygame
import sys
import time
import numpy as np
from ball import Ball
from paddle import Paddle
from settings import width, height, player_width, player_height


class Table:
    def __init__(self, screen):
        self.screen = screen
        self.game_over = False
        self.score_limit = 10
        self.winner = None
        self._generate_world()
        self.font = pygame.font.SysFont('Bauhaus 93', 60)
        self.install_font = pygame.font.SysFont('Bauhaus 93', 30)
        self.color = pygame.Color("white")

        # create and add player to the screen
    def _generate_world(self):
        self.human_player = Paddle(0, height // 2 - (player_height // 2), player_width, player_height)
        self.ai_player = Paddle(width - player_width, height // 2 - (player_height // 2), player_width, player_height)
        self.ball = Ball(width // 2 - player_width, height - player_width, player_width)

    def _ball_hit(self):
        # if ball is not hit by a player and pass through table sides
        if self.ball.rectangle.left >= width:
            self.human_player.score += 1
            self.ball.rectangle.x = width//2
            time.sleep(1)
        elif self.ball.rectangle.right <= 0:
            self.ai_player.score += 1
            self.ball.rectangle.x = width//2
            time.sleep(1)
        # if ball lands into the player
        if pygame.Rect.colliderect(self.ball.rectangle, self.human_player.rectangle):
            self.ball.direction = "right"
        if pygame.Rect.colliderect(self.ball.rectangle, self.ai_player.rectangle):
            self.ball.direction = "left"

    def _bot_opponent(self):
        if self.ball.direction == "left" and self.ball.rectangle.centery != self.human_player.rectangle.centery:
            if self.ball.rectangle.top <= self.human_player.rectangle.top:
                if self.human_player.rectangle.top > 0:
                    self.human_player.move_up()
            if self.ball.rectangle.bottom >= self.human_player.rectangle.bottom:
                if self.human_player.rectangle.bottom < height:
                    self.human_player.move_down()

    def player_move(self):
        keys = pygame.key.get_pressed()
        # for bot opponent controls
        self._bot_opponent()
        # for player controls
        if keys[pygame.K_UP]:
            if self.ai_player.rectangle.top > 0:
                self.ai_player.move_up()
        if keys[pygame.K_DOWN]:
            if self.ai_player.rectangle.bottom < height:
                self.ai_player.move_down()

    def _show_score(self):
        human_score, computer_score = str(self.human_player.score), str(self.ai_player.score)
        human_score = self.font.render(human_score, True, self.color)
        computer_score = self.font.render(computer_score, True, self.color)
        self.screen.blit(human_score, (width//4, 50))
        self.screen.blit(computer_score, ((width//4)*3, 50))

    def _game_end(self):
        if self.winner is not None:
            print(f"{self.winner} wins!")
            pygame.quit()
            sys.exit()

    def draw(self, screen):
        # Draw all game objects onto the screen.
        self.ball.draw(screen)
        self.human_player.draw(screen)
        self.ai_player.draw(screen)

    def update(self):
        self._show_score()
        self.human_player.update(self.screen)
        self.ai_player.update(self.screen)
        self._ball_hit()
        if self.human_player.score == self.score_limit:
            self.winner = "Opponent"
        elif self.ai_player.score == self.score_limit:
            self.winner = "You"
        self._game_end()
        self.ball.update(self.screen)

    def reset(self):
        self.human_player = Paddle(0, height // 2 - (player_height // 2), player_width, player_height)
        self.ai_player = Paddle(width - player_width, height // 2 - (player_height // 2), player_width, player_height)
        self.ball = Ball(width // 2 - player_width, height - player_width, player_width)

    def reset(self):
        """Reset the game state, including ball and paddle positions."""
        # Reset ball position to center
        self.ball.x = width // 2
        self.ball.y = height // 2

        # Reset ball velocity
        self.ball.vx = np.random.choice([-4, 4])  # Random direction
        self.ball.vy = np.random.choice([-3, 3])

        # Reset paddle positions
        self.human_player.y = height // 2 - player_height // 2
        self.ai_player.y = height // 2 - player_height // 2

        # Reset scores if needed
        self.score = 0
        self.ai_score = 0

    def step(self, action):
        """Updates the game state based on the action (for training)."""
        reward = 0
        done = False

        # Example: move paddle up/down based on action
        if action == 2:  # Move up
            self.human_player.y = max(0, self.human_player.y - 10)
        elif action == 3:  # Move down
            self.human_player.y = min(height - player_height, self.human_player.y + 10)

        # Update the ball and check for scoring
        self.ball.update(self.screen)

        # Check for game over conditions (e.g., ball out of bounds)
        if self.ball.x < 0 or self.ball.x > width:
            done = True  # End the episode if ball is out

        return reward, done

    def get_observation(self):
            return np.array([
            self.ball.x / width,  # Normalize by dividing by screen width
            self.ball.y / height,  # Normalize by dividing by screen height
            self.ball.vx / 10.0,  # Normalize by max expected velocity
            self.ball.vy / 10.0,
            self.human_player.y / height,
            self.ai_player.y / height
        ], dtype=np.float32)

    def get_observation(self):
        return self.screen.copy()
