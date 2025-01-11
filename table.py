import pygame
import sys
import time
from ball import Ball
from player import Player
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
        self.human_player = Player(0, height // 2 - (player_height // 2), player_width, player_height)
        self.computer_player = Player(width - player_width, height // 2 - (player_height // 2), player_width, player_height)
        self.ball = Ball(width // 2 - player_width, height - player_width, player_width)

    def _ball_hit(self):
        # if ball is not hit by a player and pass through table sides
        if self.ball.rectangle.left >= width:
            self.human_player.score += 1
            self.ball.rectangle.x = width//2
            time.sleep(1)
        elif self.ball.rectangle.right <= 0:
            self.computer_player.score += 1
            self.ball.rectangle.x = width//2
            time.sleep(1)
        # if ball lands into the player
        if pygame.Rect.colliderect(self.ball.rectangle, self.human_player.rectangle):
            self.ball.direction = "right"
        if pygame.Rect.colliderect(self.ball.rectangle, self.computer_player.rectangle):
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
            if self.computer_player.rectangle.top > 0:
                self.computer_player.move_up()
        if keys[pygame.K_DOWN]:
            if self.computer_player.rectangle.bottom < height:
                self.computer_player.move_down()

    def _show_score(self):
        human_score, computer_score = str(self.human_player.score), str(self.computer_player.score)
        human_score = self.font.render(human_score, True, self.color)
        computer_score = self.font.render(computer_score, True, self.color)
        self.screen.blit(human_score, (width//4, 50))
        self.screen.blit(computer_score, ((width//4)*3, 50))

    def _game_end(self):
        if self.winner is not None:
            print(f"{self.winner} wins!")
            pygame.quit()
            sys.exit()

    def update(self):
        self._show_score()
        self.human_player.update(self.screen)
        self.computer_player.update(self.screen)
        self._ball_hit()
        if self.human_player.score == self.score_limit:
            self.winner = "Opponent"
        elif self.computer_player.score == self.score_limit:
            self.winner = "You"
        self._game_end()
        self.ball.update(self.screen)
