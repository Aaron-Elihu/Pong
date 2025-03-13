import pygame


class Paddle:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.rectangle = pygame.Rect(self.x, self.y, width, height)
        self.color = pygame.Color("white")
        self.player_speed = 16
        self.score = 0

    def move_up(self):
        self.rectangle.y -= self.player_speed

    def move_down(self):
        self.rectangle.y += self.player_speed

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, self.rectangle)

    def update(self, screen):
        pygame.draw.rect(screen, self.color, self.rectangle)

