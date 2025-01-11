import pygame
import sys
from settings import width, height
from table import Table

pygame.init()
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Ping Pong")


class Pong:
    def __init__(self, screen):
        self.screen = screen
        self.FPS = pygame.time.Clock()

    def draw(self):
        pygame.display.flip()

    def main(self):
        # start menu
        table = Table(self.screen)  # pass to table the player_option saved to table.game_mode
        while True:
            self.screen.fill("black")
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            table.player_move()
            table.update()
            self.draw()
            self.FPS.tick(30)


if __name__ == "__main__":
    play = Pong(screen)
    play.main()



