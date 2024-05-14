from __future__ import annotations

import pygame
import pymunk
import pymunk.pygame_util
from pygame.locals import QUIT
from pymunk.vec2d import Vec2d

from src.config import get_app_config, get_stickman_config, get_swing_config
from src.data_types import AppConfigType, StickmanConfigType, SwingConfigType
from src.objects.stickman import Stickman
from src.objects.swing import Swing


class App:
    """
    Create Pygame application to run simulation of teaching 2D Stickmen AI to swing.
    Uses Pymunk as the physics engine.
    """

    def __init__(self) -> None:
        """
        Initialise App and define parameters.
        """
        self._app_config: AppConfigType
        self._swing_config: SwingConfigType
        self._stickman_config: StickmanConfigType
        self._space: pymunk.Space

        self._members: list[Stickman] = []
        self._swings: list[Swing] = []
        self._running = False

    @property
    def screen(self) -> pygame.Surface:
        return self._display_surf

    @classmethod
    def create_app(cls) -> App:
        """
        Create application and load config.

        Returns:
            app (App): App with Pygame and Pymunk config set
        """
        pygame.init()
        app = cls()
        app._load_config()
        app._configure()
        return app

    def _load_config(self) -> None:
        self._app_config = get_app_config()
        self._swing_config = get_swing_config()
        self._stickman_config = get_stickman_config()

    def _configure(self) -> None:
        pygame.display.set_caption(self._app_config.name)
        self._display_surf = pygame.display.set_mode((self._app_config.width, self._app_config.height))
        self._pg_font = pygame.font.SysFont(self._app_config.font, self._app_config.font_size)
        self._clock = pygame.time.Clock()

        self._space = pymunk.Space()
        self._space.gravity = (0.0, 900.0)
        self._draw_options = pymunk.pygame_util.DrawOptions(self.screen)

    def add_swing(self, start_pos: Vec2d, shape_filter_group: int) -> None:
        """
        Add Swing to application.

        Parameters:
            start_pos (Vec2d): Start position of top hinge of Swing
            shape_filter_group (int): Collision mask
        """
        _swing = Swing.create(self._swing_config, self._space, start_pos, shape_filter_group)
        self._swings.append(_swing)

    def add_stickman(self, start_pos: Vec2d, shape_filter_group: int) -> None:
        """
        Add Stickman to application.

        Parameters:
            start_pos (Vec2d): Start position of foot of Stickman
            shape_filter_group (int): Collision mask
        """
        _stickman = Stickman.create(self._stickman_config, self._space, start_pos, shape_filter_group)
        self._members.append(_stickman)

    def write_text(self, text: str, x: float, y: float) -> None:
        """
        Write text to the screen at the given position.

        Parameters:
            text (str): Text to write
            x (float): x coordinate of text's position
            y (float): y coordinate of text's position
        """
        _text = self._pg_font.render(text, 1, (255, 255, 255))
        self.screen.blit(_text, (x, y))

    def update(self) -> None:
        """
        Display application information to screen.
        """
        for swing in self._swings:
            swing.update()
        self._space.debug_draw(self._draw_options)
        self._space.step(1 / self._app_config.fps)

    def run(self) -> None:
        """
        Run the application and handle events.
        """
        self._running = True
        while self._running:
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    self._running = False
                    return

            self.screen.fill((180, 180, 180))

            self.update()
            pygame.display.update()
            self._clock.tick(self._app_config.fps)
