# cear_pilot/training/pygame_viewer.py
# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Optional

from cear_pilot.envs.nzone_grid import NZoneGridEnv


class PygameGridViewer:
    """
    Minimal live viewer for NZoneGridEnv during training.
    Controls:
      - Close window: stop training
      - SPACE: pause/resume
    """

    def __init__(
        self,
        width: int,
        height: int,
        cell_px: int = 40,
        fps: int = 12,
        title: str = "Live Training",
    ):
        try:
            import pygame  # type: ignore
        except Exception as e:
            raise ImportError("pygame required for --view. Install with: pip install pygame") from e

        self.pygame = pygame
        pygame.init()

        self.W = int(width)
        self.H = int(height)
        self.cell = int(cell_px)
        self.fps = int(fps)

        self.pad_top = 90
        self.screen_w = self.W * self.cell
        self.screen_h = self.H * self.cell + self.pad_top

        self.screen = pygame.display.set_mode((self.screen_w, self.screen_h))
        pygame.display.set_caption(title)

        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 18)
        self.small = pygame.font.SysFont("Arial", 14)

        self.paused = False

        self.zone_colors = [
            (85, 55, 40),
            (40, 75, 55),
            (35, 55, 90),
        ]
        self.grid_line = (25, 25, 25)
        self.agent_color = (230, 230, 230)
        self.text_color = (240, 240, 240)
        self.panel_bg = (15, 15, 15)

    def _zone_of_x(self, x: int) -> int:
        if x < self.W / 3:
            return 0
        elif x < 2 * self.W / 3:
            return 1
        return 2

    def pump(self) -> Optional[bool]:
        pygame = self.pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
        return True

    def wait_if_paused(self) -> Optional[bool]:
        while self.paused:
            ok = self.pump()
            if ok is False:
                return False
            self.clock.tick(12)
        return True

    def draw(
        self,
        env: NZoneGridEnv,
        step: int,
        episode: int,
        last_action: int,
        loss: float,
        loss_pred: float,
        loss_smooth: float,
        entropy: float,
        g_norm: float,
    ) -> Optional[bool]:
        pygame = self.pygame

        ok = self.pump()
        if ok is False:
            return False
        ok = self.wait_if_paused()
        if ok is False:
            return False

        self.screen.fill(self.panel_bg)
        pygame.draw.rect(self.screen, self.panel_bg, (0, 0, self.screen_w, self.pad_top))

        action_names = ["U", "D", "L", "R", "S"]
        zid = int(env.zone_id())
        x, y, t = int(env.x), int(env.y), int(env.t)

        line1 = (
            f"step={step}  ep={episode}  t={t}  zone={zid}  pos=({x},{y})  "
            f"a={action_names[last_action] if 0<=last_action<5 else last_action}"
        )
        line2 = (
            f"loss={loss:.4f}  pred={loss_pred:.4f}  smooth={loss_smooth:.4f}  "
            f"H={entropy:.3f}  ||g||={g_norm:.3f}   (SPACE: pause/resume)"
        )
        txt1 = self.font.render(line1, True, self.text_color)
        txt2 = self.small.render(line2, True, self.text_color)
        self.screen.blit(txt1, (10, 10))
        self.screen.blit(txt2, (10, 40))

        if self.paused:
            paused = self.font.render("PAUSED", True, (255, 220, 120))
            self.screen.blit(paused, (10, 65))

        y0 = self.pad_top

        for yy in range(self.H):
            for xx in range(self.W):
                zid_x = self._zone_of_x(xx)
                col = self.zone_colors[zid_x]
                rect = pygame.Rect(xx * self.cell, y0 + yy * self.cell, self.cell, self.cell)
                pygame.draw.rect(self.screen, col, rect)

        for xx in range(self.W + 1):
            pygame.draw.line(
                self.screen,
                self.grid_line,
                (xx * self.cell, y0),
                (xx * self.cell, y0 + self.H * self.cell),
                1,
            )
        for yy in range(self.H + 1):
            pygame.draw.line(
                self.screen,
                self.grid_line,
                (0, y0 + yy * self.cell),
                (self.W * self.cell, y0 + yy * self.cell),
                1,
            )

        ax = x * self.cell + self.cell // 2
        ay = y0 + y * self.cell + self.cell // 2
        r = max(6, self.cell // 3)
        pygame.draw.circle(self.screen, self.agent_color, (ax, ay), r)

        pygame.display.flip()
        self.clock.tick(self.fps)
        return True

    def close(self):
        try:
            self.pygame.quit()
        except Exception:
            pass
