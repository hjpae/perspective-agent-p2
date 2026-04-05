# cear_pilot/training/pygame_viewer_phase2.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pygame

from cear_pilot.envs.nzone_phase2 import NZonePhase2Env, NZonePhase2Config


# ----------------------------
# Helpers
# ----------------------------

def clamp(v, lo=0, hi=255):
    return max(lo, min(hi, int(v)))


def lerp_color(c1, c2, a: float):
    a = float(np.clip(a, 0.0, 1.0))
    return tuple(clamp((1 - a) * x + a * y) for x, y in zip(c1, c2))


def value_to_heat(v, vmin, vmax):
    """Blue -> gray -> orange style heatmap."""
    if vmax <= vmin:
        return (128, 128, 128)
    x = (float(v) - vmin) / (vmax - vmin)
    x = float(np.clip(x, 0.0, 1.0))
    if x < 0.5:
        return lerp_color((70, 95, 160), (185, 185, 185), x / 0.5)
    return lerp_color((185, 185, 185), (220, 125, 60), (x - 0.5) / 0.5)


def sigma_to_zone_color(sig, sig_left, sig_right):
    """
    Low sigma (clean/right) -> blue-green
    High sigma (noisy/left) -> warm brown-red
    """
    if sig_left <= sig_right:
        return (120, 120, 120)
    a = (sig - sig_right) / (sig_left - sig_right)
    a = float(np.clip(a, 0.0, 1.0))
    return lerp_color((30, 120, 110), (165, 90, 65), a)


# ----------------------------
# Viewer
# ----------------------------

class Phase2EnvViewer:
    def __init__(
        self,
        env: NZonePhase2Env,
        cell_px: int = 42,
        fps: int = 8,
        title: str = "CEAR Phase 2 Environment Viewer",
    ):
        pygame.init()
        pygame.font.init()

        self.env = env
        self.cfg = env.cfg
        self.W = env.W
        self.H = env.H
        self.cell = int(cell_px)
        self.fps = int(fps)

        self.pad_top = 100
        self.pad_right = 260
        self.screen_w = self.W * self.cell + self.pad_right
        self.screen_h = self.H * self.cell + self.pad_top

        self.screen = pygame.display.set_mode((self.screen_w, self.screen_h))
        pygame.display.set_caption(title)

        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 20)
        self.small = pygame.font.SysFont("Arial", 16)
        self.tiny = pygame.font.SysFont("Arial", 13)

        self.running = True
        self.paused = False
        self.auto_step = False
        self.show_patch = True
        self.show_mu = False
        self.show_sigma = True
        self.show_visited = True

        self.last_action = self.env.ACTION_STAY
        self.last_obs = None
        self.last_info = None

        self.save_dir = Path("pygame_captures")
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.action_names = {
            self.env.ACTION_UP: "UP",
            self.env.ACTION_DOWN: "DOWN",
            self.env.ACTION_LEFT: "LEFT",
            self.env.ACTION_RIGHT: "RIGHT",
            self.env.ACTION_STAY: "STAY",
        }

    def reset(self, seed: int = 0):
        obs, info = self.env.reset(seed=seed)
        self.last_obs = obs
        self.last_info = info
        self.last_action = self.env.ACTION_STAY

    def step_env(self, action: int | None = None):
        if action is None:
            action = self.last_action
        obs, _, terminated, truncated, info = self.env.step(action)
        self.last_obs = obs
        self.last_info = info
        self.last_action = action
        if terminated or truncated:
            self.reset(seed=None)

    def screenshot(self):
        fname = self.save_dir / f"phase2_env_t{self.env.t:03d}.png"
        pygame.image.save(self.screen, str(fname))
        print(f"[saved] {fname}")

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return

            if event.type == pygame.KEYDOWN:
                key = event.key

                if key == pygame.K_ESCAPE:
                    self.running = False
                    return
                elif key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif key == pygame.K_TAB:
                    self.auto_step = not self.auto_step
                elif key == pygame.K_r:
                    self.reset(seed=0)
                elif key == pygame.K_s:
                    self.screenshot()
                elif key == pygame.K_p:
                    self.show_patch = not self.show_patch
                elif key == pygame.K_m:
                    self.show_mu = not self.show_mu
                elif key == pygame.K_n:
                    self.show_sigma = not self.show_sigma
                elif key == pygame.K_v:
                    self.show_visited = not self.show_visited

                # manual agent movement
                elif key == pygame.K_UP:
                    self.step_env(self.env.ACTION_UP)
                elif key == pygame.K_DOWN:
                    self.step_env(self.env.ACTION_DOWN)
                elif key == pygame.K_LEFT:
                    self.step_env(self.env.ACTION_LEFT)
                elif key == pygame.K_RIGHT:
                    self.step_env(self.env.ACTION_RIGHT)
                elif key == pygame.K_PERIOD:
                    self.step_env(self.env.ACTION_STAY)

    def draw_header(self):
        pygame.draw.rect(self.screen, (18, 18, 20), (0, 0, self.screen_w, self.pad_top))

        info = self.last_info or {}
        p_active = int(info.get("perturbation_active", 0))
        p_trace = float(info.get("perturbation_trace", 0.0))
        n_p = int(info.get("n_perturbations", self.cfg.n_perturbations))
        sigma = float(info.get("current_sigma", self.env._sigma_map[self.env.y, self.env.x]))

        line1 = (
            f"t={self.env.t:03d}   pos=({self.env.x},{self.env.y})   "
            f"zone={self.env.report_zone_id(self.env.x)}   "
            f"action={self.action_names.get(self.last_action, self.last_action)}"
        )
        line2 = (
            f"perturb_active={p_active}   perturb_trace={p_trace:.2f}   "
            f"n_perturb={n_p}   sigma_here={sigma:.3f}"
        )
        line3 = (
            "[SPACE] pause  [TAB] auto-step  [Arrows] move  [.] stay  "
            "[S] save PNG  [R] reset  [P] patch  [M] mu  [N] sigma  [V] visited"
        )

        self.screen.blit(self.font.render(line1, True, (235, 235, 235)), (12, 10))
        self.screen.blit(self.small.render(line2, True, (210, 210, 210)), (12, 40))
        self.screen.blit(self.small.render(line3, True, (180, 180, 180)), (12, 68))

        if p_active:
            badge = self.font.render("PERTURBATION ON", True, (255, 220, 120))
            self.screen.blit(badge, (self.screen_w - 450, 65))
        elif self.paused:
            badge = self.font.render("PAUSED", True, (255, 220, 120))
            self.screen.blit(badge, (self.screen_w - 120, 14))

    def draw_grid(self):
        y0 = self.pad_top
        mu_min = float(np.min(self.env._mu_map))
        mu_max = float(np.max(self.env._mu_map))
        sig_left = float(self.cfg.sigma_left)
        sig_right = float(self.cfg.sigma_right)

        for yy in range(self.H):
            for xx in range(self.W):
                rect = pygame.Rect(xx * self.cell, y0 + yy * self.cell, self.cell, self.cell)

                mu = float(self.env._mu_map[yy, xx])
                sig = float(self.env._sigma_map[yy, xx])

                if self.show_sigma:
                    base = sigma_to_zone_color(sig, sig_left, sig_right)
                else:
                    base = (70, 70, 76)

                if self.show_mu:
                    mu_col = value_to_heat(mu, mu_min, mu_max)
                    base = lerp_color(base, mu_col, 0.50)

                pygame.draw.rect(self.screen, base, rect)

                # perturbation overlay: emphasize inversion period
                if self.last_info and int(self.last_info.get("perturbation_active", 0)) == 1:
                    inv = float(self.env._inversion_pattern[xx])  # left +1 -> right -1
                    if inv >= 0:
                        overlay = (255, 255, 255, int(55 * abs(inv)))
                    else:
                        overlay = (30, 30, 30, int(45 * abs(inv)))

                    surf = pygame.Surface((self.cell, self.cell), pygame.SRCALPHA)
                    surf.fill(overlay)
                    self.screen.blit(surf, rect.topleft)

                if self.show_visited and (xx, yy) in self.env.visited:
                    pygame.draw.circle(
                        self.screen,
                        (220, 220, 220),
                        rect.center,
                        max(2, self.cell // 10),
                    )

                pygame.draw.rect(self.screen, (30, 30, 32), rect, 1)

        # reporting zone boundaries
        for b in self.cfg.report_zone_boundaries:
            x = b * self.cell
            pygame.draw.line(
                self.screen,
                (250, 250, 250),
                (x, y0),
                (x, y0 + self.H * self.cell),
                2,
            )

        # agent
        cx = self.env.x * self.cell + self.cell // 2
        cy = y0 + self.env.y * self.cell + self.cell // 2
        pygame.draw.circle(self.screen, (245, 245, 245), (cx, cy), self.cell // 3)
        pygame.draw.circle(self.screen, (40, 40, 40), (cx, cy), self.cell // 3, 2)

    def draw_side_panel(self):
        x0 = self.W * self.cell + 12
        y0 = self.pad_top - 16
        panel_w = self.pad_right - 24
        panel_h = self.H * self.cell + 10

        pygame.draw.rect(self.screen, (24, 24, 28), (x0, y0, panel_w, panel_h), border_radius=8)
        pygame.draw.rect(self.screen, (70, 70, 78), (x0, y0, panel_w, panel_h), 1, border_radius=8)

        # patch values
        self.screen.blit(self.font.render("Local patch (8-neighbor)", True, (235, 235, 235)), (x0 + 10, y0 + 10))

        obs = self.last_obs if self.last_obs is not None else np.zeros((8,), dtype=np.float32)
        patch = np.asarray(obs[:8], dtype=np.float32)

        coords = list(self.cfg.patch_order)
        coord_to_val = {c: patch[i] for i, c in enumerate(coords)}

        mini = 40
        gx = x0 + 18
        gy = y0 + 44

        vals = list(coord_to_val.values())
        vmin, vmax = float(np.min(vals)), float(np.max(vals))

        for j, dy in enumerate([-1, 0, 1]):
            for i, dx in enumerate([-1, 0, 1]):
                rr = pygame.Rect(gx + i * mini, gy + j * mini, mini - 4, mini - 4)
                if dx == 0 and dy == 0:
                    pygame.draw.rect(self.screen, (240, 240, 240), rr, border_radius=6)
                    pygame.draw.rect(self.screen, (50, 50, 50), rr, 1, border_radius=6)
                    txt = self.small.render("A", True, (40, 40, 40))
                    self.screen.blit(txt, txt.get_rect(center=rr.center))
                else:
                    val = coord_to_val[(dx, dy)]
                    col = value_to_heat(val, vmin, vmax) if self.show_patch else (90, 90, 95)
                    pygame.draw.rect(self.screen, col, rr, border_radius=6)
                    pygame.draw.rect(self.screen, (40, 40, 42), rr, 1, border_radius=6)
                    txt = self.tiny.render(f"{val:+.2f}", True, (20, 20, 20))
                    self.screen.blit(txt, txt.get_rect(center=rr.center))

        # legends
        ly = gy + 3 * mini + 18
        lines = [
            f"start_xy = {self.cfg.start_xy}",
            f"sigma: {self.cfg.sigma_left:.2f} -> {self.cfg.sigma_right:.2f}",
            f"duration = {self.cfg.perturbation_duration}",
            f"scale = {self.cfg.perturbation_scale:.2f}",
            f"scheduled steps = {self.env.perturbation_steps}",
        ]
        for k, line in enumerate(lines):
            txt = self.small.render(line, True, (205, 205, 210))
            self.screen.blit(txt, (x0 + 10, ly + 22 * k))

    def draw(self):
        self.screen.fill((12, 12, 14))
        self.draw_header()
        self.draw_grid()
        self.draw_side_panel()
        pygame.display.flip()

    def run(self):
        while self.running:
            self.handle_events()
            if self.auto_step and not self.paused:
                self.step_env(self.last_action)
            self.draw()
            self.clock.tick(self.fps)

        pygame.quit()


def main():
    cfg = NZonePhase2Config(
        width=23,
        height=7,
        obs_dim=8,
        max_steps=300,
        start_xy=(11, 3),
        sigma_left=0.20,
        sigma_right=0.10,
        n_perturbations=4,
        perturbation_duration=15,
        perturbation_scale=0.12,
    )
    env = NZonePhase2Env(config=cfg)
    viewer = Phase2EnvViewer(env, cell_px=42, fps=8)
    viewer.reset(seed=0)
    viewer.run()


if __name__ == "__main__":
    main()