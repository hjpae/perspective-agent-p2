# cear_pilot/phase2_rollout_viewer.py
# -*- coding: utf-8 -*-

"""
Phase 2 rollout Pygames viewer (Spyder-friendly)
- Change MODE to check envs / random / learned policy. 
"""

from pathlib import Path
import sys
import numpy as np
import torch
import torch.nn.functional as F
import pygame

# =======================
# USER CONFIG
# =======================

MODE = "sweep"        # "sweep" | "random" | "checkpoint"
CKPT_PATH = None      # checkpoint path 

DEVICE = "cpu"
SEED = 0

EPISODES = 3
MAX_STEPS = 240

FPS = 8
CELL_PX = 42

GREEDY = True  # if checkpoint mode

# --- env overrides ---
MODE_SWITCH_PROB = 0.60
RUPTURE_BASE_PROB = 0.30
RUPTURE_OBS_SIGMA = 3.0
CONFLICT_LOAD_INCREMENT = 0.25
MISLEADING_REL_DELTA = -0.20

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nzone_phase2 import NZonePhase2Env, NZonePhase2Config
from agent import CEARAgent, AgentConfig
from decoder import ObsDecoder, DecoderConfig


ACTION_NAMES = ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]
MODE_NAMES = {0: "OPEN", 1: "MIXED", 2: "GUARDED"}


# =======================
# utils
# =======================

def onehot(indices, n):
    return F.one_hot(indices.long(), num_classes=n).float()


def make_proprio(last_action, n_actions, device):
    a = torch.tensor([last_action], device=device)
    return onehot(a, n_actions)


# =======================
# checkpoint loader
# =======================

def load_checkpoint(path, device, obs_dim_override=None):
    ckpt = torch.load(path, map_location=device)
    meta = ckpt["meta"]

    agent_cfg = AgentConfig(device=device)
    agent_cfg.encoder.__dict__.update(meta["agent_cfg"]["encoder"])
    agent_cfg.world.__dict__.update(meta["agent_cfg"]["world"])
    agent_cfg.state.__dict__.update(meta["agent_cfg"]["state"])
    agent_cfg.policy.__dict__.update(meta["agent_cfg"]["policy"])

    if obs_dim_override is not None:
        agent_cfg.encoder.obs_dim = int(obs_dim_override)

    agent = CEARAgent(agent_cfg)
    agent.load_state_dict(ckpt["agent_state"])
    agent.eval()

    dec_cfg = DecoderConfig(**meta["decoder_cfg"])
    if obs_dim_override is not None:
        dec_cfg.obs_dim = int(obs_dim_override)

    decoder = ObsDecoder(dec_cfg)
    decoder.load_state_dict(ckpt["decoder_state"])
    decoder.eval()

    return agent, decoder


# =======================
# Pygames viewer
# =======================

class Viewer:
    def __init__(self, env):
        pygame.init()
        pygame.font.init()

        self.env = env
        self.cell = CELL_PX
        self.pad_top = 150

        self.W, self.H = env.W, env.H

        self.screen = pygame.display.set_mode(
            (self.W * self.cell, self.H * self.cell + self.pad_top)
        )
        pygame.display.set_caption("Phase 2 Viewer")

        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 20)
        self.small = pygame.font.SysFont("Arial", 16)

        self.paused = False

    def pump(self):
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                return False
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_SPACE:
                    self.paused = not self.paused
                if e.key == pygame.K_ESCAPE:
                    return False
        return True

    def draw(self, step, action, info, ep):
        if not self.pump():
            return False

        while self.paused:
            if not self.pump():
                return False
            self.clock.tick(10)

        self.screen.fill((20, 20, 20))

        # text
        line1 = f"ep={ep} step={step} action={ACTION_NAMES[action]}"
        line2 = (
            f"rel={info.get('recent_reliability',0):+.2f} "
            f"frag={info.get('fragility',0):.2f} "
            f"rmem={info.get('rupture_memory',0):.2f} "
            f"cload={info.get('conflict_load',0):.2f}"
        )
        line3 = (
            f"enc={info.get('encounter_event',False)} "
            f"rupture={info.get('rupture',False)} "
            f"mode={MODE_NAMES.get(info.get('reliability_mode',1))}"
        )

        self.screen.blit(self.font.render(line1, True, (255,255,255)), (10, 10))
        self.screen.blit(self.small.render(line2, True, (255,255,255)), (10, 50))
        self.screen.blit(self.small.render(line3, True, (255,200,200)), (10, 80))

        y0 = self.pad_top

        # grid
        for y in range(self.H):
            for x in range(self.W):
                rect = pygame.Rect(
                    x*self.cell,
                    y0 + y*self.cell,
                    self.cell,
                    self.cell
                )
                pygame.draw.rect(self.screen, (70,90,110), rect, 0)

        # agent
        ax = int(info.get("x", 0))*self.cell + self.cell//2
        ay = y0 + int(info.get("y", 0))*self.cell + self.cell//2
        pygame.draw.circle(self.screen, (255,255,255), (ax, ay), self.cell//3)

        pygame.display.flip()
        self.clock.tick(FPS)
        return True


# =======================
# rollout policies
# =======================

def act_random(env):
    return int(env.action_space.sample())


def act_sweep(env):
    if env.x < env.W - 1:
        return 3  # RIGHT
    return 2      # LEFT


@torch.no_grad()
def act_model(obs, last_action, env, agent):
    x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    p = make_proprio(last_action, env.action_space.n, DEVICE)
    a, _ = agent.step(x, p_t=p, greedy=GREEDY)
    return int(a.item())


# =======================
# main
# =======================

def main():

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    cfg = NZonePhase2Config(
        max_steps=MAX_STEPS,
        mode_switch_prob=MODE_SWITCH_PROB,
        rupture_base_prob=RUPTURE_BASE_PROB,
        rupture_obs_sigma=RUPTURE_OBS_SIGMA,
        conflict_load_increment=CONFLICT_LOAD_INCREMENT,
        misleading_reliability_delta=MISLEADING_REL_DELTA,
    )

    env = NZonePhase2Env(cfg)

    agent = None
    if MODE == "checkpoint":
        agent, _ = load_checkpoint(CKPT_PATH, DEVICE, env.obs_dim)

    viewer = Viewer(env)

    try:
        for ep in range(EPISODES):

            obs, info = env.reset(seed=SEED+ep)
            last_action = 4

            if agent is not None:
                agent.reset(batch_size=1)

            for step in range(MAX_STEPS):

                if MODE == "random":
                    action = act_random(env)
                elif MODE == "sweep":
                    action = act_sweep(env)
                else:
                    action = act_model(obs, last_action, env, agent)

                obs, _, term, trunc, info = env.step(action)

                if not viewer.draw(step, action, info, ep):
                    return

                last_action = action

                if term or trunc:
                    break

    finally:
        pygame.quit()
        env.close()

if __name__ == "__main__":
    main()