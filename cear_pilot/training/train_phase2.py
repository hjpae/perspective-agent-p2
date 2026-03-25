# cear_pilot/training/train_phase2.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from cear_pilot.envs.nzone_common import NZoneCommonConfig, NZoneCommonEnv
from cear_pilot.models.agent import CEARAgent, AgentConfig
from cear_pilot.models.decoder import ObsDecoder, DecoderConfig


def onehot(indices: torch.Tensor, n: int) -> torch.Tensor:
    return F.one_hot(indices.long(), num_classes=n).float()


@torch.no_grad()
def make_proprio_from_last_action(last_action: int, n_actions: int, device: torch.device) -> torch.Tensor:
    a = torch.tensor([last_action], device=device)
    return onehot(a, n_actions)


def timestamp_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def save_meta(run_dir: Path, meta: Dict) -> None:
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))


def seed_everything(seed: int, deterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class EMAMeanVar:
    def __init__(self, beta: float = 0.99, eps: float = 1e-8):
        self.beta = beta
        self.eps = eps
        self.mean = None
        self.var = None

    def update(self, x: float) -> Tuple[float, float]:
        if self.mean is None:
            self.mean = x
            self.var = 0.0
        else:
            m = self.mean
            self.mean = self.beta * self.mean + (1.0 - self.beta) * x
            self.var = self.beta * self.var + (1.0 - self.beta) * (x - m) * (x - m)
        std = float(np.sqrt(max(self.var, 0.0) + self.eps))
        return float(self.mean), std


def save_checkpoint(run_dir: Path, tag: str, agent: CEARAgent, decoder: ObsDecoder, meta: Dict) -> None:
    ckpt = {
        "agent_state": agent.state_dict(),
        "decoder_state": decoder.state_dict(),
        "meta": meta,
    }
    torch.save(ckpt, run_dir / f"ckpt_{tag}.pt")


def try_save_table(rows, out_path: Path) -> Path:
    df = pd.DataFrame(rows)
    try:
        p = out_path.with_suffix(".parquet")
        df.to_parquet(p, index=False)
        return p
    except Exception:
        p = out_path.with_suffix(".csv")
        df.to_csv(p, index=False)
        return p


def action_name(a: int) -> str:
    return ["UP", "DOWN", "LEFT", "RIGHT", "STAY"][int(a)]


def load_phase1_checkpoint(
    ckpt_path: str,
    device: str,
    obs_dim_override: int | None = None,
    update_mode: str = "fixed",
    alpha_fixed: float = 0.10,
    alpha_min: float = 0.03,
    alpha_max: float = 0.30,
):
    ckpt = torch.load(ckpt_path, map_location=device)
    meta = ckpt["meta"]

    agent_cfg = AgentConfig(device=device)
    agent_cfg.encoder.__dict__.update(meta["agent_cfg"]["encoder"])
    agent_cfg.world.__dict__.update(meta["agent_cfg"]["world"])
    agent_cfg.state.__dict__.update(meta["agent_cfg"]["state"])
    agent_cfg.policy.__dict__.update(meta["agent_cfg"]["policy"])

    if obs_dim_override is not None:
        agent_cfg.encoder.obs_dim = int(obs_dim_override)

    agent_cfg.world.update_mode = str(update_mode)
    agent_cfg.world.alpha_fixed = float(alpha_fixed)
    agent_cfg.world.alpha_min = float(alpha_min)
    agent_cfg.world.alpha_max = float(alpha_max)

    agent = CEARAgent(agent_cfg)
    agent.load_state_dict(ckpt["agent_state"])

    dec_cfg = DecoderConfig(**meta["decoder_cfg"])
    if obs_dim_override is not None:
        dec_cfg.obs_dim = int(obs_dim_override)
    decoder = ObsDecoder(dec_cfg)
    decoder.load_state_dict(ckpt["decoder_state"])

    return agent, decoder, meta


def build_evidence_pressure(
    pred_now: float,
    pred_ema: float,
    info: Dict,
) -> float:
    """
    Composite evidence-integration pressure for adaptive alpha.

    This is the key signal for the maturity assay.
    It combines:
    - immediate prediction mismatch
    - accumulated mismatch
    - encounter exposure depth
    - rupture occurrence
    - misleading/reliability pressure
    - conflict accumulation
    - fragility

    Keep this scalar; the world_latent alpha-net already learns how to use it.
    """
    on_enc = float(info.get("on_encounter", 0))
    row_exposure = float(info.get("row_exposure_mult", 1.0))
    rupture = float(info.get("rupture", 0))
    reliability = float(info.get("reliability_estimate", 0.0))
    conflict_load = float(info.get("conflict_load", 0.0))
    fragility = float(info.get("fragility", 0.0))
    encounter_outcome = int(info.get("encounter_outcome", 1))

    misleading_pressure = max(0.0, -reliability)
    supportive_relief = max(0.0, reliability)

    # outcome-sensitive exposure term
    if encounter_outcome == 2:   # misleading
        outcome_term = 1.25 * on_enc * row_exposure
    elif encounter_outcome == 0: # supportive
        outcome_term = -0.50 * on_enc * row_exposure
    else:
        outcome_term = 0.30 * on_enc * row_exposure

    err_value = (
        0.30 * pred_now
        + 0.22 * pred_ema
        + 0.14 * outcome_term
        + 0.14 * rupture
        + 0.10 * misleading_pressure
        + 0.06 * conflict_load
        + 0.04 * fragility
        - 0.05 * supportive_relief
    )
    return float(max(0.0, err_value))


def print_phase2_status(
    step: int,
    episode: int,
    info: Dict,
    loss: float,
    loss_pred: float,
    loss_smooth: float,
    loss_actor: float,
    entropy: float,
    alpha: float,
    g_norm: float,
    recent_actions: deque,
    recent_x: deque,
    recent_zone: deque,
    width: int,
) -> None:
    if len(recent_actions) == 0:
        return

    arr_a = np.array(recent_actions, dtype=np.int32)
    arr_x = np.array(recent_x, dtype=np.float32)
    arr_z = np.array(recent_zone, dtype=np.int32)

    right_rate = float(np.mean(arr_a == 3))
    left_rate = float(np.mean(arr_a == 2))
    stay_rate = float(np.mean(arr_a == 4))
    vertical_rate = float(np.mean((arr_a == 0) | (arr_a == 1)))
    x_mean = float(np.mean(arr_x))
    x_norm = x_mean / max(1, width - 1)

    zone_counts = np.bincount(arr_z, minlength=5)
    zone_pref = int(np.argmax(zone_counts))

    print(
        f"[phase2] step={step:6d} ep={episode:4d} "
        f"pos=({int(info['x']):2d},{int(info['y']):1d}) zone={int(info['zone_id'])} "
        f"sigma={float(info['current_sigma']):.3f} enc={int(info.get('on_encounter', 0))} "
        f"profile={str(info.get('encounter_profile', 'none'))} outcome={int(info.get('encounter_outcome', 1))} "
        f"rel={float(info.get('reliability_estimate', 0.0)):.3f} "
        f"frag={float(info.get('fragility', 0.0)):.3f} load={float(info.get('conflict_load', 0.0)):.3f} "
        f"rup={int(info.get('rupture', 0))} "
        f"loss={loss:.4f} pred={loss_pred:.4f} smooth={loss_smooth:.4f} actor={loss_actor:.4f} "
        f"H={entropy:.3f} alpha={alpha:.3f} ||g||={g_norm:.3f} | "
        f"recent_pref: right={right_rate:.2f} left={left_rate:.2f} stay={stay_rate:.2f} vert={vertical_rate:.2f} "
        f"x_mean={x_mean:.2f} x_norm={x_norm:.2f} dominant_zone={zone_pref}"
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--phase1_ckpt", type=str, required=True)
    ap.add_argument("--steps", type=int, default=36000)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cpu")

    ap.add_argument("--w_smooth", type=float, default=0.05)
    ap.add_argument("--w_entropy", type=float, default=0.02)
    ap.add_argument("--w_actor", type=float, default=0.05)
    ap.add_argument("--actor_b", type=float, default=0.98)
    ap.add_argument("--warmup_steps", type=int, default=1000)

    ap.add_argument("--width", type=int, default=23)
    ap.add_argument("--height", type=int, default=7)
    ap.add_argument("--obs_dim", type=int, default=8)
    ap.add_argument("--max_steps", type=int, default=240)

    ap.add_argument("--phase2_sigma_left", type=float, default=0.50)
    ap.add_argument("--phase2_sigma_right", type=float, default=0.05)

    ap.add_argument("--update_mode", type=str, default="adaptive", choices=["fixed", "adaptive"])
    ap.add_argument("--alpha_fixed", type=float, default=0.10)
    ap.add_argument("--alpha_min", type=float, default=0.03)
    ap.add_argument("--alpha_max", type=float, default=0.30)

    ap.add_argument("--log_traj", action="store_true")
    ap.add_argument("--log_every", type=int, default=1)
    ap.add_argument("--save_ckpt_every", type=int, default=12000)

    ap.add_argument("--print_every", type=int, default=500)
    ap.add_argument("--print_window", type=int, default=200)

    ap.add_argument("--reset_g_every_episode", action="store_true")

    args = ap.parse_args()

    seed_everything(args.seed, deterministic=True)
    device = torch.device(args.device)

    env_cfg = NZoneCommonConfig(
        phase="phase2",
        width=args.width,
        height=args.height,
        obs_dim=args.obs_dim,
        max_steps=args.max_steps,
        use_encounter=True,
        phase2_sigma_left=args.phase2_sigma_left,
        phase2_sigma_right=args.phase2_sigma_right,
    )
    env = NZoneCommonEnv(config=env_cfg)

    obs, info = env.reset(seed=args.seed)
    try:
        env.action_space.seed(args.seed)
        env.observation_space.seed(args.seed)
    except Exception:
        pass

    agent, decoder, phase1_meta = load_phase1_checkpoint(
        ckpt_path=args.phase1_ckpt,
        device=args.device,
        obs_dim_override=args.obs_dim,
        update_mode=args.update_mode,
        alpha_fixed=args.alpha_fixed,
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
    )
    agent.to(device)
    decoder.to(device)

    params = list(agent.parameters()) + list(decoder.parameters())
    opt = torch.optim.Adam(params, lr=args.lr)

    n_actions = int(env.action_space.n)

    run_dir = Path("outputs") / "runs" / timestamp_id()
    run_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "env_type": "phase2_common",
        "seed": args.seed,
        "steps": args.steps,
        "lr": args.lr,
        "device": args.device,
        "warmup_steps": int(args.warmup_steps),
        "reset_g_every_episode": bool(args.reset_g_every_episode),
        "phase1_ckpt": str(Path(args.phase1_ckpt).resolve()),
        "phase1_meta": phase1_meta,
        "loss_weights": {
            "w_smooth": args.w_smooth,
            "w_entropy": args.w_entropy,
            "w_actor": args.w_actor,
        },
        "actor_b": args.actor_b,
        "env_cfg": asdict(env_cfg),
        "agent_cfg": {
            "encoder": asdict(agent.cfg.encoder),
            "world": asdict(agent.cfg.world),
            "state": asdict(agent.cfg.state),
            "policy": asdict(agent.cfg.policy),
        },
        "decoder_cfg": asdict(decoder.cfg),
    }
    save_meta(run_dir, meta)

    log_rows = []
    log_every = int(max(1, args.log_every))

    agent.reset(batch_size=1)
    last_action = 4
    g_prev = agent.get_latents()["g"].detach().clone()

    pred_err_ema = EMAMeanVar(beta=0.97)
    actor_baseline = None
    prev_err_t = torch.zeros((1, 1), dtype=torch.float32, device=device)

    episode = 0
    t0 = time.time()

    act_hist = np.zeros(n_actions, dtype=np.int64)
    zone_hist = np.zeros(5, dtype=np.int64)

    recent_actions = deque(maxlen=args.print_window)
    recent_x = deque(maxlen=args.print_window)
    recent_zone = deque(maxlen=args.print_window)

    try:
        for step in range(args.steps):
            x_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            p_t = make_proprio_from_last_action(last_action, n_actions, device=device)

            # alpha at this step responds to previously accumulated evidence pressure
            out = agent.forward_step(x_t, p_t, ablate_g=False, err_t=prev_err_t)
            g_t = out["g"]
            s_t = out["s"]
            logits_pred = out["logits"]
            alpha_t = out["alpha"]

            logits_act = agent.policy(s_t.detach())
            pi_act = torch.softmax(logits_act, dim=-1)
            pi_pred = torch.softmax(logits_pred, dim=-1).detach()

            a_t = agent.policy.sample_action(logits_act, greedy=False)
            a_int = int(a_t.item())

            obs_next, _, terminated, truncated, info2 = env.step(a_int)
            x_next = torch.tensor(obs_next, dtype=torch.float32, device=device).unsqueeze(0)

            xhat_all = decoder.predict_all_actions(g_t)
            xhat_exp = torch.sum(pi_pred.unsqueeze(-1) * xhat_all, dim=1)

            loss_pred = F.mse_loss(xhat_exp, x_next)
            loss_smooth = torch.mean((g_t - g_prev) ** 2)
            entropy = -torch.sum(pi_act * torch.log(pi_act + 1e-9), dim=-1).mean()

            per_a_err = torch.mean((xhat_all - x_next.unsqueeze(1)) ** 2, dim=-1).squeeze(0)
            e_chosen = per_a_err[a_int]

            with torch.no_grad():
                pred_now = float(e_chosen.detach().item())
                pred_hist, _ = pred_err_ema.update(pred_now)

                if actor_baseline is None:
                    actor_baseline = pred_now
                if args.actor_b > 0.0:
                    actor_baseline = float(args.actor_b * actor_baseline + (1.0 - args.actor_b) * pred_now)

                err_value = build_evidence_pressure(
                    pred_now=pred_now,
                    pred_ema=pred_hist,
                    info=info2,
                )
                err_t = torch.tensor([[err_value]], dtype=torch.float32, device=device)

            adv = float(e_chosen.detach().item()) - float(actor_baseline)
            logp = torch.log_softmax(logits_act, dim=-1)[0, a_int]
            loss_actor = -adv * logp

            if step < args.warmup_steps:
                loss = loss_pred + args.w_smooth * loss_smooth - args.w_entropy * entropy
            else:
                loss = (
                    loss_pred
                    + args.w_smooth * loss_smooth
                    - args.w_entropy * entropy
                    + args.w_actor * loss_actor
                )

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

            act_hist[a_int] += 1
            zone_hist[int(info2["zone_id"])] += 1

            recent_actions.append(a_int)
            recent_x.append(int(info2["x"]))
            recent_zone.append(int(info2["zone_id"]))

            if args.log_traj and (step % log_every == 0):
                row = {
                    "step": int(step),
                    "episode": int(episode),
                    "t": int(info2["t"]),
                    "x": int(info2["x"]),
                    "y": int(info2["y"]),
                    "zone_id": int(info2["zone_id"]),
                    "on_encounter": int(info2.get("on_encounter", 0)),
                    "encounter_idx": int(info2.get("encounter_idx", -1)),
                    "encounter_profile": str(info2.get("encounter_profile", "none")),
                    "encounter_outcome": int(info2.get("encounter_outcome", 1)),
                    "row_band": str(info2.get("row_band", "balanced")),
                    "row_exposure_mult": float(info2.get("row_exposure_mult", 1.0)),
                    "current_sigma": float(info2["current_sigma"]),
                    "reliability_estimate": float(info2.get("reliability_estimate", 0.0)),
                    "fragility": float(info2.get("fragility", 0.0)),
                    "rupture_memory": float(info2.get("rupture_memory", 0.0)),
                    "conflict_load": float(info2.get("conflict_load", 0.0)),
                    "rupture": int(info2.get("rupture", 0)),
                    "evidence_pressure": float(err_value),
                    "action": int(a_int),
                    "action_name": action_name(a_int),
                    "loss": float(loss.detach().item()),
                    "loss_pred": float(loss_pred.detach().item()),
                    "loss_smooth": float(loss_smooth.detach().item()),
                    "loss_actor": float(loss_actor.detach().item()),
                    "entropy": float(entropy.detach().item()),
                    "alpha": float(alpha_t.mean().detach().item()),
                    "g_norm": float(torch.norm(g_t, dim=-1).mean().detach().item()),
                }
                for i, v in enumerate(g_t.squeeze(0).detach().cpu().numpy()):
                    row[f"g_{i}"] = float(v)
                log_rows.append(row)

            if (step + 1) % args.print_every == 0:
                print_phase2_status(
                    step=step + 1,
                    episode=episode,
                    info=info2,
                    loss=float(loss.detach().item()),
                    loss_pred=float(loss_pred.detach().item()),
                    loss_smooth=float(loss_smooth.detach().item()),
                    loss_actor=float(loss_actor.detach().item()),
                    entropy=float(entropy.detach().item()),
                    alpha=float(alpha_t.mean().detach().item()),
                    g_norm=float(torch.norm(g_t, dim=-1).mean().detach().item()),
                    recent_actions=recent_actions,
                    recent_x=recent_x,
                    recent_zone=recent_zone,
                    width=args.width,
                )

            obs = obs_next
            info = info2
            last_action = a_int
            g_prev = g_t.detach().clone()
            prev_err_t = err_t.detach()

            done = bool(terminated or truncated)
            if done:
                episode += 1
                obs, info = env.reset(seed=int(args.seed + episode))
                if args.reset_g_every_episode:
                    agent.reset(batch_size=1)
                last_action = 4
                g_prev = agent.get_latents()["g"].detach().clone()
                prev_err_t = torch.zeros((1, 1), dtype=torch.float32, device=device)

            if (step + 1) % args.save_ckpt_every == 0:
                save_checkpoint(run_dir, f"{step+1}", agent, decoder, meta)

        save_checkpoint(run_dir, "final", agent, decoder, meta)

    finally:
        if len(log_rows) > 0:
            try_save_table(log_rows, run_dir / "train_log")

        summary = {
            "n_steps": int(args.steps),
            "n_episodes": int(episode + 1),
            "action_hist": act_hist.tolist(),
            "zone_hist": zone_hist.tolist(),
            "elapsed_sec": float(time.time() - t0),
        }
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        print(f"[phase2] saved to: {run_dir}")


if __name__ == "__main__":
    main()