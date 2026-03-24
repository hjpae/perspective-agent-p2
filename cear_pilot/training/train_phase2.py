# cear_pilot/training/train_phase2.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from cear_pilot.envs.nzone_phase2 import NZonePhase2Config, NZonePhase2Env
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
            self.mean = self.beta * self.mean + (1 - self.beta) * x
            self.var = self.beta * self.var + (1 - self.beta) * (x - m) * (x - m)
        std = float(np.sqrt(max(self.var, 0.0) + self.eps))
        return float(self.mean), std


def save_checkpoint(run_dir: Path, tag: str, agent: CEARAgent, decoder: ObsDecoder, meta: Dict) -> None:
    ckpt = {
        "agent_state": agent.state_dict(),
        "decoder_state": decoder.state_dict(),
        "meta": meta,
    }
    torch.save(ckpt, run_dir / f"ckpt_{tag}.pt")


def load_phase1_checkpoint(
    ckpt_path: str,
    device: str,
    obs_dim_override: int | None = None,
    g_damping_override: float | None = None,
    g_influence_override: float | None = None,
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
    if g_damping_override is not None:
        agent_cfg.world.g_damping = float(g_damping_override)
    if g_influence_override is not None:
        agent_cfg.state.g_influence = float(g_influence_override)

    agent = CEARAgent(agent_cfg)
    agent.load_state_dict(ckpt["agent_state"])

    dec_cfg = DecoderConfig(**meta["decoder_cfg"])
    if obs_dim_override is not None:
        dec_cfg.obs_dim = int(obs_dim_override)
    decoder = ObsDecoder(dec_cfg)
    decoder.load_state_dict(ckpt["decoder_state"])

    return agent, decoder, meta


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--phase1_ckpt", type=str, required=True)
    ap.add_argument("--steps", type=int, default=48000)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cpu")

    # maturity-oriented defaults: weaker actor, stronger entropy than collapse-prone version
    ap.add_argument("--w_smooth", type=float, default=0.05)
    ap.add_argument("--w_entropy", type=float, default=0.02)
    ap.add_argument("--w_actor", type=float, default=0.05)
    ap.add_argument("--actor_b", type=float, default=0.98)
    ap.add_argument("--warmup_steps", type=int, default=1000)

    # phase2 env
    ap.add_argument("--width", type=int, default=18)
    ap.add_argument("--height", type=int, default=9)
    ap.add_argument("--obs_dim", type=int, default=8)
    ap.add_argument("--max_steps", type=int, default=240)

    ap.add_argument("--zone_sigma", type=float, nargs=3, default=(0.42, 0.30, 0.20))
    ap.add_argument("--p_slip", type=float, nargs=3, default=(0.00, 0.03, 0.08))
    ap.add_argument("--p_drift", type=float, nargs=3, default=(0.00, 0.04, 0.08))
    ap.add_argument("--drift_vec", type=int, nargs=6, default=(0, 0, 1, 0, -1, 0))
    ap.add_argument("--volatile_zone", type=int, default=2)
    ap.add_argument("--volatile_period", type=int, default=20)
    ap.add_argument("--volatile_strength", type=float, default=0.10)

    ap.add_argument("--fragility_init", type=float, default=0.10)
    ap.add_argument("--fragility_decay", type=float, default=0.005)
    ap.add_argument("--rupture_memory_init", type=float, default=0.00)
    ap.add_argument("--rupture_memory_decay", type=float, default=0.96)
    ap.add_argument("--conflict_load_init", type=float, default=0.00)
    ap.add_argument("--conflict_load_decay", type=float, default=0.95)
    ap.add_argument("--zone_fragility_delta", type=float, nargs=3, default=(0.00, 0.02, 0.04))

    ap.add_argument("--mode_period", type=int, default=30)

    ap.add_argument("--rupture_base_prob", type=float, default=0.30)
    ap.add_argument("--rupture_fragility_weight", type=float, default=0.80)
    ap.add_argument("--rupture_memory_weight", type=float, default=0.30)
    ap.add_argument("--rupture_load_weight", type=float, default=0.25)
    ap.add_argument("--rupture_reliability_relief", type=float, default=0.20)

    ap.add_argument("--rupture_obs_corrupt_steps", type=int, default=4)
    ap.add_argument("--rupture_obs_sigma", type=float, default=3.0)
    ap.add_argument("--rupture_action_slip_prob", type=float, default=0.25)
    ap.add_argument("--rupture_memory_increment", type=float, default=0.30)
    ap.add_argument("--conflict_load_increment", type=float, default=0.25)

    # optional overrides for phase2 to make g a bit more influential without touching model files
    ap.add_argument("--g_damping_override", type=float, default=0.20)
    ap.add_argument("--g_influence_override", type=float, default=2.5)

    ap.add_argument("--log_traj", action="store_true")
    ap.add_argument("--log_every", type=int, default=1)
    ap.add_argument("--save_ckpt_every", type=int, default=12000)

    # default: carry g across episodes in phase2
    ap.add_argument("--reset_g_every_episode", action="store_true")

    args = ap.parse_args()

    seed_everything(args.seed, deterministic=True)
    device = torch.device(args.device)

    dv = args.drift_vec
    drift_vec = ((dv[0], dv[1]), (dv[2], dv[3]), (dv[4], dv[5]))

    env_cfg = NZonePhase2Config(
        width=args.width,
        height=args.height,
        obs_dim=args.obs_dim,
        max_steps=args.max_steps,
        zone_sigma=tuple(args.zone_sigma),
        p_slip=tuple(args.p_slip),
        p_drift=tuple(args.p_drift),
        drift_vec=drift_vec,
        volatile_zone=args.volatile_zone,
        volatile_period=args.volatile_period,
        volatile_strength=args.volatile_strength,
        fragility_init=args.fragility_init,
        fragility_decay=args.fragility_decay,
        rupture_memory_init=args.rupture_memory_init,
        rupture_memory_decay=args.rupture_memory_decay,
        conflict_load_init=args.conflict_load_init,
        conflict_load_decay=args.conflict_load_decay,
        zone_fragility_delta=tuple(args.zone_fragility_delta),
        mode_period=args.mode_period,
        rupture_base_prob=args.rupture_base_prob,
        rupture_fragility_weight=args.rupture_fragility_weight,
        rupture_memory_weight=args.rupture_memory_weight,
        rupture_load_weight=args.rupture_load_weight,
        rupture_reliability_relief=args.rupture_reliability_relief,
        rupture_obs_corrupt_steps=args.rupture_obs_corrupt_steps,
        rupture_obs_sigma=args.rupture_obs_sigma,
        rupture_action_slip_prob=args.rupture_action_slip_prob,
        rupture_memory_increment=args.rupture_memory_increment,
        conflict_load_increment=args.conflict_load_increment,
    )
    env = NZonePhase2Env(config=env_cfg)

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
        g_damping_override=args.g_damping_override,
        g_influence_override=args.g_influence_override,
    )
    agent.to(device)
    decoder.to(device)

    params = list(agent.parameters()) + list(decoder.parameters())
    opt = torch.optim.Adam(params, lr=args.lr)

    n_actions = int(env.action_space.n)

    run_dir = Path("outputs") / "runs" / timestamp_id()
    run_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "env_type": "phase2",
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

    # initialize agent ONCE; carry g by default
    agent.reset(batch_size=1)
    last_action = 4
    g_prev = agent.get_latents()["g"].detach().clone()

    ema_world = None
    pi_prev = None
    kl_ema = None
    maxpi_ema = None
    logits_norm_ema = None

    b = None
    err_stats = EMAMeanVar(beta=0.99)

    act_hist = np.zeros(n_actions, dtype=np.int64)
    zone_hist = np.zeros(3, dtype=np.int64)

    enc_hist = 0
    rup_hist = 0
    sup_hist = 0
    neu_hist = 0
    mis_hist = 0

    t0 = time.time()
    episode = 0
    t_in_ep = 0

    try:
        for step in range(args.steps):
            x_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            p_t = make_proprio_from_last_action(last_action, n_actions, device=device)

            out = agent.forward_step(x_t, p_t, ablate_g=False)
            g_t = out["g"]
            s_t = out["s"]
            logits_pred = out["logits"]

            logits_act = agent.policy(s_t.detach())
            pi_act = torch.softmax(logits_act, dim=-1)
            pi_pred = torch.softmax(logits_pred, dim=-1).detach()

            a_t = agent.policy.sample_action(logits_act, greedy=False)
            a_int = int(a_t.item())

            obs_next, _, terminated, truncated, info = env.step(a_int)
            x_next = torch.tensor(obs_next, dtype=torch.float32, device=device).unsqueeze(0)

            xhat_all = decoder.predict_all_actions(g_t)
            xhat_exp = torch.sum(pi_pred.unsqueeze(-1) * xhat_all, dim=1)

            loss_pred = F.mse_loss(xhat_exp, x_next)
            loss_smooth = torch.mean((g_t - g_prev) ** 2)
            entropy = -torch.sum(pi_act * torch.log(pi_act + 1e-9), dim=-1).mean()

            per_a_err = torch.mean((xhat_all - x_next.unsqueeze(1)) ** 2, dim=-1).squeeze(0)
            e_chosen = per_a_err[a_int]

            with torch.no_grad():
                e_val = float(e_chosen.detach().item())
                _, s = err_stats.update(e_val)

                if b is None:
                    b = e_val
                if args.actor_b > 0.0:
                    b = float(args.actor_b * b + (1.0 - args.actor_b) * e_val)

                baseline = float(b) if (args.actor_b > 0.0) else 0.0
                adv = -(e_val - baseline)
                adv = adv / (s + 1e-8)
                adv = float(np.clip(adv, -5.0, 5.0))

            logp = F.log_softmax(logits_act, dim=-1)[0, a_int]
            loss_actor = -(torch.tensor(adv, device=device) * logp)

            phase = "warmup" if step < int(args.warmup_steps) else "full"
            w_actor_eff = 0.0 if step < int(args.warmup_steps) else args.w_actor

            loss_world = loss_pred + args.w_smooth * loss_smooth
            loss = loss_world + w_actor_eff * loss_actor - args.w_entropy * entropy

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

            g_prev = g_t.detach().clone()
            obs = obs_next
            last_action = a_int

            enc_hist += int(bool(info.get("encounter_event", False)))
            rup_hist += int(bool(info.get("rupture", False)))
            if int(info.get("encounter_type", 1)) == 0:
                sup_hist += int(bool(info.get("encounter_event", False)))
            elif int(info.get("encounter_type", 1)) == 1:
                neu_hist += int(bool(info.get("encounter_event", False)))
            elif int(info.get("encounter_type", 1)) == 2:
                mis_hist += int(bool(info.get("encounter_event", False)))

            if args.log_traj and ((step % log_every) == 0):
                z = info.get("zone_id", -1)
                with torch.no_grad():
                    g_np = g_t.detach().squeeze(0).float().cpu().numpy()
                    g_norm = float(torch.linalg.vector_norm(g_t).item())
                    ent_val = float(entropy.item())

                row = {
                    "t_global": int(step),
                    "episode": int(episode),
                    "t_in_ep": int(t_in_ep),
                    "phase": str(phase),
                    "zone_id": int(z),
                    "x": int(info.get("x", -1)),
                    "y": int(info.get("y", -1)),
                    "action": int(a_int),
                    "entropy": float(ent_val),
                    "g_norm": float(g_norm),
                    "loss_pred": float(loss_pred.item()),
                    "loss_smooth": float(loss_smooth.item()),

                    "on_encounter": int(bool(info.get("on_encounter", False))),
                    "encounter_event": int(bool(info.get("encounter_event", False))),
                    "encounter_type": int(info.get("encounter_type", 1)),
                    "recent_reliability": float(info.get("recent_reliability", 0.0)),
                    "reliability_mode": int(info.get("reliability_mode", 1)),

                    "fragility": float(info.get("fragility", np.nan)),
                    "rupture_memory": float(info.get("rupture_memory", np.nan)),
                    "conflict_load": float(info.get("conflict_load", np.nan)),
                    "rupture": int(bool(info.get("rupture", False))),
                    "pending_ruptures": int(info.get("pending_ruptures", 0)),
                    "slip": int(bool(info.get("slip", False))),
                    "drift": int(bool(info.get("drift", False))),
                }
                for i, gv in enumerate(g_np):
                    row[f"g_{i}"] = float(gv)
                log_rows.append(row)

            t_in_ep += 1
            if terminated or truncated:
                obs, info = env.reset(seed=args.seed + episode + 1)

                if args.reset_g_every_episode:
                    agent.reset(batch_size=1)

                g_prev = agent.get_latents()["g"].detach().clone()
                last_action = 4
                episode += 1
                t_in_ep = 0

            with torch.no_grad():
                maxpi = float(pi_act.max(dim=-1).values.mean().item())
                if pi_prev is None:
                    kl = 0.0
                else:
                    kl_t = torch.sum(
                        pi_act * (torch.log(pi_act + 1e-9) - torch.log(pi_prev + 1e-9)),
                        dim=-1,
                    )
                    kl = float(kl_t.mean().item())
                pi_prev = pi_act.detach()

                maxpi_ema = maxpi if (maxpi_ema is None) else (0.98 * maxpi_ema + 0.02 * maxpi)
                kl_ema = kl if (kl_ema is None) else (0.98 * kl_ema + 0.02 * kl)

                ln = float(torch.mean(torch.abs(logits_act)).item())
                logits_norm_ema = ln if (logits_norm_ema is None) else (0.98 * logits_norm_ema + 0.02 * ln)

            act_hist[a_int] += 1
            z = info.get("zone_id", -1)
            if isinstance(z, (int, np.integer)) and 0 <= int(z) <= 2:
                zone_hist[int(z)] += 1

            lw = float(loss_world.item())
            ema_world = lw if ema_world is None else 0.98 * ema_world + 0.02 * lw

            if (step + 1) % 2000 == 0:
                dt = time.time() - t0
                with torch.no_grad():
                    e_det = per_a_err.detach().float().cpu().numpy()
                    e_min, e_max, e_std = float(e_det.min()), float(e_det.max()), float(e_det.std())

                act_prob = (act_hist / max(act_hist.sum(), 1)).tolist()
                zone_prob = (zone_hist / max(zone_hist.sum(), 1)).tolist()

                print(
                    f"[{step+1:>7}/{args.steps}] "
                    f"phase={phase} "
                    f"world={lw:.4f} w_ema={float(ema_world):.4f} pred={float(loss_pred.item()):.4f} "
                    f"smooth={float(loss_smooth.item()):.4f} | "
                    f"actor={float(loss_actor.item()):.4f} b={0.0 if b is None else float(b):.4f} "
                    f"H={float(entropy.item()):.3f} maxpi={float(maxpi_ema):.3f} KL={float(kl_ema):.6f} "
                    f"logits|.|={float(logits_norm_ema):.3f} "
                    f"e[min,max,std]={e_min:.3f},{e_max:.3f},{e_std:.3f} "
                    f"zone={[round(x,2) for x in zone_prob]} act={[round(x,2) for x in act_prob]} "
                    f"enc_win={enc_hist} rup_win={rup_hist} sup={sup_hist} neu={neu_hist} mis={mis_hist} "
                    f"frag={float(info.get('fragility', np.nan)):.2f} "
                    f"rmem={float(info.get('rupture_memory', np.nan)):.2f} "
                    f"rel={float(info.get('recent_reliability', 0.0)):.2f} "
                    f"carry_g={int(not args.reset_g_every_episode)} "
                    f"(ep={episode}, {dt:.1f}s)"
                )

                act_hist[:] = 0
                zone_hist[:] = 0
                enc_hist = 0
                rup_hist = 0
                sup_hist = 0
                neu_hist = 0
                mis_hist = 0
                t0 = time.time()

            if args.save_ckpt_every > 0 and ((step + 1) % args.save_ckpt_every == 0):
                save_checkpoint(run_dir, f"step{step+1}", agent, decoder, meta)

    finally:
        pass

    if args.log_traj and len(log_rows) > 0:
        df = pd.DataFrame(log_rows)
        out_parquet = run_dir / "train_traj.parquet"
        out_csv = run_dir / "train_traj.csv"
        try:
            df.to_parquet(out_parquet, index=False)
            print(f"Saved training trajectory to: {out_parquet}")
        except Exception as e:
            print(f"[WARN] Parquet failed ({type(e).__name__}: {e}). Falling back to CSV.")
            df.to_csv(out_csv, index=False)
            print(f"Saved training trajectory to: {out_csv}")

    save_checkpoint(run_dir, "final", agent, decoder, meta)
    print(f"Saved final checkpoint to: {run_dir / 'ckpt_final.pt'}")


if __name__ == "__main__":
    main()