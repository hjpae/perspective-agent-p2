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
from typing import Dict, Tuple, Iterable

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


def resolve_formation_profile(profile_name: str) -> tuple[bool, Tuple[int, ...], Tuple[str, ...]]:
    """
    Returns:
        use_encounter, encounter_columns, encounter_profiles

    Assumes the env rewrite will interpret encounter_profiles literally as:
        "misleading" or "supportive".
    """
    profile_name = str(profile_name).strip().lower()

    if profile_name == "no_encounter":
        return False, tuple(), tuple()

    # Symmetric timing baselines
    even_cols = (3, 9, 15, 21)
    early_cols = (3, 6, 9, 12)
    late_cols = (12, 15, 18, 21)

    if profile_name == "even_m4":
        return True, even_cols, ("misleading", "misleading", "misleading", "misleading")
    if profile_name == "even_s4":
        return True, even_cols, ("supportive", "supportive", "supportive", "supportive")

    if profile_name == "early_m4":
        return True, early_cols, ("misleading", "misleading", "misleading", "misleading")
    if profile_name == "early_s4":
        return True, early_cols, ("supportive", "supportive", "supportive", "supportive")

    if profile_name == "late_m4":
        return True, late_cols, ("misleading", "misleading", "misleading", "misleading")
    if profile_name == "late_s4":
        return True, late_cols, ("supportive", "supportive", "supportive", "supportive")

    raise ValueError(f"Unknown formation_profile: {profile_name}")


def load_phase1_checkpoint(
    ckpt_path: str,
    device: str,
    obs_dim_override: int | None = None,
    update_mode: str = "adaptive",
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

    # Phase 2 changes only update-law, while carrying the phase-1 backbone.
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


def build_evidence_pressure(pred_now: float, info: Dict) -> float:
    """
    Formation-stage evidence pressure.

    Keep this simple:
    - immediate prediction mismatch
    - encounter valence pressure
    - reliability / fragility / conflict / rupture-memory
    - rupture event
    """
    on_enc = float(info.get("on_encounter", 0))
    rupture = float(info.get("rupture", 0))
    reliability = float(info.get("reliability_estimate", 0.0))
    conflict_load = float(info.get("conflict_load", 0.0))
    fragility = float(info.get("fragility", 0.0))
    rupture_memory = float(info.get("rupture_memory", 0.0))
    encounter_outcome = int(info.get("encounter_outcome", 1))

    misleading_pressure = max(0.0, -reliability)
    supportive_relief = max(0.0, reliability)

    if encounter_outcome == 2:
        outcome_term = 1.00 * on_enc
    elif encounter_outcome == 0:
        outcome_term = -0.60 * on_enc
    else:
        outcome_term = 0.0

    err_value = (
        0.42 * pred_now
        + 0.16 * outcome_term
        + 0.12 * rupture
        + 0.10 * misleading_pressure
        + 0.08 * conflict_load
        + 0.06 * fragility
        + 0.06 * rupture_memory
        - 0.06 * supportive_relief
    )
    return float(max(0.0, err_value))


def freeze_module(module: torch.nn.Module) -> None:
    module.eval()
    for p in module.parameters():
        p.requires_grad = False


def unfreeze_module(module: torch.nn.Module) -> None:
    module.train()
    for p in module.parameters():
        p.requires_grad = True


def count_trainable(params: Iterable[torch.nn.Parameter]) -> int:
    return int(sum(p.numel() for p in params if p.requires_grad))


def grad_norm(params: Iterable[torch.nn.Parameter]) -> float:
    sq = 0.0
    for p in params:
        if p.grad is not None:
            g = p.grad.detach()
            sq += float(torch.sum(g * g).item())
    return float(np.sqrt(max(sq, 0.0)))


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--phase1_ckpt", type=str, required=True)
    ap.add_argument("--formation_profile", type=str, default="even_m4", choices=[
        "no_encounter",
        "even_m4", "even_s4",
        "early_m4", "early_s4",
        "late_m4", "late_s4",
    ])

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cpu")

    ap.add_argument("--steps", type=int, default=36000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--w_smooth", type=float, default=1.0)
    ap.add_argument("--w_entropy", type=float, default=0.01)

    ap.add_argument("--width", type=int, default=23)
    ap.add_argument("--height", type=int, default=7)
    ap.add_argument("--obs_dim", type=int, default=8)
    ap.add_argument("--max_steps", type=int, default=240)

    ap.add_argument("--phase2_sigma_left", type=float, default=0.50)
    ap.add_argument("--phase2_sigma_right", type=float, default=0.03)

    ap.add_argument("--update_mode", type=str, default="adaptive", choices=["fixed", "adaptive"])
    ap.add_argument("--alpha_fixed", type=float, default=0.10)
    ap.add_argument("--alpha_min", type=float, default=0.03)
    ap.add_argument("--alpha_max", type=float, default=0.30)

    ap.add_argument("--log_traj", action="store_true")
    ap.add_argument("--log_every", type=int, default=1)
    ap.add_argument("--save_ckpt_every", type=int, default=12000)
    ap.add_argument("--print_every", type=int, default=1200)

    ap.add_argument("--reset_g_every_episode", action="store_true")

    args = ap.parse_args()

    seed_everything(args.seed, deterministic=True)
    device = torch.device(args.device)

    use_encounter, encounter_columns, encounter_profiles = resolve_formation_profile(args.formation_profile)

    # Formation-stage env:
    # - no row stance manipulation
    # - no slip / drift-like confounds
    # - no extra observation corruption boosts
    # - no rupture spike confound
    env_cfg = NZonePhase2Config(
        phase="phase2",
        width=args.width,
        height=args.height,
        obs_dim=args.obs_dim,
        max_steps=args.max_steps,

        use_encounter=bool(use_encounter),
        encounter_columns=tuple(encounter_columns),
        encounter_profiles=tuple(encounter_profiles) if len(encounter_profiles) > 0 else (
            "ambiguous", "confirm", "perturb", "accumulate", "recovery"
        ),

        phase2_sigma_left=args.phase2_sigma_left,
        phase2_sigma_right=args.phase2_sigma_right,

        # Remove row-specific formation bias.
        row_sigma_offsets=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        row_exposure_mults=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),

        # Turn off mild ecology confounds for formation-stage clarity.
        use_slip=False,
        base_action_slip=0.0,
        misleading_action_slip_boost=0.0,
        supportive_action_slip_relief=0.0,

        use_obs_corruption=False,
        misleading_obs_sigma_boost=0.0,
        supportive_obs_sigma_relief=0.0,

        # Keep hidden-state dynamics, but make them cleaner.
        misleading_rupture_prob=0.0,
    )
    env = NZonePhase2Config(config=env_cfg)

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

    # Freeze the entire backbone.
    # Only alpha_net is trainable.
    freeze_module(agent)
    freeze_module(decoder)
    unfreeze_module(agent.world.alpha_net)

    trainable_params = list(agent.world.alpha_net.parameters())
    assert count_trainable(trainable_params) > 0, "alpha_net has no trainable params."
    opt = torch.optim.Adam(trainable_params, lr=args.lr)

    agent.enc.eval()
    agent.state.eval()
    agent.policy.eval()
    decoder.eval()
    agent.world.gru.eval()
    if hasattr(agent.world, "ln"):
        agent.world.ln.eval()
    agent.world.alpha_net.train()

    n_actions = int(env.action_space.n)

    run_dir = Path("outputs") / "runs" / timestamp_id()
    run_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "env_type": "phase2_formation_alpha_only",
        "formation_profile": str(args.formation_profile),
        "seed": args.seed,
        "steps": args.steps,
        "lr": args.lr,
        "device": args.device,
        "reset_g_every_episode": bool(args.reset_g_every_episode),
        "phase1_ckpt": str(Path(args.phase1_ckpt).resolve()),
        "phase1_meta": phase1_meta,
        "loss_weights": {
            "w_smooth": args.w_smooth,
            "w_entropy": args.w_entropy,
        },
        "env_cfg": asdict(env_cfg),
        "freeze_scheme": {
            "encoder": True,
            "state": True,
            "policy": True,
            "decoder": True,
            "world_gru": True,
            "world_ln": True,
            "world_alpha_net": False,
        },
        "agent_cfg": {
            "encoder": asdict(agent.cfg.encoder),
            "world": asdict(agent.cfg.world),
            "state": asdict(agent.cfg.state),
            "policy": asdict(agent.cfg.policy),
        },
        "decoder_cfg": asdict(decoder.cfg),
        "trainable_param_count": count_trainable(trainable_params),
    }
    save_meta(run_dir, meta)

    log_rows = []
    log_every = int(max(1, args.log_every))

    agent.reset(batch_size=1)
    last_action = 4
    g_prev = agent.get_latents()["g"].detach().clone()
    prev_err_t = torch.zeros((1, 1), dtype=torch.float32, device=device)

    pred_err_stats = EMAMeanVar(beta=0.97)

    episode = 0
    t_in_ep = 0
    t0 = time.time()

    act_hist = np.zeros(n_actions, dtype=np.int64)
    zone_hist = np.zeros(5, dtype=np.int64)
    enc_hist = 0
    mislead_hist = 0
    support_hist = 0
    neutral_hist = 0

    pi_prev = None
    maxpi_ema = None
    kl_ema = None
    logits_norm_ema = None
    ema_world = None
    ema_alpha = None
    ema_evidence = None
    ema_delta_g = None

    print(
        f"[phase2:init] profile={args.formation_profile} "
        f"trainable_params={count_trainable(trainable_params)} "
        f"carry_g={int(not args.reset_g_every_episode)} "
        f"update_mode={agent.cfg.world.update_mode} "
        f"alpha_range=({agent.cfg.world.alpha_min:.3f},{agent.cfg.world.alpha_max:.3f}) "
        f"phase1_ckpt={args.phase1_ckpt}"
    )

    try:
        for step in range(args.steps):
            x_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            p_t = make_proprio_from_last_action(last_action, n_actions, device=device)

            out = agent.forward_step(x_t, p_t, ablate_g=False, err_t=prev_err_t)
            g_t = out["g"]
            logits_act = out["logits"]
            alpha_t = out["alpha"]
            g_candidate = out["g_candidate"]
            s_t = out["s"]

            pi_act = torch.softmax(logits_act, dim=-1)

            # Frozen backbone policy: sampling only.
            a_t = agent.policy.sample_action(logits_act, greedy=False)
            a_int = int(a_t.item())

            obs_next, _, terminated, truncated, info2 = env.step(a_int)
            x_next = torch.tensor(obs_next, dtype=torch.float32, device=device).unsqueeze(0)

            # Frozen decoder, but gradient still flows back to g -> alpha_net.
            xhat_all = decoder.predict_all_actions(g_t)
            xhat_exp = torch.sum(pi_act.unsqueeze(-1) * xhat_all, dim=1)

            loss_pred = F.mse_loss(xhat_exp, x_next)
            loss_smooth = torch.mean((g_t - g_prev) ** 2)

            entropy = -(pi_act * torch.log(pi_act + 1e-9)).sum(dim=-1).mean()
            loss = loss_pred + args.w_smooth * loss_smooth - args.w_entropy * entropy

            per_a_err = torch.mean((xhat_all - x_next.unsqueeze(1)) ** 2, dim=-1).squeeze(0)
            e_chosen = per_a_err[a_int]

            with torch.no_grad():
                pred_now = float(e_chosen.detach().item())
                pred_ema, pred_std = pred_err_stats.update(pred_now)

                err_value = build_evidence_pressure(
                    pred_now=pred_now,
                    info=info2,
                )
                err_t = torch.tensor([[err_value]], dtype=torch.float32, device=device)

                delta_g = float(torch.linalg.vector_norm(g_t.detach() - g_prev.detach()).item())
                g_norm = float(torch.linalg.vector_norm(g_t.detach()).item())
                g_cand_norm = float(torch.linalg.vector_norm(g_candidate.detach()).item())

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

                ln = float(torch.mean(torch.abs(logits_act)).item())

                maxpi_ema = maxpi if (maxpi_ema is None) else (0.98 * maxpi_ema + 0.02 * maxpi)
                kl_ema = kl if (kl_ema is None) else (0.98 * kl_ema + 0.02 * kl)
                logits_norm_ema = ln if (logits_norm_ema is None) else (0.98 * logits_norm_ema + 0.02 * ln)

                ema_alpha = float(alpha_t.mean().item()) if (ema_alpha is None) else (
                    0.98 * ema_alpha + 0.02 * float(alpha_t.mean().item())
                )
                ema_evidence = err_value if (ema_evidence is None) else (0.98 * ema_evidence + 0.02 * err_value)
                ema_delta_g = delta_g if (ema_delta_g is None) else (0.98 * ema_delta_g + 0.02 * delta_g)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            gn = grad_norm(trainable_params)
            opt.step()

            act_hist[a_int] += 1
            z = info2.get("zone_id", -1)
            if isinstance(z, (int, np.integer)) and 0 <= int(z) <= 4:
                zone_hist[int(z)] += 1

            # Count actual encounter occupancy, not event pulses.
            enc_hist += int(bool(info2.get("on_encounter", False)))

            outcome = int(info2.get("encounter_outcome", 1))
            if outcome == 2:
                mislead_hist += 1
            elif outcome == 0:
                support_hist += 1
            else:
                neutral_hist += 1

            lw = float(loss.item())
            ema_world = lw if ema_world is None else 0.98 * ema_world + 0.02 * lw

            if args.log_traj and ((step % log_every) == 0):
                with torch.no_grad():
                    g_np = g_t.detach().squeeze(0).float().cpu().numpy()
                    s_np = s_t.detach().squeeze(0).float().cpu().numpy()
                    gc_np = g_candidate.detach().squeeze(0).float().cpu().numpy()

                row = {
                    "formation_profile": str(args.formation_profile),
                    "t_global": int(step),
                    "episode": int(episode),
                    "t_in_ep": int(t_in_ep),

                    "zone_id": int(info2.get("zone_id", -1)),
                    "x": int(info2.get("x", -1)),
                    "y": int(info2.get("y", -1)),

                    "action": int(a_int),
                    "action_name": action_name(a_int),

                    "loss": float(loss.item()),
                    "loss_pred": float(loss_pred.item()),
                    "loss_smooth": float(loss_smooth.item()),

                    "pred_now": float(pred_now),
                    "pred_ema": float(pred_ema),
                    "pred_std": float(pred_std),
                    "evidence_pressure": float(err_value),

                    "alpha": float(alpha_t.mean().item()),
                    "g_norm": float(g_norm),
                    "g_candidate_norm": float(g_cand_norm),
                    "delta_g": float(delta_g),
                    "grad_norm_alpha": float(gn),

                    "entropy": float(entropy.item()),
                    "maxpi": float(maxpi),
                    "kl_to_prev_pi": float(kl),
                    "logits_abs_mean": float(ln),

                    "current_sigma": float(info2.get("current_sigma", np.nan)),
                    "on_encounter": int(bool(info2.get("on_encounter", False))),
                    "encounter_idx": int(info2.get("encounter_idx", -1)),
                    "encounter_profile": str(info2.get("encounter_profile", "none")),
                    "encounter_outcome": int(info2.get("encounter_outcome", 1)),

                    "reliability_estimate": float(info2.get("reliability_estimate", np.nan)),
                    "fragility": float(info2.get("fragility", np.nan)),
                    "rupture_memory": float(info2.get("rupture_memory", np.nan)),
                    "conflict_load": float(info2.get("conflict_load", np.nan)),
                    "rupture": int(bool(info2.get("rupture", False))),
                }

                for i, gv in enumerate(g_np):
                    row[f"g_{i}"] = float(gv)
                for i, sv in enumerate(s_np):
                    row[f"s_{i}"] = float(sv)
                for i, cv in enumerate(gc_np):
                    row[f"g_candidate_{i}"] = float(cv)

                log_rows.append(row)

            if (step + 1) % args.print_every == 0:
                with torch.no_grad():
                    e_det = per_a_err.detach().float().cpu().numpy()
                    e_min, e_max, e_std = float(e_det.min()), float(e_det.max()), float(e_det.std())

                act_prob = (act_hist / max(act_hist.sum(), 1)).tolist()
                zone_prob = (zone_hist / max(zone_hist.sum(), 1)).tolist()

                print(
                    f"[{step+1:>7}/{args.steps}] "
                    f"profile={args.formation_profile} "
                    f"phase=formation_alpha_only "
                    f"world={lw:.4f} w_ema={float(ema_world):.4f} "
                    f"pred={float(loss_pred.item()):.4f} smooth={float(loss_smooth.item()):.4f} "
                    f"| alpha={float(alpha_t.mean().item()):.4f} a_ema={0.0 if ema_alpha is None else float(ema_alpha):.4f} "
                    f"ev={float(err_value):.4f} ev_ema={0.0 if ema_evidence is None else float(ema_evidence):.4f} "
                    f"dg={float(delta_g):.4f} dg_ema={0.0 if ema_delta_g is None else float(ema_delta_g):.4f} "
                    f"||g||={float(g_norm):.4f} ||gc||={float(g_cand_norm):.4f} "
                    f"gnorm={float(gn):.4f} "
                    f"| H={float(entropy.item()):.3f} maxpi={float(maxpi_ema):.3f} KL={float(kl_ema):.6f} "
                    f"logits|.|={float(logits_norm_ema):.3f} "
                    f"e[min,max,std]={e_min:.3f},{e_max:.3f},{e_std:.3f} "
                    f"pred_now/ema/std={float(pred_now):.3f}/{float(pred_ema):.3f}/{float(pred_std):.3f} "
                    f"zone={[round(x, 2) for x in zone_prob]} "
                    f"act={[round(x, 2) for x in act_prob]} "
                    f"enc_win={enc_hist} "
                    f"mis/sup/neu={mislead_hist}/{support_hist}/{neutral_hist} "
                    f"profile_now={str(info2.get('encounter_profile', 'none'))} "
                    f"outcome={int(info2.get('encounter_outcome', 1))} "
                    f"rel={float(info2.get('reliability_estimate', np.nan)):.2f} "
                    f"frag={float(info2.get('fragility', np.nan)):.2f} "
                    f"cload={float(info2.get('conflict_load', np.nan)):.2f} "
                    f"rmem={float(info2.get('rupture_memory', np.nan)):.2f} "
                    f"carry_g={int(not args.reset_g_every_episode)} "
                    f"(ep={episode}, t={t_in_ep}, x={int(info2.get('x', -1))}, y={int(info2.get('y', -1))}, {time.time() - t0:.1f}s)"
                )

                act_hist[:] = 0
                zone_hist[:] = 0
                enc_hist = 0
                mislead_hist = 0
                support_hist = 0
                neutral_hist = 0
                t0 = time.time()

            if args.save_ckpt_every > 0 and ((step + 1) % args.save_ckpt_every == 0):
                save_checkpoint(run_dir, f"step{step+1}", agent, decoder, meta)

            obs = obs_next
            last_action = a_int
            g_prev = g_t.detach().clone()
            prev_err_t = err_t.detach()
            t_in_ep += 1

            if truncated or terminated:
                obs, info = env.reset(seed=args.seed + episode + 1)

                if args.reset_g_every_episode:
                    agent.reset(batch_size=1)

                # Carry g by default across episodes.
                g_prev = agent.get_latents()["g"].detach().clone()
                prev_err_t = torch.zeros((1, 1), dtype=torch.float32, device=device)

                last_action = 4
                episode += 1
                t_in_ep = 0

    finally:
        if args.log_traj and len(log_rows) > 0:
            out_path = try_save_table(log_rows, run_dir / "train_traj")
            print(f"Saved training trajectory to: {out_path}")

    save_checkpoint(run_dir, "final", agent, decoder, meta)
    print(f"Saved final checkpoint to: {run_dir / 'ckpt_final.pt'}")


if __name__ == "__main__":
    main()