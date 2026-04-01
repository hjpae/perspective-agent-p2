#!/usr/bin/env python3
"""
Patch train_phase2.py to support --from_scratch training.
Run from project root: python patch_from_scratch.py
"""
from pathlib import Path

target = Path("train_phase2.py")
if not target.exists():
    print(f"ERROR: {target} not found"); exit(1)

content = target.read_text()

if "from_scratch" in content:
    print("Already patched!"); exit(0)

changes = 0

# ── PATCH 1: Make --phase1_ckpt not required, add --from_scratch ──
old1 = '    ap.add_argument("--phase1_ckpt", type=str, required=True)'
new1 = '''    ap.add_argument("--phase1_ckpt", type=str, default="")
    ap.add_argument("--from_scratch", action="store_true",
                    help="Train from random init. No Phase 1 checkpoint needed.")'''
if old1 in content:
    content = content.replace(old1, new1, 1)
    changes += 1
    print("  [1] Made --phase1_ckpt optional, added --from_scratch")
else:
    print("  [1] SKIP: could not find --phase1_ckpt line")

# ── PATCH 2: Add init_from_scratch function before load_phase1_checkpoint ──
old2 = 'def load_phase1_checkpoint(args: argparse.Namespace):'
new2 = '''def init_from_scratch(args: argparse.Namespace):
    """Initialize agent and decoder from random weights (no Phase 1 pretrain)."""
    device = str(args.device)

    agent_cfg = AgentConfig(device=device)
    agent_cfg.encoder.obs_dim = int(args.obs_dim)
    agent_cfg.world.update_mode = str(args.update_mode)
    agent_cfg.world.alpha_fixed = float(args.alpha_fixed)
    agent_cfg.world.alpha_min = float(args.alpha_min)
    agent_cfg.world.alpha_max = float(args.alpha_max)
    agent_cfg.world.energy_mode = str(args.energy_mode)
    agent_cfg.world.dyn_eta = float(args.dyn_eta)
    agent_cfg.world.confine_lambda = float(args.confine_lambda)
    agent_cfg.world.n_prototypes = int(args.n_prototypes)
    agent_cfg.world.use_error_feedback = True
    agent_cfg.world.err_dim = len(ERR_FEATURE_NAMES)

    agent = CEARAgent(agent_cfg)
    agent.to(device)

    dec_cfg = DecoderConfig(g_dim=agent_cfg.world.g_dim, obs_dim=int(args.obs_dim))
    decoder = ObsDecoder(dec_cfg)
    decoder.to(device)

    meta = {
        "from_scratch": True,
        "agent_cfg": {
            "encoder": agent_cfg.encoder.__dict__,
            "world": agent_cfg.world.__dict__,
            "state": agent_cfg.state.__dict__,
            "policy": agent_cfg.policy.__dict__,
        },
        "decoder_cfg": dec_cfg.__dict__,
        "args": vars(args),
    }

    print("[init_from_scratch] initialized agent + decoder from random weights.")
    return agent, decoder, meta


def load_phase1_checkpoint(args: argparse.Namespace):'''
if old2 in content:
    content = content.replace(old2, new2, 1)
    changes += 1
    print("  [2] Added init_from_scratch function")
else:
    print("  [2] SKIP: could not find load_phase1_checkpoint definition")

# ── PATCH 3: Branch in main() ──
old3 = '    agent, decoder, meta = load_phase1_checkpoint(args)'
new3 = '''    if args.from_scratch:
        agent, decoder, meta = init_from_scratch(args)
    else:
        if not args.phase1_ckpt:
            raise ValueError("Either --from_scratch or --phase1_ckpt is required")
        agent, decoder, meta = load_phase1_checkpoint(args)'''
if old3 in content:
    content = content.replace(old3, new3, 1)
    changes += 1
    print("  [3] Added from_scratch branch in main()")
else:
    print("  [3] SKIP: could not find load_phase1_checkpoint call in main()")

# ── PATCH 4: Fix meta reference for from_scratch (env_cfg) ──
# resolve_env_from_phase1_meta needs to handle from_scratch meta
# It should work since we include "args" in meta, but the function
# reads from args directly, not from meta. So it should be fine.

# ── PATCH 5: Fix ckpt save to handle from_scratch ──
old5 = '        "phase1_ckpt": str(Path(args.phase1_ckpt).resolve()),'
new5 = '        "phase1_ckpt": str(Path(args.phase1_ckpt).resolve()) if args.phase1_ckpt else "from_scratch",'
if old5 in content:
    content = content.replace(old5, new5, 1)
    changes += 1
    print("  [5] Fixed checkpoint save for from_scratch")

target.write_text(content)
print(f"\nDone! Applied {changes} patches to {target}")
