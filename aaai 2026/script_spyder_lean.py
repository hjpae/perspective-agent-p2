#%% initial training 
## zone0 volatile, zone2 stable
import sys
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

from cear_pilot.training.train import main

# if __name__ == "__main__":
#     sys.argv = [
#       str(Path(__file__).name),
#       "--device","cpu",
#       "--steps","48000",
#       "--seed","4",
     
#       "--w_entropy","0.001",
#       "--w_actor","0.25",
#       "--actor_b","0.98",
      
#       # "--use_slip",
#       # "--p_slip","0.60","0.30","0.0",

#       # "--mirror_x", 
#       # "--mirror_actions",

#       # "--view",
#       # "--view_every", "2",
#       # "--view_fps", "20",
#       # "--view_cell_px", "42",
#     ]
#     main()

BASE_ARGS = [
    "--device","cpu",
    "--steps","48000",

    "--w_entropy","0.001",
    "--w_actor","0.25",
    "--actor_b","0.98",
    
    "--log_traj",
    "--log_every","1",
]

if __name__ == "__main__":
    script_name = str(Path(__file__).name)

    for seed in [1, 2, 3, 4, 5]:
        sys.argv = [script_name] + BASE_ARGS + ["--seed", str(seed)]
        print(f"\n===== Running seed={seed} =====")
        main()

#%%
# script_switch_sweep_eval_spyder.py
# -*- coding: utf-8 -*-

from pathlib import Path
import os, sys, subprocess, time

# -----------------------
# 0) Spyder-safe setup
# -----------------------
PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

RUNS_DIR = PROJECT_ROOT / "outputs" / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)


def newest_run_dir() -> Path:
    dirs = [p for p in RUNS_DIR.iterdir() if p.is_dir()]
    if not dirs:
        raise FileNotFoundError("No run dirs found")
    return max(dirs, key=lambda p: p.stat().st_mtime)


def run_module(module: str, args: list[str]):
    cmd = [sys.executable, "-m", module] + args
    print("\nRunning:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def safe_sleep():
    time.sleep(0.5)

# -----------------------
# 1) Checkpoint
# -----------------------
TRAIN_ID = "seed5"   # <-- change if needed
CKPT = PROJECT_ROOT / "outputs" / "runs" / TRAIN_ID / "ckpt.pt"
assert CKPT.exists(), f"Missing ckpt: {CKPT}"

# -----------------------
# 2) Experiment settings
# -----------------------
T_TOTAL = 550
WARMUP  = 150
PERIODS = [20, 40, 80]

SIGMA_A = (0.60, 0.30, 0.05)
SIGMA_B = (0.05, 0.30, 0.60)

DEVICE = "cpu"
SEED   = "0"
GREEDY = True

# figure params
PRE_WINDOW = 80
ALPHA = 0.05
CONSEC = 3
L = 60
POLICY_SIGNAL = "entropy"

# -----------------------
# 3) Run sweep + figures
# -----------------------
for P in PERIODS:
    print("\n" + "=" * 80)
    print(f"=== period = {P} ===")

    before = set(p.name for p in RUNS_DIR.iterdir() if p.is_dir())

    args_collect = [
        "--ckpt", str(CKPT),
        "--device", DEVICE,
        "--seed", SEED,
        "--T", str(T_TOTAL),
        "--warmup", str(WARMUP),
        "--period", str(P),
        "--sigma_A", str(SIGMA_A[0]), str(SIGMA_A[1]), str(SIGMA_A[2]),
        "--sigma_B", str(SIGMA_B[0]), str(SIGMA_B[1]), str(SIGMA_B[2]),
        "--max_steps", str(T_TOTAL),
    ]
    if GREEDY:
        args_collect.append("--greedy")
    
    # 1) Collect
    run_module("cear_pilot.experiments.run_switch_sweep", args_collect)
    safe_sleep()
    
    # 2) Detect run_dir
    after = [p for p in RUNS_DIR.iterdir() if p.is_dir() and p.name not in before]
    run_dir = max(after, key=lambda p: p.stat().st_mtime) if after else newest_run_dir()
    
    # 3) Make figure
    run_module("cear_pilot.analysis.figure_switch_eval", [
        "--run_dir", str(run_dir),
        "--warmup", str(WARMUP),
        "--pre_window", str(PRE_WINDOW),
        "--alpha", str(ALPHA),
        "--consec", str(CONSEC),
        "--L", str(L),
        "--policy_signal", POLICY_SIGNAL,
    ])
    
    # 4) Console lag table (summary)
    # -----------------------
    print("\n" + "=" * 80)
    print("FINAL LAG SUMMARY TABLE")
    
    args_table = [
        "--root_dir", str(RUNS_DIR),
        "--periods", *[str(p) for p in PERIODS],
        "--warmup", str(WARMUP),
        "--L", str(L),
        "--signed_g",
    ]
    run_module("cear_pilot.analysis.print_switch_lag_table", args_table)

    print(f"[OK] figure saved:")
    print(run_dir / "figs" / f"fig_switch_eval_{POLICY_SIGNAL}.png")


print("\nALL DONE.")

#%% pygame gif (testing runs)
# -----------------------
# PYGAME ROLLOUT (Spyder)
# -----------------------
from pathlib import Path
import os, sys
import numpy as np
import torch

from cear_pilot.envs.nzone_grid import NZoneConfig, NZoneGridEnv
from cear_pilot.models.agent import CEARAgent, AgentConfig
from cear_pilot.models.decoder import ObsDecoder, DecoderConfig
from cear_pilot.training.pygame_viewer import PygameGridViewer


def _onehot(idx: int, n: int) -> np.ndarray:
    v = np.zeros((n,), dtype=np.float32)
    v[int(idx)] = 1.0
    return v


def _build_from_ckpt(ckpt_path: Path, device: str = "cpu", max_steps: int = 400):
    ckpt = torch.load(ckpt_path, map_location=device)
    meta = ckpt["meta"]

    # Env
    env_cfg = NZoneConfig(**meta["env_cfg"])
    env_cfg.max_steps = int(max_steps)
    env = NZoneGridEnv(config=env_cfg)

    # Agent
    agent_cfg = AgentConfig(device=device)
    agent_cfg.encoder.__dict__.update(meta["agent_cfg"]["encoder"])
    agent_cfg.world.__dict__.update(meta["agent_cfg"]["world"])
    agent_cfg.state.__dict__.update(meta["agent_cfg"]["state"])
    agent_cfg.policy.__dict__.update(meta["agent_cfg"]["policy"])
    agent = CEARAgent(agent_cfg)

    # Decoder (not required for viewing, but ckpt has it)
    dec_cfg = DecoderConfig(**meta["decoder_cfg"])
    decoder = ObsDecoder(dec_cfg)

    agent.load_state_dict(ckpt["agent_state"])
    decoder.load_state_dict(ckpt["decoder_state"])

    agent.to(device).eval()
    decoder.to(device).eval()
    return agent, decoder, env


def _entropy_from_logits(logits: np.ndarray) -> float:
    ex = np.exp(logits - np.max(logits))
    p = ex / (np.sum(ex) + 1e-12)
    return float(-np.sum(p * np.log(p + 1e-12)))


def run_pygame_rollout(
    ckpt_path: str,
    T: int = 400,
    greedy: bool = True,
    device: str = "cpu",
    seed: int = 0,
    fps: int = 12,
    cell_px: int = 40,
    sigma: tuple[float, float, float] | None = None,
    n_episodes: int = 10,          # <-- add
    episode_sleep_sec: float = 0.4 # <-- add (optional)
):
    ckpt_path = Path(ckpt_path)
    assert ckpt_path.exists(), f"Missing ckpt: {ckpt_path}"

    rng = np.random.default_rng(seed)
    agent, decoder, env = _build_from_ckpt(ckpt_path, device=device, max_steps=T)

    if sigma is not None:
        env._zone_sigma = np.array(list(sigma), dtype=np.float32)

    viewer = PygameGridViewer(
        width=env.cfg.width,
        height=env.cfg.height,
        cell_px=cell_px,
        fps=fps,
        title=f"Testing runs | {ckpt_path.parent.name}",
    )

    try:
        for ep in range(int(n_episodes)):
            # New episode seed each time
            ep_seed = int(rng.integers(0, 1_000_000))
            obs, info = env.reset(seed=ep_seed)
            agent.reset(batch_size=1)
            last_action = 4

            for t_global in range(int(T)):
                x_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                p_t = torch.tensor(_onehot(last_action, env.action_space.n), dtype=torch.float32, device=device).unsqueeze(0)

                with torch.no_grad():
                    action, out = agent.step(x_t, p_t, greedy=bool(greedy), ablate_g=False)

                a_int = int(action.item())
                logits = out["logits"].squeeze(0).detach().cpu().numpy()
                g_vec = out["g"].squeeze(0).detach().cpu().numpy()

                obs, _, terminated, truncated, info2 = env.step(a_int)

                ent = _entropy_from_logits(logits)
                g_norm = float(np.linalg.norm(g_vec))

                ok = viewer.draw(
                    env=env,
                    step=t_global,
                    episode=ep,           # <-- this will show episode count in HUD (viewer already supports it)
                    last_action=a_int,
                    loss=0.0,
                    loss_pred=0.0,
                    loss_smooth=0.0,
                    entropy=ent,
                    g_norm=g_norm,
                )
                if ok is False:
                    return

                last_action = a_int
                if terminated or truncated:
                    break

            # tiny pause between episodes (optional)
            if episode_sleep_sec > 0:
                import time
                time.sleep(float(episode_sleep_sec))

    finally:
        viewer.close()


# -----------------------
# CALL IT (edit this)
# -----------------------
CKPT = str((Path(__file__).resolve().parent / "outputs" / "runs" / "20260128_010139" / "ckpt.pt").resolve())

run_pygame_rollout(
    ckpt_path=CKPT,
    T=60,
    greedy=False,
    device="cpu",
    n_episodes=10,
    seed=2,
    fps=24,
    cell_px=40,
    #sigma=(0.60, 0.30, 0.05),  # optional: force regime A
    #sigma=(0.05, 0.30, 0.60),  # optional: force regime B
)
