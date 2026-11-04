"""The server's PPO controller (server/ppo_follow.py) must choose exactly what the trained policy chose in the simulator:
same observation code (follow_obs.py), same network (models/follow_ppo.onnx, exported from the SB3 model), same rules.
Drives sim/env.py with the SB3 model and replays every cycle's inputs into PPOFollow.decide.

    python3 tests/sim/ppo_server_parity_test.py      (needs models/follow_ppo.onnx and eval/rl/ppo_follow_budget.zip)
"""
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, os.path.join(ROOT, "sim"))
sys.path.insert(0, os.path.join(ROOT, "server"))
onnx, sb3 = os.path.join(ROOT, "models", "follow_ppo.onnx"), os.path.join(ROOT, "eval", "rl", "ppo_follow_budget.zip")
if not (os.path.exists(onnx) and os.path.exists(sb3)):
    print("skipped: no trained PPO policy on this machine (per-install artifact)")
    sys.exit(0)
import env as E  # noqa: E402
from ppo_follow import PPOFollow  # noqa: E402
from sb3_contrib import RecurrentPPO  # noqa: E402

model = RecurrentPPO.load(sb3, device="cpu")
P = PPOFollow.__new__(PPOFollow)
import onnxruntime as ort  # noqa: E402
P.sess = ort.InferenceSession(onnx, providers=["CPUExecutionProvider"])
bad = n = 0
for seed in (31, 32):
    e = E.FollowEnv(episode_s=1800)
    obs, _ = e.reset(seed=seed)
    P.reset()
    st, start, done = None, np.ones((1,), bool), False
    while not done:
        a, st = model.predict(obs, state=st, episode_start=start, deterministic=True)
        start = np.zeros((1,), bool)
        mine = P.decide(obs)
        bad += list(map(int, a)) != list(mine)
        n += 1
        obs, r, term, trunc, info = e.step(a)
        done = term or trunc
assert bad == 0, f"{bad} of {n} decisions differ"
print(f"ppo server parity ok: {n} decisions, the server controller chose exactly what the trained policy chose")
