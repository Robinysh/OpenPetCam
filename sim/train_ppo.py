#!/usr/bin/env python3
"""The RL challenger: a recurrent PPO policy (sb3-contrib RecurrentPPO, small LSTM) trained on randomised rooms of
the calibrated simulator, judged on the SAME held-out rooms and ground-truth metrics as sim/train_policy.py.

    python3 sim/train_ppo.py [timesteps]     -> eval/rl/ppo_follow.zip + eval/rl/ppo_report.json
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import env as E  # noqa: E402
import train_policy as TP  # noqa: E402
from room import load_params  # noqa: E402

HOURS = TP.HOURS


def evaluate_ppo(model, bank):
    res = []
    for seed, free in bank:
        e = E.FollowEnv(episode_s=HOURS * 3600)
        obs, _ = e.reset(options={"room": (seed, free)})
        state, start, done = None, np.ones((1,), dtype=bool), False
        while not done:
            a, state = model.predict(obs, state=state, episode_start=start, deterministic=True)
            obs, r, term, trunc, info = e.step(a)
            start = np.zeros((1,), dtype=bool)
            done = term or trunc
        sc = E.episode_score(info, HOURS)
        res.append((sc["in_view"], sc["moves_per_h"], sc["fast_min_per_h"], sc["shielded"]))
    return res


def main(timesteps=600_000):
    from sb3_contrib import RecurrentPPO
    from stable_baselines3.common.env_util import make_vec_env
    P = load_params()
    venv = make_vec_env(lambda: E.FollowEnv(randomize=True), n_envs=8, seed=1)
    model = RecurrentPPO("MlpLstmPolicy", venv, n_steps=256, batch_size=512, learning_rate=3e-4, gamma=0.995,
                         ent_coef=0.01, policy_kwargs={"lstm_hidden_size": 64, "net_arch": [64]}, seed=1, verbose=0)
    t0 = time.time()
    done = 0
    while done < timesteps:
        model.learn(total_timesteps=min(100_000, timesteps - done), reset_num_timesteps=False)
        done += min(100_000, timesteps - done)
        print(f"{done} steps, {time.time() - t0:.0f} s", flush=True)
    model.save(os.path.join(ROOT, "eval", "rl", "ppo_follow"))
    test = TP.rooms(32, 90_000, P)
    res = evaluate_ppo(model, test)
    report = {"timesteps": timesteps, "train_s": round(time.time() - t0),
              "in_view": round(float(np.mean([r[0] for r in res])), 3),
              "robust": round(float(np.mean(sorted(r[0] for r in res)[:8])), 3),
              "moves_per_h": round(float(np.mean([r[1] for r in res])), 1),
              "fast_min_per_h": round(float(np.mean([r[2] for r in res])), 1),
              "shielded_per_room": round(float(np.mean([r[3] for r in res])), 1)}
    json.dump(report, open(os.path.join(ROOT, "eval", "rl", "ppo_report.json"), "w"), indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 600_000)
