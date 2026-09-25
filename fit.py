#!/usr/bin/env python3
"""
fit.py  -  fit one system's parameters to your research log.

    python fit.py research/hospital_queue.json --out models/hospital_queue.json
    python fit.py research/hospital_queue.json --holdout 1      # keep last run for testing
    python fit.py research/power_grid.json --free L0,f0         # also fit sentinel params
    python fit.py research/market.json --generic                # use the generic linear-lag model

Research log format (collect_runs.py writes this; you can also write it by hand):
{
  "family": "hospital_queue",
  "runs": [
    {"initial": {"wait_time": 12.0, "queue": 40.0, "discharges": 8.0},
     "actions":      [ {"staffing": 20.0, ...}, ... ],
     "observations": [ {"wait_time": ..., "queue": ..., "discharges": ...}, ... ]}
  ]
}
One run = one reset() followed by N step() calls. actions[i] produced observations[i].

What it does
- Starts from DEFAULTS[family] in predict.py.
- Minimises (model - observed) / sigma over every step of every training run,
  where sigma is each observable's spread in your data. That mirrors the
  competition's per-observable scaling.
- Uses a soft-L1 loss so a few noisy points can't drag the fit.
- Freezes parameters whose current value is exactly 0.0 (the "take from
  initial" sentinels and switched-off effects) unless you --free them.
- Prints a proxy score: mean of 1/(1+|err|/sigma), the competition formula
  with your own sigma estimate. Not the real score, but it moves the same way.
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import predict as P


def load_log(path):
    with open(path) as f:
        log = json.load(f)
    runs = []
    for r in log["runs"]:
        n = min(len(r["actions"]), len(r["observations"]))
        if n > 0:
            runs.append({"initial": r["initial"], "actions": r["actions"][:n],
                         "observations": r["observations"][:n]})
    return log["family"], runs


def obs_names(runs):
    return list(runs[0]["initial"].keys())


def sigma_per_obs(runs, names):
    sig = {}
    for n in names:
        vals = np.array([o[n] for r in runs for o in r["observations"]], dtype=float)
        s = float(vals.std()) if len(vals) > 1 else 0.0
        sig[n] = s if s > 1e-9 else max(abs(float(vals.mean())) * 0.1, 1e-6)
    return sig


def generic_defaults(runs, names):
    p = {}
    action_names = list(runs[0]["actions"][0].keys())
    for n in names:
        vals = [o[n] for r in runs for o in r["observations"]]
        p["k__" + n] = 0.05
        p["b__" + n] = float(np.mean(vals))
        for a in action_names:
            p["w__" + n + "__" + a] = 0.0
    return p


def residuals(x, keys, fixed, family, runs, names, sig):
    p = dict(fixed)
    p.update(zip(keys, x))
    res = []
    for r in runs:
        sim = P.simulate(family, r["initial"], r["actions"], p)
        for row, obs in zip(sim, r["observations"]):
            for n in names:
                res.append((row[n] - float(obs[n])) / sig[n])
    return np.asarray(res)


def proxy_score(p, family, runs, names, sig):
    """Competition formula with your own sigma. Higher is better, max 1."""
    if not runs:
        return float("nan")
    tot, cnt = 0.0, 0
    for r in runs:
        sim = P.simulate(family, r["initial"], r["actions"], p)
        for row, obs in zip(sim, r["observations"]):
            for n in names:
                tot += 1.0 / (1.0 + abs(row[n] - float(obs[n])) / sig[n])
                cnt += 1
    return tot / cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--out", default=None, help="where to write model.json (default models/<family>.json)")
    ap.add_argument("--holdout", type=int, default=0, help="number of runs (from the end) kept for testing")
    ap.add_argument("--free", default="", help="comma-separated params to fit even though they are 0.0")
    ap.add_argument("--generic", action="store_true", help="fit the generic linear-lag model instead")
    ap.add_argument("--start", default=None, help="existing model.json to start from instead of defaults")
    ap.add_argument("--max-nfev", type=int, default=3000)
    args = ap.parse_args()

    family, runs = load_log(args.log)
    model_family = "generic" if args.generic else family
    if model_family not in P.ADVANCE:
        sys.exit(f"unknown family {family!r}; use --generic")
    names = obs_names(runs)
    train = runs[:len(runs) - args.holdout] if args.holdout else runs
    hold = runs[len(runs) - args.holdout:] if args.holdout else []
    sig = sigma_per_obs(runs, names)

    if model_family == "generic":
        p0 = generic_defaults(train, names)
    else:
        p0 = dict(P.DEFAULTS[family])
    if args.start:
        p0.update(P.load_params(model_family, args.start))

    free = set(k for k in args.free.split(",") if k)
    keys = [k for k, v in p0.items() if v != 0.0 or k in free or k.startswith(("b__", "w__"))]
    fixed = {k: v for k, v in p0.items() if k not in keys}
    x0 = np.array([p0[k] for k in keys], dtype=float)
    lower = np.array([-np.inf if k.startswith(("b__", "w__")) else 0.0 for k in keys])
    upper = np.full(len(keys), np.inf)
    x0 = np.maximum(x0, lower)

    print(f"family={family} model={model_family} runs={len(runs)} train={len(train)} holdout={len(hold)}")
    print(f"steps in training data: {sum(len(r['actions']) for r in train)}")
    print(f"sigma estimates: { {k: round(v, 4) for k, v in sig.items()} }")
    if fixed:
        print(f"frozen (value 0.0): {sorted(fixed)}")
    before_tr = proxy_score(p0, model_family, train, names, sig)
    before_ho = proxy_score(p0, model_family, hold, names, sig)
    print(f"proxy score before: train={before_tr:.4f} holdout={before_ho:.4f}")

    sol = least_squares(residuals, x0, args=(keys, fixed, model_family, train, names, sig),
                        bounds=(lower, upper), loss="soft_l1", f_scale=1.0,
                        x_scale="jac", max_nfev=args.max_nfev)
    p = dict(fixed)
    p.update(zip(keys, sol.x))
    after_tr = proxy_score(p, model_family, train, names, sig)
    after_ho = proxy_score(p, model_family, hold, names, sig)
    print(f"proxy score after:  train={after_tr:.4f} holdout={after_ho:.4f}   ({sol.message})")

    print("fitted parameters:")
    for k in keys:
        print(f"  {k:24s} {p0[k]:12.5g} -> {p[k]:12.5g}")

    out = args.out or os.path.join("models", f"{family}.json")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump({"family": model_family, "params": p, "sigma_estimate": sig,
                   "proxy_score_train": after_tr, "proxy_score_holdout": after_ho}, f, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
