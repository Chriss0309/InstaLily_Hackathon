#!/usr/bin/env python3
"""
collect_runs.py  -  run one planned experiment and append it to research/<family>.json

    python collect_runs.py hospital_queue hold:recovery --steps 60
    python collect_runs.py hospital_queue pulse --pre 20 --dur 20 --post 40
    python collect_runs.py hospital_queue hold:mid --steps 60
    python collect_runs.py hospital_queue step:overtime=1.0 --pre 30 --steps 90
    python collect_runs.py hospital_queue --dry-run pulse        # print the plan, spend nothing

Plans
    hold:recovery   hold the brief's reference recovery action for --steps
    hold:pulse      hold the brief's reference pulse action for --steps
    hold:mid        hold the midpoint between recovery and pulse for --steps
    pulse           recovery for --pre, pulse for --dur, recovery for --post
    step:NAME=VAL   hold mid for --pre, then set NAME=VAL (others stay at mid) until --steps
    file:PATH#KEY   run the segments in PATH[KEY], a JSON list of [steps, {control: value}];
                    unset controls stay at the recovery action. KEY is SYSTEM or SYSTEM.variant

Credentials come from gateway.py (env vars, or the JSON in secrets/).

Every call to step() costs one credit. The script refuses to start if the
plan needs more steps than you have left, prints the budget before and
after, and writes the log after every step so no paid observation is lost.

The log stores the brief too, so the kit's own fit.py can read it.
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import predict as P


def build_plan(family, plan, steps, pre, dur, post):
    ref = P.REFERENCE[family]
    rec, pul = ref["recovery"], ref["pulse"]
    mid = {k: 0.5 * (rec[k] + pul[k]) for k in rec}
    if plan == "hold:recovery":
        return [dict(rec)] * steps
    if plan == "hold:pulse":
        return [dict(pul)] * steps
    if plan == "hold:mid":
        return [dict(mid)] * steps
    if plan == "pulse":
        return [dict(rec)] * pre + [dict(pul)] * dur + [dict(rec)] * post
    if plan.startswith("step:"):
        name, val = plan[5:].split("=")
        if name not in P.BOUNDS[family]:
            sys.exit(f"{name!r} is not a control of {family}; options: {list(P.BOUNDS[family])}")
        lo, hi = P.BOUNDS[family][name]
        val = float(val)
        if not lo <= val <= hi:
            sys.exit(f"{name}={val} is outside bounds [{lo}, {hi}]")
        changed = dict(mid); changed[name] = val
        return [dict(mid)] * pre + [changed] * max(steps - pre, 0)
    if plan.startswith("file:"):
        path, key = plan[5:].rsplit("#", 1)
        if key.split(".")[0] != family:
            sys.exit(f"schedule {key!r} is not for {family}")
        out = []
        for n, over in json.load(open(path))[key]:
            for name, v in over.items():
                if name not in P.BOUNDS[family]:
                    sys.exit(f"{key}: {name!r} is not a control of {family}")
                lo, hi = P.BOUNDS[family][name]
                if not lo <= v <= hi:
                    sys.exit(f"{key}: {name}={v} is outside [{lo}, {hi}]")
            out += [{**rec, **over}] * n
        return out
    sys.exit(f"unknown plan {plan!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("family", choices=list(P.BOUNDS))
    ap.add_argument("plan")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--pre", type=int, default=20)
    ap.add_argument("--dur", type=int, default=20)
    ap.add_argument("--post", type=int, default=40)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    actions = build_plan(args.family, args.plan, args.steps, args.pre, args.dur, args.post)
    print(f"{args.family}: plan={args.plan} -> {len(actions)} steps ({len(actions)} credits)")
    if args.dry_run:
        for i, a in enumerate(actions):
            if i == 0 or a != actions[i - 1]:
                print(f"  from step {i + 1:4d}: {a}")
        return

    from gateway import open_client
    out_path = args.out or os.path.join(HERE, "research", f"{args.family}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    log = {"family": args.family, "runs": []}
    if os.path.exists(out_path):
        with open(out_path) as f:
            log = json.load(f)

    with open_client() as sim:
        remaining = sim.budget(args.family)["remaining"]
        if len(actions) > remaining:
            sys.exit(f"plan needs {len(actions)} steps but only {remaining} remain; nothing spent")
        print("budget before:", remaining)
        log["brief"] = sim.brief(args.family)
        run = sim.reset(args.family)
        run_id, initial = run["run_id"], run["observation"]
        observations = []
        for a in actions:
            observations.append(sim.step(run_id, a)["observation"])
            partial = {"plan": args.plan, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                       "initial": initial, "actions": actions[:len(observations)],
                       "observations": observations}
            with open(out_path + ".tmp", "w") as f:
                json.dump({**log, "runs": log["runs"] + [partial]}, f)
            os.replace(out_path + ".tmp", out_path)   # atomic swap: an interrupt never truncates the log
        print("budget after: ", sim.budget(args.family)["remaining"])

    print(f"saved run #{len(log['runs']) + 1} ({len(observations)} steps) to {out_path}")
    print("first obs:", initial)
    print("last obs: ", observations[-1])


if __name__ == "__main__":
    main()
