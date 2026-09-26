#!/usr/bin/env python3
"""
score_zip.py  -  score uploaded ZIPs on our paid research runs. Free, spends no steps.

    python score_zip.py uploads/2026-09-25_D_all_ten.zip
    python score_zip.py uploads/2026-09-25_D_all_ten.zip uploads/2026-09-25_E_all_ten.zip market epidemic

Runs each system folder's own predict.py and model.json the way the scorer does, on every
run in research/<system>.json and research/<system>_*.json, and prints a proxy score per log
file: mean of 1/(1+|error|/sigma), under two sigmas, shown as std/d1:
  std: each observable's spread over all our runs for that system (the same proxy as fit.py)
  d1:  5.6 x the spread of its one-step changes. Tracks the public scores of our uploads
       better (CLAUDE.md Findings, "The ruler"). Trust a change only if it wins on both.
Every ZIP is scored with the same sigmas, so the numbers compare versions.
Runs a version was not fitted on (e.g. a new round) are the honest comparison.
"""
import glob
import importlib.util
import json
import os
import statistics
import sys
import tempfile
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))


def load_runs(fam):
    logs = {}
    for path in sorted(glob.glob(os.path.join(HERE, "research", f"{fam}.json"))
                       + glob.glob(os.path.join(HERE, "research", f"{fam}_*.json"))):
        runs = []
        for r in json.load(open(path))["runs"]:
            n = min(len(r["actions"]), len(r["observations"]))
            if n:
                runs.append((r["initial"], r["actions"][:n], r["observations"][:n]))
        logs[os.path.basename(path)] = runs
    return logs


def main():
    zips = [a for a in sys.argv[1:] if a.endswith(".zip")]
    only = [a for a in sys.argv[1:] if not a.endswith(".zip")]
    if not zips:
        sys.exit(__doc__)
    tmp = tempfile.mkdtemp()
    for zi, zpath in enumerate(zips):
        zipfile.ZipFile(zpath).extractall(os.path.join(tmp, str(zi)))
    fams = sorted({d for zi in range(len(zips)) for d in os.listdir(os.path.join(tmp, str(zi)))
                   if os.path.isdir(os.path.join(tmp, str(zi), d))})
    for fam in [f for f in fams if not only or f in only]:
        logs = load_runs(fam)
        if not logs:
            print(f"== {fam}: no research logs")
            continue
        names = list(next(iter(logs.values()))[0][0])
        sig = {k: statistics.pstdev([o[k] for runs in logs.values() for _, _, obs in runs for o in obs])
               or 1e-6 for k in names}
        sig1 = {k: 5.6 * statistics.pstdev([obs[t][k] - obs[t - 1][k] for runs in logs.values()
                                            for _, _, obs in runs for t in range(1, len(obs))]) or 1e-6
                for k in names}
        print(f"== {fam}   (std/d1)")
        for zi, zpath in enumerate(zips):
            folder = os.path.join(tmp, str(zi), fam)
            if not os.path.exists(os.path.join(folder, "predict.py")):
                print(f"   {os.path.basename(zpath):32s} (not in this zip)")
                continue
            spec = importlib.util.spec_from_file_location(f"predict_{zi}_{fam}", os.path.join(folder, "predict.py"))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            ctx = {"family": fam, "observables": names, "intervention_bounds": {}, "brief": "",
                   "documents": [], "protocol": "", "revision": ""}
            cols, tot, tot1, cnt = [], 0.0, 0.0, 0
            for log, runs in logs.items():
                s, s1, c = 0.0, 0.0, 0
                for initial, actions, obs in runs:
                    for row, o in zip(mod.predict(initial, actions, ctx), obs):
                        s += sum(1 / (1 + abs(row[k] - o[k]) / sig[k]) for k in names)
                        s1 += sum(1 / (1 + abs(row[k] - o[k]) / sig1[k]) for k in names)
                        c += len(names)
                cols.append(f"{log.replace(fam, '').replace('.json', '') or 'base'} {s / c:.3f}/{s1 / c:.3f}")
                tot += s; tot1 += s1; cnt += c
            print(f"   {os.path.basename(zpath):32s} all {tot / cnt:.3f}/{tot1 / cnt:.3f} | " + " | ".join(cols))


if __name__ == "__main__":
    main()
