#!/usr/bin/env python3
"""
build_submission.py  -  package predict.py + fitted parameters into submission.zip

    python build_submission.py                       # all ten systems
    python build_submission.py hospital_queue market # only those

For each system it creates  submission/<system>/predict.py  (a copy of this
folder's predict.py) and  submission/<system>/model.json  (copied from
models/<system>.json if that exists, otherwise the built-in defaults are used).
Then it imports every folder's predict.py the way the runner does, runs one
4,000-step episode on a random schedule, and checks the output is complete
and finite. A folder that fails is left out of the zip and reported, so a
crash never reaches the portal.
"""
import importlib.util
import json
import math
import os
import shutil
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import predict as P

ALL = list(P.BOUNDS)


def smoke_test(folder, family):
    spec = importlib.util.spec_from_file_location(f"predict_{family}", os.path.join(folder, "predict.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    context = {"family": family, "observables": list(P.TEST_INITIAL[family]),
               "intervention_bounds": {k: list(v) for k, v in P.BOUNDS[family].items()},
               "brief": "", "documents": [], "protocol": "", "revision": ""}
    sched = P.random_schedule(family, 4000, seed=7)
    out = mod.predict(P.TEST_INITIAL[family], sched, context)
    assert len(out) == 4000, f"returned {len(out)} rows, expected 4000"
    assert mod.predict(P.TEST_INITIAL[family], sched, None) == out, "folder-name family fallback broken"
    for row in out:
        for k in P.TEST_INITIAL[family]:
            v = row[k]
            assert isinstance(v, (int, float)) and math.isfinite(v), f"bad value for {k}: {v!r}"
    return True


def main():
    systems = sys.argv[1:] or ALL
    bad = [s for s in systems if s not in ALL]
    if bad:
        sys.exit(f"unknown system(s): {bad}. Choose from {ALL}")
    root = os.path.join(HERE, "submission")
    shutil.rmtree(root, ignore_errors=True)
    included = []
    for fam in systems:
        folder = os.path.join(root, fam)
        os.makedirs(folder)
        shutil.copy(os.path.join(HERE, "predict.py"), os.path.join(folder, "predict.py"))
        src = os.path.join(HERE, "models", f"{fam}.json")
        if os.path.exists(src) and "params" not in json.load(open(src)):
            print(f"FAIL  {fam:18s} models/{fam}.json is not from our fit.py (kit format?) -> left out of the zip")
            shutil.rmtree(folder)
            continue
        if os.path.exists(src):
            shutil.copy(src, os.path.join(folder, "model.json"))
            note = "fitted model.json"
        else:
            note = "built-in defaults (no models/%s.json)" % fam
        try:
            smoke_test(folder, fam)
            included.append(fam)
            print(f"ok    {fam:18s} {note}")
        except Exception as e:
            print(f"FAIL  {fam:18s} {e}  -> left out of the zip")
            shutil.rmtree(folder)

    zpath = os.path.join(HERE, "submission.zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for fam in included:
            for name in ("predict.py", "model.json"):
                path = os.path.join(root, fam, name)
                if os.path.exists(path):
                    z.write(path, arcname=f"{fam}/{name}")
    size = os.path.getsize(zpath) / 1024
    print(f"\nwrote submission.zip ({size:.0f} KiB) with {len(included)} system(s): {included}")
    if len(included) < len(ALL):
        print(f"missing (will score 0 unless already submitted this phase): "
              f"{[s for s in ALL if s not in included]}")


if __name__ == "__main__":
    main()
