#!/usr/bin/env python3
"""
compare_zips.py  -  prove two ZIPs give the same forecasts, system by system. Free.

    python compare_zips.py submission.zip uploads/2026-09-26_G_six.zip market
    python compare_zips.py submission.zip uploads/2026-09-25_E_all_ten.zip

Runs each system folder of both ZIPs the way the scorer does (its own predict.py and
model.json) on every paid run in research/ and on three random 4,000-step schedules, and
reports the largest absolute difference per system. "same" means every value is exactly
equal, so a system that was proven as a public upload in one ZIP is proven in the other.
Only systems present in both ZIPs are compared.
"""
import importlib.util
import os
import sys
import tempfile
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import predict as P
from score_zip import load_runs


def load(folder, tag):
    spec = importlib.util.spec_from_file_location(f"predict_{tag}", os.path.join(folder, "predict.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    zips = [a for a in sys.argv[1:] if a.endswith(".zip")]
    only = [a for a in sys.argv[1:] if not a.endswith(".zip")]
    if len(zips) != 2:
        sys.exit(__doc__)
    tmp = tempfile.mkdtemp()
    for zi, zpath in enumerate(zips):
        zipfile.ZipFile(zpath).extractall(os.path.join(tmp, str(zi)))
    fams = [f for f in P.BOUNDS if all(os.path.exists(os.path.join(tmp, str(zi), f, "predict.py")) for zi in (0, 1))]
    for fam in [f for f in fams if not only or f in only]:
        a = load(os.path.join(tmp, "0", fam), f"a_{fam}")
        b = load(os.path.join(tmp, "1", fam), f"b_{fam}")
        names = list(P.TEST_INITIAL[fam])
        ctx = {"family": fam, "observables": names, "intervention_bounds": {}, "brief": "",
               "documents": [], "protocol": "", "revision": ""}
        cases = [(init, acts) for runs in load_runs(fam).values() for init, acts, _ in runs]
        cases += [(P.TEST_INITIAL[fam], P.random_schedule(fam, 4000, seed=s)) for s in (1, 2, 3)]
        worst = 0.0
        for init, acts in cases:
            for ra, rb in zip(a.predict(init, acts, ctx), b.predict(init, acts, ctx)):
                worst = max(worst, max(abs(ra[k] - rb[k]) for k in names))
        print(f"{'same ' if worst == 0 else 'DIFF '} {fam:18s} max |diff| {worst:g} over {len(cases)} runs")


if __name__ == "__main__":
    main()
