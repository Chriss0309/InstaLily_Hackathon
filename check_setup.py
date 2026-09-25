#!/usr/bin/env python3
"""
check_setup.py  -  confirm everything works. Spends ZERO simulator steps.

    python check_setup.py            # environment + model self-test + gateway (free reads)
    python check_setup.py --gemma    # also send Gemma one tiny test prompt

What it checks
1. Python 3.12 and the exact package versions the scoring sandbox uses.
2. predict.py runs 4,000 steps for all ten systems and stays finite.
3. Gateway (free calls only): budget for every system, and saves each brief
   and its documents to docs/. Then compares the live bounds and observable
   names with the ones hard-coded in predict.py and flags any difference.
"""
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import predict as P
import gateway

SANDBOX = {"numpy": "2.3.5", "scipy": "1.16.3", "sklearn": "1.7.2", "joblib": "1.5.2"}


def check_environment():
    ok = True
    v = sys.version_info
    if v[:2] != (3, 12):
        print(f"WARN  Python {v.major}.{v.minor}; scoring runs 3.12")
        ok = False
    for mod, want in SANDBOX.items():
        try:
            have = __import__(mod).__version__
        except ImportError:
            print(f"WARN  {mod} not installed (sandbox has {want})")
            ok = False
            continue
        if have != want:
            print(f"WARN  {mod} {have}, sandbox has {want}")
            ok = False
    print(("ok    " if ok else "WARN  ") + f"python {v.major}.{v.minor}.{v.micro} and packages")
    return ok


def check_models():
    ok = True
    for fam in P.BOUNDS:
        t0 = time.time()
        out = P.simulate(fam, P.TEST_INITIAL[fam], P.random_schedule(fam, 4000), P.DEFAULTS[fam])
        finite = len(out) == 4000 and all(math.isfinite(x) for row in out for x in row.values())
        ok &= finite
        print(("ok    " if finite else "FAIL  ") + f"model {fam:18s} 4000 steps in {time.time() - t0:.2f}s")
    return ok


def check_gateway():
    url, _ = gateway.credentials()
    if not url:
        print("SKIP  gateway: no credentials. Copy your credentials JSON into secrets/.")
        return None
    docs = os.path.join(HERE, "docs")
    os.makedirs(docs, exist_ok=True)
    ok = True
    with gateway.open_client() as sim:
        for fam in P.BOUNDS:
            budget = sim.budget(fam)
            brief = sim.brief(fam)
            documents = sim.documents(fam)
            with open(os.path.join(docs, f"{fam}_brief.json"), "w") as f:
                json.dump(brief, f, indent=2)
            with open(os.path.join(docs, f"{fam}_documents.json"), "w") as f:
                json.dump(documents, f, indent=2)
            live_bounds = {k: tuple(float(x) for x in v) for k, v in brief.get("interventions", {}).items()}
            live_obs = list(brief.get("observables", []))
            flags = []
            if live_bounds != P.BOUNDS[fam]:
                flags.append(f"bounds differ: live={live_bounds}")
            if live_obs != list(P.TEST_INITIAL[fam]):
                flags.append(f"observables differ: live={live_obs}")
            n_docs = len(documents) if isinstance(documents, list) else len(documents.get("documents", documents))
            ok &= not flags
            print(("ok    " if not flags else "FLAG  ") +
                  f"gateway {fam:18s} steps left={budget.get('simulator_steps_remaining', budget.get('remaining'))} "
                  f"documents={n_docs}" + ("".join("\n      " + x for x in flags)))
    print(f"      briefs and documents saved to {docs}")
    return ok


def check_gemma():
    key = gateway.gemma_key()
    if not key:
        print("SKIP  gemma: set GEMMA_API_KEY or put the key in secrets/gemma_key.txt")
        return None
    from google import genai
    with genai.Client(api_key=key) as g:
        reply = g.models.generate_content(model="gemma-4-26b-a4b-it", contents="Reply with the single word OK.")
    print(f"ok    gemma replied: {reply.text.strip()[:40]!r}")
    return True


if __name__ == "__main__":
    results = [check_environment(), check_models(), check_gateway()]
    if "--gemma" in sys.argv:
        results.append(check_gemma())
    print("\nall good" if all(r is not False for r in results) else "\nsee WARN/FAIL/FLAG lines above")
