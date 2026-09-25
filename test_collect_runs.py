"""Free check of collect_runs.py's file: plan and save path. No gateway, no credits.

    python test_collect_runs.py
"""
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import collect_runs
import gateway
import predict as P

tmp = tempfile.mkdtemp()
plan = os.path.join(tmp, "plan.json")
json.dump({"market": [[3, {}], [2, {"interest_rate": 0.1}]],
           "market.bad": [[2, {"interest_rate": 0.5}]],
           "market.typo": [[2, {"interest": 0.1}]]}, open(plan, "w"))

# valid schedule: unset controls stay at recovery
acts = collect_runs.build_plan("market", f"file:{plan}#market", 0, 0, 0, 0)
rec = P.REFERENCE["market"]["recovery"]
assert acts == [rec] * 3 + [{**rec, "interest_rate": 0.1}] * 2, acts

# every guard stops before any step is spent
for key, fam in (("market.bad", "market"), ("market.typo", "market"), ("market", "traffic")):
    try:
        collect_runs.build_plan(fam, f"file:{plan}#{key}", 0, 0, 0, 0)
        raise AssertionError(f"{key} on {fam} should have been refused")
    except SystemExit:
        pass


class Fake:  # stands in for the gateway client
    def __enter__(self): return self
    def __exit__(self, *a): pass
    def budget(self, fam): return {"remaining": 99}
    def brief(self, fam): return {}
    def reset(self, fam): return {"run_id": "r", "observation": {"price": 1.0}}
    def step(self, rid, a): return {"observation": {"price": 2.0}}


gateway.open_client = lambda: Fake()
out = os.path.join(tmp, "log.json")
for _ in range(2):
    sys.argv = ["collect_runs.py", "market", f"file:{plan}#market", "--out", out]
    collect_runs.main()
log = json.load(open(out))
assert [len(r["actions"]) for r in log["runs"]] == [5, 5]
assert not os.path.exists(out + ".tmp")
print("\ncollect_runs checks passed")
