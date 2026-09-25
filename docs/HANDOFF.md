# Handoff from the claude.ai session (Sep 24, 2026)

Everything from the planning session that isn't a hard rule. `CLAUDE.md` holds the rules and
current status; this file holds the reasoning. Official rules: `kit/PROMPT.md`, `kit/briefs.md`,
`kit/FAQ.md`, `kit/README.md`.

## What happened in that session

1. Chris shared the problem-statement PDF and portal. We broke the challenge down from first
   principles, several times, in simpler terms each time. He prefers plain explanations and visuals.
2. Mapped each of the ten systems to the textbook model it resembles (table below).
3. Built one `predict.py` covering all ten systems, plus `fit.py`, `collect_runs.py`,
   `build_submission.py`. Tested fitting on synthetic data.
4. Chris shared the participant kit and credentials. Verified bounds/reference actions against the
   official briefs (all match), fixed `collect_runs.py` to the kit's real API, added
   `check_setup.py`, `gateway.py`, `setup.ps1`, and packaged the project for
   `C:\Users\ooich\groundtruth-hackathon`.
5. Chris asked for things claude.ai couldn't do: create the folder on his machine, run a subagent
   workflow ("ultracode"), use `/karpathy-guidelines`. That's why the work moved to Claude Code.

## The problem in one paragraph

Each system is a hidden program with knobs (controls, bounded) and dials (observables, noisy).
We get 2,000 paid steps per system to poke it. Then our code gets a starting reading plus 4,000
future knob settings and must predict every dial at every step, with no peeking. The systems have
memory (past actions change future responses) and two hidden extra rules each. So the job is:
build a small simulator that behaves like theirs, from very little data.

## Strategy (first principles)

- **All ten on the board first.** Missing = 0 in a 10-way average. Ten mediocre beats seven great.
- **Structure beats black boxes here.** 2,000 noisy steps can't train a big model, but they can
  pin down 10 to 20 numbers in a textbook-shaped model. Synthetic check: structured hospital model
  scored 0.95 on holdout vs 0.72 for a generic linear-lag fit on the same 200 steps.
- **Settle points dominate.** 4,000 steps mostly sit near wherever the system settles under held
  controls. Getting steady states and time constants right earns most of the score.
- **Spend steps where the test lives.** Recovery scenarios use 70 to 100% of the way from the
  brief's recovery action to its pulse action. The brief's "compare X vs Y" sentences describe
  the order/recovery/composition test episodes. Design experiments around them.
- **The scoring curve** `1/(1+e/sigma)` gives 0.5 at 1 sigma and 0.25 at 3 sigma. Consistent small
  drift over 4,000 steps costs more than one bad spike.
- **Hidden state resets to a fixed convention** and only observables are randomized, so research
  runs are directly comparable to test episodes, and `START[fam]` can rebuild hidden state the same way.
- **Latest upload counts, not best.** Only upload a model that beats the current one on holdout.

## Standard experiment plan (per system, about 290 steps)

| Run | Plan | Steps | What it tells you |
|---|---|---|---|
| A | `hold:recovery --steps 60` | 60 | Calm baseline, settle time, noise level per observable |
| B | `pulse --pre 20 --dur 20 --post 40` | 80 | One recovery-history episode in miniature; mechanism signatures |
| C | `hold:mid --steps 60` | 60 | A loaded operating point; base for one-knob tests |
| D | `step:KNOB=VAL --pre 30 --steps 90` | 90 | Holdout run for `fit.py --holdout 1` |

After fitting, look at where residuals are worst, then spend on: one-knob steps from `mid`, the
brief's paired comparisons (same totals, different order or spacing), and repeated pulses with
short vs long rest. Keep about 20% of each budget to validate before finals.

Reading run B: does the output keep moving after the pulse ends (memory)? Is recovery slower than
run A's drain (fatigue-like)? Does a second bump appear later (delayed returns)? Does a response
ramp up instead of jumping (handover/orientation-like)?

## Per-system notes

Mechanism candidates are named in the brief only for traffic, supply chain and hospital queue.
For the rest, the brief lists memory effects; exactly two of three are active in every system.
"v1" = what `predict.py` does now. "Gaps" = what the brief describes that v1 ignores.

**epidemic**: SIR with a hospital compartment. v1: single age group, school/mask scale contact,
vaccination moves S to R. Gaps: three age groups, clinic workforce shared by vaccination and
hospital pressure, bed waiting list, behavior/immunity/postponed-gathering memory. Brief compares:
closure vs masking at similar case counts; vaccination before vs after a restriction pulse.

**market**: mean-reverting price, volume, depth. v1: each observable lags toward a target shifted
by rate and tax. Gaps: producer/consumer groups, working cash, order pipeline (new policy doesn't
cancel commitments), dealer books and settlement, risk capacity, momentum-chasing. Memory
candidates: funding tied up until settlement, risk capacity cut by adverse moves, exposure
shifting toward recent winners.

**traffic**: two approach queues into a shared junction, exit buffers that spill back. v1: that,
with toll/ramp/lane/freight/clearance scaling capacity and demand. Gaps: light vs heavy classes,
diversion, reported-speed definition. Named mechanisms: route learning, crew fatigue/switching
costs, persistent spillback fronts. Brief compares: toll vs ramp prep with similar totals;
clearance vs signal reversal after stopping arrivals.

**power_grid**: price-responsive load with rebound, reserves with state of charge, governor droop,
interconnector heating. Gaps: thermostat population detail, heterogeneous reserves, curtailment.
Every reset starts from the same asynchronous load population at price 0.8.

**supply_chain**: production to supplier stock to transit (delay) to receiving to retail. v1 adds
machine health from wear/maintenance. Gaps: two goods classes, rework loop, shared utilities.
Named mechanisms: congested transport and rework, machine heat/wear, adaptive production
commitments. Brief compares: maintenance vs idle pause at fixed mix; rush before vs after
dispatch; equal orders in opposite production sequences.

**wildlife**: Lotka-Volterra with logistic prey, two regions, corridor migration. Gaps: habitat
patches, juveniles, animals in transit, settlement competition. Brief compares: habitat recovery
with corridors closed vs open; harvest before vs after protection.

**reservoir**: water balance plus algae, screen fouling, quality lag. Gaps: stratified layers,
**seasonal inflow** (tick = 1 day, so 4,000 ticks is about 11 years of seasons; v1 has none),
delayed groundwater/contaminant returns, aeration remobilizing deep material.

**ad_auction**: saturating win rate in bid, budget pacing, fulfilment-limited conversions,
audience fatigue. Memory candidates: rival capital moving between audiences, repeated exposure
removing reachable people, broad introduction changing later follow-up.

**social_contagion**: Bass-style diffusion in two communities linked by bridge outreach, capped
onboarding, churn with delayed reconsideration. Memory candidates: credibility, incentive
expectations, cross-community relationships. Brief compares: local vs bridge campaigns; incentive
before vs after recruitment; recovery with outreach stopped.

**hospital_queue**: queue served by effective staff. v1: handover lag on staffing, fatigue from
overtime, return pool when follow-up is low, overflow cap, `wait = queue / capacity`. Gaps:
`urgent_priority` ignored, case types, assessment vs treatment stages, finite chairs/beds. Named
mechanisms: fatigue, handover, returning case mix. Brief compares: equal staff-hours with
different overtime spacing; diagnostic allocation at fixed staffing; follow-up after the same
discharge burst.

## Testing without spending steps

The kit's `Client` accepts `transport=`, and `gateway.open_client(transport=...)` passes it
through. In the claude.ai session a fake gateway was built with `httpx.MockTransport` serving
`/reset`, `/step`, `/budget/<sys>`, `/brief/<sys>`, `/documents/<sys>`, backed by our own models
with changed numbers plus noise. It verified the whole pipeline and the budget/bounds guards.
That harness wasn't shipped; rebuild it in `tests/` if needed. Never point tests at the real gateway.

## Gemma

Optional research helper. Key from Google AI Studio in `GEMMA_API_KEY` or `secrets/gemma_key.txt`.
Models: `gemma-4-26b-a4b-it`, or `gemma-4-31b-it` if enabled. Earns no points; useful for turning
a brief plus measurements into competing hypotheses and a separating experiment.

## Subagents (Chris wanted this)

The systems are independent, so one agent per system is the natural split: each owns
`research/<system>.json`, `models/<system>.json` and its family block in `predict.py`. Watch out:
`predict.py` is one shared file. Either have agents propose diffs to their own block only and
merge centrally, or split the families into per-system modules that `build_submission.py` copies
into each folder. Rule 1 in `CLAUDE.md` applies to every agent: no steps without Chris's OK.

## Open questions

- What do `documents(system)` contain? Answered Sep 24, see Findings in `CLAUDE.md`.
- Which two mechanisms are active in each system? Unknown until experiments.
- What sigma does the organizer use per observable? Hidden; `fit.py` uses our data's std as a proxy.