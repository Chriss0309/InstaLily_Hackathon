# Groundtruth hackathon (Instalily x Google DeepMind, Toronto 26)

Individual competition, Sep 22 to 30, 2026. Chris is the participant. Research ten black-box
simulators with a limited step budget, then submit a numerical `predict()` per system that
forecasts 4,000 steps ahead with no feedback. Background, reasoning and per-system notes live in
`docs/HANDOFF.md` (not auto-loaded). Read it before starting research or model work on a system.

## Hard rules (read these first)

1. **Never spend simulator steps without Chris's explicit OK in this session.** Every `step()` costs
   1 credit, 2,000 per system for the whole event, never refunded. Anything that calls `step()`
   costs: `collect_runs.py` (unless `--dry-run`), `kit/collect.py`, or direct client code.
   Free: `check_setup.py`, `--dry-run`, and `brief()`, `documents()`, `budget()`, `reset()`.
   Before asking, state the system, the plan, and the exact step count.
2. **Never open, print, copy or commit anything in `secrets/`.** Credentials come only through
   `gateway.py`. No keys in code, logs, model files, commits or the submission ZIP.
3. **Chris uploads manually** in the portal. Never claim something was uploaded.
4. **Every change to `predict.py` must pass** `python check_setup.py` and `python build_submission.py`
   before you call it done. A crash or timeout scores 0 for that system.
5. **Think before coding. Simplest change that works. Surgical diffs. Verify.** Use
   `/karpathy-guidelines` if it's installed. Don't add features nobody asked for.

## How to talk to Chris

- Direct, casual, plain words. No corporate language, no em dashes, no filler.
- He asked repeatedly for simple, first-principles explanations. Say what a thing is and why it
  matters before details. Visuals help.
- Flag anything new you discover (rule changes, API surprises, data that contradicts a model) and
  add it to the Findings section below.
- Windows + PowerShell machine. Show commands for PowerShell with the venv active.

## Competition contract (what the scorer does)

- Entry point per system folder: `predict(initial, interventions, context)`. `initial` = noisy obs
  dict before step 1. `interventions` = 4,000 action dicts. Return 4,000 dicts, every observable,
  finite floats, no `initial` row. Start state fresh from `initial` every call.
- `context` keys: `protocol`, `revision`, `family`, `observables`, `intervention_bounds`, `brief`,
  `documents`. Static; no hidden info.
- Sandbox: Python 3.12, numpy 2.3.5, scipy 1.16.3, scikit-learn 1.7.2, joblib 1.5.2, stdlib. No
  network, no Gemma, no simulator. 2 CPUs, 3 GiB RAM, 1,200 s for load + all 40 episodes.
- ZIP: system folders at the root (no enclosing folder), 30 MiB zipped, 300 MiB expanded. Load
  files relative to `__file__`. Shared helpers must be copied into every folder.
- Score per observable per step: `1 / (1 + |error| / sigma)` vs noiseless truth. Averaged over
  observables, steps, 40 episodes. Four equal categories: sustained operation, action order,
  recovery history, composition. Overall = mean of all 10 systems; missing or crashed = 0.
- Same physical parameters and the same 2-of-3 active mechanisms for everyone, in every phase.
  Reset puts hidden state at a fixed convention; only observables are randomized.
- Test schedules stay inside bounds. Recovery scenarios use 70 to 100% of the distance from the
  brief's reference recovery action to its pulse action, per control.
- Uploads: 3 accepted per system per Toronto day, shared by public and final. **Latest accepted
  counts, not best.** Public uploads do NOT carry into finals.
- Dates (America/Toronto): public daily sweep 23:30. **Finals open Sep 28 12:00. Close Sep 30 12:00.**

## Project layout

| Path | Purpose |
|---|---|
| `predict.py` | The submitted forecaster. One file for all ten systems, branches on `context["family"]`. |
| `fit.py` | Fits one system's params to `research/<system>.json`, writes `models/<system>.json`. |
| `collect_runs.py` | Planned experiments (`hold:recovery`, `hold:pulse`, `hold:mid`, `pulse`, `step:NAME=VAL`, `file:PATH#KEY`). COSTS STEPS. |
| `schedules/` | `file:` plans: JSON `{KEY: [[steps, {control: value}], ...]}`, unset controls at recovery. |
| `test_collect_runs.py` | Free check of the `file:` plan guards and the save path (fake client). |
| `build_submission.py` | Copies `predict.py` + `models/<system>.json` into `submission/<system>/`, smoke-tests, zips. |
| `score_zip.py` | Free: runs each folder of one or more uploaded ZIPs on every paid run, proxy score per log file. |
| `check_setup.py` | Env/version check, 4,000-step self-test for all ten, free gateway reads into `docs/`. |
| `gateway.py` | Credentials (env vars, then `secrets/*credentials*.json`); loads `kit/client.py` by file path. |
| `kit/` | Organizer kit + rule docs. Untouched. Its `predict.py`/`fit.py` are a different model format. |
| `research/` | Paid experiment logs. Irreplaceable. Never delete or overwrite. |
| `models/` `docs/` `secrets/` | Fitted params / saved briefs + documents / credentials (off limits). |

## predict.py conventions

- Each family has `DEFAULTS[fam]` (params), `START[fam](initial, p)` (rebuild hidden state),
  `ADVANCE[fam](state, action, p)` (one tick, returns obs dict). Equations live here; numbers
  come from `fit.py`.
- Param value `0.0` on a level-type param (`L0`, `p0`, `K_n`, ...) means "take it from `initial`".
  `fit.py` freezes 0.0 params unless passed `--free name`.
- `simulate()` replaces non-finite values with the previous step's value; `predict()` falls back
  to holding `initial` if a model raises. Keep both guards.
- Plain Python floats, imports limited to `json`, `math`, `os`. About 0.01 s per 4,000 steps.
- `BOUNDS`, `REFERENCE` (recovery/pulse actions) and the observable names in `TEST_INITIAL` match
  the official briefs for all 10 systems (re-checked Sep 24). `check_setup.py` re-checks bounds and
  names against live briefs, not `REFERENCE`.
- `fit.py` minimizes soft-L1 residuals scaled by each observable's std, and reports a proxy score
  (the competition formula with our own sigma). Use `--holdout 1` to score on the last run.

## Commands (PowerShell, venv active)

```powershell
python check_setup.py --gemma                     # free: versions, self-test, budgets, save briefs/docs, Gemma ping
python collect_runs.py SYSTEM --dry-run PLAN      # free preview of an experiment
python collect_runs.py SYSTEM PLAN [--steps N]    # COSTS STEPS, needs Chris's OK
python fit.py research\SYSTEM.json --holdout 1
python build_submission.py [SYSTEM ...]           # writes submission.zip
python score_zip.py uploads\A.zip uploads\B.zip [SYSTEM ...]   # free: compare versions on paid runs
```

## Status at handoff (Sep 24, end of claude.ai session)

- Built and tested end to end against a fake gateway only: setup, collect, fit, build.
- Sep 24 late: `.venv` works (Python 3.12.6, sandbox package versions), `check_setup.py` all good,
  2,000 steps left on all ten systems (none spent). `submission.zip` (all ten on defaults) built
  and verified Sep 24, rebuilt ~12:45 Toronto with the context fallback. Rebuilt ~16:35 Toronto
  with the fitted hospital_queue model (other nine still on defaults).
- Upload 1 (Public, Sep 24 Toronto, all ten on textbook defaults) scored: ad_auction 0.5318,
  epidemic 0.1360, hospital_queue 0.4579, market 0.2219, power_grid 0.5042, reservoir 0.3845,
  social_contagion 0.4510, supply_chain 0.5970, traffic 0.2976, wildlife 0.1890 (mean 0.377).
  These are the baselines to beat. Public scores run well below our proxy S1/S2 for the same
  models, so the official sigma is probably tighter than our std-based proxy.
- Round D (Sep 25, scratch `D/`): every system now ships a model fitted on the first-look data,
  chosen by held-out S1/S2 among candidates that passed a one-control-at-a-time 4,000-step
  robustness gate. Physics blocks for epidemic (SEIRS + bed cap, 0.87/0.91), market (price lag,
  0.83/0.90), traffic (queue model, 0.87/0.99), supply_chain (0.84/0.99), wildlife (prey + food,
  0.87/0.92), reservoir (seasonal + spillway, 0.87/0.90), ad_auction (0.92/0.95),
  social_contagion (0.35/0.97); power_grid is a params-only refit (0.68/0.80). The generic ridge
  model failed the gate everywhere (negative outputs, 12-30x single-control blow-ups). Zip saved
  as `uploads/2026-09-25_D_all_ten.zip`; keep a copy of every uploaded zip in `uploads/`.
- Steps spent (Chris approved): 500 on every system on Sep 24 (pulse 150/100/150 + hold:recovery
  100). Round C on Sep 25 (one control at a time, `schedules/c1.json`, 4,830 steps) went to
  separate files `research/<system>_c1.json`. Left: epidemic 1100, market 800, traffic 1260,
  power_grid 800, supply_chain 1150, wildlife 1020, reservoir 1140, ad_auction 1060,
  social_contagion 1080, hospital_queue 760. Copies in `research/backup/`.
  Run `python check_setup.py` first; it shows steps left per system.
- Round E (Sep 25 Toronto, scratch `E/`, xhigh effort): every system refit on first look + round C
  with structural fixes, chosen by leave-one-run-out and shipped only if it beat the round-D model
  on unseen round-C data. All ten shipped. Round-D model on unseen C data -> E held-out: epidemic
  0.81 -> 0.84, market 0.89 -> 0.91, traffic 0.81 -> 0.87, reservoir 0.87 -> 0.90, ad_auction
  0.83 -> 0.91, wildlife 0.81 -> 0.86, hospital 0.78 -> 0.92 (D had overtime backwards),
  supply_chain 0.47 -> 0.48 (the mid-action regime is still poorly predicted). power_grid now
  simulates 960 thermostatic loads: 40 episodes take 20-42 s locally (limit 1,200 s). Zip saved as
  `uploads/2026-09-25_E_all_ten.zip`.
- Upload E (public, Sep 25 Toronto) scored: ad_auction 0.8780, epidemic 0.6870, hospital_queue
  0.6981, market 0.6218, power_grid 0.7804, reservoir 0.8447, social_contagion 0.4971,
  supply_chain 0.7938, traffic 0.8103, wildlife 0.6560 (mean 0.727, Upload 1 was 0.377). All ten
  beat Upload 1, so nothing was reverted. Smallest gain: social_contagion (+0.05), now the weakest.
  E was fitted on Round C, so Round C is no longer a holdout for it; Round F is.

## Next steps

1. Done Sep 24: `check_setup.py` passed, documents read (see Findings).
2. Done Sep 24: `submission.zip` built with all ten on defaults, verified by scorer emulation and a
   rules audit. Chris uploads it in the Public tab.
3. Per system: ~200 steps of the standard plan (see handoff), fit, check residuals, upload.
4. Prioritize by expected gain. Keep ~20% of each budget for validating final models.
5. From Sep 28 12:00: Chris must upload finals explicitly in the Final tab.
6. Round F (planned Sep 25, approved by Chris Sep 25, run it locally): order and recovery-spacing tests,
   `schedules/f1.json`, 5,015 steps, every system keeps >= 400. Per system three runs from reset:
   `xy` = stress X then Y, `yx` = Y then X, `spacing` = the reference pulse repeated with a short
   then a longer rest (market and hospital fit only two gaps' worth). Save to
   `research/<system>_f1.json`. Before refitting, score the current models on these runs with
   `score_zip.py`: they were fitted without them, so that is the honest order/spacing check.

   | System | X vs Y (order runs) | Pulse x gaps | Steps | Left after |
   |---|---|---|---|---|
   | epidemic | vaccination vs school+mask (brief) | 3x40, rest 20 / 80 | 700 | 400 |
   | market | interest rate vs tax | 2x30, rest 15 | 400 | 400 |
   | traffic | loaded road, signal 0.15 vs 0.85 (brief: reversal) | 3x30, rest 15 / 45 | 435 | 825 |
   | power_grid | price 0 vs reserve 150 + charging 0 + interconnector 0.2 | 3x30, rest 10 / 40 | 400 | 400 |
   | supply_chain | orders 80, production 1.5 vs 0.5 (brief) | 3x30, rest 20 / 60 | 555 | 595 |
   | wildlife | hunting 7 vs habitat 0.1 (brief) | 3x30, rest 15 / 60 | 565 | 455 |
   | reservoir | release+irrigation vs deep withdrawal+no aeration | 3x40, rest 20 / 80 | 620 | 520 |
   | ad_auction | breadth 0.775 vs bid 5 + budget 100 (brief) | 3x30, rest 10 / 40 | 440 | 620 |
   | social_contagion | incentive 2 vs seeding 9 (brief) | 3x40, rest 10 / 40 | 540 | 540 |
   | hospital_queue | staff 5+overtime+no follow-up vs electives+diag 0.75+urgent | 3x15, rest 15 / 45 | 360 | 400 |

   ```powershell
   python collect_runs.py SYSTEM --dry-run "file:schedules/f1.json#SYSTEM.xy"
   foreach ($k in "xy","yx","spacing") { python collect_runs.py SYSTEM "file:schedules/f1.json#SYSTEM.$k" --out research\SYSTEM_f1.json }
   ```

## Findings (append new ones here)

- Kit API: `reset()` returns `{run_id, observation}`; `step()` returns `{observation}`; `budget()`
  has `remaining` and `simulator_steps_remaining`; `brief()` has `observables` (list) and
  `interventions` ({name: [lo, hi]}).
- Portal pages (guide, briefs, challenge, faq) matched the kit markdown word for word on Sep 24.
- `documents(system)` (read Sep 24): 3 per system, the same list already inside `brief()` and passed
  to `predict()` as `context["documents"]`. Two are boilerplate, identical for all ten. The third
  repeats the brief text word for word and adds one new thing: the range each initial observable is
  sampled from at reset. `initial` adds research noise on top, so the range is a free prior on it.
- `TEST_INITIAL` values sit outside those reset ranges for 9 of 10 systems (market depth 1000 vs
  80-120, power_grid load 500 vs 90-120). Only the self-tests use it; scoring never does.
- v1 `START` contradicts the documented start state in three systems: traffic seeds queues and exit
  buffers from the initial reading (docs: roads and crossings start empty), supply_chain puts
  `shipments * lead0` in transit (docs: buffers and conveyors start empty), ad_auction sets
  `pending = conversions` (docs: no pending purchases). Hospital services also start empty. Expect
  a start-up transient; the first ~10 steps of run A will show how big it is.
- Chris's machine clock is UTC+8, 12 h ahead of Toronto (EDT). Local time: the upload-slot day
  resets at noon, the 23:30 public sweep is 11:30 next morning, finals open Sep 29 00:00 and close
  Oct 1 00:00. Dates in this file are Toronto dates.
- Live portal re-read Sep 24 12:01 Toronto: matches the kit docs sentence for sentence. Upload
  mechanics: every system inside a ZIP uses one of its 3 daily slots; one out-of-slot system
  rejects the whole ZIP (nothing charged); acceptance happens before execution, so a crashing
  upload still uses the slot and scores 0; final receipts show acceptance only, so a crashing final
  silently replaces a good one. Build only changed systems (`build_submission.py SYSTEM ...`) and
  prove a ZIP as a public upload before using it as a final. Final standings come at the first
  sweep after close.
- Defaults at realistic starts are valid but far off: epidemic cases decay to ~0, wildlife goes
  extinct under the pulse action, hospital queue hits its 500 cap under pulse, supply_chain
  inventories hit 2000 or 0, reservoir level hits 1500 or 0, traffic speeds pin at vfree=60 vs a
  30-45 reset range, market just holds `initial` at zero controls. Compare these first against run A.
- `predict()` takes the family from `context["family"]` if `context` has `.get` (PROMPT/FAQ say
  "plain dictionary", README says "plain mapping"), else from the folder name, else generic.
  Added Sep 24; the `build_submission.py` smoke test asserts the folder fallback gives the same output.
- The sandbox also has numpy, scipy, scikit-learn, joblib. `predict.py`'s docstring line "which is
  all the sandbox allows" is wrong; json/math/os only is our own choice.
- Every accepted public upload gets its own score ("your own public result appears when its
  asynchronous evaluation finishes", `kit/PROMPT.md:134`): 40 hidden episodes, real sigma, no steps.
  Up to 3 per system per Toronto day, about 12 before finals open. Use Public freely to compare
  models; HANDOFF's "only upload if it beats holdout" applies to Final only.
- `collect_runs.py:104` reopens `research/<system>.json` with "w" (truncate) on every step. An
  interrupt mid-write can wipe every earlier paid run for that system. Fix (temp file + os.replace)
  before spending steps.
- `fit.py --generic` saves `"family": "generic"`, and `load_params` ignores a model.json whose
  family differs from `context["family"]`, so a generic fit silently ships textbook defaults.
- Textbook `START`s copy physical constants from the noisy first reading (wildlife K = 2x prey,
  social N = 5x adopters, grid L0 and f0, reservoir inflow0/q0/level_max, market p0/v0/d0). Physics
  are fixed per system, so fit these as constants (`fit.py --free`) unless data shows the first
  reading matters.
- Independent review Sep 24: the standard plan's 60-90 step runs are too short for 4,000-step
  scoring, and HANDOFF's "structure 0.95 vs generic 0.72" came from our own fake-gateway data, so
  it proves nothing about the real systems.
- `collect_runs.py` save fixed Sep 24 (temp file + `os.replace`), tested with a fake client.
  Real collection runs about 0.4 s per step.
- Observation noise (hospital_queue) is about 0.5% of the level, multiplicative: exact zeros stay
  zero. Research data and `initial` are nearly the truth.
- hospital_queue, first 500 steps: the first reading only matters for ~15 steps; two runs with
  different starts are identical after that. Recovery action: settles by step ~15 at queue 23,
  discharges 11.5/step, wait 0. Services start empty, so discharges are exactly 0 for steps 1-5,
  then ~15 at steps 8-10. Pulse: discharges drop to 0 at once for ~20 steps, then 1-3/step; queue
  +27/step (electives add ~1 arrival per unit) and caps near 330; wait climbs ~1.2/step to 117.
  Recovery: discharges back to ~11 within 10 steps but never above ~12.4, so the queue drains only
  ~2/step (330 -> 102 after 150 steps, not recovered). The textbook default has arrivals 6 (real
  11.5), no service delay, cap 500, and recovers in ~20 steps. MAE textbook vs do-nothing: wait
  14.7 vs 26.5, queue 84 vs 110, discharges 5.2 vs 3.6.
- First look, all nine others (Sep 24, analyst reports in scratch `B/<system>/`). All systems are
  near noise-free (0.03-0.5%) and deterministic. Held-out scores (S1 = fit 1-250, predict unseen
  recovery 251-400; S2 = fit run 1, predict run 2), textbook defaults vs best found:
  epidemic 0.47/0.49 vs SEIRS + bed cap 155 + waiting list 0.87/0.91 (oscillates, period ~155,
  never settles); market 0.48/0.47 vs textbook refit with p0/v0/d0 free 0.83/0.88; traffic
  0.40/0.38 vs generic 0.66/0.97 (recovery action = empty road: flows 0, speeds 48.9); power_grid
  0.57/0.71 vs generic 0.70/0.81 (undamped post-pulse oscillation); supply_chain 0.48/0.65 vs
  refit 0.66/0.75 (supplier caps ~362, orders don't drain it); wildlife 0.41/0.44 vs generic
  0.87/0.84 (textbook drives prey to 0 forever); reservoir 0.50/0.56 vs physics_v2 (seasonal
  inflow, spillway ~941, delivery cap) 0.87/0.90; ad_auction 0.58/0.57 vs structural 0.86/0.90;
  social_contagion 0.34/0.84 vs structural 0.35/0.97 (S1 hard for all: post-pulse crash).
- Every first-look run moved all controls together, so no per-control effect is identified yet
  (market refit coefficients cancel only on the joint pulse). All nine analysts asked for
  one-control-at-a-time runs from the recovery baseline (240-700 steps each). `collect_runs.py`
  can't express that yet: `step:` holds the MID action, not recovery.
- hospital_queue model contest (Sep 24, scratch `A/`): the "minimal" fluid queue won and is
  shipped (predict.py hospital block + `models/hospital_queue.json`). S1/S2: minimal 0.854/0.989,
  brief-faithful 0.756/0.985, do-nothing 0.585/0.855, textbook refit 0.583/0.920, textbook
  defaults 0.535/0.763, generic 0.460/0.939. Generic fails here because rise and recovery are not
  mirror images; no model type wins everywhere, so pick per system on held-out data.
  Unidentified: overtime's sign (fitted g = -0.42, against the brief's hint) and follow-up
  diversion vs fatigue. Decisive test, ~40 steps: from calm, staffing 15 alone; this model
  predicts the queue grows ~2.5/step, the fatigue story predicts no growth.
- Kit `fit.py` model.json has no `params` key; our `predict.py` would silently use defaults, so
  `build_submission.py` refuses it. Kit `predict.py` raises on family mismatch (crash = 0).
- Right-click compressing `submission/` adds an enclosing folder and breaks the layout. Use
  `build_submission.py`.
- Reservoir tick = 1 day, so 4,000 ticks is about 11 years. The brief mentions seasonal river
  supply; the v1 model has no seasonality.
- `.claude/settings.json` denies reading `secrets/` and asks before collect commands. That is a
  backstop only (Bash rules are prefix matches). Rules 1 and 2 above are what count.
- D was fitted on first-look data only, so the Round C runs are a holdout for it. D does not carry
  over from joint pulses to single controls. Proxy (std sigma), first-look vs Round C: supply_chain
  0.97 -> 0.47, social_contagion 0.97 -> 0.68, hospital_queue 0.94 -> 0.78, power_grid 0.79 ->
  0.70, epidemic / traffic / wildlife ~0.93 -> 0.81, ad_auction 0.94 -> 0.83, reservoir 0.91 ->
  0.87, market 0.91 -> 0.89. Composition is weak and order/spacing are untested.
- Upload 1's textbook code is not in the repo or `uploads/`. Only power_grid's textbook equations
  survive (its `DEFAULTS`; D only changed its model.json). Reverting any other system to Upload 1
  needs that zip from Chris's machine.
- What the models predict for Round F: D had no order effect at all for supply_chain; E has a
  small one (retail 0.3 sigma). Both D and E predict none for reservoir quality (the memory term
  treats all controls alike). If the data shows one, that structure is wrong.
- E on all paid runs (proxy, std sigma) is 0.89-0.95 for every system, yet public runs 0.50-0.88.
  The biggest gap is social_contagion (proxy 0.93, public 0.50): the test episodes hit behavior
  our runs never showed. That is what Round F is for.
- This cloud session cannot collect: the network policy blocks gt-gateway-wavddee32q-uc.a.run.app
  and no gateway credentials are set. Collection runs on Chris's machine.
- Round F on Sep 26 (Chris's machine): hospital_queue through supply_chain finished, then
  wildlife.xy crashed partway with WinError 5 on `os.replace` (the log file held open for a moment
  by an editor, indexer or antivirus). The partial run is real paid data: it may sit in
  `research/wildlife_f1.json` and/or a leftover `.tmp`. `collect_runs.py` now retries locked saves
  and writes a `_rescue_` file instead of crashing.
- The documents say each phase's 40 episodes are exactly 10 per category: sustained operation,
  intervention order, recovery spacing, joint intervention.


## More info please refer to the webpage: 
https://gt-portal-wavddee32q-uc.a.run.app/guide
https://gt-portal-wavddee32q-uc.a.run.app/briefs
https://gt-portal-wavddee32q-uc.a.run.app/challenge
https://gt-portal-wavddee32q-uc.a.run.app/faq