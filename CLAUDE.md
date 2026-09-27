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
| `score_zip.py` | Free: runs each folder of one or more uploaded ZIPs on every paid run, proxy score per log file under both rulers (std/d1). |
| `compare_zips.py` | Free: proves two ZIPs give exactly the same forecasts per system (paid runs + random 4,000-step schedules). Use it before a final upload. |
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

- Round G (Sep 26 Toronto, workflow of 6 agents, scratch `G/`): each system refit on all paid data
  (first look + C + F), judged by one shared harness (frozen sigma, hold out one Round F variant,
  refit on the rest, rotate). Ship rule: held-out Round F mean >= E + 0.01, in-sample first-look
  and Round C each no more than 0.01 below E, 4,000-step crash/blow-up gate passes. Every claimed
  win was re-run independently with the agent's own fit code before merging. Merged into
  predict.py + models: hospital_queue (held-out F 0.795 -> 0.839), supply_chain (0.695 -> 0.789),
  wildlife (0.836 -> 0.874), traffic (0.745 -> 0.764), market (0.880 -> 0.899), and
  social_contagion (0.812 -> 0.862; the agent's best candidate refit with B's per-member organic
  growth bounded <= 0, which also cleared the Round C floor). The other four (epidemic,
  power_grid, reservoir, ad_auction) are byte-identical to E. Upload zip with only the six
  changed systems: `uploads/2026-09-26_G_six.zip` (not uploaded yet; public scores pending).

- Upload G (public, Sep 26 Toronto, `uploads/2026-09-26_G_six.zip`) scored: hospital_queue 0.7105
  (E 0.6981), market 0.6654 (0.6218), social_contagion 0.5114 (0.4971), supply_chain 0.8003
  (0.7938), traffic 0.8167 (0.8103), wildlife 0.6527 (0.6560). Mean of all ten 0.735 (E 0.727).
  Wildlife dropped, so the repo's wildlife block and params are back to E (byte-identical to E's
  output); no upload needed for that, the final zip will carry it. Gains were far smaller than on
  our held-out Round F runs.

- Round I (Sep 26 Toronto, 4 agents, scratch `G/I/`): refit to the Round H long holds, each
  re-verified independently before merging. Long-run fit (our ruler) current -> new: market 0.64 ->
  0.94 (a "dealer hold" that ties up funding during a slow fall, gated on interest rate
  0.035-0.085: a guess, since only the long run opens it), social_contagion 0.78 -> 0.98 (organic
  growth fills a finite relationship-led audience, A ~92 / B ~79 ceiling), wildlife 0.80 -> 0.95
  (food sets capacity, harvest saturates, predators die in transit; E structure replaced),
  hospital_queue 0.86 -> 0.94 (part of overtime fatigue only bites after overtime stops).
  Held-out Round F: market 0.899 -> 0.897, social 0.862 -> 0.864, wildlife 0.836 -> 0.861,
  hospital 0.839 -> 0.839. Upload zip: `uploads/2026-09-26_I_four.zip` (not uploaded yet).
  Revert sources if any drops: market / social / hospital from `uploads/2026-09-26_G_six.zip`,
  wildlife from `uploads/2026-09-25_E_all_ten.zip`.

- Upload I (public, Sep 26 Toronto, `uploads/2026-09-26_I_four.zip`) scored: wildlife 0.7485
  (E 0.6560), hospital_queue 0.7328 (G 0.7105), social_contagion 0.5144 (G 0.5114), market
  0.6215 (G 0.6654). Market dropped, so its `predict.py` block and `models/market.json` are back
  to G (byte-identical code); wildlife, hospital and social stay on I. Best public per system:
  ad_auction 0.878, epidemic 0.687, hospital 0.733, market 0.665, power_grid 0.780, reservoir
  0.845, social 0.514, supply_chain 0.800, traffic 0.817, wildlife 0.749 (mean 0.747).
  `uploads/2026-09-26_final_candidate_v1.zip` = all ten at their best public version (E:
  epidemic, power_grid, reservoir, ad_auction; G: market, supply_chain, traffic; I: wildlife,
  social, hospital). `compare_zips.py` proved every folder gives exactly the same forecasts as
  the public upload it came from. No public re-upload of market is needed: G's market score is
  on record and public does not carry into finals.

- Round K (Sep 26 Toronto, 6 agents on the Round J data, scratch `J/`, every win re-run
  independently with the shared harness `J/harness.py`). Ship rule: both rulers (std and d1);
  honest J = the structure fitted WITHOUT the J run must predict J better than the shipped model
  (+0.01 for a new structure); no older log more than 0.01 worse; 4,000-step gate and hold scan.
  Merged into predict.py + models (honest J gain std/d1; all-logs std/d1 before -> after):
  ad_auction B, purchases outside the core audience (breadth > 0.55) need 7x the fulfilment work
  (+0.021/+0.026; 0.912/0.891 -> 0.926/0.909). power_grid B, reserve cap ~70-96 set by
  interconnector + charging, slow + fast conventional generation with a ~55 floor, demand lags
  price (+0.024/+0.026; 0.879/0.866 -> 0.912/0.901). reservoir B, outlet capacity ~ sqrt(level),
  quality from surface + deep layers with flushing (+0.034/+0.030; 0.896/0.902 -> 0.918/0.934).
  supply_chain B, two goods classes sold separately on the shelf, class 2 waits 12 steps before
  the shared queue, e2 held >= 0.0046 (+0.054/+0.070; 0.869/0.769 -> 0.891/0.805). traffic A,
  same equations refit (+0.010/+0.004; 0.862/0.853 -> 0.874/0.871). social_contagion A, overload
  churn above ~340 members (A + B), cross-validated +0.004/+0.016 (0.923/0.766 -> 0.936/0.803).
  No change: epidemic, market, wildlife, hospital_queue. Upload zip: `uploads/2026-09-26_K_six.zip`
  (not uploaded yet). Revert source for any of the six: `uploads/2026-09-26_final_candidate_v1.zip`
  (the block and model.json as of commit 3079e7a).

- Upload K (public, Sep 27 Toronto, just after midnight, `uploads/2026-09-26_K_six.zip`) scored:
  social_contagion 0.5512 (I 0.5144), power_grid 0.8169 (E 0.7804), traffic 0.8305 (G 0.8167),
  supply_chain 0.8085 (G 0.8003), ad_auction 0.8811 (E 0.8780), reservoir 0.8275 (E 0.8447).
  Five of six up; reservoir dropped, so its block and params are back to E (byte-identical
  forecasts, checked with `compare_zips.py`). Best public per system: ad_auction 0.881 (K),
  epidemic 0.687 (E), hospital 0.733 (I), market 0.665 (G), power_grid 0.817 (K), reservoir
  0.845 (E), social 0.551 (K), supply_chain 0.809 (K), traffic 0.831 (K), wildlife 0.749 (I);
  mean 0.757 (was 0.747). Honest-J gains predicted the sign for 5 of 6 but not the size
  (supply +0.054 honest -> +0.008 public; social +0.016 CV -> +0.037 public).
  `uploads/2026-09-27_final_candidate_v2.zip` = all ten at their best public version (K: ad,
  power, social, supply, traffic; E: epidemic, reservoir; G: market; I: hospital, wildlife),
  every folder proven identical in forecasts to the upload it was scored from. This is what the
  repo builds now, and the default final.
- Reservoir V1P (Sep 27, scratch `J/reservoir_v1*`): E with only the outflow capacity switched
  to c0 + c1*sqrt(level) and the water params refit on all four logs (season length P held at
  E's 67.773; the fit's 67.82 scores the same on our data but drifts the inflow cycle ~3 steps
  by step 4,000). Quality is E's, byte for byte, so a public test isolates round K's water fix
  from its two-layer quality. All logs 0.8955/0.9019 -> 0.9055/0.9273, honest J +0.004/+0.013
  (quality unchanged by design; level on J 0.796 -> 0.843 d1). Test zip:
  `uploads/2026-09-27_L_reservoir_v1p.zip` (not uploaded yet). If it beats 0.8447, splice
  `J/reservoir_v1p` into the repo; if not, reservoir stays on E.
- Upload V1P (public, Sep 27 Toronto morning) scored reservoir 0.8613 (E 0.8447, K's B 0.8275):
  the water fix is real and K's two-layer quality was what lost. Spliced into the repo. Best
  public per system: ad_auction 0.881 (K), epidemic 0.687 (E), hospital 0.733 (I), market 0.665
  (G), power_grid 0.817 (K), reservoir 0.861 (V1P), social 0.551 (K), supply_chain 0.809 (K),
  traffic 0.831 (K), wildlife 0.749 (I); mean 0.758. `uploads/2026-09-27_final_candidate_v3.zip`
  = all ten at those versions, every folder proven identical in forecasts to the upload it was
  scored from. This is the default final.
- Reservoir V2 (Sep 27, scratch `J/reservoir_v2`): V1P plus fitted weights for release and
  irrigation in E's quality stress drive (fit 0.26 each; E had 1; rounds C and F: alone they
  leave quality unchanged). Quality params refit on all four logs. All logs 0.9055/0.9273 ->
  0.9127/0.9314; honest J +0.012/+0.006 (misses the +0.01 structural bar on d1 by 0.004); long
  holds move at most 0.3 d1-sigma, all toward the J data (70% hold quality 0.926 -> 0.940), none
  of round K's big no-aeration drops. Public test zip `uploads/2026-09-27_M_reservoir_v2.zip`;
  keep only if it beats V1P's 0.8613.
- Traffic B2 (Sep 27, scratch `J/traffic_L/B2`, agent + my re-check): speed equation only.
  Moving vehicles count 0.78 of a stopped one; speed recovers at 0.06/step while a route still
  has cars and 0.19/step once it is empty (data: speed stays low while the queue drains, then
  jumps); lane closure lowers route b's free speed (5.4 x lane, ~2x the empty-road data, but set
  by the loaded pulses). ct stays 0.5; route a's exit capacity X_a kept at K's 66.8 (the fit ran it
  to its 2,000 bound, but no log changes anywhere between 67 and 1,995). Honest L +0.013/+0.034;
  all logs 0.8614/0.8578 -> 0.8684/0.8724; worst older log _j1 -0.008/-0.004; ramp-alone speeds
  30.2/31.1 (data 31.5/30.6, K 27.6/28.2). Test zip `uploads/2026-09-27_M_traffic_b2.zip`; keep
  only if it beats K's 0.8305. The persistent speed drop after pulses (data 2.35 after a 100-step
  pulse, ~1 after J and after four 30-step pulses) and a journey-time speed are still unmodeled.
- Supply_chain two-intake test (Sep 27, scratch `J/supply_chain_L2/B`): K's fixed 12-step class-2
  delay replaced by two intakes (class 1 served first on the shared transport, class 2 on what is
  left, each with its own ceiling) and one shelf slope for both classes. Fails the ship rules
  (_f1 -0.016/-0.027: class 2 now reaches the shelf ~step 5 of a full pulse instead of ~41; all
  +0.001/+0.004) but predicts round L far better unseen (+0.021/+0.032) and bounds retail at the
  70% hold with low mix (369-431 vs K's 1,184-4,519). A public test decides which regime the hidden
  set weighs more: `uploads/2026-09-27_M_supply_twointake.zip`; keep only if it beats 0.8085.
  Unmodeled in both: ~17-25 steps after every pulse ends, shipments burst to 50-62/step for 5-13
  steps (150-650 goods onto the shelf; the brief's rework path?), and round L's faster flow with
  earlier supplier refill says production or congestion depends on product mix.
- Uploads M (public, Sep 27 Toronto): traffic B2 0.8377 (K 0.8305) and supply_chain two-intake
  0.8120 (K 0.8085); both spliced into the repo. Reservoir V2 not uploaded yet (reservoir 1 slot
  left on Sep 27). `uploads/2026-09-27_final_candidate_v4.zip` = all ten at their best public
  version (traffic B2, supply two-intake, reservoir V1P, ad / power / social K, epidemic E,
  market G, hospital / wildlife I), every folder proven identical in forecasts to its scored
  upload. Mean of best public 0.7594. This is the default final.

## Next steps

1. Done Sep 24: `check_setup.py` passed, documents read (see Findings).
2. Done Sep 24: `submission.zip` built with all ten on defaults, verified by scorer emulation and a
   rules audit. Chris uploads it in the Public tab.
3. Per system: ~200 steps of the standard plan (see handoff), fit, check residuals, upload.
4. Prioritize by expected gain. Keep ~20% of each budget for validating final models.
5. From Sep 28 12:00: Chris must upload finals explicitly in the Final tab.
6. Done Sep 26 (see Findings): Round F, order and recovery-spacing tests,
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

7. Done Sep 26 (approved and run by Chris, see Findings): Round H, one long hold from reset per weak system,
   `schedules/h1.json`, 2,050 steps: social_contagion recovery 500 (leaves 40), epidemic recovery
   400 (leaves 0), market 70% joint pulse 400 (leaves 0), wildlife 70% pulse 350 (leaves 13),
   hospital_queue 70% pulse 400 (leaves 0). Public uploads replace the "final check" reserve.
   `foreach ($s in ...) { python collect_runs.py $s "file:schedules/h1.json#$s.long" --out "research\${s}_h1.json" }`

8. Done Sep 26 (approved and run by Chris, see Findings): Round J, the five systems with steps left, one
   run each from reset: 250 steps at the 70% pulse, then 150 at recovery (`schedules/j1.json`,
   2,000 steps, validated free against the plan parser). Leaves traffic 425, ad_auction 220,
   supply_chain 195, reservoir 120, power_grid 0. Why: wildlife's +0.09 came from the Round H
   70% hold; supply_chain's 70% prediction (retail 852) has never been observed; the 150-step
   tail measures recovery after a long 70% stress, which every recovery-history episode has.
   `foreach ($s in "traffic","power_grid","supply_chain","reservoir","ad_auction") { python collect_runs.py $s "file:schedules/j1.json#$s.long" --out "research\${s}_j1.json" }`

9. Done Sep 27 (approved and run by Chris, see Findings): Round L, three runs aimed at the biggest guesses
   the Round K hold scans found (`schedules/l1.json`, 600 steps, validated free):
   supply_chain.mix, 150 steps at the 70% settings but product_mix left at 0.5, then 45 at
   recovery (195, leaves 0): does class-2 stock pile up on the shelf? Retail at step 150:
   G 798, K 611, K without the e2 floor 703 (d1 sigma 49). traffic.ramp, ramp_metering 1 alone
   for 100 steps, then 30 at recovery (130): pins toll's demand effect ct. Speeds while ramp is
   on: K (ct 0.5) 27.6 / 28.2, ct 0.74 gives 33.9 / 34.4. traffic.train, four full pulses of 30
   with 20-step rests, then 85 at recovery (275): does the persistent speed_a drop keep growing
   with each pulse? Traffic keeps 20 steps. ad_auction (220), reservoir (120), social (40) and
   wildlife (13) are left unspent.
   `python collect_runs.py supply_chain "file:schedules/l1.json#supply_chain.mix" --out research\supply_chain_l1.json`
   `foreach ($k in "ramp","train") { python collect_runs.py traffic "file:schedules/l1.json#traffic.$k" --out research\traffic_l1.json }`

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
- Round F collected Sep 26 Toronto (02:48-03:23): all 30 runs complete and match the schedule.
  Wildlife also has the crashed partial xy (run 0 of `research/wildlife_f1.json`, 91 steps; the
  92-step copy moved to `research/backup/`). Steps left: epidemic 400, market 400, traffic 825,
  power_grid 400, supply_chain 595, wildlife 363, reservoir 520, ad_auction 620,
  social_contagion 540, hospital_queue 400.
- E on unseen Round F (proxy, std sigma): epidemic 0.92, reservoir 0.89, market 0.88, ad_auction
  0.87, power_grid 0.84, wildlife 0.83, social_contagion 0.80, hospital_queue 0.79, traffic 0.75,
  supply_chain 0.70. D beats E there on hospital (0.81), social (0.82) and supply_chain (0.75):
  the Round C refit bought single-control accuracy at some cost in order/spacing.
- Round F misses, data vs E: traffic blocks for real (pulse 1: flow_b 0 vs model 11; pulse 2:
  flow_a 0, flow_b 42 vs 6/11; order effect 1.0-1.6 sigma, model ~0 on route b), so the shared
  junction / spillback mechanism is active and unmodeled. supply_chain: under the pulse supplier
  stock drains to ~9 (model 142) and retail builds to 84-250 (model 0). hospital_queue:
  discharges are exactly 0 in every pulse (model 1.8), only 5 after a 15-step rest (model 12,
  queue 301 vs 230), 15.7 after a long one (model 10.9). social_contagion: the post-pulse drop
  is too deep in the model (after a 10-step rest A 93 vs 61, B 47 vs 28), peaks low by 10-25.
  wildlife: prey is back to 98/91 after a 15-step rest (model 36/26) and the model overshoots
  after long rests (138-149 vs 121-123). reservoir: quality shows a real order effect (0.57
  sigma), the model none. market: price is the weak observable (rate-then-tax falls further
  than the model, joint pulses fall less). power_grid: frequency is the weak observable.
  epidemic and ad_auction hold up.
- E's code, checked Sep 26: 40 random 4,000-step episodes per system (holds of 1 to 4,000 steps)
  are all finite and non-negative. power_grid is the slowest at 16 s for 40 (limit 1,200 s).
- Round G findings (agents' notes, checked against data where stated):
  hospital_queue: the slow recovery after every pulse is staff orientation that fades over ~22
  steps (same after 15, 25 and 100-step pulses), not fatigue that builds up. Follow-up 0 at calm
  changes nothing for 130 steps, so returning case mix is probably inactive. Discharges under
  stress come in lumps (mostly 0, batches of 3-9), so the model shrinks low discharge
  predictions toward the median (a scoring choice). traffic: flows are deterministic lumps and
  cap any smooth model near 0.70-0.76 on flows; speeds carry the gains. The junction holds
  crossing vehicles ~5 steps plus an exit store shared by both routes; route b's delay depends on
  lane closure (11 steps at 0.325, 16 at 0.65). Toll's effect on demand is unidentified (no run
  has ramp > 0 with toll 5). supply_chain: forward transport is the bottleneck and its rate moves
  (~34/step under stress, 25-27 at recovery); throughput collapses under long stress at high
  receiving effort (machine heat/wear), so 4,000 steps at receiving 1.5 + maintenance 0 now
  predict ~2.4/step shipments (untested beyond ~80 steps). wildlife: prey regrows at ~0.18-0.20
  per step after any pulse; the overshoot size depends on how long prey stayed low (food sets
  capacity); hunting removes about a fixed count per step with a floor near 7. market: joint
  rate+tax pulses hold price flat ~12 steps then fall to a 65.9 floor; rate alone plateaus at
  72.2; while tax is on, committed price moves don't reverse (xy keeps falling after the rate
  ends); volume/depth only move while price moves.
- social_contagion: the agent's best candidate fixed E's too-fast post-pulse crash (held-out F
  0.812 -> 0.860) but missed the Round C floor by 0.0016; refit with bb <= 0 it passed (0.862,
  c1 0.9271) and shipped. Still open, in E and in G: at the recovery action B keeps growing for thousands of steps (E: B 321 at
  step 4,000, G: 240; data after every pulse settles A ~45-75, B ~35-65, and growth slows as B rises).
  Our runs are <= 420 steps, so the long-run ceiling is unidentified. This may be most of the
  0.497 public score (10 of 40 episodes are long holds).
- Public vs our proxy after G: social 0.51 vs ~0.90, epidemic 0.69 vs 0.92, wildlife 0.65 vs 0.87,
  market 0.67 vs 0.90, hospital 0.71 vs 0.84; traffic, supply_chain, power_grid, reservoir and
  ad_auction are within ~0.1. Every current model settles to a fixed point by ~step 400 (the
  epidemic's waves damp to a flat ~101 cases at recovery; social's B is the only drift, to 240).
  Our runs stop at 420-700 steps, so 90% of every 4,000-step episode is extrapolation. The big
  gaps are the slow, long-memory systems.
- Current-model predictions for the Round H runs (compare after collecting): social recovery
  A/B t100 64/50, t300 94/96, t500 102/127; epidemic recovery cases/hospital t100 46/40, t200
  113/88, t400 100/72; market 70% joint price t100 78.8, t400 75.7, depth ~56; wildlife 70% prey
  N/S t100 34/28, t350 35/29; hospital 70% queue ~318, discharges t100 2.85, t400 1.96, wait 78.
- Round H collected Sep 26 (~11:02-11:09 Toronto): all 5 long holds complete and match
  `schedules/h1.json`. Steps left: social_contagion 40, wildlife 13, epidemic / market /
  hospital_queue 0. Data vs current models (score on our ruler):
  epidemic recovery 400: waves fade to ~100 cases by step 250, the model matches (0.95), so the
  epidemic's weak public score is not its long recovery behavior.
  social recovery 500: both communities saturate, A ~91 and B ~74 at step 500, still creeping;
  the model keeps climbing (A 102, B 125 at 500) (0.78).
  market 70% joint pulse 400: price falls slowly (~0.1-0.15/step) for ~250 steps and stops near
  77; depth erodes to 7 while price falls, then rebounds to 27 by step 400; volume ~3 while price
  moves, 1.6 after. The model drops price fast then freezes, depth stuck at 56 (0.64).
  wildlife 70% pulse 350: prey settles at ~20 / ~18, predators ~1.76; E rebounds to 35 / 29 and
  predators 2.14 (0.80). The Round G wildlife candidate scores 0.71 on this run, which fits its
  small public drop.
  hospital 70% pulse 400: discharges ~0 for ~25 steps then flat at ~4.4; the model decays to 2.0
  (0.86). Wait levels at 74.5, queue at ~323.
- Round I (Sep 26, workflow of 4 agents: market, social, wildlife, hospital): refit to the long
  runs. Ship rule: long run +0.02, held-out Round F no worse than -0.005, first look / C / F
  in-sample no worse than -0.01, gate passes. Harness baseline = the current repo models.
- The documents say each phase's 40 episodes are exactly 10 per category: sustained operation,
  intervention order, recovery spacing, joint intervention.
- Upload I lesson (market): the dealer hold fitted the Round H run (0.67 -> 0.95 on our ruler)
  and was flat on held-out F, yet lost 0.044 publicly. Its gate was a guess (rate 0.035-0.085)
  and its settlement takes ~200 steps, so the hidden set's 70-100% pulses spend hundreds of
  steps in a regime no run constrains. Rule: before shipping a new mechanism, run it on
  4,000-step holds (recovery, 70%, 100%, each control alone) next to the incumbent; a big
  change where we have no data is a bet, and a public upload is the only way to test it.
- G -> I moved social's B level at recovery from 240 to 79 by step 4,000 (data ~75 at step
  500) but the public score moved only +0.003. So hidden episodes do not sit at recovery for
  thousands of steps: they are busy schedules, and what scores is the response to each change
  and the settle level of each regime, not the 4,000-step asymptote.
- 4,000-step hold scan of the shipped models (Sep 26, free, scratch): every system is flat by
  step ~400 except reservoir (seasonal), power_grid (tiny undamped wobble) and social under
  incentive 2 alone, where B climbs 75 -> 134 -> 193 -> 282 (loyal recruits convert to
  incentive-led members, which frees room under the Mb cap, so organic growth refills for
  ever). Our longest incentive-only segment is 70 steps, so that is unconstrained, not wrong.
  supply_chain at the 70% pulse: retail builds to 852 (sigma 313) and supplier stock swings
  0 -> 320 -> 361; never observed.
- Social misses still in I (F `yx`, seeding then incentive): A over by 35 during the incentive
  phase (220 vs 184) and under by 28 after it (41 vs 69); the first-look pulse: A 227 vs 202
  at its step 100. The model converts too many seeded recruits to incentive-led members when
  membership is high (brief: the onboarding workforce is shared with existing members). The
  `xy` order and the 500-step recovery hold fit well. That is the next social model round.
- Round J collected Sep 26 (~13:21-13:29 Toronto): all 5 runs complete (400 steps each) and
  match `schedules/j1.json`. Steps left: traffic 425, ad_auction 220, supply_chain 195,
  reservoir 120, power_grid 0. Current models on unseen J (std proxy): ad_auction 0.875,
  reservoir 0.855, supply_chain 0.836, power_grid 0.828, traffic 0.827. Misses at 70%:
  supply_chain supplier stock empties in 3 steps (model 50) and refills at step ~75-100 (model
  150+), retail peaks ~740 (model 844); traffic flows 14-15 (model 9-12), speed_a 12.5 (model
  10), queue released faster after the stress; power_grid frequency ~0.5 too low all phase;
  reservoir quality holds 0.94-0.96 (model drifts to 0.93); ad_auction conversions spike to 7
  at step 20 then settle 4.2 (model flat 5.3).
- The ruler (Sep 26, free, scratch `J/ruler*.py`): for every public-scored version (E, G, I; 20
  system-version pairs) we scored that exact code on our runs under several sigma definitions.
  sigma = 5.6 x (pooled std of one-step changes), "d1", tracks public best: on runs a version
  had not seen, mean miss 0.067 and correlation +0.74 (our std sigma: 0.088 at its best scale,
  correlation +0.42; unscaled std sigma: 0.105). d1 hits social exactly (E 0.499 vs public
  0.497, G 0.512 vs 0.511) and market E (0.623 vs 0.622). Still off: epidemic (d1 0.836 vs
  public 0.687, so the hidden episodes show epidemic behavior our runs never did) and
  supply_chain (d1 too harsh by ~0.1-0.19). Under d1 lumpy observables (traffic flows, hospital
  discharges, shipments) are cheap to miss and smooth ones (speeds, wait time, prey, adopters,
  price, depth, inventories, level) are expensive. From now on a candidate must win on both
  rulers. Frozen sigmas: scratch `J/sigmas.json`; harness: scratch `J/harness.py`.
- Round K data findings (agents' notes, the numbers re-checked):
  power_grid: the reserve delivered at 70% (request 105) equals the full pulse's (150), so the
  cap is below 105; the old model's two errors cancelled at 100% and only the 70% hold exposed
  them. Interconnector vs charging never moved separately, so the cap's split is a guess.
  supply_chain: sales ~28/step with both goods classes on the shelf, ~16 with class 1 alone;
  the fitted class-2 stock effect e2 was ~0 and let retail run to 8,000+ at the 70% hold with
  product_mix 0.5 (37,000 at mix 0), caught by a hold scan at the mix extremes and fixed by
  holding e2 at the honest fit's 0.0046 (same scores, bounded: ~1,200 at mix 0.5). Shipments
  step up 32 -> 35.5 once the dispatch queue fills. reservoir: capped outflow is one function of
  level (16.6 at 921 down to 11.0 at 276), no sign of screen fouling; deep water starts ~0.007
  better after reset; aeration 0.3 keeps the surface as clean as aeration 1; season ~67.8 steps.
  traffic: route b looks the same at 70% and 100%; the persistent speed_a drop after pulses
  grows with repeated pulses (~2 after two, ~3.7 after three), unmodeled. ad_auction:
  conversions are exactly 0 on steps 1-2 after every reset; breadth 0.55 alone converts 7.3/step
  with no backlog. social: the joint pulse levels off at A + B ~ 336-344; the reset dip is
  proportional to the first reading (unmodeled); the seeding-then-incentive miss is still open.
- Same-structure refits under the d1 ruler (Sep 26): market does not beat G (G is at its best
  for its structure). epidemic's two observables keep the same ratio under both rulers, so the
  weighting barely matters there. hospital gains +0.0096 d1 but its refit moved the follow-up
  program so capacity at full follow-up fell just under arrivals: at the recovery action the
  queue climbed 23 -> 105 after step 400 (no data that long); freezing tP and phi fixes the
  drift but then std drops 0.008, so not shipped. Rule: always run the 4,000-step hold scan.
- Round L collected Sep 27 (~19:06-19:08 Chris's time): all 3 runs complete and match
  `schedules/l1.json`. Steps left: traffic 20, ad_auction 220, reservoir 120, social 40,
  wildlife 13, every other system 0. Data vs predictions made before collecting:
  supply_chain at the 70% settings with product_mix 0.5: retail peaks at ~417 (step 100) and
  then FALLS to 371 at step 150 while the stress continues (K 446 -> 607, G 714 -> 797, K
  without the e2 floor 470 -> 698); lower mix means less stock on the shelf (the mix-0.71 J run
  peaked ~740), so class 2 sells faster than K assumes. Shipments 36 early, sagging to ~30 by
  step 100 (all models flat 32). Score on this unseen run: K 0.797/0.661, G 0.658/0.504.
  traffic ramp metering alone at toll 5: flows 12.0 / 11.9 (K 13.5 / 12.2, ct 0.74 gives 7.4),
  so toll's demand effect is near K's ct 0.5; speeds 31.5 / 30.6 (K 27.6 / 28.2), ~3 higher
  than K at that load. Train of four full pulses with 20-step rests: speed_a stays 12-16 all
  through (rests too short to recover) and is back to 48.0 after 85 steps of rest (48.3 before
  the train); K settles at 47.0, so the persistent drop does not grow with repeated pulses and
  mostly fades. K scores best on both traffic runs (ramp 0.883/0.818, train 0.780/0.799).
- Supply_chain after Round L (params-only refit agent, Sep 27, scratch `J/supply_chain_L`): no
  params-only refit passes (best +0.003 std). Diagnosis: class 2 sells faster than K assumes
  (~13.7/step at ~180 on the shelf vs K 12.0); at mix 0.5 class-1 arrivals (~15/step) are below
  class-1 sales, so retail stays modest and falls. K's fixed 12-step class-2 delay is wrong (class
  2 reaches the shelf ~step 8 at mix 0.5-0.65) and throughput depends on mix (36.5/step at 0.5 vs
  32-35.5 at 0.71; K has one fixed cap of 32). Consequence: at the 70% hold K's retail settles at
  ~1,180 (mix 0.5), ~2,400 (0.3), ~4,500 (0), far above the data's 371-and-falling. A structural
  fix (two transport queues, class 2 on leftover capacity, faster class-2 shelf) is in progress.
- Upload windows (kit/PROMPT.md): public and final uploads both stay open until Sep 30 12:00
  Toronto and share the 3 slots per system per day. So public tests can continue Mon-Wed; keep
  1 slot per system per day for a final update.


## More info please refer to the webpage: 
https://gt-portal-wavddee32q-uc.a.run.app/guide
https://gt-portal-wavddee32q-uc.a.run.app/briefs
https://gt-portal-wavddee32q-uc.a.run.app/challenge
https://gt-portal-wavddee32q-uc.a.run.app/faq