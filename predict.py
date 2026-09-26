#!/usr/bin/env python3
"""
predict.py  -  first-guess forecasters for all ten hackathon systems, in ONE file.

The organizers' runner imports this file and calls

    predict(initial, interventions, context)

once per test episode. context["family"] names the system, so the SAME file
goes into every system folder. Each folder gets its own model.json with the
fitted numbers for that system. No model.json -> built-in defaults, so the
file always runs.

Every system follows one pattern:

    DEFAULTS[family]              first-guess parameter values (fit.py overwrites these)
    START[family](initial, p)     rebuild hidden state from the first observation
    ADVANCE[family](state, a, p)  one tick: mutate state, return the observation dict

You change the equations here. fit.py changes the numbers.

Conventions
- A parameter whose default is 0.0 and whose name is a level (L0, p0, K_n, ...)
  means "take it from the initial observation". fit.py keeps 0.0-valued
  parameters frozen unless you pass --free, so that convention survives fitting.
- Plain Python floats everywhere. 4,000 steps x 40 episodes runs in seconds.
- Imports only json/math/os, which is all the sandbox allows.
"""
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
INF = float("inf")

DEFAULTS = {}
START = {}
ADVANCE = {}


# ------------------------------------------------------------------ helpers

def _pos(x):
    return x if x > 0.0 else 0.0


def _clip(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def _f(d, key, default=0.0):
    """float(d[key]) with a fallback; actions and observations are dicts."""
    v = d.get(key, default)
    try:
        return float(v)
    except (TypeError, ValueError):
        return float(default)


# ------------------------------------------------------------------ epidemic
# SEIRS with a 2-stage latent period (cases lag actions by 2 steps), waning
# immunity (the epidemic comes back in waves), a referral pipeline that starts
# empty, and a hard bed cap with a waiting list (hospital pins at ~155).
# Round C (single-control holds) added: waiting patients leave the list at rate wl
# (plateau exactly at cap, drain on time); behavior: contacts fall as hospital fills
# (1 / (1 + bh * H/cap)); referrals per case rise with school closure (hs, older age
# mix) and with vaccination (hv, shared clinic workforce). Masks carry most of the
# restriction effect, school closure little. Age groups otherwise ignored.
# Fitted on all three research runs (scratchpad E/epidemic).
DEFAULTS["epidemic"] = dict(
    N=18646.41367192948,       # population
    s0=1.0,                    # susceptible fraction cap at reset (not binding: S = N - E - I)
    rI=5.050754028127234,      # infectious per initial daily case
    beta=0.44806734996298114,  # transmission per step at full contact
    sigma=0.40623995640848354, # latent stage exit rate (2 stages)
    gamma=0.14472651613652904, # recovery rate per step
    omega=0.009601018340951832, # immunity waning rate per step
    a_s=0.036083109003154354,  # contact reduction at full school closure
    a_m=0.38188557470949186,   # exposure reduction at full mask mandate
    v_eff=0.6787575878795006,  # fraction of vaccination_rate*S immunized per step
    h=0.05451534864245651,     # hospital referrals per onset
    k_c=0.1376297200054615,    # referral pipeline rate per step
    d=0.07619603943561191,     # hospital discharge rate per step
    cap=155.18613733120998,    # hard bed cap
    wl=0.03573089168068964,    # share of the bed waiting list that leaves per step
    bh=0.31885890020465174,    # behavior: contact cut per unit hospital pressure H/cap
    hs=0.06482224419159772,    # extra referrals per onset at full school closure (fraction)
    hv=0.09257700840017434,    # extra referrals per onset at full vaccination (fraction)
)


def start_epidemic(init, p):
    N = max(p["N"], 1.0)
    c0 = _pos(_f(init, "daily_cases"))
    E1 = E2 = c0 / max(p["sigma"], 1e-6)   # onsets = sigma * E2
    I = p["rI"] * c0
    S = min(p["s0"], 1.0) * N
    S = min(S, N - E1 - E2 - I)
    R = _pos(N - S - E1 - E2 - I)
    return dict(S=S, E1=E1, E2=E2, I=I, R=R, C=0.0, W=0.0, H=_pos(_f(init, "hospital_load")))


def adv_epidemic(s, a, p):
    N = max(p["N"], 1.0)
    sc = _f(a, "school_closure"); vr = _f(a, "vaccination_rate")
    b = p["beta"] * (1 - p["a_s"] * sc) * (1 - p["a_m"] * _f(a, "mask_mandate")) \
        / (1 + p["bh"] * s["H"] / p["cap"])
    S, E1, E2, I, R = s["S"], s["E1"], s["E2"], s["I"], s["R"]
    inf = min(b * S * I / N, S)
    vax = min(p["v_eff"] * vr * S, S - inf)
    on = p["sigma"] * E2      # new onsets = daily_cases
    mv = p["sigma"] * E1
    rec = p["gamma"] * I
    wane = p["omega"] * R
    s["S"] = S + wane - inf - vax
    s["E1"] = E1 + inf - mv
    s["E2"] = E2 + mv - on
    s["I"] = I + on - rec
    s["R"] = R + rec + vax - wane
    hh = p["h"] * (1 + p["hs"] * sc) * (1 + p["hv"] * vr / 0.003)
    ref = p["k_c"] * s["C"]   # referrals leave the clinical pipeline
    s["C"] += hh * on - ref
    s["W"] += ref             # and wait for a bed
    Hd = (1 - p["d"]) * s["H"]
    adm = min(s["W"], _pos(p["cap"] - Hd))
    s["W"] = (s["W"] - adm) * (1 - p["wl"])
    s["H"] = Hd + adm
    return {"daily_cases": on, "hospital_load": s["H"]}


START["epidemic"], ADVANCE["epidemic"] = start_epidemic, adv_epidemic


# ------------------------------------------------------------------ market
# Round G (scratchpad G/market). Price follows its target through a two-stage lag:
# stage 1 (pf, committed orders) is faster falling (k_p) than recovering (k_pu), stage 2
# (execution) follows pf at k_p2. Round F showed transaction tax changes how the rate
# moves price: joint pulses fall later but deeper (floor ~65.9 vs ~72.2 for rate alone),
# and a rate drop keeps going under a following tax (no recovery while tax is on). So:
#   c_t1: tax slows stage 1 in both directions (k / (1 + c_t1 * tax/0.05)).
#   a_amp: tax deepens the committed move at execution: (p0 - pf) * (1 + a_amp * tax/0.05).
#   a_rate_p, a_amp are held at the observed plateaus (c1 rate-alone 72.2, base joint 65.9)
#   so long holds settle where the data settled.
#   b_vp, b_dp: volume rises and depth falls while price is moving (per unit |price step|).
# Volume and depth otherwise relax first-order. Calm levels are fitted constants.
# a_tax_v and a_rate_v stay tied (a_tax_v = 2 * a_rate_v); keep the tie when refitting.
DEFAULTS["market"] = dict(
    p0=94.18772,  # calm price level (0 = take from initial)
    v0=1.84674,  # calm volume level (0 = take from initial)
    d0=90.97057,  # calm depth level (0 = take from initial)
    a_rate_p=2.33,  # price target falls a_rate_p * interest_rate (fixed: rate plateau)
    a_tax_v=1.04083,  # volume target falls a_tax_v * tax (tied: 2 * a_rate_v)
    a_rate_v=0.52041,  # volume target falls a_rate_v * interest_rate
    a_vol_d=-0.0129,  # depth target change per unit of volume above baseline
    a_tax_d=10.80974,  # depth target falls a_tax_d * tax
    k_p=0.10511,  # price lag stage 1, target below pf (falling)
    k_pu=0.01744,  # price lag stage 1, target above pf (recovering)
    k_p2=0.03586,  # price lag stage 2
    k_v=0.30562,  # volume relaxation rate
    k_d=0.12142,  # depth relaxation rate
    c_t1=2.63671,  # tax slows price stage 1
    a_amp=0.289,  # tax deepens the committed price move (fixed: joint floor)
    b_vp=1.79483,  # volume target rise per unit |price step|
    b_dp=7.91325,  # depth target drop per unit |price step|
)


def start_market(init, p):
    price = _f(init, "price", 1.0)
    vol = _pos(_f(init, "volume"))
    dep = _pos(_f(init, "depth"))
    return dict(price=price, volume=vol, depth=dep, pf=price,
                p0=p["p0"] if p["p0"] > 0 else price,
                v0=p["v0"] if p["v0"] > 0 else max(vol, 1e-6),
                d0=p["d0"] if p["d0"] > 0 else max(dep, 1e-6))


def adv_market(s, a, p):
    r = _pos(_f(a, "interest_rate"))
    tax = _pos(_f(a, "transaction_tax"))
    g = tax / 0.05
    tp = s["p0"] * _pos(1 - p["a_rate_p"] * r)
    k1 = p["k_pu"] if tp > s["pf"] else p["k_p"]
    s["pf"] += _clip(k1, 0, 1) / (1 + _pos(p.get("c_t1", 0.0)) * g) * (tp - s["pf"])
    q = _pos(s["p0"] - (s["p0"] - s["pf"]) * (1 + _pos(p.get("a_amp", 0.0)) * g))
    old = s["price"]
    s["price"] += _clip(p["k_p2"], 0, 1) * (q - s["price"])
    dp = abs(s["price"] - old)
    tv = s["v0"] * _pos(1 - p["a_tax_v"] * tax - p["a_rate_v"] * r) + _pos(p.get("b_vp", 0.0)) * dp
    td = s["d0"] * _pos(1 - p["a_vol_d"] * (s["volume"] / s["v0"] - 1) - p["a_tax_d"] * tax) \
        - _pos(p.get("b_dp", 0.0)) * dp
    s["volume"] += _clip(p["k_v"], 0, 1) * (tv - s["volume"])
    s["depth"] += _clip(p["k_d"], 0, 1) * (_pos(td) - s["depth"])
    return {"price": s["price"], "volume": s["volume"], "depth": s["depth"]}


START["market"], ADVANCE["market"] = start_market, adv_market


# ------------------------------------------------------------------ traffic
# Round G (Sep 26): E's queue model plus a shared junction (scratchpad G/traffic, fit_g.py).
# Round J (Sep 26): same equations, params refit on first look + round C + round F + round J (250 steps
# at the 70% pulse, then 150 at recovery; scratchpad J/traffic, fit_A.py). Per route r in (a, b):
#   arrivals  = lam_r * ramp * (1 - ct * toll/5)   ramp = admitted demand; recovery (ramp 0) = empty road
#   pipeline  arrivals reach the approach queue after D_r + dl_r * max(0, lane - lz) - T steps (= free-flow
#             time to first exit; round C: b exits 11 steps after demand at lane 0.325, 16 at 0.65)
#   approach  Q_r <= qmax_r (excess rejected)
#   admission = min(Q_r, C_r * share_r * lf_r)     share_a = signal_timing, share_b = 1 - signal_timing
#             lf_r = 1 - l_r * lane_closure
#   junction  admitted vehicles cross for T steps, then wait in an exit store R_r that empties at
#             X_r * lf_r per step (flow_r). Crossing + waiting vehicles of BOTH routes share Jtot of
#             room; when it is full, admissions of both routes are scaled down (one route obstructs the
#             other). Round F: after a signal reversal the old route keeps flowing ~5-10 steps, and the
#             drained stock after a sig 0.85 pulse is 1,500 vs ~940 after sig 0.15. Room is checked
#             before this step's departures, so a full junction admits in bursts (period ~6 flow pattern
#             under heavy demand at lane 0); real flows are just as lumpy and smoothing it scored worse.
#   speed     EMA (alpha) toward (vfree_r - M_r) / (1 + n_r / qref_r), n_r = Q_r + junction_r +
#             max(w * pipeline now, pipeline kd steps ago): speeds fall on the first pulse step, but
#             stay low ~5 steps after demand stops (round C). M_b = 0.
#             The first speed reading's offset from free speed decays separately at alpha0 (~0.28/step
#             in every run; E tied it to alpha).
#   memory    M_a += g * Q_a / (Q_a + qref_a) * (Mmax - M_a), never decays.
# Fixed, not fitted: D_a, D_b, dl_b, lz, kd, T, w, ct (toll's demand effect: no run has ramp > 0 at
# toll > 2.5, so a fitted ct only moves an unobserved regime; kept at E's 0.5).
# The initial flow reading is ignored: roads start empty.
DEFAULTS["traffic"] = dict(
    D_a=11.0,
    D_b=11.0,
    dl_a=0.0,
    dl_b=15.4,
    lz=0.325,
    kd=5.0,
    T=5.0,
    ct=0.5,
    lam_a=27.02263109417189,
    lam_b=24.365439918156284,
    C_a=67.16894615915552,
    C_b=29.37029654889871,
    l_a=0.001522387918088238,
    l_b=0.6127555410605433,
    qmax_a=515.4252828714582,
    qmax_b=138.21474865463236,
    qref_a=190.52116304575256,
    qref_b=183.69919863080182,
    vfree_a=49.1715512266661,
    vfree_b=48.7283718081734,
    alpha=0.13974926702184665,
    alpha0=0.29216846258712104,
    g=0.6656356208115545,
    Mmax=2.156616799149093,
    X_a=66.77297083384761,
    X_b=16.285982611881536,
    Jtot=187.06334764173235,
    w=1.0,
)

_TRAFFIC_RING = 64


def start_traffic(init, p):
    s = {"M": 0.0, "t": 0}
    kd = max(int(round(p["kd"])), 0)
    T = max(int(round(p["T"])), 0)
    for r in ("a", "b"):
        v = _f(init, "speed_" + r, p["vfree_" + r])
        s["V" + r] = p["vfree_" + r]   # hidden state: empty road at free speed
        s["O" + r] = v - p["vfree_" + r] if math.isfinite(v) else 0.0   # reading offset, decays at alpha0
        s["Q" + r] = 0.0
        s["P" + r] = [0.0] * _TRAFFIC_RING   # arrivals due at step (index mod ring)
        s["F" + r] = 0.0   # vehicles in the pipeline
        s["H" + r] = [0.0] * kd   # pipeline load, kd steps late
        s["X" + r] = [0.0] * T    # vehicles crossing the junction
        s["S" + r] = 0.0          # sum of X_r
        s["R" + r] = 0.0          # crossed, waiting for exit space
    return s


def adv_traffic(s, a, p):
    sig = _clip(_f(a, "signal_timing", 0.5), 0, 1)
    lane = _clip(_f(a, "lane_closure"), 0, 1)
    toll = _clip(_f(a, "toll"), 0, 5)
    ramp = _clip(_f(a, "ramp_metering"), 0, 1)
    dem = ramp * _pos(1 - p["ct"] * toll / 5)
    share = {"a": sig, "b": 1 - sig}
    lf, adm = {}, {}
    t = s["t"]
    s["t"] = t + 1
    T = max(int(round(p["T"])), 0)
    for r in ("a", "b"):
        x = p["lam_" + r] * dem
        ring = s["P" + r]
        d = int(round(p["D_" + r] - T + p["dl_" + r] * _pos(lane - p["lz"])))
        d = 1 if d < 1 else _TRAFFIC_RING - 1 if d > _TRAFFIC_RING - 1 else d
        ring[(t + d) % _TRAFFIC_RING] += x
        s["F" + r] += x
        y = ring[t % _TRAFFIC_RING]
        ring[t % _TRAFFIC_RING] = 0.0
        s["F" + r] -= y
        s["Q" + r] = min(s["Q" + r] + y, p["qmax_" + r])
        lf[r] = _pos(1 - p["l_" + r] * lane)
        adm[r] = min(s["Q" + r], p["C_" + r] * share[r] * lf[r])
    room = _pos(p["Jtot"] - s["Sa"] - s["Sb"] - s["Ra"] - s["Rb"])
    tot = adm["a"] + adm["b"]
    if tot > room:
        k = room / tot
        adm["a"] *= k
        adm["b"] *= k
    out = {}
    for r in ("a", "b"):
        s["Q" + r] -= adm[r]
        cross = s["X" + r]
        if cross:
            cross.append(adm[r])
            s["S" + r] += adm[r]
            z = cross.pop(0)
            s["S" + r] -= z
        else:
            z = adm[r]
        s["R" + r] += z
        served = min(s["R" + r], p["X_" + r] * lf[r])
        s["R" + r] -= served
        f = _pos(s["F" + r])
        hist = s["H" + r]
        if hist:
            hist.append(f)
            f = max(p["w"] * _pos(s["F" + r]), hist.pop(0))
        n = s["Q" + r] + _pos(s["S" + r]) + s["R" + r] + f
        if r == "a":
            m = s["Qa"]
            s["M"] += p["g"] * m / (m + p["qref_a"]) * (p["Mmax"] - s["M"])
            vf = p["vfree_a"] - s["M"]
        else:
            vf = p["vfree_b"]
        s["V" + r] += p["alpha"] * (vf / (1 + n / p["qref_" + r]) - s["V" + r])
        s["O" + r] *= 1 - p["alpha0"]
        out["flow_" + r] = served
        out["speed_" + r] = s["V" + r] + s["O" + r]
    return out


START["traffic"], ADVANCE["traffic"] = start_traffic, adv_traffic


# ------------------------------------------------------------------ power grid
# Round J refit (scratchpad J/power_grid, candidate B), all four logs (first look,
# round C, round F, round J 70% hold).
# Load = base demand + a fixed population of thermostatic cooling loads (2 classes,
# 480 each, spread thermal time constants). Each load cools while on, warms while
# off, and switches the moment its temperature reaches the deadband limit that price
# shifts (exact crossing inside the step, so loads with different time constants
# drift apart and a synchronized rebound dies out). The reading is the share of the
# step each load ran. Every reset starts from the same asynchronous population at
# price 0.8.
# Base demand lags its price-set desired level (brief: price reduces DESIRED
# demand): it starts at the initial load reading and closes a fraction 1-rho of the
# gap per step (data: every price step moves load ~80% at once, the rest over ~15
# steps; step 1 of every run tracks the initial reading).
# Frequency: supply - demand drives it, damping pulls it back. Conventional
# generation = slow dispatch setpoint (rate ks ~1/57, displaced by delivered
# reserve) + fast governor droop (rate kg), inside output limits gmin..gmax (data:
# after a reserve hold generation takes ~50 steps to come back; renewable share
# tracks load swings within a few steps). Fitted floor ~55: under any reserve
# hold conventional output sits at the floor, so frequency follows
# supply - load through damping alone (data: ~34 power units per Hz there).
# Reserve delivers min(request, cap) with ramp kq. cap = q0 + qa*(ic+charging)/2
# (data: renewable curtailment, a proxy for delivered reserve, is the same at the
# 70% pulse (request 105) as at the full pulse (150), so the cap is below 105;
# more reserve arrives at interconnector 1 + charging 1: fitted cap ~70 at the
# pulse settings, ~96 at recovery settings. The two controls always moved
# together under reserve, so the even split between them is a guess.)
# Renewables r0 + r1*interconnector, curtailed by delivered reserve (cq).
# Share = renewables / (renewables + G + reserve).
_PG_N = 480                 # cooling loads per class
_PG_GOLD = 0.6180339887498949

DEFAULTS["power_grid"] = dict(
    s0=0.5997461081462461,        # thermostat band centre at price 0.8 (normalized temperature)
    db=0.12309073296195919,       # thermostat deadband width
    kap=0.05076375588023307,      # band shift per unit price above 0.8
    tau0=123.63141517011486,      # thermal time constant, class 0 (steps)
    tau1=95.4467759072381,        # thermal time constant, class 1 (steps)
    h=0.3803191813288195,         # +/- spread of time constants within a class
    W0=67.30766742065039,         # total power of class 0 cooling loads
    W1=22.483453547704116,        # total power of class 1 cooling loads
    B0=108.93430536399443,        # load at price 0.8 (base demand + running cooling loads)
    e=0.13886019548554632,        # desired base-demand drop per unit price above 0.8 (fraction of B0)
    rho=0.8342667685781021,       # base demand keeps this fraction of its gap to desired, per step
    kf=0.013798400468984642,      # Hz per unit power imbalance per step
    df=0.46910896332402163,       # frequency damping per step
    g0=67.63170664806327,         # conventional dispatch setpoint without reserve
    droop=3.050960083364232,      # conventional power per Hz below 50 (fast governor)
    kg=0.11813579579628114,       # fast governor response per step
    ks=0.017490331169999856,      # slow dispatch response per step
    disp=0.3071785088986796,      # dispatch setpoint drop per unit delivered reserve
    gmin=54.83456083365457,       # conventional output floor
    gmax=150.0,                   # conventional output ceiling (not binding in any run)
    r0=11.115024965489186,        # local renewables
    r1=26.814185778609957,        # remote renewables at interconnector 1
    cq=0.00796327898964801,       # renewable curtailment per unit reserve
    q0=67.2616135647811,          # reserve cap at interconnector 0 and charging 0
    qa=28.567583145838718,        # extra reserve cap at interconnector 1 and charging 1
    kq=0.9999999980000012,        # reserve ramp per step
)


def start_power_grid(init, p):
    lo0 = min(max(p["s0"] - p["db"] / 2, 1e-3), 0.999)
    hi0 = min(max(p["s0"] + p["db"] / 2, lo0 + 1e-4), 0.9995)
    aa, tau, th, on, w = [], [], [], [], []
    for c in range(2):
        tc, wc = p["tau%d" % c], p["W%d" % c] / _PG_N
        for n in range(_PG_N):
            tk = tc * (1 + p["h"] * ((n + 0.5) / _PG_N * 2 - 1))
            ph = ((n + 0.5) * _PG_GOLD + 0.123 * c) % 1.0
            t_on = tk * math.log(hi0 / lo0)
            t_off = tk * math.log((1 - lo0) / (1 - hi0))
            tt = ph * (t_on + t_off)
            o = tt < t_on
            aa.append(math.exp(-1.0 / tk))
            tau.append(tk)
            th.append(hi0 * math.exp(-tt / tk) if o else 1 - (1 - lo0) * math.exp(-(tt - t_on) / tk))
            on.append(o)
            w.append(wc)
    base = 0.0
    for k in range(len(on)):
        if on[k]:
            base += w[k]
    return dict(a=aa, tau=tau, th=th, on=on, w=w, base=base, d=_f(init, "load", p["B0"]) - base,
                x=_f(init, "frequency", 50.0) - 50.0, gs=p["g0"], gf=0.0, Q=0.0)


def adv_power_grid(s, a, p):
    price = _clip(_f(a, "price_signal", 0.8), 0.0, 2.0)
    res = _clip(_f(a, "reserve_dispatch"), 0.0, 150.0)
    ch = _clip(_f(a, "charging_allowance", 1.0), 0.0, 1.0)
    ic = _clip(_f(a, "interconnector"), 0.0, 1.0)
    # thermostatic loads: exact threshold crossing inside the step
    sp = p["s0"] + p["kap"] * (price - 0.8)
    lo, hi = sp - p["db"] / 2, sp + p["db"] / 2
    aa, tau, th, on, w = s["a"], s["tau"], s["th"], s["on"], s["w"]
    tot = 0.0
    for k in range(len(aa)):
        x = th[k]
        o = on[k]
        if o and x <= lo:
            o = False
        elif not o and x >= hi:
            o = True
        if o:
            y = x * aa[k]
            if y > lo:
                th[k] = y
                tot += w[k]
            else:
                t1 = tau[k] * math.log(x / lo)
                th[k] = 1 - (1 - lo) * math.exp(-(1 - t1) / tau[k])
                o = False
                tot += w[k] * t1
        else:
            y = 1 - (1 - x) * aa[k]
            if y < hi:
                th[k] = y
            else:
                t1 = tau[k] * math.log((1 - x) / (1 - hi))
                th[k] = hi * math.exp(-(1 - t1) / tau[k])
                o = True
                tot += w[k] * (1 - t1)
        on[k] = o
    # base demand lags its desired level
    s["d"] += (1.0 - p["rho"]) * (p["B0"] * (1 - p["e"] * (price - 0.8)) - s["base"] - s["d"])
    load = s["d"] + tot
    # reserve, renewables, conventional (slow dispatch + fast governor), frequency
    cap = _pos(p["q0"] + p["qa"] * 0.5 * (ic + ch))
    s["Q"] += p["kq"] * (min(res, cap) - s["Q"])
    ren = (p["r0"] + p["r1"] * ic) * _pos(1.0 - p["cq"] * s["Q"])
    s["gs"] += p["ks"] * (p["g0"] - p["disp"] * s["Q"] - s["gs"])
    s["gf"] += p["kg"] * (-p["droop"] * s["x"] - s["gf"])
    g = _clip(s["gs"] + s["gf"], p["gmin"], p["gmax"])
    sup = ren + g + s["Q"]
    s["x"] += p["kf"] * (sup - load) - p["df"] * s["x"]
    share = ren / sup if sup > 1e-6 else 0.0
    return {"load": load, "frequency": 50.0 + s["x"], "renewable_share": _clip(share, 0.0, 1.0)}


START["power_grid"], ADVANCE["power_grid"] = start_power_grid, adv_power_grid


# ------------------------------------------------------------------ supply chain
# Round J (scratchpad J/supply_chain), fitted to first look + round C + round F + round J.
# Chain as in G: production (2-step delay) -> supplier stock (ceiling) -> orders withdraw
# available stock -> dispatch queue (withdrawals stop when it is full) -> forward transport
# (3-step conveyor) -> receiving buffer -> receiving (rate x receiving_effort) -> retail -> sales.
# Idle supplier stock turns unavailable; maintenance restores it. Transport shares drive service
# with receiving and maintenance, and slows as machine heat builds at high receiving effort.
# J change 1, two goods classes on the shelf, each sold separately: every run sells ~28/step in
# its first 3 steps (both classes, fixed initial shares), ~15-18 once the shelf holds class 1
# alone (pulse, 70% hold, recovery tails), ~28 again when class-2 goods arrive. Product mix sets
# the class-1 share of each dispatch; classes keep their order through queue and buffer.
# J change 2, class-2 goods spend 12 steps in their own intake before the shared queue, so under
# congestion they reach the shelf late (~38 steps into the pulse, ~20 at the 70% hold, at once in
# round C where nothing queues). The 70% hold's retail peak then eases toward class-1 balance.
# J refit: production at partial maintenance (nm; only rounds C and J have 0 < maintenance < 1)
# and the congestion ceiling q_max; the 70% hold refills supplier stock at ~step 90, not ~150.
# e2 held at >= 0.0046 (the fit without round J): the data pin it only loosely, and near 0 it lets
# class-2 stock grow without limit at low product mix with high throughput (no run goes there).
DEFAULTS["supply_chain"] = dict(
    kp=32.26912568747066,     # production per step at production_effort 1, maintenance 0 (2-step delay)
    dm=0.6611589768053079,    # share of production lost at full maintenance
    nm=2.1567601323545484,    # maintenance exponent on that loss (loss = dm * maintenance**nm)
    s_cap=361.8,              # supplier stock ceiling (calm reading)
    s_res=0.0,                # supplier stock that orders cannot withdraw
    ag=0.358267119804474,     # share of available supplier stock that turns unavailable per step while it sits
    tr=0.48422747215464135,   # share of unavailable stock made available again per step per unit maintenance
    q_max=714.5843597563409,  # dispatched goods not yet transported where new withdrawals stop (congestion)
    trans=55.75717518194261,  # forward transport per step at zero receiving effort, maintenance and heat (3-step conveyor)
    b_max=192.43913634684716, # receiving buffer size where transport stops
    kr=51.51447025719483,     # receiving per step per unit receiving_effort
    dr=0.3558915209717783,    # share of receiving taken by full maintenance (shared drive service)
    d1=14.721648553383538,    # class-1 retail sales per step at an empty shelf
    e1=0.009023432268020146,   # extra class-1 sales per step per unit of class-1 shelf stock
    d2=11.190944989045953,    # class-2 retail sales per step at an empty shelf
    e2=0.004600018015166459, # extra class-2 sales per step per unit of class-2 shelf stock
    f2=0.4972468764725786,     # class-2 share of the initial retail stock
    n2=12.0,                  # steps class-2 goods spend in their own intake before the shared queue
    tre=0.30292847130594425,  # share of transport lost per unit receiving_effort (shared drive service)
    tm=0.0896066244575284,    # share of transport lost per unit maintenance
    h0=3876.91706028572,      # machine heat where transport halves; 0 = off
    hc=0.0013096697217733662, # share of heat lost per step at maintenance 0
    hm=0.21545111404719985,   # extra share of heat lost per step per unit maintenance
    hk=3.351275846420814,     # heat input = transport x receiving_effort**hk (drive service load)
)


def start_supply(init, p):
    s = _pos(_f(init, "inventory_supplier"))
    r = _pos(_f(init, "inventory_retail"))
    n2 = int(round(_clip(p["n2"], 0.0, 50.0)))
    return dict(s=s, a=s, r1=r * (1.0 - p["f2"]), r2=r * p["f2"], pp=[0.0, 0.0],
                q=[], qs=0.0, c2=[0.0] * n2, conv=[(0.0, 0.0)] * 3, b=[], bs=0.0, h=0.0)


def _take(fifo, amt):
    """Remove amt from the front of a FIFO list of [total, class2]; return the class-2 part."""
    got2 = 0.0
    while amt > 1e-12 and fifo:
        tot, two = fifo[0]
        if tot <= amt:
            got2 += two
            amt -= tot
            fifo.pop(0)
        else:
            f = amt / tot
            got2 += two * f
            fifo[0] = [tot - amt, two * (1.0 - f)]
            amt = 0.0
    return got2


def adv_supply(s, a, p):
    oq = _clip(_f(a, "order_quantity"), 0.0, 80.0)
    mix = _clip(_f(a, "product_mix", 0.5), 0.0, 1.0)
    pe = _clip(_f(a, "production_effort", 1.0), 0.0, 1.5)
    re = _clip(_f(a, "receiving_effort", 1.0), 0.0, 1.5)
    mt = _clip(_f(a, "maintenance"), 0.0, 1.0)
    av = s["a"]
    av += -p["ag"] * av + p["tr"] * mt * _pos(s["s"] - av)
    un = _pos(s["s"] - av)
    cap, res = p["s_cap"], p["s_res"]
    if un != 0.0:
        cap, res = cap - un, _pos(res - un)
    s["pp"].append(p["kp"] * pe * _pos(1.0 - p["dm"] * mt ** p["nm"]))
    av = min(av + s["pp"].pop(0), max(av, cap))
    held = s["qs"] + sum(s["c2"])
    w = min(oq, _pos(av - res), _pos(p["q_max"] - held))
    av -= w
    s["a"] = av
    s["s"] = av + un
    # product mix sets the class-1 share of the dispatch; class 2 waits in its own intake first
    w2 = w * (1.0 - mix)
    if s["c2"]:
        s["c2"].append(w2)
        w2 = s["c2"].pop(0)
        w = w - w * (1.0 - mix) + w2
    if w > 0.0:
        s["q"].append([w, w2])
        s["qs"] += w
    tcap = p["trans"] * _pos(1.0 - p["tre"] * re - p["tm"] * mt)
    if p["h0"] > 0.0:
        tcap /= 1.0 + (s["h"] / p["h0"]) ** 4
    t = min(s["qs"], tcap, _pos(p["b_max"] - s["bs"]))
    # machine heat: builds with transport work x drive load, cools slowly, maintenance cools it faster
    s["h"] = _pos(s["h"] + t * re ** p["hk"] - (p["hc"] + p["hm"] * mt) * s["h"])
    t2 = _take(s["q"], t)
    s["qs"] = _pos(s["qs"] - t)
    s["conv"].append((t, t2))
    arr, arr2 = s["conv"].pop(0)
    if arr > 0.0:
        s["b"].append([arr, arr2])
        s["bs"] += arr
    recv = min(s["bs"], p["kr"] * re * _pos(1.0 - p["dr"] * mt))
    recv2 = min(_take(s["b"], recv), recv)
    s["bs"] = _pos(s["bs"] - recv)
    # the shelf sells each class separately
    r1 = s["r1"] + recv - recv2
    r2 = s["r2"] + recv2
    s["r1"] = _pos(r1 - min(r1, p["d1"] + p["e1"] * s["r1"]))
    s["r2"] = _pos(r2 - min(r2, p["d2"] + p["e2"] * s["r2"]))
    return {"shipments": recv, "inventory_supplier": s["s"], "inventory_retail": s["r1"] + s["r2"]}


START["supply_chain"], ADVANCE["supply_chain"] = start_supply, adv_supply


# ------------------------------------------------------------------ wildlife
# Prey + hidden food stock per region, predators, and a corridor transit pool. Round I (scratchpad
# G/I/wildlife): fitted to first look + round C + round F + the round H 350-step 70% pulse hold.
# Food sets the crowding capacity, not the birth rate of a thin herd: births b / (1 + prey / (Pb*food)).
# After any crash prey regrows at the same pace whatever the history, while food that built up during a
# long low spell gives the big overshoot (reset, 100-step pulse) and a short spell a small one. Under a
# sustained pulse food refills but prey stays thin, so there is no rebound (round H: 70% pulse holds
# prey at ~20 / ~18 for 300 steps).
# Hunting takes about a fixed count per step at high prey (hq*quota*Hs) with a refuge at low prey (Ph), so
# harvest per animal peaks near 20 prey. Strong hunting (the 70-100% pulse box) pushes prey to a low floor
# (~20 at 70%, ~7 at the full pulse) instead of a proportional decline.
# Habitat protection acts through prey deaths (hm) and food renewal (hk), more in the north; the fitted
# shelter from hunting (sh) is near zero.
# Corridor: animals leave both regions while it is open and wait in a transit pool that settles in the
# other region at 1/tp, 1/td per step, so closing it still brings animals home. Predators in transit
# die at tmd per step, so an open corridor keeps predators low for as long as it stays open
# (round C: 2.33 -> 1.67 in 60 steps; round H: 1.76 at 70% for 200 steps).
# Predator crowding saturates at high density (reset predators 8-15 fall fast, then settle near 2.3).
DEFAULTS["wildlife"] = dict(
    b=0.6362021787044908,  # prey births per capita at low density (b - mu ~ 0.14/step: regrowth of a thin herd)
    mu=0.4999951619736701,  # prey death rate (b and mu act as a pair; the fit sits at fit_i's cap mu <= 0.5)
    hm_n=0.12324542792794141,  # extra prey deaths at zero habitat protection, north: mu*(1 + hm*(1-hab))
    hm_s=0.08572935810245555,  # same, south
    rho_n=0.00413835708177833,  # food renewal, north
    rho_s=0.0028053682271417554,  # food renewal, south
    hk_n=0.9641170108811695,  # habitat boost to food renewal, north
    hk_s=0.6700595767282168,  # habitat boost to food renewal, south
    cons=0.00024005825752208948,  # food eaten per prey
    F0=0.4420735707361732,  # food level at reset (fraction of capacity)
    Pb=2034.0758036218936,  # prey crowding of births, per unit food
    hq=0.07512128021510996,  # harvest per unit quota
    Ph=20.32150445273686,  # harvest refuge: prey level where harvest halves per capita
    Hs=20.075573650419937,  # harvest saturation: with Ph ~ Hs, harvest levels off near hq*quota*expo*Hs animals per step
    sh_n=0.028176737367711778,  # shelter: hunting exposure 1 - sh*hab, north
    sh_s=0.0007775495732185203,  # shelter, south
    a=0.04905029861810504,  # predator growth at abundant prey
    Hp=1.0548383106089296,  # prey level for half predator growth
    m=7.612037624790867e-05,  # predator death rate
    k=0.03254795359959699,  # predator crowding
    Dk=4.101246942704927,  # predator level where crowding per predator halves
    Hv=2.09556997964094e-08,  # prey level where half the predators are counted
    ep_n=0.046769319563386876,  # prey leaving the north per step at full corridor access
    ep_s=0.06084258159428294,  # prey leaving the south per step at full corridor access
    ed_n=0.03575732963153821,  # predators leaving the north per step at full corridor access
    ed_s=0.03655340930869652,  # predators leaving the south per step at full corridor access
    tp=33.04544506024473,  # prey transit pool: 1/tp of it settles in the other region per step
    td=13.92757536048174,  # predator transit pool: 1/td settles per step
    tmd=0.02344325674642063,  # predators in transit that die per step
)


def start_wildlife(init, p):
    pn = _pos(_f(init, "prey_north")); ps = _pos(_f(init, "prey_south"))
    hv = p["Hv"]
    dn = _pos(_f(init, "predator_north")) * (pn + hv) / max(pn, 1e-6)
    ds = _pos(_f(init, "predator_south")) * (ps + hv) / max(ps, 1e-6)
    f0 = min(p["F0"], 1.0)
    # go = animals that left last step (prey to north, prey to south, predators to north, to south)
    return dict(pn=pn, ps=ps, dn=dn, ds=ds, fn=f0, fs=f0, go=(0.0, 0.0, 0.0, 0.0),
                wpn=0.0, wps=0.0, wdn=0.0, wds=0.0)


def adv_wildlife(s, a, p):
    quota = _clip(_f(a, "hunting_quota", 0.0), 0.0, 8.0)
    hab = _clip(_f(a, "habitat_protection", 1.0), 0.0, 1.0)
    cor = _clip(_f(a, "corridor_access", 0.0), 0.0, 1.0)
    for reg in ("n", "s"):
        prey = s["p" + reg]; food = s["f" + reg]; pred = s["d" + reg]
        rho = p["rho_" + reg] * (1 + p["hk_" + reg] * hab)
        eat = p["cons"] * prey * food
        s["f" + reg] = min(max(food + rho * (1.0 - food) - eat, 0.0), 1.0)
        expo = _clip(1.0 - p["sh_" + reg] * hab, 0.0, 1.0)
        harvest = p["hq"] * quota * expo * prey * prey / (prey + p["Ph"]) / (1.0 + prey / p["Hs"])
        birth = p["b"] / (1.0 + prey / (p["Pb"] * max(food, 1e-9)))
        death = p["mu"] * (1.0 + p["hm_" + reg] * (1.0 - hab))
        v = prey + prey * (birth - death) - harvest
        s["p" + reg] = v if v > 1e-6 else 1e-6
        crowd = p["k"] * pred / (1.0 + pred / p["Dk"])
        v = pred + pred * (p["a"] * prey / (prey + p["Hp"]) - p["m"] - crowd)
        s["d" + reg] = v if v > 1e-6 else 1e-6
    # corridor: journeys start only while it is open; animals already travelling still arrive
    xpn = _clip(p["ep_n"] * cor, 0.0, 1.0) * s["pn"]; xps = _clip(p["ep_s"] * cor, 0.0, 1.0) * s["ps"]
    xdn = _clip(p["ed_n"] * cor, 0.0, 1.0) * s["dn"]; xds = _clip(p["ed_s"] * cor, 0.0, 1.0) * s["ds"]
    s["pn"] -= xpn; s["ps"] -= xps; s["dn"] -= xdn; s["ds"] -= xds
    tpn, tps, tdn, tds = s["go"]
    s["go"] = (xps, xpn, xds, xdn)
    s["wpn"] += tpn; s["wps"] += tps; s["wdn"] += tdn; s["wds"] += tds
    surv = 1.0 - _clip(p["tmd"], 0.0, 1.0)
    s["wdn"] *= surv; s["wds"] *= surv
    tp = max(p["tp"], 1.0); td = max(p["td"], 1.0)
    rel = s["wpn"] / tp; s["wpn"] -= rel; s["pn"] += rel
    rel = s["wps"] / tp; s["wps"] -= rel; s["ps"] += rel
    rel = s["wdn"] / td; s["wdn"] -= rel; s["dn"] += rel
    rel = s["wds"] / td; s["wds"] -= rel; s["ds"] += rel
    hv = p["Hv"]
    return {"prey_north": s["pn"], "predator_north": s["dn"] * s["pn"] / (s["pn"] + hv),
            "prey_south": s["ps"], "predator_south": s["ds"] * s["ps"] / (s["ps"] + hv)}


START["wildlife"], ADVANCE["wildlife"] = start_wildlife, adv_wildlife


# ------------------------------------------------------------------ reservoir
# Round J refit (Sep 26) on every paid run: first look, round C (one control at a time),
# round F (order + spacing) and round J (250 steps at the 70% pulse, then 150 at recovery).
# Water: seasonal river inflow (sinusoid in steps since reset), spillway cap at 941 (excess
# leaves as spill in outflow), loss e0 + e1*level.
# Delivery = min(request, outlet capacity, water available). Capacity grows with the square
# root of the head, c0 + c1*sqrt(level) (orifice flow). Round J showed it: at levels 275-400
# the old straight line was up to 0.7/step off; one curve fits 275-941 for every depth and
# aeration setting.
# Bank storage: the reservoir exchanges kx*(H - level) per step with an aquifer whose head H
# starts at H0 on every reset and relaxes toward the level at rate kh. A falling level draws
# water in (a share ro of it shows in the inflow reading), a rising level loses some.
# Quality: two stored layers, read at the outlet by withdrawal depth w (0 = surface, 1 = deep):
# quality = qc - (1-w)*S - w*D, plus a start-up transient from the first reading (tq).
# S = surface deficit: slow calm drift s0; grows only when aeration is near zero,
# sa*(1-aer)^na (round J: aeration 0.3 for 250 steps left the surface clean), faster during
# deep withdrawal (1 + sw*w: a deep release changes the stored layers; round F order runs).
# D = deep deficit: starts at D0 < 0 (the reset profile has better deep water: deep withdrawal
# reads ~0.007 higher early on), grows in proportion to missing aeration (da) and relaxes
# toward the surface at kd. Both recover at their own rate plus flushing, kf times the share
# of stored water leaving per step (round J: at level ~300 with ~11/step out, quality levels
# off within ~100 steps). Release and irrigation alone leave quality unchanged (round C, F).
DEFAULTS["reservoir"] = dict(
    A=11.311063112274228, B=2.2254484682304327, P=67.82051362601999, phi=0.023315293531304966,   # river = A + B*sin(2*pi*t/P + phi)
    c0=4.568004913413258, c1=0.39201795729701744,   # outlet capacity c0 + c1*sqrt(level)
    e0=-0.023485257219978194, e1=0.0012596644753478103,   # loss per step e0 + e1*level
    Lcap=941.0,   # spillway level (measured, held fixed in the fit)
    kx=0.005729233064621423, kh=0.03787895141137731, H0=435.59773769709494, ro=0.6413732880317944,   # bank storage: exchange rate, head relaxation, start head, share seen in inflow
    qc=0.9557237762358463, tq=4.1289578456719145,   # calm quality, start-up time constant
    s0=3.455440037355148e-05, sa=0.00012015648578010433, na=8.0, sw=0.7705429686376665, ks=0.0029492133683308723,   # surface: drift, no-aeration drive, its shape (fixed), deep-withdrawal boost, recovery
    da=0.0003393066798193797, kd=0.010343836182189627, D0=-0.010176188908178603, kf=0.06443387496311355,   # deep: no-aeration drive, relaxation toward surface, reset value; flushing
)


def start_reservoir(init, p):
    return dict(t=0, level=_pos(_f(init, "level")), q0=_f(init, "quality"), H=p["H0"], S=0.0, D=p["D0"])


def adv_reservoir(s, a, p):
    aer = _clip(_f(a, "aeration", 1.0), 0.0, 1.0); irr = _pos(_f(a, "irrigation_allocation"))
    rel = _pos(_f(a, "release_rate", 2.0)); w = _clip(_f(a, "withdrawal_depth"), 0.0, 1.0)
    s["t"] += 1; t = s["t"]; L = s["level"]
    river = p["A"] + p["B"] * math.sin(2 * math.pi * t / p["P"] + p["phi"])
    G = p["kx"] * (s["H"] - L)
    s["H"] += p["kh"] * (L - s["H"])
    inflow = river + (p["ro"] * G if G > 0 else 0.0)
    wet = river + G
    loss = p["e0"] + p["e1"] * L
    cap = p["c0"] + p["c1"] * math.sqrt(L)
    dlv = min(rel + irr, max(cap, 0.0), max(L + wet - loss, 0.0))
    L = L + wet - dlv - loss
    spill = 0.0
    if L > p["Lcap"]:
        spill = L - p["Lcap"]; L = p["Lcap"]
    if L < 0:
        L = 0.0
    s["level"] = L
    F = (dlv + spill) / max(L, 1.0)   # share of the stored water leaving this step (flushing)
    S = s["S"]; D = s["D"]
    s["S"] = S + p["s0"] + p["sa"] * (1.0 - aer) ** p["na"] * (1.0 + p["sw"] * w) - (p["ks"] + p["kf"] * F) * S
    s["D"] = D + p["da"] * (1.0 - aer) - p["kd"] * (D - S) - p["kf"] * F * D
    q = p["qc"] - (1.0 - w) * s["S"] - w * s["D"] + (s["q0"] - p["qc"]) * math.exp(-t / p["tq"])
    return {"level": L, "inflow": inflow, "outflow": dlv + spill, "quality": q}


START["reservoir"], ADVANCE["reservoir"] = start_reservoir, adv_reservoir


# ------------------------------------------------------------------ ad auction
# Round J structural model: Round E + audience-dependent fulfilment work, fitted on all
# research runs (first look, round C, round F order/spacing, round J 70% hold).
# Audience x in [0, 1] is cut into nested bins; breadth w targets x < w (a partly covered
# bin counts fractionally). Each bin has reach R (repeated exposure removes people, back in
# tauR), converted-unavailable V (back in tauV), attention A and pending purchases Q.
# Win rate per bin saturates in bid, scaled by a conserved pool of rival capital K that
# drifts toward where we bid and where reachable people are, and is boosted where reach
# is thin. Budget pacing cuts win_rate by pace**kap. Purchase starts need follow-up
# exposure (k1) plus a spontaneous part (k0); broader audiences convert less (cq).
# Completions share one fulfilment capacity F (core purchases per step). A purchase from
# outside the core audience (x > 0.55) needs W1 times the work: round F showed breadth
# 0.55 alone reaching 7.3 conversions/step with no backlog, while every pulse reaching
# past 0.55 plateaus at 5.2-5.8 and keeps converting 2-7 steps after it ends (round J:
# spike to 7.6 while the backlog is mostly core, plateau 5.8, backlog empty at step 61).
# W1 is held flat beyond 0.775 (no run targets that far). Hidden state starts at the
# fixed reset convention (full reach, empty pipeline, uniform rivals); the initial
# reading is ignored.
AD_EDGES = [0.0, 0.55, 0.775, 1.0]
AD_D = [AD_EDGES[i + 1] - AD_EDGES[i] for i in range(len(AD_EDGES) - 1)]
DEFAULTS["ad_auction"] = dict(
    wmax=0.6976841856757215,    # win_rate ceiling at high bid (fresh reach, rival pressure 1)
    b0=4.196706126225343,       # bid scale of the win curve (scaled by local rival pressure K)
    g=0.23506937881524861,      # win boost where reachable people are thin
    vp=240.59025109200323,      # spend per unit reached-and-won at bid 1.5
    pe=0.30106956762787446,     # price exponent in bid
    kap=0.9568440530526832,     # pacing: win_rate x pace**kap (kap<1: bid shading, not pure throttling)
    f=0.07985536088271657,      # reach lost per step per unit exposure
    tauR=90.11929453188766,     # reach recovery time (steps)
    a=1.1860742404258042,       # attention gained per unit impressions (x100)
    k1=0.03879977863657136,     # follow-up: purchase starts per attention at exposure xr
    k0=0.16803944679906646,     # spontaneous purchase starts per attention
    xr=0.26,                    # reference exposure for k1 (fixed)
    k2=0.0964909558652435,      # pending -> completed rate
    F=10.388282856804741,       # fulfilment capacity (core purchases per step)
    W1=7.071413158784073,       # fulfilment work per purchase outside the core audience (core = 1)
    v=0.0019641043057240797,    # converted customers made unavailable per conversion (per unit breadth)
    tauV=22.06587768896608,     # time for converted customers to return (steps)
    cq=2.9219323419764,         # conversion propensity falls as exp(-cq*x) across the audience
    phi=0.31923921522105014,    # price rises with local rival pressure K**phi
    rho=0.37102537507148814,    # rival capital pulled toward where we bid (conserved pool)
    tauK=26.133068834258516,    # rival capital relocation time (steps)
    mu=0.5963098102001345,      # rival capital pulled toward reachable people
)


def _ad_binavg_exp(c, e0, e1):
    if abs(c) < 1e-9:
        return 1.0
    return (math.exp(c * e1) - math.exp(c * e0)) / (c * (e1 - e0))


def start_ad(init, p):
    nb = len(AD_D)
    return dict(R=[1.0] * nb, V=[0.0] * nb, A=[0.0] * nb, Q=[0.0] * nb, K=[1.0] * nb,
                qx=[_ad_binavg_exp(-p["cq"], AD_EDGES[i], AD_EDGES[i + 1]) for i in range(nb)],
                wk=[1.0] + [p["W1"]] * (nb - 1))


def adv_ad(s, a, p):
    nb = len(AD_D)
    R = s["R"]; V = s["V"]; A = s["A"]; Q = s["Q"]; K = s["K"]; qx = s["qx"]
    wmax, b0, g, phi, kap = p["wmax"], p["b0"], p["g"], p["phi"], p["kap"]
    k1, k0, xr, k2, F = p["k1"], p["k0"], p["xr"], p["k2"], p["F"]
    f, tauR, v, tauV, aa = p["f"], p["tauR"], p["v"], p["tauV"], p["a"]
    tauK = p["tauK"]
    b = _pos(_f(a, "bid", 1.5)); cap = _pos(_f(a, "budget_cap", 20.0))
    w = _clip(_f(a, "targeting_breadth", 0.55), 0.0, 1.0)
    pb = p["vp"] * (b / 1.5) ** p["pe"]
    u = [0.0] * nb; av = [0.0] * nb; vol = [0.0] * nb; Wu = [0.0] * nb
    Su = 0.0; Vt = 0.0
    for i in range(nb):
        ui = _clip((w - AD_EDGES[i]) / AD_D[i], 0.0, 1.0)
        u[i] = ui
        x = _pos(R[i] - V[i])
        av[i] = x
        vol[i] = ui * AD_D[i] * x
        Ki = K[i] if K[i] > 1e-6 else 1e-6
        wi = wmax * (1.0 - math.exp(-b / (b0 * Ki))) * (1.0 + g * (1.0 - x))
        Wu[i] = wi if wi < 1.0 else 1.0
        pr = pb * Ki ** phi if phi != 0.0 else pb
        Su += vol[i] * Wu[i] * pr; Vt += vol[i]
    pace = cap / Su if Su > cap else 1.0
    wf = pace ** kap if kap != 1.0 else pace
    spend = Su * pace
    It = 0.0
    for i in range(nb):
        It += vol[i] * Wu[i]
    win = It * wf / Vt if Vt > 0 else 0.0
    D = 0.0
    wk = s["wk"]
    for i in range(nb):
        D += k2 * Q[i] * wk[i]
    sF = F / D if D > F else 1.0
    conv = 0.0
    bf = b * pace ** (1.0 - kap) / 1.5
    mu = p["mu"]; rho = p["rho"]
    tg = [((av[i] ** mu if mu > 0 else 1.0) * (1.0 + rho * (u[i] * bf))) for i in range(nb)]
    nrm = 0.0
    for i in range(nb):
        nrm += tg[i] * AD_D[i]
    for i in range(nb):
        x = u[i] * Wu[i] * wf
        ci = k2 * Q[i] * sF
        conv += ci
        st = A[i] * (k0 + k1 * x / xr)
        Q[i] += st - ci
        A[i] += aa * 100.0 * vol[i] * Wu[i] * wf * qx[i] - st
        R[i] = _clip(R[i] - f * x * av[i] * R[i] + (1.0 - R[i]) / tauR, 0.0, 1.0)
        V[i] = _pos(V[i] + v * ci / AD_D[i] - V[i] / tauV)
        K[i] += (tg[i] / nrm - K[i]) / tauK
    return {"win_rate": win, "spend": spend, "conversions": conv}


START["ad_auction"], ADVANCE["ad_auction"] = start_ad, adv_ad


# ------------------------------------------------------------------ social contagion
# Fitted structural model (round J: round I + overload churn). Per community: core members K (the initial (1-f) share,
# never leave), loyal recruits L (organic + bridge introductions), seeded recruits S,
# incentive-led members J (the initial f share, promised cohorts, converts), disappointed former
# members D, and people who could still join P = N - members - D - queued (finite community N).
# Seeding x (1 + m*u) x P/N fills a 2-stage onboarding queue (`tq` steps per stage), u =
# incentive/2: A gets sa*seed*(1-br), B gets sb*seed, of which the bridge share br goes through
# a slower introduction queue (`tq2` per stage) into L. Seeded onboarding is also throttled by
# one shared workforce capacity `Nt` (A + B members). Promises accompany waiting cohorts: a share
# phi*u of each queue entry graduates into J instead of S (or L). While incentive is on, L and S
# convert to J at kc*u; J leaves at lam*(1-u), S churns at lr*(1-u), and L and S churn at
# lamM*(E - 2u) while incentive sits below its expectation E (EMA, `tau_e`). Everyone who leaves
# becomes disappointed and returns to P after `tau_d` steps on average. Organic growth
# (a + b*(S + J)) x workforce room x P/N x (1 - (K + L)/M) goes to L: it fills a finite
# relationship-led audience M (round H long hold: no outreach settles near A 92, B 75-79), and
# seeded / incentive-led members crowd it out (b <= 0 in practice).
# Round J, overload churn: the onboarding workforce also serves existing members. Above `No`
# members (A + B) it can't keep up, and every non-core member (L, S, J) leaves at
# co x ((A + B)/No - 1) per step, incentive or not. Fitted co is steep, so this is a ceiling
# near No: the base joint pulse levels off at A 201 + B 135 = 336 (round I kept climbing to
# A 275 + B 237 = Nt on long pulse holds), and c1's seeding alone peaked at 344.
DEFAULTS["social_contagion"] = dict(
    f=0.5115883429778939,         # share of initial members who are incentive-led (leave at reset)
    lam=0.07802907283601557,       # incentive-led drain per step at zero incentive
    tq=5.248659047777971,         # steps per local onboarding stage (2 stages)
    tq2=27.910082690290107,       # steps per bridge-introduction stage (2 stages)
    sa=1.4080054258024688,         # A queue entries per unit seeding (times 1 - bridge)
    sb=0.32362953490601726,       # B queue entries per unit seeding (bridge share goes via introductions)
    Nt=1059.6505720129899,         # shared onboarding capacity (A + B members)
    aa=0.9534459880045454,        # organic growth A per step (times room)
    ab=0.6543247756090418,       # organic growth B per step (times room)
    ba=-0.011191491090078892,     # organic growth per seeded / incentive-led member A (crowding)
    bb=-7.881267527842448e-07,      # organic growth per seeded / incentive-led member B (crowding)
    kc=0.07802913833631368,       # recruits converted to incentive-led per step at full incentive
    lr=0.010641580595212169,      # seeded-recruit churn per step at zero incentive
    lamM=0.1376270736230053,      # recruit churn per step per unit of unmet incentive expectation
    tau_e=12.524193149468623,     # incentive expectation time constant (steps)
    phi=0.1406473875740199,                      # share of queue entries promised at full incentive
    m=0.2198384342939869,                        # extra seeded recruitment at full incentive (x (1 + m*u))
    Na=350.43093056749103,                       # community size A
    Nb=300.8264516873275,                       # community size B
    tau_d=1.0000000000184888,                   # steps before a disappointed former member reconsiders
    kr=52.661194558649385,                       # churn reduction from cross-community relationships (1/(1 + kr*R))
    tau_r=3204.049653164439,                   # relationship memory R: EMA of bridge outreach over tau_r steps
    Ma=93.25997717485042,                     # relationship-led audience A: organic growth stops as core + loyal reach it
    Mb=79.33291466553214,                     # relationship-led audience B
    co=1.5425675743912917,                                  # overload churn per step per unit of members above No (round J)
    No=340.23902190699846,                                # members (A + B) the shared workforce can support (round J)
)


def start_social(init, p):
    Aa = _pos(_f(init, "adopters_a")); Ab = _pos(_f(init, "adopters_b"))
    f = p["f"]
    s = dict(E=0.0, R=0.0)
    for c, A0 in (("a", Aa), ("b", Ab)):
        s["K" + c] = (1 - f) * A0; s["J" + c] = f * A0
        for k in ("L", "S", "D", "Q1", "Q2", "P1", "P2", "Q1p", "Q2p", "P1p", "P2p"):
            s[k + c] = 0.0
    return s


def adv_social(s, a, p):
    seed = _clip(_f(a, "seeding"), 0, 10); u = _clip(_f(a, "incentive"), 0, 2) / 2
    br = _clip(_f(a, "bridge_outreach"), 0, 1)
    drain = p["lam"] * (1 - u)
    pr = p["phi"] * u
    s["E"] += (2 * u - s["E"]) / p["tau_e"]
    s["R"] += (br - s["R"]) / p["tau_r"]
    keep = 1 / (1 + p["kr"] * s["R"])
    churnE = p["lamM"] * max(0.0, s["E"] - 2 * u)
    A = {c: s["K" + c] + s["L" + c] + s["S" + c] + s["J" + c] for c in ("a", "b")}
    roomS = max(0.0, 1 - (A["a"] + A["b"]) / p["Nt"])
    over = p["co"] * max(0.0, (A["a"] + A["b"]) / p["No"] - 1)
    g = seed * roomS * (1 + p["m"] * u)
    out = {}
    for c, loc, obs in (("a", p["sa"] * (1 - br), "adopters_a"), ("b", p["sb"] * (1 - br), "adopters_b")):
        queued = s["Q1" + c] + s["Q2" + c] + s["P1" + c] + s["P2" + c]
        free = max(0.0, 1 - (A[c] + s["D" + c] + queued) / p["N" + c])
        ql = loc * g * free
        qx = p["sb"] * br * g * free if c == "b" else 0.0
        tq, tq2 = p["tq"], p["tq2"]
        o1 = s["Q1" + c] / tq; o2 = s["Q2" + c] / tq
        o1p = s["Q1p" + c] / tq; o2p = s["Q2p" + c] / tq
        r1 = s["P1" + c] / tq2; r2 = s["P2" + c] / tq2
        r1p = s["P1p" + c] / tq2; r2p = s["P2p" + c] / tq2
        s["Q1" + c] += ql - o1; s["Q2" + c] += o1 - o2
        s["Q1p" + c] += pr * ql - o1p; s["Q2p" + c] += o1p - o2p
        s["P1" + c] += qx - r1; s["P2" + c] += r1 - r2
        s["P1p" + c] += pr * qx - r1p; s["P2p" + c] += r1p - r2p
        L = s["L" + c]; S = s["S" + c]; J = s["J" + c]; D = s["D" + c]
        org = (p["a" + c] + p["b" + c] * (S + J)) * roomS * free * max(0.0, 1 - (s["K" + c] + L) / p["M" + c])
        convL = p["kc"] * u * L; convS = p["kc"] * u * S
        leaveJ = min(J, (drain + over) * J); leaveS = min(S, (p["lr"] * (1 - u) * keep + churnE + over) * S)
        leaveL = min(L, (churnE + over) * L)
        s["J" + c] = max(0.0, J + o2p + r2p + convL + convS - leaveJ)
        s["S" + c] = max(0.0, S + (o2 - o2p) - convS - leaveS)
        s["L" + c] = max(0.0, L + (r2 - r2p) + org - convL - leaveL)
        s["D" + c] = D + leaveJ + leaveS + leaveL - D / p["tau_d"]
        out[obs] = s["K" + c] + s["L" + c] + s["S" + c] + s["J" + c]
    return out


START["social_contagion"], ADVANCE["social_contagion"] = start_social, adv_social


# ------------------------------------------------------------------ hospital queue
# Fitted fluid queue (round I: round G's structure plus "later" fatigue, refit on all runs
# incl. the Round H 400-step hold). Patients arrive (base `lam` plus
# electives into a list capped at `Emax`), wait, get admitted while occupancy < `Cs`,
# and are discharged from step `dead` on, at most `mu` work units per step; an elective
# needs `we` units. Arrivals past `Qmax` are referred elsewhere; waiting patients leave
# at rate `r`.
# mu = k * s_eff * diag_balance**hx * (1 + bo*ot) * (1 - fF*(F - fl*min(F, target))) * (1 - phi*P)
#   s_eff: added staff, and staff moved between assessment and treatment by a diag
#          change (|d change| * staffing), are only `eta` effective while orienting.
#          tO > 0 (round G): that orientation load fades exponentially over tO steps, and
#          staff cuts remove orienting staff first. tO = 0: E's hard window of To steps.
#   F: fatigue. Lags target = overtime * min(1, waiting / Wf) over tF steps, so overtime
#      only tires staff while patients are waiting. Round I: a share fl of the fatigue that
#      matches the current overtime costs nothing, so a steady overtime hold stays flat
#      (Round H: discharges flat at 4.3 for 350 steps at overtime 0.7) and the cost shows
#      once overtime is cut ("overtime can create later fatigue").
#   P: follow-up program load, lags followup_capacity over tP steps (starts empty).
# wait = EMA(alpha) of cw * waiting / (smoothed discharges + reneging). urgent_priority ignored.
# Reported discharges below ~dm are shrunk by up to dq (the real ones come in lumps there).
DEFAULTS["hospital_queue"] = dict(
    lam=11.458291441286233,      # base arrivals per step
    Emax=82.17273118516361,      # elective waiting-list cap
    Qmax=329.6258149033101,      # total queue cap (overflow referred elsewhere)
    r=0.01656084785507731,       # reneging fraction of waiting patients per step
    Cs=96.06206437719332,        # service occupancy cap (chairs + beds)
    k=1.0112045225693582,        # work per staff per step (regular-patient units) at balanced diag
    phi=0.4014574479129759,      # staff share diverted by a full follow-up program
    tP=66.05634059049068,        # follow-up program fill/empty time (steps)
    To=105.95822557579072,       # orientation window (steps), used only when tO = 0
    eta=0.5785203542250029,      # effectiveness of staff during orientation
    bo=0.2626271284154454,       # overtime work boost at full overtime
    fF=0.4355301805808154,       # capacity lost at full fatigue
    tF=95.3489814474512,         # fatigue build/recovery time (steps)
    Wf=0.6456960680299912,       # waiting patients at which overtime fully fatigues
    we=2.0782915089581793,       # work per elective patient (regular patient = 1)
    hx=0.6864400696018882,       # diag balance exponent (1 = linear tent peaked at 0.4)
    cw=2.6028250939966933,       # wait scale
    tb=5.944179388578482,        # smoothing of recent discharges (steps)
    alpha=0.04283839671885026,   # wait EMA weight
    dead=2.0,                    # steps before the first discharge after reset
    tO=24.587229407365562,       # >0: orientation fades exponentially over tO steps (To unused)
    dq=0.9,                      # reported-discharge shrink at low rates (0 = off)
    dm=2.0,                      # discharge rate below which the shrink applies
    fl=0.31233372437850215,      # share of fatigue that costs nothing while overtime continues (0 = round G)
)


def start_hospital(init, p):
    return dict(W=max(0.0, _f(init, "queue")), E=0.0, Sr=0.0, Se=0.0, Nr=0.0, Ne=0.0,
                w=_f(init, "wait_time"), Dbar=_f(init, "discharges"), P=0.0, F=0.0,
                pend=[], mv=[], Na=0.0, Nm=0.0, s_prev=20.0, d_prev=0.4, wprev=0.0, t=0)


def adv_hospital(s, a, p):
    st = _f(a, "staffing", 20.0); ot = _f(a, "overtime"); dg = _f(a, "diagnostic_allocation", 0.4)
    To = p["To"]
    tO = p.get("tO", 0.0)
    pend, mv = s["pend"], s["mv"]
    ds = st - s["s_prev"]
    dd = abs(dg - s["d_prev"]) * st
    if tO > 0.0:
        # orientation load that fades exponentially over tO steps: added staff (cuts remove
        # orienting staff first) plus staff moved by a diag change
        s["Na"] = max(0.0, s["Na"] + ds)
        s["Nm"] += dd
        Np = s["Na"] + s["Nm"]
        dec = math.exp(-1.0 / tO)
        s["Na"] *= dec
        s["Nm"] *= dec
    else:
        # orientation cohorts [age, amount]: added staff (cuts remove the newest first) and moved staff
        if ds > 0:
            pend.append([0, ds])
        elif ds < 0:
            cut = -ds
            while cut > 1e-12 and pend:
                if pend[-1][1] <= cut:
                    cut -= pend.pop()[1]
                else:
                    pend[-1][1] -= cut
                    cut = 0.0
        Np = 0.0
        if dd > 0:
            mv.append([0, dd])
        for c in pend + mv:
            Np += c[1] * min(1.0, max(0.0, To - c[0]))
    Np = min(Np, st)
    s_eff = st - (1.0 - p["eta"]) * Np
    s["s_prev"] = st
    s["d_prev"] = dg
    s["P"] += (_f(a, "followup_capacity") - s["P"]) / max(p["tP"], 1.0)
    h = max(0.0, min(dg / 0.4, (1.0 - dg) / 0.6))
    if p["hx"] != 1.0:
        h = h ** p["hx"]
    tgt = ot * min(1.0, s["wprev"] / max(p["Wf"], 1e-6))
    s["F"] += (tgt - s["F"]) / max(p["tF"], 1.0)
    fat = s["F"] - p.get("fl", 0.0) * min(s["F"], tgt)
    otf = (1.0 + p["bo"] * ot) * max(0.0, 1.0 - p["fF"] * fat)
    mu = max(0.0, p["k"] * s_eff * h * otf * (1.0 - p["phi"] * s["P"]))
    W, E, Sr, Se = s["W"], s["E"], s["Sr"], s["Se"]
    Qprev = W + E + Sr + Se + s["Nr"] + s["Ne"]
    D = 0.0
    if s["t"] >= p["dead"]:
        S = Sr + Se
        if S > 0.0:
            D = min(S, mu / ((Sr + p["we"] * Se) / S))
            f = D / S
            Sr -= Sr * f
            Se -= Se * f
    Sr += s["Nr"]
    Se += s["Ne"]
    e_in = min(max(0.0, _f(a, "elective_scheduling")), max(0.0, p["Emax"] - E))
    arr = p["lam"] + e_in
    room = max(0.0, p["Qmax"] - Qprev)
    if arr > room:
        e_in *= room / arr
        arr = room
    W += arr - e_in
    E += e_in
    tot = W + E
    adm = min(tot, max(0.0, p["Cs"] - Sr - Se))
    Nr = Ne = 0.0
    if tot > 0.0:
        Nr = adm * W / tot
        Ne = adm * E / tot
        keep = 1.0 - adm / tot
        W *= keep
        E *= keep
    W *= 1.0 - p["r"]
    E *= 1.0 - p["r"]
    waiting = W + E
    s["wprev"] = waiting
    s["Dbar"] += (D - s["Dbar"]) / max(p["tb"], 1.0)
    out_rate = s["Dbar"] + p["r"] * waiting
    T = p["cw"] * waiting / out_rate if out_rate > 1e-9 else 0.0
    s["w"] += p["alpha"] * (T - s["w"])
    for c in pend:
        c[0] += 1
    while pend and pend[0][0] >= To + 1.0:
        pend.pop(0)
    for c in mv:
        c[0] += 1
    while mv and mv[0][0] >= To + 1.0:
        mv.pop(0)
    s["W"], s["E"], s["Sr"], s["Se"], s["Nr"], s["Ne"] = W, E, Sr, Se, Nr, Ne
    s["t"] += 1
    dq = p.get("dq", 0.0)
    if dq > 0.0:
        # real discharges come in lumps when service is slow (mostly 0, now and then a batch);
        # the score rewards the typical value there, so report low rates shrunk toward 0
        D *= 1.0 - dq * math.exp(-(D / max(p.get("dm", 3.0), 1e-6)) ** 2)
    return {"wait_time": s["w"], "queue": W + E + Sr + Se + Nr + Ne, "discharges": D}


START["hospital_queue"], ADVANCE["hospital_queue"] = start_hospital, adv_hospital


# ------------------------------------------------------------------ generic fallback
# Each observable relaxes toward a target that is linear in the controls:
#   y += k * (b + sum_j w_j * u_j - y)
# Parameter keys: k__<obs>, b__<obs>, w__<obs>__<action>. With no parameters it
# holds every observable at its initial value, which is a safe day-one submission.
DEFAULTS["generic"] = {}


def start_generic(init, p):
    y = {k: _f(init, k) for k in init}
    b = {k: p.get("b__" + k, y[k]) for k in y}
    return {"y": y, "b": b}


def adv_generic(s, a, p):
    y = s["y"]
    for n in y:
        k = _clip(p.get("k__" + n, 0.05), 0, 1)
        t = s["b"][n]
        for an in a:
            w = p.get("w__" + n + "__" + an)
            if w:
                t += w * _f(a, an)
        y[n] += k * (t - y[n])
    return dict(y)


START["generic"], ADVANCE["generic"] = start_generic, adv_generic


# ------------------------------------------------------------------ shared loop

def simulate(family, initial, actions, p):
    """Run the model forward. Returns one observation dict per action.
    Any non-finite value is replaced by the previous step's value, so the
    output is always complete and finite (a hard requirement of the runner)."""
    start = START[family]; adv = ADVANCE[family]
    names = list(initial.keys())
    state = start(initial, p)
    last = {k: _f(initial, k) for k in names}
    out = []
    for a in actions:
        obs = adv(state, a, p)
        row = {}
        for k in names:
            v = obs.get(k)
            if v is None or v != v or v == INF or v == -INF:
                v = last[k]
            row[k] = float(v)
        out.append(row)
        last = row
    return out


def load_params(family, path=None):
    """Defaults, overridden by model.json next to this file (or an explicit path)."""
    p = dict(DEFAULTS.get(family, {}))
    path = path or os.path.join(HERE, "model.json")
    if os.path.exists(path):
        try:
            with open(path) as f:
                saved = json.load(f)
            if saved.get("family", family) == family:
                p.update(saved.get("params", {}))
        except Exception:
            pass
    return p


def predict(initial, interventions, context):
    family = str(context.get("family")) if hasattr(context, "get") else ""
    if family not in ADVANCE:
        family = os.path.basename(HERE)   # folder name is the system id
    if family not in ADVANCE:
        family = "generic"
    p = load_params(family)
    try:
        return simulate(family, initial, interventions, p)
    except Exception:
        # Never crash: a crash scores 0. Fall back to holding the initial values.
        return simulate("generic", initial, interventions, {})


# ------------------------------------------------------------------ reference data
# Published control bounds and the brief's reference actions. Used by the
# research and test scripts; harmless to ship inside each folder.
BOUNDS = {
    "epidemic": {"school_closure": (0, 1), "mask_mandate": (0, 1), "vaccination_rate": (0, 0.003)},
    "market": {"interest_rate": (0, 0.1), "transaction_tax": (0, 0.05)},
    "traffic": {"signal_timing": (0.1, 0.9), "lane_closure": (0, 0.75), "toll": (0, 5),
                "ramp_metering": (0, 1), "freight_priority": (0, 1), "clearance_effort": (0, 1)},
    "power_grid": {"price_signal": (0, 2), "reserve_dispatch": (0, 150),
                   "charging_allowance": (0, 1), "interconnector": (0, 1)},
    "supply_chain": {"order_quantity": (0, 80), "lead_time_buy": (0, 1), "product_mix": (0, 1),
                     "production_effort": (0, 1.5), "receiving_effort": (0, 1.5), "maintenance": (0, 1)},
    "wildlife": {"hunting_quota": (0, 8), "habitat_protection": (0, 1), "corridor_access": (0, 1)},
    "reservoir": {"release_rate": (0, 12), "irrigation_allocation": (0, 8),
                  "withdrawal_depth": (0, 1), "aeration": (0, 1)},
    "ad_auction": {"bid": (0, 5), "budget_cap": (0, 100), "targeting_breadth": (0.1, 1)},
    "social_contagion": {"seeding": (0, 10), "incentive": (0, 2), "bridge_outreach": (0, 1)},
    "hospital_queue": {"staffing": (1, 20), "elective_scheduling": (0, 20), "diagnostic_allocation": (0.1, 0.8),
                       "urgent_priority": (0, 1), "overtime": (0, 1), "followup_capacity": (0, 1)},
}

REFERENCE = {
    "epidemic": {
        "recovery": {"mask_mandate": 0.0, "school_closure": 0.0, "vaccination_rate": 0.0},
        "pulse": {"mask_mandate": 1.0, "school_closure": 1.0, "vaccination_rate": 0.003}},
    "market": {
        "recovery": {"interest_rate": 0.0, "transaction_tax": 0.0},
        "pulse": {"interest_rate": 0.1, "transaction_tax": 0.05}},
    "traffic": {
        "recovery": {"clearance_effort": 1.0, "freight_priority": 0.5, "lane_closure": 0.0,
                     "ramp_metering": 0.0, "signal_timing": 0.5, "toll": 5.0},
        "pulse": {"clearance_effort": 0.0, "freight_priority": 1.0, "lane_closure": 0.65,
                  "ramp_metering": 1.0, "signal_timing": 0.15, "toll": 0.0}},
    "power_grid": {
        "recovery": {"charging_allowance": 1.0, "interconnector": 1.0, "price_signal": 1.5, "reserve_dispatch": 0.0},
        "pulse": {"charging_allowance": 0.0, "interconnector": 0.2, "price_signal": 0.0, "reserve_dispatch": 150.0}},
    "supply_chain": {
        "recovery": {"lead_time_buy": 1.0, "maintenance": 1.0, "order_quantity": 0.0, "product_mix": 0.5,
                     "production_effort": 1.0, "receiving_effort": 1.5},
        "pulse": {"lead_time_buy": 0.2, "maintenance": 0.0, "order_quantity": 80.0, "product_mix": 0.8,
                  "production_effort": 1.5, "receiving_effort": 0.35}},
    "wildlife": {
        "recovery": {"corridor_access": 0.0, "habitat_protection": 1.0, "hunting_quota": 0.0},
        "pulse": {"corridor_access": 1.0, "habitat_protection": 0.1, "hunting_quota": 7.0}},
    "reservoir": {
        "recovery": {"aeration": 1.0, "irrigation_allocation": 0.0, "release_rate": 2.0, "withdrawal_depth": 0.0},
        "pulse": {"aeration": 0.0, "irrigation_allocation": 8.0, "release_rate": 12.0, "withdrawal_depth": 1.0}},
    "ad_auction": {
        "recovery": {"bid": 1.5, "budget_cap": 20.0, "targeting_breadth": 0.55},
        "pulse": {"bid": 5.0, "budget_cap": 100.0, "targeting_breadth": 0.775}},
    "social_contagion": {
        "recovery": {"bridge_outreach": 0.0, "incentive": 0.0, "seeding": 0.0},
        "pulse": {"bridge_outreach": 0.6, "incentive": 2.0, "seeding": 9.0}},
    "hospital_queue": {
        "recovery": {"diagnostic_allocation": 0.4, "elective_scheduling": 0.0, "followup_capacity": 1.0,
                     "overtime": 0.0, "staffing": 20.0, "urgent_priority": 0.6},
        "pulse": {"diagnostic_allocation": 0.75, "elective_scheduling": 20.0, "followup_capacity": 0.0,
                  "overtime": 1.0, "staffing": 5.0, "urgent_priority": 1.0}},
}

# Midpoints of each observable's reset range (docs/*_documents.json). Only used by self-tests.
TEST_INITIAL = {
    "epidemic": {"daily_cases": 170.0, "hospital_load": 45.0},
    "market": {"price": 100.0, "volume": 100.0, "depth": 100.0},
    "traffic": {"flow_a": 37.5, "flow_b": 37.5, "speed_a": 37.5, "speed_b": 37.5},
    "power_grid": {"load": 105.0, "frequency": 50.0, "renewable_share": 0.3},
    "supply_chain": {"shipments": 27.5, "inventory_supplier": 100.0, "inventory_retail": 100.0},
    "wildlife": {"prey_north": 85.0, "predator_north": 11.5, "prey_south": 85.0, "predator_south": 11.5},
    "reservoir": {"level": 500.0, "inflow": 10.0, "outflow": 6.0, "quality": 0.86},
    "ad_auction": {"win_rate": 0.375, "spend": 20.0, "conversions": 0.95},
    "social_contagion": {"adopters_a": 50.0, "adopters_b": 37.5},
    "hospital_queue": {"wait_time": 3.5, "queue": 50.0, "discharges": 11.5},
}


def random_schedule(family, n=4000, seed=0, hold=50):
    """Piecewise-constant random actions inside the published bounds."""
    import random
    rng = random.Random(seed)
    b = BOUNDS[family]
    out = []
    cur = None
    for t in range(n):
        if t % hold == 0:
            cur = {k: lo + (hi - lo) * rng.random() for k, (lo, hi) in b.items()}
        out.append(cur)
    return out


if __name__ == "__main__":
    # Self-test: every family runs 4,000 steps on a random schedule with the
    # defaults, stays finite, and reports its runtime.
    import time
    for fam in BOUNDS:
        t0 = time.time()
        out = simulate(fam, TEST_INITIAL[fam], random_schedule(fam), DEFAULTS[fam])
        dt = time.time() - t0
        ok = all(math.isfinite(v) for row in out for v in row.values())
        print(f"{fam:18s} steps={len(out)} finite={ok} time={dt:.2f}s "
              f"first={ {k: round(v, 2) for k, v in out[0].items()} } "
              f"last={ {k: round(v, 2) for k, v in out[-1].items()} }")
