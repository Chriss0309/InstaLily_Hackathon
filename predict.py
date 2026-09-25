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
# Round E fit on all three research runs (first look + round C single-control run).
# Price follows its target through a two-stage lag (lag, ramp, plateau), stage 1 is faster falling (k_p)
# than recovering (k_pu): round C rate on t50 = 29 steps, rate off t50 = 70 steps.
# Volume and depth relax first-order. Calm levels are fitted constants, not the first reading.
# a_tax_v and a_rate_v stay tied (a_tax_v = 2 * a_rate_v); keep the tie when refitting.
DEFAULTS["market"] = dict(
    p0=93.81467991644752,  # calm price level (0 = take from initial)
    v0=1.886058677676455,  # calm volume level (0 = take from initial)
    d0=90.63434257445736,  # calm depth level (0 = take from initial)
    a_rate_p=3.1467327979958792,  # price target falls a_rate_p * interest_rate
    a_tax_v=-1.0829051252047697,  # volume target falls a_tax_v * tax (tied: 2 * a_rate_v)
    a_rate_v=-0.5414525626023848,  # volume target falls a_rate_v * interest_rate
    a_vol_d=-0.012858346757379126,  # depth target change per unit of volume above baseline
    a_tax_d=10.975679806459524,  # depth target falls a_tax_d * tax
    k_p=0.04372360873249523,  # price lag stage 1, target below pf (falling)
    k_pu=0.023888161002877387,  # price lag stage 1, target above pf (recovering)
    k_p2=0.04137618323073983,  # price lag stage 2
    k_v=0.305597708318972,  # volume relaxation rate
    k_d=0.12230677632502787,  # depth relaxation rate
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
    tp = s["p0"] * (1 - p["a_rate_p"] * r)
    tv = s["v0"] * _pos(1 - p["a_tax_v"] * tax - p["a_rate_v"] * r)
    td = s["d0"] * _pos(1 - p["a_vol_d"] * (s["volume"] / s["v0"] - 1) - p["a_tax_d"] * tax)
    k1 = p["k_pu"] if tp > s["pf"] else p["k_p"]
    s["pf"] += _clip(k1, 0, 1) * (tp - s["pf"])
    s["price"] += _clip(p["k_p2"], 0, 1) * (s["pf"] - s["price"])
    s["volume"] += _clip(p["k_v"], 0, 1) * (tv - s["volume"])
    s["depth"] += _clip(p["k_d"], 0, 1) * (td - s["depth"])
    return {"price": s["price"], "volume": s["volume"], "depth": s["depth"]}


START["market"], ADVANCE["market"] = start_market, adv_market


# ------------------------------------------------------------------ traffic
# Queue model, refitted on all research runs incl. round C (Sep 25). Per route r in (a, b):
#   arrivals  = lam_r * ramp * (1 - ct * toll/5)   ramp = admitted demand; recovery (ramp 0) = empty road
#   pipeline  arrivals reach the junction queue after D_r steps
#   capacity  = C_r * share_r * (1 - l_r * lane)   share_a = signal_timing, share_b = 1 - signal_timing
#   queue     Q_r <= qmax_r, flow_r = min(Q_r, capacity)
#   speed     EMA toward (vfree_r - M_r) / (1 + (Q_r + F_r) / qref_r), M_b = 0
#             F_r = vehicles in the pipeline as they were kd steps ago (round C: speeds move ~5 steps late)
#   memory    M_a += g * Q_a / (Q_a + qref_a) * (Mmax - M_a), never decays. Only a standing queue on a
#             feeds it: speed_a stays ~2.3 low after the pulse, but not after 120 steps of free-flowing mid traffic.
# The initial flow reading is ignored: roads start empty.
DEFAULTS["traffic"] = dict(
    D_a=11.0,
    D_b=16.0,
    kd=5.0,
    ct=0.5,
    lam_a=26.28574311063386,
    lam_b=27.705894747007093,
    C_a=38.292724019749464,
    C_b=33.13338646725439,
    l_a=7.297199579789583e-09,
    l_b=0.9138174180679676,
    qmax_a=442.43203037474916,
    qmax_b=123.49544644002185,
    qref_a=176.48953469147457,
    qref_b=306.1687847839545,
    vfree_a=49.03225604777348,
    vfree_b=48.60161068384281,
    alpha=0.20189943691122356,
    g=0.06158071225790588,
    Mmax=3.3192311035196593,
)


def start_traffic(init, p):
    s = {"M": 0.0}
    kd = max(int(round(p["kd"])), 0)
    for r in ("a", "b"):
        v = _f(init, "speed_" + r, p["vfree_" + r])
        s["V" + r] = v if math.isfinite(v) else p["vfree_" + r]
        s["Q" + r] = 0.0
        s["P" + r] = [0.0] * max(int(round(p["D_" + r])), 1)
        s["F" + r] = 0.0   # vehicles in the pipeline
        s["H" + r] = [0.0] * kd   # pipeline load, kd steps late
    return s


def adv_traffic(s, a, p):
    sig = _clip(_f(a, "signal_timing", 0.5), 0, 1)
    lane = _clip(_f(a, "lane_closure"), 0, 1)
    toll = _clip(_f(a, "toll"), 0, 5)
    ramp = _clip(_f(a, "ramp_metering"), 0, 1)
    dem = ramp * _pos(1 - p["ct"] * toll / 5)
    share = {"a": sig, "b": 1 - sig}
    out = {}
    for r in ("a", "b"):
        x = p["lam_" + r] * dem
        pipe = s["P" + r]
        pipe.append(x)
        s["F" + r] += x
        y = pipe.pop(0)
        s["F" + r] -= y
        q = min(s["Q" + r] + y, p["qmax_" + r])
        cap = p["C_" + r] * share[r] * _pos(1 - p["l_" + r] * lane)
        served = min(q, cap)
        s["Q" + r] = q - served
        f = _pos(s["F" + r])
        hist = s["H" + r]
        if hist:
            hist.append(f)
            f = hist.pop(0)
        n = s["Q" + r] + f
        if r == "a":
            m = s["Qa"]
            s["M"] += p["g"] * m / (m + p["qref_a"]) * (p["Mmax"] - s["M"])
            vf = p["vfree_a"] - s["M"]
        else:
            vf = p["vfree_b"]
        s["V" + r] += p["alpha"] * (vf / (1 + n / p["qref_" + r]) - s["V" + r])
        out["flow_" + r] = served
        out["speed_" + r] = s["V" + r]
    return out


START["traffic"], ADVANCE["traffic"] = start_traffic, adv_traffic


# ------------------------------------------------------------------ power grid
# Fitted to all three research runs (first look + round C), scratchpad E/power_grid.
# Load = price-elastic base demand + a fixed population of thermostatic cooling
# loads (2 classes, 480 each, spread thermal time constants). Each load cools while on,
# warms while off, and switches the moment its temperature reaches the deadband
# limit that price shifts (exact crossing inside the step, so loads with different
# time constants drift apart and a synchronized rebound dies out instead of
# locking to whole-step cycles). The reading is the share of the step each load
# ran. Every reset starts from the same asynchronous population at price 0.8; the
# initial load reading's offset from base decays at rate rho.
# Frequency: supply - demand drives it, damping pulls it back. Conventional
# generation G follows a governor (droop, response kg) around
# g0 and is displaced by reserve. Reserve delivers min(request, q0) with ramp kq.
# Charging allowance has no modeled effect (never moved alone in research; no
# reserve depletion seen over 100 steps even at charging 0). Renewables
# r0 + r1*interconnector, curtailed by delivered reserve (cq).
# Share = renewables / (renewables + G + reserve).
_PG_N = 480                 # cooling loads per class
_PG_GOLD = 0.6180339887498949

DEFAULTS["power_grid"] = dict(
    s0=0.5942231497307183,    # thermostat band centre at price 0.8 (normalized temperature)
    db=0.1461719451654106,    # thermostat deadband width
    kap=0.06077917097107874,  # band shift per unit price above 0.8
    tau0=104.21868686039369,  # thermal time constant, class 0 (steps)
    tau1=81.43321232025225,   # thermal time constant, class 1 (steps)
    h=0.3871541479382904,     # +/- spread of time constants within a class
    W0=63.11771398715735,     # total power of class 0 cooling loads
    W1=24.463065643241904,    # total power of class 1 cooling loads
    B0=108.00856678810136,    # base load at price 0.8
    e=0.1161950882666221,     # base-load drop per unit price above 0.8 (fraction)
    rho=0.8161574306145789,   # per-step decay of the initial load reading's offset
    kf=0.010831787230847283,  # Hz per unit power imbalance per step
    df=0.21977813679052322,   # frequency damping per step
    g0=66.42739154839518,     # conventional generation setpoint
    droop=11.90888852049393,  # conventional power per Hz below 50
    kg=0.18205496777007926,   # governor response per step
    disp=0.5090986285343971,  # conventional displaced per unit reserve
    r0=13.369593164414749,    # local renewables
    r1=23.185024789034003,    # remote renewables at interconnector 1
    cq=0.00597333061367036,   # renewable curtailment per unit reserve
    q0=117.20437539180821,    # reserve power cap
    kq=0.7265069035629227,    # reserve ramp per step
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
    return dict(a=aa, tau=tau, th=th, on=on, w=w, base=base, e0=_f(init, "load", p["B0"]) - p["B0"],
                x=_f(init, "frequency", 50.0) - 50.0, G=p["g0"], Q=0.0)


def adv_power_grid(s, a, p):
    price = _clip(_f(a, "price_signal", 0.8), 0.0, 2.0)
    res = _clip(_f(a, "reserve_dispatch"), 0.0, 150.0)
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
    s["e0"] *= p["rho"]
    load = p["B0"] * (1 - p["e"] * (price - 0.8)) + tot - s["base"] + s["e0"]
    # reserve, renewables, conventional, frequency
    s["Q"] += p["kq"] * (min(res, _pos(p["q0"])) - s["Q"])
    ren = (p["r0"] + p["r1"] * ic) * _pos(1.0 - p["cq"] * s["Q"])
    gt = _pos(p["g0"] - p["disp"] * s["Q"] - p["droop"] * s["x"])
    s["G"] += p["kg"] * (gt - s["G"])
    sup = ren + s["G"] + s["Q"]
    s["x"] += p["kf"] * (sup - load) - p["df"] * s["x"]
    share = ren / sup if sup > 1e-6 else 0.0
    return {"load": load, "frequency": 50.0 + s["x"], "renewable_share": _clip(share, 0.0, 1.0)}


START["power_grid"], ADVANCE["power_grid"] = start_power_grid, adv_power_grid


# ------------------------------------------------------------------ supply chain
# Fitted to research/supply_chain.json + supply_chain_c1.json (3 runs, scratchpad E/supply_chain).
# Chain: production (2-step delay) -> supplier stock (ceiling) -> orders withdraw available stock
# -> dispatch queue (withdrawals stop when it is full: congestion) -> forward transport (3-step
# conveyor) -> receiving buffer -> receiving (rate x receiving_effort) -> retail -> sales.
# Idle supplier stock turns unavailable; maintenance restores it.
# Maintenance takes part of the receiving capacity (round C: shipments +7 the step it stops).
# Sales grow with retail stock (round C: ~27/step at stock 170, ~37/step at stock 1150).
# Conveyors and buffers start empty, so shipments read 0 until orders flow.
DEFAULTS["supply_chain"] = dict(
    kp=30.475112042903152,      # production per step at production_effort 1, maintenance 0 (2-step delay)
    dm=0.6422716868477619,      # share of production lost at full maintenance
    nm=2.824576042800507,       # maintenance exponent on that loss (loss = dm * maintenance**nm)
    s_cap=361.8,                # supplier stock ceiling (calm reading)
    s_res=0.0,                  # supplier stock that orders cannot withdraw
    ag=0.40909647277548145,     # share of available supplier stock that turns unavailable per step while it sits
    tr=0.18669021390133464,     # share of unavailable stock made available again per step per unit maintenance
    q_max=281.9821335794766,    # dispatch queue size where new withdrawals stop (congestion)
    trans=67.9568444498031,     # forward transport per step at zero wear (3-step conveyor)
    b_max=1225.3989644420017,   # receiving buffer size where transport stops
    kr=55.01904875596659,       # receiving per step per unit receiving_effort
    dr=0.6385018911433132,      # share of receiving taken by full maintenance (shared drive service)
    dem=26.07451530226024,      # base retail sales per step
    e=0.008425148350236628,     # extra retail sales per step per unit of retail stock
)


def start_supply(init, p):
    s = _pos(_f(init, "inventory_supplier"))
    return dict(s=s, a=s, r=_pos(_f(init, "inventory_retail")), pp=[0.0, 0.0], q=0.0,
                conv=[0.0, 0.0, 0.0], b=0.0)


def adv_supply(s, a, p):
    oq = _clip(_f(a, "order_quantity"), 0.0, 80.0)
    pe = _clip(_f(a, "production_effort", 1.0), 0.0, 1.5)
    re = _clip(_f(a, "receiving_effort", 1.0), 0.0, 1.5)
    mt = _clip(_f(a, "maintenance"), 0.0, 1.0)
    # supplier stock = available part a + unavailable part un. Available stock that sits turns
    # unavailable at rate ag; maintenance makes it available again. Everything is available at reset.
    av = s["a"]
    av += -p["ag"] * av + p["tr"] * mt * _pos(s["s"] - av)
    un = _pos(s["s"] - av)
    cap, res = p["s_cap"], p["s_res"]
    if un != 0.0:
        cap, res = cap - un, _pos(res - un)
    s["pp"].append(p["kp"] * pe * _pos(1.0 - p["dm"] * mt ** p["nm"]))
    av = min(av + s["pp"].pop(0), max(av, cap))
    w = min(oq, _pos(av - res), _pos(p["q_max"] - s["q"]))
    av -= w
    s["a"] = av
    s["s"] = av + un
    s["q"] += w
    t = min(s["q"], p["trans"], _pos(p["b_max"] - s["b"]))
    s["q"] -= t
    s["conv"].append(t)
    s["b"] += s["conv"].pop(0)
    recv = min(s["b"], p["kr"] * re * _pos(1.0 - p["dr"] * mt))
    s["b"] -= recv
    s["r"] = _pos(s["r"] + recv - min(s["r"] + recv, p["dem"] + p["e"] * s["r"]))
    return {"shipments": recv, "inventory_supplier": s["s"], "inventory_retail": s["r"]}


START["supply_chain"], ADVANCE["supply_chain"] = start_supply, adv_supply


# ------------------------------------------------------------------ wildlife
# Prey + hidden food stock per region, predators, and a corridor transit pool. Fitted to all research
# runs (first look + round C single-control run, scratchpad E/wildlife).
# Food refills while prey is low, which gives the prey overshoot after every recovery.
# Hunting: protected habitat shelters prey (exposure 1 - sh*hab, more in the north).
# Low habitat protection raises prey deaths in both regions within a few steps.
# Corridor: animals leave both regions while it is open and wait in a transit pool that settles in the
# other region at 1/tp, 1/td per step, so closing it still brings animals home (both regions dip, then rebound).
# Predator crowding saturates at high density (reset predators 8-15 fall ~5% per step).
DEFAULTS["wildlife"] = dict(
    b=0.26781299869299846,  # prey births per unit food
    mu=0.09378084869219784,  # prey death rate
    hm_n=0.2576395951605718,  # extra prey deaths at zero habitat protection, north: mu*(1 + hm*(1-hab))
    hm_s=0.18540464161864234,  # same, south
    rho_n=0.022985768034225865,  # food renewal, north
    rho_s=0.02109503717824554,  # food renewal, south
    hk_n=0.3374597311757939,  # habitat boost to food renewal, north
    hk_s=0.09220229754644624,  # habitat boost to food renewal, south
    cons=0.0003492328513884398,  # food eaten per prey
    F0=0.7332502500176639,  # food level at reset (fraction of capacity)
    Pb=626.7976256737691,  # prey crowding of births
    hq=0.017334337798512452,  # harvest per unit quota
    Ph=4.123764333241913,  # harvest refuge: prey level where harvest halves per capita
    sh_n=0.38748040073742684,  # shelter: hunting exposure 1 - sh*hab, north
    sh_s=0.30993373356878345,  # shelter, south
    a=0.060447236101810194,  # predator growth at abundant prey
    Hp=1.2493526797510561,  # prey level for half predator growth
    m=0.029621192537206093,  # predator death rate
    k=0.016763889930661843,  # predator crowding
    Dk=8.558428510509604,  # predator level where crowding per predator halves
    Hv=2.09556997964094e-08,  # prey level where half the predators are counted
    ep_n=0.011431681705489506,  # prey leaving the north per step at full corridor access
    ep_s=0.01842451280556638,  # prey leaving the south per step at full corridor access
    ed_n=0.017825653865054313,  # predators leaving the north per step at full corridor access
    ed_s=0.017748456905400296,  # predators leaving the south per step at full corridor access
    tp=49.97683271277038,  # prey transit pool: 1/tp of it settles in the other region per step
    td=28.48842990500357,  # predator transit pool: 1/td settles per step
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
        harvest = p["hq"] * quota * expo * prey * prey / (prey + p["Ph"])
        birth = p["b"] * food / (1.0 + prey / p["Pb"])
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
# Grey-box fit to research data, first look + round C (Sep 25). Seasonal river inflow (sinusoid in
# steps since reset), water balance with a hard spillway cap (excess leaves as spill in outflow),
# delivery min(request, c0 + c1*level, water available), loss e0 + e1*level.
# Bank storage: the reservoir exchanges kx*(H - level) per step with an aquifer whose head H starts
# at H0 on every reset and relaxes toward the level at rate kh. A falling level draws water in (a
# share ro of it shows in the inflow reading), a rising level loses some. No irrigation return
# flow (round C: none seen in 90 steps of irrigation 8).
# Quality: calm level qc, start-up transient from the first reading, a slow stress
# memory m driven only by a joint push: u = excess of the mean 0..1 position (recovery -> pulse)
# over th = 0.25, the most any one control alone can give.
DEFAULTS["reservoir"] = dict(
    A=11.279344763228828, B=2.2546514118561714, P=67.77337141928736, phi=0.015263764397348854,   # river = A + B*sin(2*pi*t/P + phi)
    c0=9.082030387128691, c1=0.007593384533294581,   # delivery cap c0 + c1*level
    e0=-0.19622625214888292, e1=0.0016530556531527793,   # loss per step e0 + e1*level
    Lcap=941.0,   # spillway level (measured, held fixed in the fit)
    kx=0.01604774298401246, kh=0.1382750495811783, H0=533.7342773766807, ro=0.6840276588459931,   # bank storage: exchange rate, head relaxation, start head, share seen in inflow
    qc=0.9596712525080192, qa=0.09123982053881292, tq=4.660674394054027,   # calm quality, memory weight, start-up time constant
    g0=0.0013703030433965336, g=0.0035950058192401185, d=0.006089635459834917, th=0.25,   # memory: calm drive, stress drive, decay, drive shape
)


def start_reservoir(init, p):
    return dict(t=0, level=_pos(_f(init, "level")), q0=_f(init, "quality"), m=0.0, H=p["H0"])


def adv_reservoir(s, a, p):
    aer = _f(a, "aeration", 1.0); irr = _pos(_f(a, "irrigation_allocation"))
    rel = _pos(_f(a, "release_rate", 2.0)); depth = _f(a, "withdrawal_depth")
    s["t"] += 1; t = s["t"]; L = s["level"]
    river = p["A"] + p["B"] * math.sin(2 * math.pi * t / p["P"] + p["phi"])
    G = p["kx"] * (s["H"] - L)
    s["H"] += p["kh"] * (L - s["H"])
    inflow = river + (p["ro"] * G if G > 0 else 0.0)
    wet = river + G
    loss = p["e0"] + p["e1"] * L
    dlv = min(rel + irr, max(p["c0"] + p["c1"] * L, 0.0), max(L + wet - loss, 0.0))
    L = L + wet - dlv - loss
    spill = 0.0
    if L > p["Lcap"]:
        spill = L - p["Lcap"]; L = p["Lcap"]
    if L < 0:
        L = 0.0
    s["level"] = L
    ub = _clip(((rel - 2.0) / 10.0 + irr / 8.0 + depth + (1.0 - aer)) / 4.0, 0.0, 1.0)
    u = max(ub - p["th"], 0.0) / (1.0 - p["th"])
    m = s["m"]
    s["m"] = m + ((p["g0"] + p["g"] * u) * (1 - m) - p["d"] * m)
    q = p["qc"] - p["qa"] * s["m"] + (s["q0"] - p["qc"]) * math.exp(-t / p["tq"])
    return {"level": L, "inflow": inflow, "outflow": dlv + spill, "quality": q}


START["reservoir"], ADVANCE["reservoir"] = start_reservoir, adv_reservoir


# ------------------------------------------------------------------ ad auction
# Round E structural model, fitted on all research runs (first look + round C).
# Audience x in [0, 1] is cut into nested bins; breadth w targets x < w (a partly covered
# bin counts fractionally). Each bin has reach R (repeated exposure removes people, back in
# tauR), converted-unavailable V (back in tauV), attention A and pending purchases Q.
# Win rate per bin saturates in bid, scaled by a conserved pool of rival capital K that
# drifts toward where we bid and where reachable people are, and is boosted where reach
# is thin. Budget pacing cuts win_rate by pace**kap. Purchase starts need follow-up
# exposure (k1) plus a spontaneous part (k0); broader audiences convert less (cq);
# completions share one fulfilment capacity F. Hidden state starts at the fixed reset
# convention (full reach, empty pipeline, uniform rivals); the initial reading is ignored.
AD_EDGES = [0.0, 0.55, 0.775, 1.0]
AD_D = [AD_EDGES[i + 1] - AD_EDGES[i] for i in range(len(AD_EDGES) - 1)]
DEFAULTS["ad_auction"] = dict(
    wmax=0.6264306850620251,    # win_rate ceiling at high bid (fresh reach, rival pressure 1)
    b0=3.748648874522589,       # bid scale of the win curve (scaled by local rival pressure K)
    g=0.35222224813857317,      # win boost where reachable people are thin
    vp=255.02490192982265,      # spend per unit reached-and-won at bid 1.5
    pe=0.26954681761367283,     # price exponent in bid
    kap=0.8966676216323695,     # pacing: win_rate x pace**kap (kap<1: bid shading, not pure throttling)
    f=0.1077324468008594,       # reach lost per step per unit exposure
    tauR=92.36064954122939,     # reach recovery time (steps)
    a=1.1479499540027378,       # attention gained per unit impressions (x100)
    k1=0.05129176694504404,     # follow-up: purchase starts per attention at exposure xr
    k0=0.11709938423992625,     # spontaneous purchase starts per attention
    xr=0.26,                    # reference exposure for k1 (fixed)
    k2=0.12437010286627279,     # pending -> completed rate
    F=5.3429898348924985,       # fulfilment capacity (conversions per step)
    v=0.0015854977437210016,    # converted customers made unavailable per conversion (per unit breadth)
    tauV=20.512622290026798,    # time for converted customers to return (steps)
    cq=2.641907315937337,       # conversion propensity falls as exp(-cq*x) across the audience
    phi=0.5948239341169798,     # price rises with local rival pressure K**phi
    rho=0.7081650682086872,     # rival capital pulled toward where we bid (conserved pool)
    tauK=33.02093949898502,     # rival capital relocation time (steps)
    mu=0.7060571762624779,      # rival capital pulled toward reachable people
)


def _ad_binavg_exp(c, e0, e1):
    if abs(c) < 1e-9:
        return 1.0
    return (math.exp(c * e1) - math.exp(c * e0)) / (c * (e1 - e0))


def start_ad(init, p):
    nb = len(AD_D)
    return dict(R=[1.0] * nb, V=[0.0] * nb, A=[0.0] * nb, Q=[0.0] * nb, K=[1.0] * nb,
                qx=[_ad_binavg_exp(-p["cq"], AD_EDGES[i], AD_EDGES[i + 1]) for i in range(nb)])


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
    for i in range(nb):
        D += k2 * Q[i]
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
# Fitted structural model (round E, R2_forms_sharedroom). Per community: core members K (the initial
# (1-f) share, never leave), loyal recruits L (organic + bridge introductions), locally
# seeded recruits S, incentive-led members J (the initial f share, plus converts).
# Seeding fills a 2-stage onboarding queue (`tq` steps per stage): A gets sa*seed*(1-br),
# B gets sb*seed, of which the bridge share br goes through a slower introduction queue
# (`tq2` per stage) into L. Seeded onboarding is throttled by one shared capacity `Nt`
# (A + B members). Incentive adds no recruits; while it is on, L and S convert to J at
# kc*u (u = incentive/2) and J stays; when it is off J leaves at lam*(1-u) (the ~25% start-up
# dip and the crash back to the core after any incentive period), S churns at lr*(1-u), and
# L and S churn at lamM*(E - incentive) while it sits below its expectation E (EMA, `tau_e`).
# Organic growth (a + b*A)*room goes to L; room is the shared capacity room.
# Untested: incentive held between 0 and 2 (research used only 0 and 2); there conversion (kc*u) and
# drain (lam*(1-u)) both run, so members slowly leak away.
DEFAULTS["social_contagion"] = dict(
    f=0.3937928344280045,         # share of initial members who are incentive-led (leave at reset)
    lam=0.0821023072374982,       # incentive-led drain per step at zero incentive
    tq=4.997774906771391,         # steps per local onboarding stage (2 stages)
    tq2=40.142792333100665,       # steps per bridge-introduction stage (2 stages)
    sa=1.025974357057521,         # A queue entries per unit seeding (times 1 - bridge)
    sb=0.34928196883019497,       # B queue entries per unit seeding (bridge share goes via introductions)
    Nt=440.7082879958301,         # shared onboarding capacity (A + B members)
    aa=0.7053333177480016,        # organic growth A per step (times room)
    ab=0.12972760227206098,       # organic growth B per step (times room)
    ba=-0.005548776280779653,     # organic growth per member A
    bb=0.003910881753617038,      # organic growth per member B
    kc=0.04327380200237239,       # recruits converted to incentive-led per step at full incentive
    lr=0.010966633225799948,      # seeded-recruit churn per step at zero incentive
    lamM=0.4775763072954847,      # recruit churn per step per unit of unmet incentive expectation
    tau_e=13.278237661719984,     # incentive expectation time constant (steps)
)


def start_social(init, p):
    Aa = _pos(_f(init, "adopters_a")); Ab = _pos(_f(init, "adopters_b"))
    f = p["f"]
    s = dict(E=0.0)
    for c, A0 in (("a", Aa), ("b", Ab)):
        s["K" + c] = (1 - f) * A0; s["J" + c] = f * A0
        for k in ("L", "S", "Q1", "Q2", "P1", "P2"):
            s[k + c] = 0.0
    return s


def adv_social(s, a, p):
    seed = _clip(_f(a, "seeding"), 0, 10); u = _clip(_f(a, "incentive"), 0, 2) / 2
    br = _clip(_f(a, "bridge_outreach"), 0, 1)
    drain = p["lam"] * (1 - u)
    s["E"] += (2 * u - s["E"]) / p["tau_e"]
    churnE = p["lamM"] * max(0.0, s["E"] - 2 * u)
    A = {c: s["K" + c] + s["L" + c] + s["S" + c] + s["J" + c] for c in ("a", "b")}
    roomS = max(0.0, 1 - (A["a"] + A["b"]) / p["Nt"])
    out = {}
    for c, loc, obs in (("a", p["sa"] * (1 - br), "adopters_a"), ("b", p["sb"] * (1 - br), "adopters_b")):
        g = seed * roomS
        ql = loc * g
        qx = p["sb"] * br * g if c == "b" else 0.0
        o1 = s["Q1" + c] / p["tq"]; o2 = s["Q2" + c] / p["tq"]
        s["Q1" + c] = s["Q1" + c] + ql - o1
        r1 = s["P1" + c] / p["tq2"]; r2 = s["P2" + c] / p["tq2"]
        s["P1" + c] = s["P1" + c] + qx - r1; s["P2" + c] = s["P2" + c] + r1 - r2
        s["Q2" + c] = s["Q2" + c] + o1 - o2
        L = s["L" + c]; S = s["S" + c]; J = s["J" + c]
        org = (p["a" + c] + p["b" + c] * A[c]) * roomS
        convL = p["kc"] * u * L; convS = p["kc"] * u * S
        leaveJ = drain * J
        leaveS = p["lr"] * (1 - u) * S
        dJ = convL + convS - leaveJ
        dS = o2 - convS - leaveS - churnE * S
        dL = r2 + org - convL - churnE * L
        s["J" + c] = max(0.0, J + dJ); s["S" + c] = max(0.0, S + dS); s["L" + c] = max(0.0, L + dL)
        out[obs] = s["K" + c] + s["L" + c] + s["S" + c] + s["J" + c]
    return out


START["social_contagion"], ADVANCE["social_contagion"] = start_social, adv_social


# ------------------------------------------------------------------ hospital queue
# Fitted fluid queue (round E, all research runs). Patients arrive (base `lam` plus
# electives into a list capped at `Emax`), wait, get admitted while occupancy < `Cs`,
# and are discharged from step `dead` on, at most `mu` work units per step; an elective
# needs `we` units. Arrivals past `Qmax` are referred elsewhere; waiting patients leave
# at rate `r`.
# mu = k * s_eff * diag_balance**hx * (1 + bo*ot) * (1 - fF*F) * (1 - phi*P)
#   s_eff: added staff, and staff moved between assessment and treatment by a diag
#          change (|d change| * staffing), are only `eta` effective for `To` steps.
#          Staff cuts are instant and remove the newest added staff first.
#   F: fatigue. Lags overtime * min(1, waiting / Wf) over tF steps, so overtime only
#      tires staff while patients are waiting.
#   P: follow-up program load, lags followup_capacity over tP steps (starts empty).
# wait = EMA(alpha) of cw * waiting / (smoothed discharges + reneging). urgent_priority ignored.
DEFAULTS["hospital_queue"] = dict(
    lam=11.515616926112909,     # base arrivals per step
    Emax=68.41379828715105,     # elective waiting-list cap
    Qmax=331.68039170150433,    # total queue cap (overflow referred elsewhere)
    r=0.016408766002698423,     # reneging fraction of waiting patients per step
    Cs=76.30531098409011,       # service occupancy cap (chairs + beds)
    k=1.0171433535619543,       # work per staff per step (regular-patient units) at balanced diag
    phi=0.4308252550576974,     # staff share diverted by a full follow-up program
    tP=64.21499393686628,       # follow-up program fill/empty time (steps)
    To=105.95822557579072,      # orientation time for added or moved staff (steps)
    eta=0.9247507525185236,     # effectiveness of staff during orientation
    bo=0.31680848046122956,     # overtime work boost at full overtime
    fF=0.654498202867752,       # capacity lost at full fatigue
    tF=44.716260056911,         # fatigue build/recovery time (steps)
    Wf=10.004613708556624,      # waiting patients at which overtime fully fatigues
    we=1.8049617307561252,      # work per elective patient (regular patient = 1)
    hx=0.6952436800831798,      # diag balance exponent (1 = linear tent peaked at 0.4)
    cw=2.416268718423094,       # wait scale
    tb=18.04323790268739,       # smoothing of recent discharges (steps)
    alpha=0.0836048260060405,   # wait EMA weight
    dead=2.0,                   # steps before the first discharge after reset
)


def start_hospital(init, p):
    return dict(W=max(0.0, _f(init, "queue")), E=0.0, Sr=0.0, Se=0.0, Nr=0.0, Ne=0.0,
                w=_f(init, "wait_time"), Dbar=_f(init, "discharges"), P=0.0, F=0.0,
                pend=[], mv=[], s_prev=20.0, d_prev=0.4, wprev=0.0, t=0)


def adv_hospital(s, a, p):
    st = _f(a, "staffing", 20.0); ot = _f(a, "overtime"); dg = _f(a, "diagnostic_allocation", 0.4)
    To = p["To"]
    # orientation cohorts [age, amount]: added staff (cuts remove the newest first) and moved staff
    pend, mv = s["pend"], s["mv"]
    ds = st - s["s_prev"]
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
    for c in pend:
        Np += c[1] * min(1.0, max(0.0, To - c[0]))
    dd = abs(dg - s["d_prev"]) * st
    if dd > 0:
        mv.append([0, dd])
    for c in mv:
        Np += c[1] * min(1.0, max(0.0, To - c[0]))
    Np = min(Np, st)
    s_eff = st - (1.0 - p["eta"]) * Np
    s["s_prev"] = st
    s["d_prev"] = dg
    s["P"] += (_f(a, "followup_capacity") - s["P"]) / max(p["tP"], 1.0)
    h = max(0.0, min(dg / 0.4, (1.0 - dg) / 0.6))
    if p["hx"] != 1.0:
        h = h ** p["hx"]
    s["F"] += (ot * min(1.0, s["wprev"] / max(p["Wf"], 1e-6)) - s["F"]) / max(p["tF"], 1.0)
    otf = (1.0 + p["bo"] * ot) * max(0.0, 1.0 - p["fF"] * s["F"])
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
