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
# empty, and a hard bed cap with a waiting list (hospital pins at ~156).
# Age groups ignored. Fitted on both research runs (scratchpad D/epidemic).
DEFAULTS["epidemic"] = dict(
    N=17421.053803070754,     # population
    s0=0.9778848532734722,    # susceptible fraction at reset
    rI=4.754999566339632,     # infectious per initial daily case
    beta=0.3805778967893401,  # transmission per step at full contact
    sigma=0.5599206789890795, # latent stage exit rate (2 stages)
    gamma=0.16486579572727397, # recovery rate per step
    omega=0.012901500007569764, # immunity waning rate per step
    a_s=0.18883987840434188,  # contact reduction at full school closure
    a_m=0.16468882532176157,  # exposure reduction at full mask mandate
    v_eff=0.5792619490903622, # fraction of vaccination_rate*S immunized per step
    h=0.04248979959389576,    # hospital referrals per onset
    k_c=0.19679882448224792,  # referral pipeline rate per step
    d=0.06868069865781783,    # hospital discharge rate per step
    cap=158.20990081918694,   # hard bed cap
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
    b = p["beta"] * (1 - p["a_s"] * _f(a, "school_closure")) * (1 - p["a_m"] * _f(a, "mask_mandate"))
    S, E1, E2, I, R = s["S"], s["E1"], s["E2"], s["I"], s["R"]
    inf = min(b * S * I / N, S)
    vax = min(p["v_eff"] * _f(a, "vaccination_rate") * S, S - inf)
    on = p["sigma"] * E2      # new onsets = daily_cases
    mv = p["sigma"] * E1
    rec = p["gamma"] * I
    wane = p["omega"] * R
    s["S"] = S + wane - inf - vax
    s["E1"] = E1 + inf - mv
    s["E2"] = E2 + mv - on
    s["I"] = I + on - rec
    s["R"] = R + rec + vax - wane
    ref = p["k_c"] * s["C"]   # referrals leave the clinical pipeline
    s["C"] += p["h"] * on - ref
    s["W"] += ref             # and wait for a bed
    Hd = (1 - p["d"]) * s["H"]
    adm = min(s["W"], _pos(p["cap"] - Hd))
    s["W"] -= adm
    s["H"] = Hd + adm
    return {"daily_cases": on, "hospital_load": s["H"]}


START["epidemic"], ADVANCE["epidemic"] = start_epidemic, adv_epidemic


# ------------------------------------------------------------------ market
# Refit textbook: price follows its target through a two-stage lag (lag, ramp, plateau),
# volume and depth relax first-order. Calm levels are fitted constants, not the first reading.
# a_tax_v and a_rate_v are tied (each gives half the joint-pulse volume effect): the data
# only ever moved both controls together. Keep a_tax_v = 2 * a_rate_v when refitting.
DEFAULTS["market"] = dict(
    p0=94.19155717682646,     # calm price level (0 = take from initial)
    v0=2.265499698920218,     # calm volume level (0 = take from initial)
    d0=89.81421259466852,     # calm depth level (0 = take from initial)
    a_rate_p=4.149757184759864,     # price target falls a_rate_p * interest_rate
    a_tax_v=0.49142246420393343,    # volume target falls a_tax_v * tax
    a_rate_v=0.24571123210196671,   # volume target falls a_rate_v * interest_rate
    a_vol_d=-0.017663002652874894,  # depth target change per unit of volume above baseline
    a_tax_d=11.221344173754414,     # depth target falls a_tax_d * tax
    k_p=0.02347471604832667,        # price lag stage 1 (fraction of gap closed per step)
    k_p2=0.030537635487408933,      # price lag stage 2
    k_v=0.3097084357024772, k_d=0.10824352228498302,
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
    s["pf"] += _clip(p["k_p"], 0, 1) * (tp - s["pf"])
    s["price"] += _clip(p["k_p2"], 0, 1) * (s["pf"] - s["price"])
    s["volume"] += _clip(p["k_v"], 0, 1) * (tv - s["volume"])
    s["depth"] += _clip(p["k_d"], 0, 1) * (td - s["depth"])
    return {"price": s["price"], "volume": s["volume"], "depth": s["depth"]}


START["market"], ADVANCE["market"] = start_market, adv_market


# ------------------------------------------------------------------ traffic
# Queue model fitted to research data (Sep 25). Per route r in (a, b):
#   arrivals  = lam_r * ramp * (1 - ct * toll/5)   ramp = admitted demand; recovery (ramp 0) = empty road
#   pipeline  arrivals reach the junction queue after D_r steps (11 / 16 measured)
#   capacity  = C_r * share_r * (1 - l_r * lane)   share_a = signal_timing, share_b = 1 - signal_timing
#   queue     Q_r <= qmax_r, flow_r = min(Q_r, capacity)
#   speed     EMA toward (vfree_r - M_r) / (1 + N_r / qref_r), N_r = queue + pipeline, M_b = 0
#   memory    M_a += g * N_a / (N_a + qref_a) * (Mmax - M_a), never decays (speed_a stays low after a pulse)
# The initial flow reading is ignored: roads start empty.
DEFAULTS["traffic"] = dict(
    D_a=11.0,
    D_b=16.0,
    ct=0.5,
    lam_a=10.761537575035316,
    lam_b=13.196740732446036,
    C_a=36.021986324205,
    C_b=26.16488827228393,
    l_a=8.006221324097651e-14,
    l_b=0.7030652840548853,
    qmax_a=526.9070032096531,
    qmax_b=142.38097597697734,
    qref_a=82.58281497622234,
    qref_b=118.11907626201868,
    vfree_a=48.89039995981071,
    vfree_b=48.76096955522898,
    alpha=0.2063879841815445,
    g=0.01750648103431229,
    Mmax=2.7667276406306756,
)


def start_traffic(init, p):
    s = {"M": 0.0}
    for r in ("a", "b"):
        v = _f(init, "speed_" + r, p["vfree_" + r])
        s["V" + r] = v if math.isfinite(v) else p["vfree_" + r]
        s["Q" + r] = 0.0
        s["P" + r] = [0.0] * max(int(round(p["D_" + r])), 1)
        s["F" + r] = 0.0   # vehicles in the pipeline
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
        n = s["Q" + r] + _pos(s["F" + r])
        if r == "a":
            s["M"] += p["g"] * n / (n + p["qref_a"]) * (p["Mmax"] - s["M"])
        tgt = (p["vfree_" + r] - (s["M"] if r == "a" else 0.0)) / (1 + n / p["qref_" + r])
        s["V" + r] += p["alpha"] * (tgt - s["V" + r])
        out["flow_" + r] = served
        out["speed_" + r] = s["V" + r]
    return out


START["traffic"], ADVANCE["traffic"] = start_traffic, adv_traffic


# ------------------------------------------------------------------ power grid
# Textbook: price-responsive load with deferred-demand rebound, reserves with a
# state of charge, governor droop on frequency, interconnector that heats with use.
DEFAULTS["power_grid"] = dict(
    L0=0.0,            # baseline load at price 0.8; 0 = take from initial
    a_price=0.15,      # load target drops a_price per unit price above 0.8
    k_load=0.10,       # load response speed
    k_def=0.05,        # deferred demand builds while load is below baseline ...
    k_rel=0.02,        # ... and leaks away ...
    k_reb=0.5,         # ... and pushes the load target back up (rebound)
    f0=0.0,            # nominal frequency; 0 = take from initial
    k_freq=0.002,      # Hz per unit power imbalance per step
    k_fdamp=0.10,      # frequency damping
    droop=20.0,        # extra conventional generation per Hz below nominal
    k_gov=0.05,        # governor response speed
    soc_max=600.0,     # reserve energy capacity
    charge_rate=10.0,  # refill per step at full charging allowance
    r_local=0.0,       # local renewable as fraction of L0; 0 = from initial share
    r_remote=0.0,      # remote renewable via interconnector as fraction of L0
    ic_heat=0.02, ic_cool=0.05, ic_derate=0.5,   # interconnector thermal limit
)


def start_power_grid(init, p):
    load = _f(init, "load", 100.0)
    L0 = p["L0"] if p["L0"] > 0 else max(load, 1e-6)
    f0 = p["f0"] if p["f0"] > 0 else _f(init, "frequency", 50.0)
    share0 = _clip(_f(init, "renewable_share"), 0, 1)
    r_local = p["r_local"] if p["r_local"] > 0 else share0
    gen0 = _pos(L0 * (1 - r_local))
    return dict(L=load, L0=L0, f0=f0, df=_f(init, "frequency", f0) - f0, D=0.0,
                gen=gen0, gen0=gen0, soc=p["soc_max"], temp=0.0,
                R_local=r_local * L0, R_remote=p["r_remote"] * L0)


def adv_power_grid(s, a, p):
    price = _f(a, "price_signal", 0.8)
    res = _pos(_f(a, "reserve_dispatch"))
    ch = _clip(_f(a, "charging_allowance"), 0, 1)
    ic = _clip(_f(a, "interconnector"), 0, 1)
    L0 = s["L0"]
    L_target = L0 * (1 - p["a_price"] * (price - 0.8)) + p["k_reb"] * s["D"]
    s["L"] += _clip(p["k_load"], 0, 1) * (L_target - s["L"])
    s["D"] = _pos(s["D"] + p["k_def"] * (L0 - s["L"]) - p["k_rel"] * s["D"])
    out = min(res, s["soc"])
    s["soc"] -= out
    charge = ch * p["charge_rate"] * _pos(1 - s["soc"] / max(p["soc_max"], 1e-6))
    s["soc"] += charge
    remote = s["R_remote"] * ic * _pos(1 - p["ic_derate"] * s["temp"])
    s["temp"] = _clip(s["temp"] + p["ic_heat"] * ic - p["ic_cool"] * s["temp"], 0, 1)
    gen_target = _pos(s["gen0"] - p["droop"] * s["df"])
    s["gen"] += _clip(p["k_gov"], 0, 1) * (gen_target - s["gen"])
    supply = s["gen"] + out + s["R_local"] + remote
    demand = s["L"] + charge
    s["df"] += p["k_freq"] * (supply - demand) - _clip(p["k_fdamp"], 0, 1) * s["df"]
    share = (s["R_local"] + remote) / max(supply, 1e-6)
    return {"load": demand, "frequency": s["f0"] + s["df"], "renewable_share": _clip(share, 0, 1)}


START["power_grid"], ADVANCE["power_grid"] = start_power_grid, adv_power_grid


# ------------------------------------------------------------------ supply chain
# Fitted to research/supply_chain.json (2 runs). Chain: production (2-step delay) -> supplier
# stock (ceiling) -> orders withdraw only stock above a reserve -> dispatch queue (withdrawals
# stop when it is full: congestion) -> forward transport (fixed rate, 3-step conveyor) ->
# receiving buffer -> receiving (rate x receiving_effort) -> retail -> sales.
# Conveyors and buffers start empty, so shipments read 0 until orders flow.
DEFAULTS["supply_chain"] = dict(
    kp=39.86426356591622,       # production per step at production_effort 1, maintenance 0
    dm=0.7247879406366704,      # share of production lost at full maintenance
    s_cap=361.8,                # supplier stock ceiling (calm reading)
    s_res=287.3101903821416,    # supplier stock that orders cannot withdraw
    q_max=1196.542781990602,    # dispatch queue size where new withdrawals stop
    trans=19.563455582353996,   # forward transport per step
    kr=53.199612412715624,      # receiving per step per unit receiving_effort
    dem=27.7,                   # retail sales per step (calm drain rate)
    b_max=100.0,                # receiving buffer size where transport stops (guard; data peak ~91)
)


def start_supply(init, p):
    return dict(s=_pos(_f(init, "inventory_supplier")), r=_pos(_f(init, "inventory_retail")),
                pp=[0.0, 0.0], q=0.0, conv=[0.0, 0.0, 0.0], b=0.0)


def adv_supply(s, a, p):
    oq = _clip(_f(a, "order_quantity"), 0.0, 80.0)
    pe = _clip(_f(a, "production_effort", 1.0), 0.0, 1.5)
    re = _clip(_f(a, "receiving_effort", 1.0), 0.0, 1.5)
    mt = _clip(_f(a, "maintenance"), 0.0, 1.0)
    s["pp"].append(p["kp"] * pe * _pos(1.0 - p["dm"] * mt))
    s["s"] = min(s["s"] + s["pp"].pop(0), max(s["s"], p["s_cap"]))
    w = min(oq, _pos(s["s"] - p["s_res"]), _pos(p["q_max"] - s["q"]))
    s["s"] -= w
    s["q"] += w
    t = min(s["q"], p["trans"], _pos(p["b_max"] - s["b"]))
    s["q"] -= t
    s["conv"].append(t)
    s["b"] += s["conv"].pop(0)
    recv = min(s["b"], p["kr"] * re)
    s["b"] -= recv
    s["r"] = _pos(s["r"] + recv - p["dem"])
    return {"shipments": recv, "inventory_supplier": s["s"], "inventory_retail": s["r"]}


START["supply_chain"], ADVANCE["supply_chain"] = start_supply, adv_supply


# ------------------------------------------------------------------ wildlife
# Prey + hidden food stock per region (fitted to research/wildlife.json, both runs).
# Food refills while prey is low, which gives the +60% prey overshoot after every recovery.
# Harvest per capita saturates at low prey (refuge), so a pulse floors prey instead of 0.
# Predators relax slowly; the counted share P/(P+Hv) reacts fast to prey crashes.
DEFAULTS["wildlife"] = dict(
    b=0.26827089890725925,  # prey birth rate per unit food
    mu=0.09590513736464606,  # prey death rate
    rho_n=0.023275396046376906,  # food renewal, north
    rho_s=0.02360487013137204,  # food renewal, south
    hk_n=0.3283619188222149,  # habitat boost to food renewal, north
    hk_s=6.72860722906197e-08,  # habitat boost to food renewal, south
    cons=0.00033999569281078357,  # food eaten per prey
    F0=0.7323078893157154,  # food level at reset (fraction of capacity)
    hq=0.02405674656053212,  # harvest per unit quota
    Ph=6.035700056359326,  # harvest refuge: prey level where harvest halves per capita
    a=0.056073609737726744,  # predator growth at abundant prey
    Hp=0.7561595746184241,  # prey level for half predator growth
    m=0.034363976966760294,  # predator death rate
    k=0.008970282476489611,  # predator crowding
    mig=0.0017185230119157672,  # corridor mixing rate at full access
    Hv=1.9543913481754216,  # prey level where half the predators are counted
    Pb=655.1621846319042,  # prey crowding of births
)


def start_wildlife(init, p):
    pn = _pos(_f(init, "prey_north")); ps = _pos(_f(init, "prey_south"))
    hv = p["Hv"]
    dn = _pos(_f(init, "predator_north")) * (pn + hv) / max(pn, 1e-6)
    ds = _pos(_f(init, "predator_south")) * (ps + hv) / max(ps, 1e-6)
    f0 = min(p["F0"], 1.0)
    return dict(pn=pn, ps=ps, dn=dn, ds=ds, fn=f0, fs=f0)


def adv_wildlife(s, a, p):
    quota = _clip(_f(a, "hunting_quota", 0.0), 0.0, 8.0)
    hab = _clip(_f(a, "habitat_protection", 1.0), 0.0, 1.0)
    cor = _clip(_f(a, "corridor_access", 0.0), 0.0, 1.0)
    for reg in ("n", "s"):
        prey = s["p" + reg]; food = s["f" + reg]; pred = s["d" + reg]
        rho = p["rho_" + reg] * (1 + p["hk_" + reg] * hab)
        s["f" + reg] = min(max(food + rho * (1 - food) - p["cons"] * prey * food, 0.0), 1.0)
        harvest = p["hq"] * quota * prey * prey / (prey + p["Ph"])
        birth = p["b"] * food / (1.0 + prey / p["Pb"])
        s["p" + reg] = max(prey + prey * (birth - p["mu"]) - harvest, 1e-6)
        s["d" + reg] = max(pred + pred * (p["a"] * prey / (prey + p["Hp"]) - p["m"] - p["k"] * pred), 1e-6)
    fp = p["mig"] * cor * (s["pn"] - s["ps"])
    fd = p["mig"] * cor * (s["dn"] - s["ds"])
    s["pn"] -= fp; s["ps"] += fp
    s["dn"] -= fd; s["ds"] += fd
    hv = p["Hv"]
    return {"prey_north": s["pn"], "predator_north": s["dn"] * s["pn"] / (s["pn"] + hv),
            "prey_south": s["ps"], "predator_south": s["ds"] * s["ps"] / (s["ps"] + hv)}


START["wildlife"], ADVANCE["wildlife"] = start_wildlife, adv_wildlife


# ------------------------------------------------------------------ reservoir
# Grey-box fit to research data (Sep 25). Seasonal river inflow (sinusoid in steps since reset),
# water balance with a hard spillway cap (excess leaves as spill in outflow), head-limited
# delivery min(request, c0 + c1*level, water available), loss e0 + e1*level, irrigation
# return flow through a two-tank delay (rt steps, fraction rf), and a slow stress memory m
# that pulls quality down. m is driven by u = mean of each control's 0..1 position from the
# brief's recovery action to its pulse action.
DEFAULTS["reservoir"] = dict(
    A=11.2652372008, B=2.2341170372, P=67.8564523494, phi=0.0452866329,   # inflow = A + B*sin(2*pi*t/P + phi) + return flow
    c0=8.9207735962, c1=0.0074794656,                  # delivery cap c0 + c1*level
    e0=1.0995387127, e1=0.0005090763,                  # loss per step e0 + e1*level
    Lcap=941.0,                                   # spillway level (measured, held fixed in the fit)
    qc=0.9555117055, qa=0.0629294055, tq=4.3817157529,       # calm quality, memory weight, start-up time constant
    g0=0.0011331768, g=0.0056554373, d=0.0052646445,    # memory: calm drive, stress drive, decay
    rf=0.0497159246, rt=8.022753209,                  # irrigation return fraction and delay
)


def start_reservoir(init, p):
    return dict(t=0, level=_pos(_f(init, "level")), q0=_f(init, "quality"),
                m=0.0, s1=0.0, s2=0.0)


def adv_reservoir(s, a, p):
    aer = _f(a, "aeration", 1.0); irr = _pos(_f(a, "irrigation_allocation"))
    rel = _pos(_f(a, "release_rate", 2.0)); depth = _f(a, "withdrawal_depth")
    s["t"] += 1; t = s["t"]; L = s["level"]; rt = p["rt"]
    s["s1"] += irr - s["s1"] / rt
    s["s2"] += s["s1"] / rt - s["s2"] / rt
    inflow = p["A"] + p["B"] * math.sin(2 * math.pi * t / p["P"] + p["phi"]) + p["rf"] * s["s2"] / rt
    loss = p["e0"] + p["e1"] * L
    dlv = min(rel + irr, max(p["c0"] + p["c1"] * L, 0.0), max(L + inflow - loss, 0.0))
    L = L + inflow - dlv - loss
    spill = 0.0
    if L > p["Lcap"]:
        spill = L - p["Lcap"]; L = p["Lcap"]
    if L < 0:
        L = 0.0
    s["level"] = L
    u = _clip(((aer - 1.0) / (0.0 - 1.0) + (irr - 0.0) / (8.0 - 0.0)
               + (rel - 2.0) / (12.0 - 2.0) + (depth - 0.0) / (1.0 - 0.0)) / 4.0, 0.0, 1.0)
    m = s["m"]
    s["m"] = m + (p["g0"] + p["g"] * u) * (1 - m) - p["d"] * m
    q = p["qc"] - p["qa"] * s["m"] + (s["q0"] - p["qc"]) * math.exp(-t / p["tq"])
    return {"level": L, "inflow": inflow, "outflow": dlv + spill, "quality": q}


START["reservoir"], ADVANCE["reservoir"] = start_reservoir, adv_reservoir


# ------------------------------------------------------------------ ad auction
# Structural model fitted to research runs 1+2. Hidden state starts at the fixed reset
# convention (full reach, empty pipeline); the initial reading is ignored.
# reach R depleted by exposure, unavailable pool V of converted customers, budget pacing,
# rivals back off as reach thins (g), attention A -> pending purchases Q -> fulfilment cap F.
DEFAULTS["ad_auction"] = dict(
    wmax=0.45874519488492105,   # win_rate ceiling at high bid
    b0=2.821663064831036,       # bid scale of the win curve
    g=0.5591357598462097,       # win_rate boost as available reach thins
    vp=263.495190775299,        # spend scale at bid 1.5
    pe=0.5311882905519455,      # price exponent in bid
    f=0.1708015179227022,       # reach lost per step per unit exposure
    tauR=38.8664257197138,      # reach recovery time (steps)
    a=0.6271373826674157,       # attention gained per unit impressions (x100)
    k1=0.15954119749096352,     # attention -> started purchase rate
    k2=0.12256620140696878,     # pending -> completed rate
    F=5.307863115442049,        # fulfilment cap (conversions per step)
    v=0.002177616146042035,     # converted customers made unavailable per conversion
    tauV=40.71399300675976,     # time for converted customers to return (steps)
)


def start_ad(init, p):
    return dict(R=1.0, A=0.0, Q=0.0, V=0.0)


def adv_ad(s, a, p):
    b = _pos(_f(a, "bid", 1.5)); cap = _pos(_f(a, "budget_cap", 20.0))
    w = _clip(_f(a, "targeting_breadth", 0.55), 0, 1)
    avail = _pos(s["R"] - s["V"])
    wr = p["wmax"] * (1 - math.exp(-b / max(p["b0"], 1e-9))) * (1 + p["g"] * (1 - avail))
    wr = min(wr, 1.0)
    price = p["vp"] * (b / 1.5) ** p["pe"]
    S = price * w * avail * wr
    pace = min(1.0, cap / S) if S > 0 else 1.0
    spend = S * pace; wro = wr * pace
    imps = w * avail * wro
    conv = min(p["k2"] * s["Q"], p["F"])
    started = p["k1"] * s["A"]
    s["Q"] += started - conv
    s["A"] += p["a"] * imps * 100 - started
    R = s["R"]
    R += -p["f"] * wro * avail * R + (1 - R) / p["tauR"]
    s["R"] = _clip(R, 0.0, 1.0); s["V"] = _pos(s["V"] + p["v"] * conv - s["V"] / p["tauV"])
    return {"win_rate": wro, "spend": spend, "conversions": conv}


START["ad_auction"], ADVANCE["ad_auction"] = start_ad, adv_ad


# ------------------------------------------------------------------ social contagion
# Fitted structural model (structural_v1). Each community has loyal members M and
# incentive-led members J. J leaves at rate `lam` when incentive is off (the ~25%
# start-up dip and the post-pulse crash). Seeding x (1 + incentive) sends people into
# a 2-stage onboarding queue (`tq` steps per stage), throttled by room left in a
# finite pool (`Na`, `Nb`); bridge outreach splits seeding between A (1-br) and B (br).
# Graduates join J in share phi*incentive/2, else M. Organic growth (a + b*M)*room.
# E = incentive expectation (EMA over `tau_e`); when incentive falls below it, loyal
# members churn at lamM*(E - incentive) (the undershoot below the pre-pulse level).
DEFAULTS["social_contagion"] = dict(
    f=0.4422848322457665,        # share of initial members who are incentive-led
    lam=0.08240899902833297,     # incentive-led drain per step at zero incentive
    tq=3.9837598952545807,       # steps per onboarding stage (2 stages)
    sa=0.6612618471037612,       # queue entries per unit seeding in A
    sb=0.15333397405304403,      # queue entries per unit seeding in B
    phi=0.9414663451326307,      # share of graduates who are incentive-led at full incentive
    Na=203.9363013809878,        # pool size A
    Nb=146.51461482935395,       # pool size B
    aa=0.7094417818728652,       # organic growth A per step (times room)
    ab=0.34448439254204555,      # organic growth B per step (times room)
    ba=-0.006154526865592056,    # organic growth per loyal member A
    bb=9.377757789382251e-05,    # organic growth per loyal member B
    lamM=0.04507760515500967,    # loyal churn per step per unit of unmet incentive expectation
    tau_e=9.891084543251234,     # incentive expectation time constant (steps)
)


def start_social(init, p):
    Aa = _pos(_f(init, "adopters_a")); Ab = _pos(_f(init, "adopters_b"))
    return dict(Ja=p["f"] * Aa, Jb=p["f"] * Ab, Ma=(1 - p["f"]) * Aa, Mb=(1 - p["f"]) * Ab,
                Q1a=0.0, Q1b=0.0, Q2a=0.0, Q2b=0.0, E=0.0)


def adv_social(s, a, p):
    seed = _clip(_f(a, "seeding"), 0, 10); inc = _clip(_f(a, "incentive"), 0, 2)
    br = _clip(_f(a, "bridge_outreach"), 0, 1)
    s["E"] += (inc - s["E"]) / p["tau_e"]
    jfrac = p["phi"] * min(inc / 2, 1)
    drain = p["lam"] * max(0.0, 1 - inc / 2)
    churn = p["lamM"] * max(0.0, s["E"] - inc)
    out = {}
    for c, share, obs in (("a", 1 - br, "adopters_a"), ("b", br, "adopters_b")):
        M = s["M" + c]; J = s["J" + c]
        room = max(0.0, 1 - (M + J) / p["N" + c])
        q_in = p["s" + c] * seed * share * (1 + inc) * room
        o1 = s["Q1" + c] / p["tq"]; o2 = s["Q2" + c] / p["tq"]
        s["Q1" + c] += q_in - o1; s["Q2" + c] += o1 - o2
        dJ = o2 * jfrac - drain * J
        dM = o2 * (1 - jfrac) + (p["a" + c] + p["b" + c] * M) * room - churn * M
        s["J" + c] = max(0.0, J + dJ); s["M" + c] = max(0.0, M + dM)
        out[obs] = s["M" + c] + s["J" + c]
    return out


START["social_contagion"], ADVANCE["social_contagion"] = start_social, adv_social


# ------------------------------------------------------------------ hospital queue
# Fitted fluid queue (winner "minimal"). Patients arrive (base `lam` plus electives
# into a list capped at `Emax`), wait, get admitted while occupancy < `Cs`, and are
# discharged from step `dead` on at most `mu` per step. Arrivals past `Qmax` are
# referred elsewhere; waiting patients leave at rate `r`.
# mu = k * (staffing - N) * diag_balance * (1 + g*overtime) * (1 - phi*P)
#   N: handover (hO staff-equivalents per added staff, decays over th steps).
#   P: follow-up program load, lags followup_capacity over tP steps.
# wait = EMA(alpha) of cw * waiting / (smoothed discharges + reneging). urgent_priority ignored.
DEFAULTS["hospital_queue"] = dict(
    lam=11.452924253952201,    # base arrivals per step
    Emax=56.430645440203776,   # elective waiting-list cap
    Qmax=330.1859055462481,    # total queue cap
    r=0.013778164740254234,    # reneging fraction per step
    Cs=74.26987901138362,      # service occupancy cap
    k=1.365737870516677,       # discharges per step per staff at balanced diag
    g=-0.4162710282890295,     # net overtime effect on capacity
    hO=1.1594629833006804,     # staff-equivalents lost per added staff
    th=21.785354113531003,     # orientation recovery time (steps)
    phi=0.5623421955576099,    # staff share diverted by a full follow-up program
    tP=11.031247309233406,     # follow-up program fill/empty time (steps)
    cw=2.316957775540321,      # wait scale
    tb=21.327243176884295,     # smoothing of recent discharges (steps)
    alpha=0.09782701141000551, # wait EMA weight
    dead=3.0,                  # steps before the first discharge after reset
)


def start_hospital(init, p):
    return dict(W=max(0.0, _f(init, "queue")), E=0.0, S=0.0, Snew=0.0,
                w=_f(init, "wait_time"), Dbar=_f(init, "discharges"), N=0.0, P=0.0,
                s_prev=20.0, t=0, decN=math.exp(-1.0 / max(p["th"], 1e-6)))


def adv_hospital(s, a, p):
    st = _f(a, "staffing", 20.0); ot = _f(a, "overtime"); dg = _f(a, "diagnostic_allocation", 0.4)
    s["N"] = min(st, max(0.0, s["N"] + p["hO"] * (st - s["s_prev"])))
    s["s_prev"] = st
    s["P"] += (_f(a, "followup_capacity") - s["P"]) / max(p["tP"], 1.0)
    h = max(0.0, min(dg / 0.4, (1.0 - dg) / 0.6))
    mu = max(0.0, p["k"] * (st - s["N"]) * h * (1.0 + p["g"] * ot) * (1.0 - p["phi"] * s["P"]))
    W, E, S = s["W"], s["E"], s["S"]
    Qprev = W + E + S + s["Snew"]
    D = min(S, mu) if s["t"] >= p["dead"] else 0.0
    S += s["Snew"] - D
    e_in = min(max(0.0, _f(a, "elective_scheduling")), max(0.0, p["Emax"] - E))
    arr = p["lam"] + e_in
    room = max(0.0, p["Qmax"] - Qprev)
    if arr > room:
        e_in *= room / arr
        arr = room
    W += arr - e_in
    E += e_in
    tot = W + E
    Snew = min(tot, max(0.0, p["Cs"] - S))
    if tot > 0.0:
        keep = 1.0 - Snew / tot
        W *= keep
        E *= keep
    W *= 1.0 - p["r"]
    E *= 1.0 - p["r"]
    waiting = W + E
    s["Dbar"] += (D - s["Dbar"]) / max(p["tb"], 1.0)
    out_rate = s["Dbar"] + p["r"] * waiting
    T = p["cw"] * waiting / out_rate if out_rate > 1e-9 else 0.0
    s["w"] += p["alpha"] * (T - s["w"])
    s["N"] *= s["decN"]
    s["W"], s["E"], s["S"], s["Snew"] = W, E, S, Snew
    s["t"] += 1
    return {"wait_time": s["w"], "queue": W + E + S + Snew, "discharges": D}


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
