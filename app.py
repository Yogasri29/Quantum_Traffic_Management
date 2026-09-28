"""
Quantum Smart Traffic Management  -  final build (robust quantum layer)
=======================================================================
Run:   pip install -r requirements.txt
       streamlit run app.py

Everything is a SIMULATION (synthetic traffic, no real signal control).
The QAOA layer uses Qiskit when it imports cleanly; otherwise it automatically falls back to a
built-in NumPy statevector QAOA simulator, so the app never breaks on a missing/incompatible qiskit.
"""
import copy
import heapq
import html
import importlib
import importlib.util
import math
import sqlite3
import time
from collections import defaultdict, deque
from datetime import datetime
from itertools import product

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Quantum Smart Traffic Management",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CONSTANTS
# ============================================================
BASE_TRAFFIC = {
    "J1": {"North": 40, "South": 25, "East": 60, "West": 30},
    "J2": {"North": 20, "South": 50, "East": 35, "West": 45},
    "J3": {"North": 70, "South": 40, "East": 25, "West": 55},
    "J4": {"North": 30, "South": 65, "East": 45, "West": 20},
}
PLANS = {"P1": 20, "P2": 30, "P3": 40, "P4": 50}      # North-South green seconds
CYCLE = 60
SATURATION = 220
JUNCTION_CAPACITY = 220
DIRECTIONS = ["North", "South", "East", "West"]
JUNCTIONS = list(BASE_TRAFFIC)
BASELINE_PLAN = {j: "P2" for j in JUNCTIONS}

ROADS = {
    ("J1", "J2"): {"km": 2.0, "speed": 50, "cap": 120, "axis": "NS"},
    ("J2", "J3"): {"km": 2.2, "speed": 50, "cap": 110, "axis": "NS"},
    ("J3", "J4"): {"km": 2.0, "speed": 50, "cap": 120, "axis": "EW"},
    ("J1", "J4"): {"km": 3.0, "speed": 60, "cap": 150, "axis": "EW"},
}
NETWORK_EDGES = list(ROADS)
NODE_XY = {"J1": (155, 220), "J2": (400, 90), "J4": (645, 220), "J3": (400, 350)}
EDGE_SHARE = 0.5
EMERGENCY_SPEED_FACTOR = 0.8
PREEMPT_LEAD = 10
PREEMPT_CLEAR = 8

ADAPTIVE_SCENARIOS = {
    "Normal": {"factors": {"North": 1.00, "South": 1.00, "East": 1.00, "West": 1.00},
               "priority": "Normal", "focus": "Balanced network demand",
               "description": "Reference traffic condition."},
    "Morning School Rush": {"factors": {"North": 1.55, "South": 1.45, "East": 0.85, "West": 0.90},
                            "priority": "School / College", "focus": "North / South school-bus demand",
                            "description": "Simulation overlay for morning school and college movement."},
    "Morning Office Rush": {"factors": {"North": 1.25, "South": 1.15, "East": 1.10, "West": 1.00},
                            "priority": "Office Rush", "focus": "Mixed inbound commuter demand",
                            "description": "Simulation overlay for morning office commuting."},
    "Evening Office Rush": {"factors": {"North": 0.90, "South": 0.85, "East": 1.45, "West": 1.50},
                            "priority": "Office Rush", "focus": "East / West commuter demand",
                            "description": "Simulation overlay for evening office traffic."},
    "School + Office": {"factors": {"North": 1.35, "South": 1.25, "East": 1.25, "West": 1.30},
                        "priority": "School + Office", "focus": "Combined school and commuter demand",
                        "description": "Combined school/college and office traffic experiment."},
    "Festival / Shopping": {"factors": {"North": 1.10, "South": 1.05, "East": 1.45, "West": 1.50},
                            "priority": "Festival / Event", "focus": "Shopping and event activity",
                            "description": "Simulation overlay for event and shopping-season demand."},
    "Heavy Rain": {"factors": {"North": 1.15, "South": 1.15, "East": 1.15, "West": 1.15},
                   "priority": "Weather", "focus": "Uniform demand increase",
                   "description": "Synthetic high-demand weather scenario."},
    "Road Incident": {"factors": {"North": 1.00, "South": 1.00, "East": 1.75, "West": 0.85},
                      "priority": "Incident", "focus": "East approach disruption",
                      "description": "Synthetic incident affecting the East approach."},
}

DEFAULT_STATE = {
    "traffic": {j: dict(d) for j, d in BASE_TRAFFIC.items()},
    "traffic_version": 0,
    "scenario": "Normal",
    "qaoa_results": {},
    "network_plan": None,
    "network_score": None,
    "experiment_log": [],
    "adaptive_results": None,
    "last_action": "System initialized",
    "page": "Overview",
    "emergency_active": False,
    "emergency_completed": False,
    "ambulance_progress": 0,
    "em_origin": "J1",
    "em_dest": "J4",
    "blocked_labels": [],
    "od_origin": "J1",
    "od_dest": "J3",
    "bn_origin": "J1",
    "bn_dest": "J4",
    "sim_minute": 0,
}
for _k, _v in DEFAULT_STATE.items():
    if _k not in st.session_state:
        st.session_state[_k] = copy.deepcopy(_v)

# ============================================================
# STYLE
# ============================================================
st.markdown("""
<style>
.stApp{background:radial-gradient(circle at 5% 5%,rgba(0,220,255,.10),transparent 28%),radial-gradient(circle at 95% 10%,rgba(130,80,255,.12),transparent 30%),#070b14;color:#edf7ff}
.block-container{max-width:1450px;padding-top:1.2rem;padding-bottom:4rem}
h1,h2,h3{color:#f4f8ff!important}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#090e1a,#070b14);border-right:1px solid rgba(85,230,255,.12)}
.stButton>button,.stDownloadButton>button{border-radius:12px!important;border:1px solid rgba(85,230,255,.35)!important;background:linear-gradient(135deg,#087eaa,#6546b8)!important;color:white!important;font-weight:700!important}
[data-testid="stSidebar"] .stButton>button{font-size:14px;padding:.35rem .6rem;text-align:left}
[data-testid="stSidebar"] button[kind="primary"],[data-testid="stSidebar"] [data-testid="stBaseButton-primary"]{background:linear-gradient(135deg,#ffd54a,#ff9f1c)!important;border:2px solid #fff0a6!important;color:#241600!important;box-shadow:0 0 18px rgba(255,180,40,.5)}
.card{border:1px solid rgba(85,230,255,.16);border-radius:20px;background:linear-gradient(135deg,rgba(18,31,52,.86),rgba(9,14,27,.82));box-shadow:0 15px 45px rgba(0,0,0,.25);padding:20px;margin-bottom:14px}
.hero{border:1px solid rgba(85,230,255,.20);border-radius:24px;background:radial-gradient(circle at 85% 20%,rgba(85,230,255,.13),transparent 30%),linear-gradient(135deg,rgba(15,30,55,.95),rgba(8,13,25,.95));padding:35px;margin-bottom:22px}
.kicker{color:#55e6ff;letter-spacing:2px;font-size:12px;font-weight:700}.big{font-size:clamp(35px,5vw,60px);font-weight:800;line-height:1.02}
.badge{display:inline-block;padding:5px 9px;border-radius:999px;background:rgba(85,230,255,.08);border:1px solid rgba(85,230,255,.18);color:#8deeff;font-size:11px;font-weight:700}
.metric-card{border:1px solid rgba(85,230,255,.16);border-radius:18px;background:rgba(12,20,35,.78);padding:17px;min-height:110px}.metric-label{color:#8298ad;font-size:11px;letter-spacing:1.2px;font-weight:700}.metric-value{color:#f4fbff;font-size:26px;font-weight:700;margin-top:5px}
.timeline{display:flex;width:100%;height:30px;border-radius:8px;overflow:hidden;background:#111827;margin:8px 0 16px}.ns{background:#1f9dca}.ew{background:#6546b8}.timeline span{font-size:11px;font-weight:700;display:flex;align-items:center;justify-content:center;color:white}
.network{position:relative;border:1px solid rgba(85,230,255,.16);border-radius:22px;background:#09111f;padding:8px;overflow:hidden}.network svg{width:100%;height:410px}
.status{padding:10px 14px;border-radius:12px;background:rgba(85,230,255,.07);border:1px solid rgba(85,230,255,.15);margin-bottom:10px}
.status.crit{background:rgba(255,83,100,.10);border-color:rgba(255,83,100,.40)}
.nav-section{color:#55e6ff;font-weight:800;font-size:12px;letter-spacing:1px;margin-top:12px;margin-bottom:4px}
.tl-row{display:flex;align-items:center;margin:6px 0}.tl-name{width:90px;font-size:12px;font-weight:700;color:#cfe6ff}
.tl-track{position:relative;flex:1;height:26px;background:#111827;border-radius:6px;overflow:hidden}
.tl-green{position:absolute;top:0;bottom:0;background:linear-gradient(90deg,#1fbf6a,#22e07a);color:#04210f;font-size:11px;font-weight:800;display:flex;align-items:center;justify-content:center;white-space:nowrap;overflow:hidden}
.tl-now{position:absolute;top:0;bottom:0;width:3px;background:#ff5364;box-shadow:0 0 8px #ff5364}
</style>
""", unsafe_allow_html=True)


# ============================================================
# SMALL UI HELPERS
# ============================================================
def show_df(df, **kw):
    try:
        st.dataframe(df, width="stretch", **kw)
    except Exception:
        st.dataframe(df, use_container_width=True, **kw)


def btn(label, key=None, primary=False, container=None, **kw):
    target = container if container is not None else st
    kwargs = dict(key=key, type="primary" if primary else "secondary", **kw)
    try:
        return target.button(label, width="stretch", **kwargs)
    except Exception:
        return target.button(label, use_container_width=True, **kwargs)


def metric_card(label, value):
    return f"<div class='metric-card'><div class='metric-label'>{label}</div><div class='metric-value'>{value}</div></div>"


# ============================================================
# TRAFFIC MODEL
# ============================================================
def vehicles(d):
    return sum(d[x] for x in DIRECTIONS)


def _group(v, g):
    lam = g / CYCLE
    cap = SATURATION * lam
    x = v / cap if cap > 0 else 9.0
    xe = min(x, 0.95)
    uniform = 0.5 * CYCLE * (1 - lam) ** 2 / (1 - xe * lam)
    overflow = 60.0 * max(0.0, x - 0.85) ** 2
    stops = min(1.5, (1 - lam) / (1 - xe * lam))
    queue = v * (1 - lam) + max(0.0, v - cap)
    return uniform + overflow, stops, queue


def _phases(d, plan):
    ns, ew = d["North"] + d["South"], d["East"] + d["West"]
    g = PLANS[plan]
    return _group(ns, g), _group(ew, CYCLE - g), ns, ew


def traffic_cost(d, plan):
    (dn, _, _), (de, _, _), ns, ew = _phases(d, plan)
    return ns * dn + ew * de


def queue_indicator(d, plan):
    (_, _, qn), (_, _, qe), _, _ = _phases(d, plan)
    return qn + qe


def stops_indicator(d, plan):
    (_, sn, _), (_, se, _), ns, ew = _phases(d, plan)
    return (ns * sn + ew * se) / max(1, ns + ew)


def travel_time_indicator(d, plan):
    return 60.0 + traffic_cost(d, plan) / max(1, vehicles(d))


def simulate_plan(d, plan):
    waiting = float(traffic_cost(d, plan))
    queue = float(queue_indicator(d, plan))
    travel = float(travel_time_indicator(d, plan))
    stops = float(stops_indicator(d, plan))
    return {"Waiting Time": waiting, "Queue Length": queue, "Travel Time": travel,
            "Stops / Vehicle": stops, "Objective / Cost": waiting + 10 * queue + 50 * stops}


def classical_solution(d):
    costs = {p: simulate_plan(d, p)["Objective / Cost"] for p in PLANS}
    return min(costs, key=costs.get), costs


def network_kpis(traffic, plan):
    ms = [simulate_plan(traffic[j], plan[j]) for j in JUNCTIONS]
    veh = max(1, sum(vehicles(traffic[j]) for j in JUNCTIONS))
    waiting = sum(m["Waiting Time"] for m in ms)
    return {"waiting": waiting, "queue": sum(m["Queue Length"] for m in ms),
            "avg_delay": waiting / veh,
            "stops": sum(m["Stops / Vehicle"] * vehicles(traffic[j]) for m, j in zip(ms, JUNCTIONS)) / veh}


def reduction_pct(base, new):
    return 0.0 if base <= 0 else (base - new) / base * 100.0


def preprocess_traffic(traffic):
    rows = []
    for j, d in traffic.items():
        total = max(1, vehicles(d))
        rows.append({"Junction": j, "Vehicles": total, **{f"{x} %": round(d[x] / total * 100, 1) for x in DIRECTIONS}})
    return pd.DataFrame(rows)


def scenario_traffic(name):
    f = ADAPTIVE_SCENARIOS[name]["factors"]
    return {j: {d: max(0, int(round(data[d] * f[d]))) for d in DIRECTIONS} for j, data in BASE_TRAFFIC.items()}


# ============================================================
# ROAD NETWORK: congestion, Dijkstra, incidents
# ============================================================
def ekey(a, b):
    return tuple(sorted((a, b)))


def edge_flow(a, b, traffic):
    return EDGE_SHARE * (vehicles(traffic[a]) + vehicles(traffic[b])) / 2.0


def edge_ratio(a, b, traffic):
    return edge_flow(a, b, traffic) / ROADS[ekey(a, b)]["cap"]


def edge_time(a, b, traffic):
    r = ROADS[ekey(a, b)]
    t0 = r["km"] / r["speed"] * 3600.0
    return t0 * (1 + 0.5 * edge_ratio(a, b, traffic) ** 4)


def signal_wait(plan, node, axis):
    g = PLANS[plan[node]] if axis == "NS" else CYCLE - PLANS[plan[node]]
    red = CYCLE - g
    return red * red / (2.0 * CYCLE)


def blocked_edges():
    out = []
    for lab in st.session_state.blocked_labels:
        a, b = lab.split("–")
        out.append(ekey(a.strip(), b.strip()))
    return out


def road_label(e):
    return f"{e[0]}–{e[1]}"


def dijkstra(traffic, src, dst, plan, blocked=(), emergency=False, free_flow=False):
    blk = {ekey(*e) for e in blocked}
    adj = {n: [] for n in JUNCTIONS}
    for (a, b) in ROADS:
        if (a, b) in blk:
            continue
        adj[a].append(b)
        adj[b].append(a)

    def leg_cost(u, v):
        r = ROADS[ekey(u, v)]
        if free_flow:
            return 0.0, r["km"] / r["speed"] * 3600.0
        travel = edge_time(u, v, traffic)
        if emergency:
            return 0.0, travel * EMERGENCY_SPEED_FACTOR
        return signal_wait(plan, u, r["axis"]), travel

    dist = {n: math.inf for n in JUNCTIONS}
    dist[src] = 0.0
    prev = {}
    heap = [(0.0, src)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        if u == dst:
            break
        for v in adj[u]:
            w, t = leg_cost(u, v)
            nd = d + w + t
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(heap, (nd, v))
    if dist[dst] == math.inf:
        return None
    path = [dst]
    while path[-1] != src:
        path.append(prev[path[-1]])
    path.reverse()
    legs, clock = [], 0.0
    for u, v in zip(path, path[1:]):
        w, t = leg_cost(u, v)
        legs.append({"from": u, "to": v, "axis": ROADS[ekey(u, v)]["axis"], "wait": w, "travel": t,
                     "depart": clock + w, "arrive": clock + w + t, "km": ROADS[ekey(u, v)]["km"]})
        clock += w + t
    return {"path": path, "total": clock, "legs": legs, "km": sum(l["km"] for l in legs)}


def corridor_windows(route, traffic):
    path, legs = route["path"], route["legs"]
    windows = []
    for i, node in enumerate(path):
        arr = 0.0 if i == 0 else legs[i - 1]["arrive"]
        axis = legs[i]["axis"] if i < len(legs) else legs[-1]["axis"]
        start, end = arr - PREEMPT_LEAD, arr + PREEMPT_CLEAR
        d = traffic[node]
        cross = (d["East"] + d["West"]) if axis == "NS" else (d["North"] + d["South"])
        rate = cross / 60.0
        windows.append({"node": node, "axis": axis, "eta": arr, "start": start, "end": end,
                        "cross_held": rate * (end - start), "cross_vs": rate * (end - start) ** 2 / 2.0})
    return windows


# ============================================================
# MAX-FLOW / MIN-CUT (Edmonds-Karp)
# ============================================================
def max_flow(caps, src, dst):
    res = defaultdict(lambda: defaultdict(float))
    for (a, b), c in caps.items():
        res[a][b] += c
        res[b][a] += c
    flow = 0.0
    while True:
        parent = {src: None}
        q = deque([src])
        while q and dst not in parent:
            u = q.popleft()
            for v, c in res[u].items():
                if c > 1e-9 and v not in parent:
                    parent[v] = u
                    q.append(v)
        if dst not in parent:
            break
        b, v = math.inf, dst
        while parent[v] is not None:
            b = min(b, res[parent[v]][v])
            v = parent[v]
        v = dst
        while parent[v] is not None:
            u = parent[v]
            res[u][v] -= b
            res[v][u] += b
            v = u
        flow += b
    reach = set(parent)
    cut = [e for e in caps if (e[0] in reach) != (e[1] in reach)]
    return flow, cut, reach


def road_capacities(blocked=(), boost=None):
    blk = {ekey(*e) for e in blocked}
    caps = {}
    for e, r in ROADS.items():
        if e in blk:
            continue
        caps[e] = r["cap"] * (1 + (boost or {}).get(e, 0.0))
    return caps


# ============================================================
# NETWORK SIGNAL OPTIMISATION
# ============================================================
def coordination_penalty(plan_a, plan_b, flow):
    return flow * abs(PLANS[plan_a] - PLANS[plan_b]) * 0.25


def coordination_total(plan, traffic):
    return sum(coordination_penalty(plan[a], plan[b], edge_flow(a, b, traffic)) for a, b in NETWORK_EDGES)


def network_classical_solution(traffic):
    local = {j: {p: simulate_plan(traffic[j], p) for p in PLANS} for j in JUNCTIONS}
    best, best_score, best_coord = None, math.inf, 0.0
    for combo in product(list(PLANS), repeat=len(JUNCTIONS)):
        plan = dict(zip(JUNCTIONS, combo))
        coord = coordination_total(plan, traffic)
        score = sum(local[j][plan[j]]["Objective / Cost"] for j in JUNCTIONS) + coord
        if score < best_score:
            best, best_score, best_coord = plan, score, coord
    return best, best_score, [local[j][best[j]] for j in JUNCTIONS], best_coord


# ============================================================
# QUANTUM LAYER  (Qiskit if available, built-in NumPy QAOA otherwise)
# ============================================================
QISKIT_MODULES = ["qiskit", "qiskit_algorithms", "qiskit_optimization"]
PENALTY = 20.0            # one-hot penalty (scaled costs are <= 10)


def _qiskit_probe():
    """Return (ok, message). Actually tries the imports the QAOA code needs."""
    try:
        for m in QISKIT_MODULES:
            if importlib.util.find_spec(m) is None:
                return False, f"module '{m}' is not installed in this Python environment"
        importlib.import_module("qiskit_optimization")
        from qiskit_algorithms import QAOA  # noqa: F401
        from qiskit_algorithms.optimizers import COBYLA  # noqa: F401
        from qiskit.primitives import StatevectorSampler  # noqa: F401
        from qiskit_optimization.algorithms import MinimumEigenOptimizer  # noqa: F401
        return True, "Qiskit stack imported successfully"
    except Exception as exc:  # ImportError, version mismatch, etc.
        return False, f"{type(exc).__name__}: {exc}"


QISKIT_OK, QISKIT_MSG = _qiskit_probe()
QUANTUM_LABEL = "QISKIT READY" if QISKIT_OK else "BUILT-IN SIMULATOR"
_QAOA_CACHE = {}


def _scaled_costs(d):
    objs = {p: simulate_plan(d, p)["Objective / Cost"] for p in PLANS}
    m = max(objs.values()) or 1.0
    return objs, {p: 10.0 * v / m for p, v in objs.items()}


def _numpy_qaoa(scaled, reps=1):
    """Pure NumPy statevector QAOA on the 4-qubit one-hot QUBO. Deterministic."""
    names = list(PLANS)
    n = len(names)
    diag = np.zeros(2 ** n)
    for idx in range(2 ** n):
        bits = [(idx >> (n - 1 - i)) & 1 for i in range(n)]
        diag[idx] = sum(scaled[names[i]] * bits[i] for i in range(n)) + PENALTY * (sum(bits) - 1) ** 2

    def state(params):
        psi = np.full(2 ** n, 1 / math.sqrt(2 ** n), dtype=complex)
        for r in range(reps):
            gamma, beta = params[2 * r], params[2 * r + 1]
            psi = psi * np.exp(-1j * gamma * diag)
            t = psi.reshape([2] * n)
            c, s = math.cos(beta), -1j * math.sin(beta)
            for ax in range(n):
                a0, a1 = np.take(t, 0, axis=ax), np.take(t, 1, axis=ax)
                t = np.stack([c * a0 + s * a1, s * a0 + c * a1], axis=ax)
            psi = t.reshape(-1)
        return psi

    def energy(params):
        return float(np.sum(np.abs(state(params)) ** 2 * diag))

    grid = np.linspace(0, math.pi, 21)
    best = min(((energy([g, b] * reps), [g, b] * reps) for g in grid for b in grid), key=lambda x: x[0])
    params = best[1]
    try:
        from scipy.optimize import minimize
        params = list(minimize(energy, params, method="COBYLA", options={"maxiter": 60}).x)
    except Exception:
        pass
    probs = np.abs(state(params)) ** 2
    order = np.argsort(-probs)
    # like Qiskit's MinimumEigenOptimizer: among the most probable samples keep the best feasible (one-hot) one
    feasible = []
    for idx in order[:8]:
        bits = [(int(idx) >> (n - 1 - i)) & 1 for i in range(n)]
        if sum(bits) == 1:
            feasible.append(names[bits.index(1)])
    return min(feasible, key=lambda p: scaled[p]) if feasible else None


def _qiskit_qaoa(scaled):
    from qiskit_optimization import QuadraticProgram
    from qiskit_optimization.converters import QuadraticProgramToQubo
    from qiskit_algorithms import QAOA
    from qiskit_algorithms.optimizers import COBYLA
    from qiskit.primitives import StatevectorSampler
    from qiskit_optimization.algorithms import MinimumEigenOptimizer

    qp = QuadraticProgram()
    for p in PLANS:
        qp.binary_var(name=p)
    qp.linear_constraint(linear={p: 1 for p in PLANS}, sense="==", rhs=1, name="one_plan")
    qp.minimize(linear=scaled)
    qubo = QuadraticProgramToQubo().convert(qp)
    qaoa = QAOA(sampler=StatevectorSampler(), optimizer=COBYLA(maxiter=5), reps=1)
    result = MinimumEigenOptimizer(qaoa).solve(qubo)
    for i, variable in enumerate(qubo.variables):
        if round(float(result.x[i])) == 1:
            return variable.name
    raise RuntimeError("QAOA did not return a valid one-hot plan.")


def qaoa_solution(d):
    objs, scaled = _scaled_costs(d)
    key = (QISKIT_OK,) + tuple(round(v, 3) for v in objs.values())
    cached = _QAOA_CACHE.get(key)
    if cached is None:
        backend, selected, note = "NumPy simulator", None, ""
        if QISKIT_OK:
            try:
                selected, backend = _qiskit_qaoa(scaled), "Qiskit (StatevectorSampler)"
            except Exception as exc:
                note = f"Qiskit run failed ({type(exc).__name__}); used built-in simulator."
        if selected is None:
            try:
                selected, backend = _numpy_qaoa(scaled), "NumPy simulator"
            except Exception as exc:
                return {"Plan": None, "NS_Green": None, "EW_Green": None, "Cost": None,
                        "Status": "QAOA unavailable", "Error": str(exc), "Backend": "none"}
        cached = (selected, backend, note)
        _QAOA_CACHE[key] = cached
    selected, backend, note = cached
    if selected is None:
        return {"Plan": None, "NS_Green": None, "EW_Green": None, "Cost": None,
                "Status": "QAOA unavailable", "Error": "No valid one-hot plan found", "Backend": backend}
    return {"Plan": selected, "NS_Green": PLANS[selected], "EW_Green": CYCLE - PLANS[selected],
            "Cost": round(objs[selected], 1), "Status": "QAOA completed", "Backend": backend, "Note": note}


def _manual_qubo_text(linear, quad=None, title="QUBO"):
    """Readable QUBO (objective + one-hot penalty) built without Qiskit."""
    lines = [f"{title} (built-in formulation)", "minimize  sum_i c_i x_i"
             + ("  +  sum coordination(x_a, x_b)" if quad else "") + f"  +  {PENALTY:g} * (sum x - 1)^2 per junction", ""]
    lines += [f"  c[{k}] = {v:.3f}" for k, v in linear.items()]
    if quad:
        lines.append("")
        lines += [f"  q[{a}, {b}] = {v:.3f}" for (a, b), v in list(quad.items())[:40]]
        if len(quad) > 40:
            lines.append(f"  ... {len(quad) - 40} more quadratic terms")
    return "\n".join(lines)


def build_qubo(d):
    _, scaled = _scaled_costs(d)
    if QISKIT_OK:
        try:
            from qiskit_optimization import QuadraticProgram
            from qiskit_optimization.converters import QuadraticProgramToQubo
            qp = QuadraticProgram()
            for p in PLANS:
                qp.binary_var(name=p)
            qp.linear_constraint(linear={p: 1 for p in PLANS}, sense="==", rhs=1, name="one_plan")
            qp.minimize(linear=scaled)
            return str(QuadraticProgramToQubo().convert(qp)), None
        except Exception:
            pass
    return _manual_qubo_text(scaled, title="Junction QUBO"), None


def build_network_qubo(traffic):
    linear = {f"{j}_{p}": simulate_plan(traffic[j], p)["Objective / Cost"] for j in JUNCTIONS for p in PLANS}
    quad = {}
    for a, b in NETWORK_EDGES:
        fl = edge_flow(a, b, traffic)
        for pa in PLANS:
            for pb in PLANS:
                if pa != pb:
                    quad[(f"{a}_{pa}", f"{b}_{pb}")] = coordination_penalty(pa, pb, fl)
    if QISKIT_OK:
        try:
            from qiskit_optimization import QuadraticProgram
            from qiskit_optimization.converters import QuadraticProgramToQubo
            qp = QuadraticProgram()
            for name in linear:
                qp.binary_var(name=name)
            for j in JUNCTIONS:
                qp.linear_constraint(linear={f"{j}_{p}": 1 for p in PLANS}, sense="==", rhs=1, name=f"one_plan_{j}")
            qp.minimize(linear=linear, quadratic=quad)
            qubo = QuadraticProgramToQubo().convert(qp)
            return str(qubo), len(qubo.variables), None
        except Exception:
            pass
    return _manual_qubo_text(linear, quad, "Network QUBO"), len(linear), None


# ============================================================
# SENSOR FEED (SQLite), HOTSPOTS, FORECASTING
# ============================================================
def get_db():
    if "db" not in st.session_state:
        conn = sqlite3.connect(":memory:", check_same_thread=False)
        conn.execute("CREATE TABLE readings(minute INTEGER, junction TEXT, north INT, south INT, east INT, west INT, total INT)")
        conn.execute("CREATE INDEX idx_readings ON readings(junction, minute)")
        st.session_state.db = conn
        st.session_state.sim_minute = 0
    return st.session_state.db


def diurnal(minute):
    h = (6 + minute / 60.0) % 24
    return 0.55 + 0.55 * math.exp(-((h - 9) / 1.6) ** 2) + 0.65 * math.exp(-((h - 18) / 1.8) ** 2)


def sim_clock(minute):
    t = 6 * 60 + int(minute)
    return f"{(t // 60) % 24:02d}:{t % 60:02d}"


def generate_readings(n, spike=None, spike_factor=1.7):
    conn = get_db()
    rows = []
    for _ in range(n):
        m = st.session_state.sim_minute
        rng = np.random.default_rng(1000 + m)
        for j in JUNCTIONS:
            vals = {}
            for d in DIRECTIONS:
                f = diurnal(m) * rng.normal(1.0, 0.07) * (spike_factor if spike == j else 1.0)
                vals[d] = max(0, int(round(BASE_TRAFFIC[j][d] * f)))
            rows.append((m, j, vals["North"], vals["South"], vals["East"], vals["West"], sum(vals.values())))
        st.session_state.sim_minute = m + 1
    conn.executemany("INSERT INTO readings VALUES (?,?,?,?,?,?,?)", rows)
    conn.commit()


def ensure_history(minimum=60):
    get_db()
    if st.session_state.sim_minute < minimum:
        generate_readings(minimum - st.session_state.sim_minute)


HOTSPOT_SQL = """SELECT junction, ROUND(AVG(total),1) AS avg_load, MAX(total) AS peak, COUNT(*) AS samples
FROM readings WHERE minute >= ? GROUP BY junction ORDER BY avg_load DESC LIMIT ?"""


def severity(avg_load):
    r = avg_load / JUNCTION_CAPACITY
    return "🔴 Critical" if r >= 0.95 else "🟠 High" if r >= 0.80 else "🟡 Moderate" if r >= 0.60 else "🟢 Low"


def hotspot_table(window, k):
    since = st.session_state.sim_minute - window
    df = pd.read_sql_query(HOTSPOT_SQL, get_db(), params=(since, k))
    df["utilisation %"] = (df["avg_load"] / JUNCTION_CAPACITY * 100).round(1)
    df["severity"] = df["avg_load"].apply(severity)
    return df, since


def sensor_window_traffic(window):
    since = st.session_state.sim_minute - window
    df = pd.read_sql_query(
        "SELECT junction, AVG(north) n, AVG(south) s, AVG(east) e, AVG(west) w FROM readings WHERE minute >= ? GROUP BY junction",
        get_db(), params=(since,))
    out = {}
    for _, r in df.iterrows():
        out[r["junction"]] = {"North": int(round(r["n"])), "South": int(round(r["s"])),
                              "East": int(round(r["e"])), "West": int(round(r["w"]))}
    return out if set(out) == set(JUNCTIONS) else None


def junction_series(junction, last=120):
    df = pd.read_sql_query("SELECT minute, total FROM readings WHERE junction=? ORDER BY minute DESC LIMIT ?",
                           get_db(), params=(junction, last))
    return df.sort_values("minute").reset_index(drop=True)


def holt(series, alpha=0.3, beta=0.1, horizon=15):
    y = [float(v) for v in series]
    level, trend = y[0], (y[1] - y[0] if len(y) > 1 else 0.0)
    one_step = [y[0]]
    for t in range(1, len(y)):
        one_step.append(level + trend)
        prev = level
        level = alpha * y[t] + (1 - alpha) * (level + trend)
        trend = beta * (level - prev) + (1 - beta) * trend
    return one_step, [max(0.0, level + (h + 1) * trend) for h in range(horizon)]


# ============================================================
# STATE HELPERS
# ============================================================
def log_event(action, result):
    st.session_state.experiment_log.insert(0, {
        "Time": datetime.now().strftime("%H:%M:%S"), "Scenario": st.session_state.scenario,
        "Action": action, "Result": str(result)})
    st.session_state.experiment_log = st.session_state.experiment_log[:100]
    st.session_state.last_action = action


def set_traffic(traffic, scenario_name, action=None):
    st.session_state.traffic = {j: {d: int(traffic[j][d]) for d in DIRECTIONS} for j in JUNCTIONS}
    st.session_state.scenario = scenario_name
    st.session_state.qaoa_results = {}
    st.session_state.network_plan = None
    st.session_state.network_score = None
    st.session_state.traffic_version += 1
    log_event(action or "Traffic updated", scenario_name)


def set_scenario(name):
    if name in ADAPTIVE_SCENARIOS:
        set_traffic(scenario_traffic(name), name, "Scenario changed")


def run_network_optimization():
    plan, score, metrics, coord = network_classical_solution(st.session_state.traffic)
    st.session_state.network_plan = plan
    st.session_state.network_score = score
    log_event("Classical network optimization", plan)
    return plan, score, metrics, coord


def get_plan():
    if st.session_state.network_plan is None:
        run_network_optimization()
    return st.session_state.network_plan


def run_qaoa_all():
    results = {j: qaoa_solution(st.session_state.traffic[j]) for j in JUNCTIONS}
    st.session_state.qaoa_results = results
    log_event("QAOA validation", {j: r.get("Plan") for j, r in results.items()})
    return results


def emergency_routes():
    tr, plan, blk = st.session_state.traffic, get_plan(), blocked_edges()
    o, d = st.session_state.em_origin, st.session_state.em_dest
    normal = dijkstra(tr, o, d, plan, blk)
    priority = dijkstra(tr, o, d, plan, blk, emergency=True)
    return normal, priority


def junction_status_rows(windows, t_now):
    rows = []
    for w in windows:
        if t_now < w["start"]:
            state = f"⏳ Normal plan — pre-empt in {w['start'] - t_now:.0f}s"
        elif t_now <= w["end"]:
            state = "🟢 GREEN LINE ACTIVE"
        else:
            state = "✔ Passed — resuming plan"
        rows.append({"Junction": w["node"], "Ambulance axis": w["axis"], "ETA (s)": round(w["eta"], 1),
                     "Green window": f"{w['start']:.0f}s → {w['end']:.0f}s", "Status": state,
                     "Cross traffic held (veh)": round(w["cross_held"], 1)})
    return pd.DataFrame(rows)


def active_nodes(windows, t_now):
    return {w["node"] for w in windows if w["start"] <= t_now <= w["end"]}


# ============================================================
# SVG / HTML RENDERING
# ============================================================
def _pos_on_path(pts, weights, frac):
    total = sum(weights) or 1.0
    target, acc = frac * total, 0.0
    for i, w in enumerate(weights):
        if acc + w >= target or i == len(weights) - 1:
            f = 0.0 if w == 0 else max(0.0, min(1.0, (target - acc) / w))
            (x1, y1), (x2, y2) = pts[i], pts[i + 1]
            return x1 + (x2 - x1) * f, y1 + (y2 - y1) * f, x2 - x1, y2 - y1
        acc += w
    return pts[0][0], pts[0][1], 1, 0


def ambulance_shape():
    return ("<g transform='scale(.85)'>"
            "<rect x='-30' y='-16' width='44' height='28' rx='3' fill='#fff' stroke='#333' stroke-width='1.5'/>"
            "<polygon points='14,-16 14,12 32,12 32,-2 24,-16' fill='#fff' stroke='#333' stroke-width='1.5'/>"
            "<polygon points='17,-13 23,-13 29,-3 17,-3' fill='#7fd6f5'/>"
            "<rect x='-30' y='0' width='62' height='4' fill='#e53935'/>"
            "<rect x='-20' y='-11' width='12' height='4' fill='#e53935'/><rect x='-16' y='-15' width='4' height='12' fill='#e53935'/>"
            "<rect x='-6' y='-22' width='8' height='6' rx='2' fill='#ff1f1f'>"
            "<animate attributeName='opacity' values='1;0.2;1' dur='0.6s' repeatCount='indefinite'/></rect>"
            "<circle cx='-16' cy='14' r='5' fill='#222'/><circle cx='20' cy='14' r='5' fill='#222'/></g>")


def network_svg(traffic=None, plan=None, path=None, weights=None, blocked=(), cut=(), progress=None,
                corridor=False, green_nodes=(), label=None):
    traffic = traffic or st.session_state.traffic
    blk = {ekey(*e) for e in blocked}
    cutset = {ekey(*e) for e in cut}
    s = ["<div class='network'><svg viewBox='0 0 800 450' xmlns='http://www.w3.org/2000/svg'>",
         "<defs><filter id='glow'><feGaussianBlur stdDeviation='4' result='b'/>"
         "<feMerge><feMergeNode in='b'/><feMergeNode in='SourceGraphic'/></feMerge></filter></defs>"]
    for (a, b) in ROADS:
        (x1, y1), (x2, y2) = NODE_XY[a], NODE_XY[b]
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        if (a, b) in cutset:
            s.append(f"<line x1='{x1}' y1='{y1}' x2='{x2}' y2='{y2}' stroke='#b084ff' stroke-width='16' opacity='.45'/>")
        if (a, b) in blk:
            s.append(f"<line x1='{x1}' y1='{y1}' x2='{x2}' y2='{y2}' stroke='#ff3b4e' stroke-width='6' stroke-dasharray='10 8'/>")
            s.append(f"<text x='{mx}' y='{my + 6}' text-anchor='middle' fill='#ff3b4e' font-size='24' font-weight='800'>✕</text>")
            s.append(f"<text x='{mx}' y='{my + 26}' text-anchor='middle' fill='#ff8b98' font-size='11' font-weight='700'>BLOCKED</text>")
        else:
            r = edge_ratio(a, b, traffic)
            col = "#2fbf71" if r < 0.6 else "#ffb52e" if r < 0.85 else "#ff5364"
            s.append(f"<line x1='{x1}' y1='{y1}' x2='{x2}' y2='{y2}' stroke='{col}' stroke-width='6' opacity='.85'/>")
            s.append(f"<text x='{mx}' y='{my - 10}' text-anchor='middle' fill='#8298ad' font-size='11'>{r * 100:.0f}% load</text>")
    pts = []
    if path and len(path) > 1:
        pts = [NODE_XY[n] for n in path]
        border = "#22e07a" if corridor else "#55e6ff"
        poly = " ".join(f"{x},{y}" for x, y in pts)
        s.append(f"<polyline points='{poly}' fill='none' stroke='{border}' stroke-width='26' stroke-linejoin='round' opacity='.35' filter='url(#glow)'/>")
        s.append(f"<polyline points='{poly}' fill='none' stroke='#0f1b2b' stroke-width='18' stroke-linejoin='round'/>")
        s.append(f"<polyline points='{poly}' fill='none' stroke='{border}' stroke-width='4' stroke-dasharray='16 12'>"
                 "<animate attributeName='stroke-dashoffset' from='0' to='-28' dur='0.8s' repeatCount='indefinite'/></polyline>")
    glow_attr = " filter='url(#glow)'"
    for n, (x, y) in NODE_XY.items():
        g = n in green_nodes
        ring = "#22e07a" if g else ("#55e6ff" if not (path and n in path) else "#8dffc0" if corridor else "#8deeff")
        extra = glow_attr if g else ""
        s.append(f"<circle cx='{x}' cy='{y}' r='30' fill='#101d30' stroke='{ring}' stroke-width='{5 if g else 3}'{extra}/>")
        s.append(f"<text x='{x}' y='{y + 6}' text-anchor='middle' fill='white' font-size='18' font-weight='700'>{n}</text>")
        s.append(f"<circle cx='{x + 24}' cy='{y - 24}' r='6' fill='{'#22e07a' if g else '#ffb52e'}'/>")
        if plan:
            ns = PLANS[plan[n]]
            ty = y - 42 if n == "J2" else y + 50
            s.append(f"<text x='{x}' y='{ty}' text-anchor='middle' fill='#8deeff' font-size='11'>{plan[n]} · N-S {ns}s / E-W {CYCLE - ns}s</text>")
    if pts and progress is not None:
        w = weights or [math.dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
        x, y, dx, dy = _pos_on_path(pts, w, progress / 100.0)
        th = math.degrees(math.atan2(dy, dx))
        tf = f"translate({x:.1f},{y:.1f}) rotate({th:.1f})" if dx >= 0 else f"translate({x:.1f},{y:.1f}) scale(-1,1) rotate({180 - th:.1f})"
        s.append(f"<g transform='{tf}'>{ambulance_shape()}</g>")
    txt = label or "4-junction simulated traffic network"
    s.append(f"<text x='400' y='437' text-anchor='middle' fill='#ffcf57' font-size='14' font-weight='700'>{html.escape(txt)}</text>")
    s.append("<text x='14' y='437' fill='#2fbf71' font-size='11'>● free</text><text x='60' y='437' fill='#ffb52e' font-size='11'>● busy</text>"
             "<text x='106' y='437' fill='#ff5364' font-size='11'>● congested</text>")
    s.append("</svg></div>")
    return "".join(s)


def signal_bar(plan):
    ns = PLANS[plan]
    ew = CYCLE - ns
    return (f"<div class='timeline'><span class='ns' style='width:{ns / CYCLE * 100:.1f}%'>{ns}s N-S</span>"
            f"<span class='ew' style='width:{ew / CYCLE * 100:.1f}%'>{ew}s E-W</span></div>")


def corridor_timeline_html(windows, total, now=None):
    tmin, tmax = -PREEMPT_LEAD, total + PREEMPT_CLEAR
    span = max(1.0, tmax - tmin)
    rows = []
    for w in windows:
        left = (w["start"] - tmin) / span * 100
        width = (w["end"] - w["start"]) / span * 100
        marker = f"<div class='tl-now' style='left:{(now - tmin) / span * 100:.1f}%'></div>" if now is not None else ""
        rows.append(f"<div class='tl-row'><div class='tl-name'>{w['node']} · {w['axis']}</div><div class='tl-track'>"
                    f"<div class='tl-green' style='left:{left:.1f}%;width:{width:.1f}%'>{w['start']:.0f}s → {w['end']:.0f}s</div>{marker}</div></div>")
    return "".join(rows)


def scenario_traffic_df():
    return pd.DataFrame([{"Junction": j, **d, "Total": vehicles(d)} for j, d in st.session_state.traffic.items()])


# ============================================================
# EXPERIMENT SWEEP
# ============================================================
def adaptive_experiment(run_qaoa=False):
    cases = [(n, scenario_traffic(n), ADAPTIVE_SCENARIOS[n]["priority"]) for n in ADAPTIVE_SCENARIOS]
    if st.session_state.scenario not in ADAPTIVE_SCENARIOS:
        cases.append((f"Current: {st.session_state.scenario}", copy.deepcopy(st.session_state.traffic), "Custom"))
    rows = []
    for name, tr, prio in cases:
        plan, score, metrics, coord = network_classical_solution(tr)
        base = network_kpis(tr, BASELINE_PLAN)
        opt = network_kpis(tr, plan)
        qmatch, qj = "Not run", "-"
        if run_qaoa:
            qj = max(JUNCTIONS, key=lambda j: vehicles(tr[j]))
            qp = qaoa_solution(tr[qj]).get("Plan")
            qmatch = "Yes" if qp and qp == classical_solution(tr[qj])[0] else "No / unavailable"
        rows.append({"Scenario": name, "Priority": prio, "Network Plan": " / ".join(plan.values()),
                     "Fixed-time Waiting": round(base["waiting"], 1), "Optimized Waiting": round(opt["waiting"], 1),
                     "Waiting Reduction %": round(reduction_pct(base["waiting"], opt["waiting"]), 1),
                     "Avg Delay (s/veh)": round(opt["avg_delay"], 1), "QAOA Spot-check": qmatch,
                     "QAOA Junction": qj, "Objective": round(score, 1)})
    st.session_state.adaptive_results = pd.DataFrame(rows)
    log_event("Adaptive experiment", f"{len(rows)} scenarios; QAOA spot-check={run_qaoa}")
    return st.session_state.adaptive_results


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================
SECTIONS = {
    "🏠 OVERVIEW": ["Overview"],
    "🚦 TRAFFIC INTELLIGENCE": ["Scenario Engine", "Hotspot Detection", "Traffic Prediction"],
    "🗺️ ROUTING & SAFETY": ["Route & Incident Planner", "Emergency Green Corridor", "Bottleneck Analysis"],
    "🎛️ SIGNAL OPTIMIZATION": ["Adaptive Control", "Network Optimizer", "Junction Control", "Signal Plan", "Traffic Simulator"],
    "⚛️ QUANTUM LAYER": ["QUBO Lab", "QAOA Lab", "Quantum Validation"],
    "📊 VALIDATION": ["Metrics", "Dataset Validation", "Adaptive Experiments", "Action Log"],
}


def goto(p):
    st.session_state.page = p


st.sidebar.markdown("# 🚦 Quantum Traffic")
st.sidebar.caption("Adaptive hybrid traffic command center")
for heading, pages_ in SECTIONS.items():
    st.sidebar.markdown(f"<div class='nav-section'>{heading}</div>", unsafe_allow_html=True)
    for p_ in pages_:
        btn(p_, key=f"nav_{p_}", primary=(st.session_state.page == p_), container=st.sidebar, on_click=goto, args=(p_,))
st.sidebar.divider()
st.sidebar.markdown(f"**Scenario:** {st.session_state.scenario}")
st.sidebar.markdown(f"**Emergency:** {'🚨 ACTIVE' if st.session_state.emergency_active else 'inactive'}")
st.sidebar.markdown(f"**Blocked roads:** {', '.join(st.session_state.blocked_labels) or 'none'}")
st.sidebar.markdown(f"**Quantum backend:** {'Qiskit' if QISKIT_OK else 'built-in NumPy'}")
st.sidebar.caption("Prototype: synthetic traffic + simulated signals. No real-world signal control is performed.")

page = st.session_state.page
TR = st.session_state.traffic


def quantum_backend_note():
    if QISKIT_OK:
        st.success("Qiskit stack detected — QAOA runs on qiskit StatevectorSampler.")
    else:
        st.info("Qiskit could not be loaded, so QAOA runs on the app's built-in NumPy statevector simulator "
                "(same QUBO, same one-hot constraint). To use Qiskit, install it in the same Python that runs Streamlit: "
                "`python -m pip install qiskit qiskit-algorithms qiskit-optimization docplex`.")
        with st.expander("Why Qiskit was not loaded"):
            st.code(QISKIT_MSG)


# ============================================================
# PAGES
# ============================================================
if page == "Overview":
    st.markdown("<div class='hero'><div class='kicker'>QUANTUM-OPTIMIZED ADAPTIVE TRAFFIC CONTROL</div>"
                "<div class='big'>Quantum Smart Traffic Management</div>"
                "<p>Adaptive signal optimisation, Dijkstra routing with incident rerouting, an ambulance GREEN LINE corridor, "
                "sensor-driven hotspot detection, max-flow bottleneck analysis and forecasting — with QUBO/QAOA validation.</p></div>",
                unsafe_allow_html=True)
    plan_now = st.session_state.network_plan or BASELINE_PLAN
    total = sum(vehicles(d) for d in TR.values())
    cols = st.columns(5)
    vals = [("NETWORK", f"4 Junctions · {len(ROADS)} Roads"), ("VEHICLES / MIN", f"{total:,}"), ("SCENARIO", st.session_state.scenario),
            ("QAOA", QUANTUM_LABEL), ("EMERGENCY", "ACTIVE" if st.session_state.emergency_active else "INACTIVE")]
    for c, (lab, val) in zip(cols, vals):
        c.markdown(metric_card(lab, val), unsafe_allow_html=True)
    st.markdown("### 🌐 Live Network")
    st.markdown(network_svg(plan=st.session_state.network_plan, blocked=blocked_edges()), unsafe_allow_html=True)
    st.markdown("### 🌍 Real-world problems this dashboard addresses")
    show_df(pd.DataFrame([
        {"Real-world problem": "Ambulances lose minutes at red lights", "Feature": "Emergency Green Corridor", "Technique": "Dijkstra + signal pre-emption windows"},
        {"Real-world problem": "Accidents / road closures create gridlock", "Feature": "Route & Incident Planner", "Technique": "Dynamic graph update + re-routing"},
        {"Real-world problem": "Fixed-time signals waste green time", "Feature": "Adaptive Control / Network Optimizer", "Technique": "Webster-style delay model + 4⁴ search + QUBO/QAOA"},
        {"Real-world problem": "Nobody knows where congestion builds", "Feature": "Hotspot Detection", "Technique": "SQLite sensor feed, sliding window, top-k"},
        {"Real-world problem": "Which road limits city throughput?", "Feature": "Bottleneck Analysis", "Technique": "Edmonds-Karp max-flow / min-cut + upgrade what-if"},
        {"Real-world problem": "Control is reactive, not proactive", "Feature": "Traffic Prediction", "Technique": "Holt forecasting with backtest vs persistence"},
    ]))
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 🔗 Connected workflow")
        st.markdown("`Scenario / Sensors → Optimizer → QUBO → QAOA → Signal Plan → Simulator → Metrics`")
        st.markdown("`Graph → Dijkstra → Incident reroute → Green-line corridor`")
        st.info(f"Last action: {st.session_state.last_action}")
    with c2:
        st.markdown("### 🚦 Current signal plan")
        if st.session_state.network_plan is None:
            st.warning("Run network optimization from Network Optimizer (showing fixed-time 30/30 baseline meanwhile).")
        for j, p in plan_now.items():
            st.markdown(f"**{j} — {p}**")
            st.markdown(signal_bar(p), unsafe_allow_html=True)

elif page == "Scenario Engine":
    st.title("🚦 Scenario Engine")
    names = list(ADAPTIVE_SCENARIOS)
    selected = st.selectbox("Select traffic scenario", names,
                            index=names.index(st.session_state.scenario) if st.session_state.scenario in names else 0)
    info = ADAPTIVE_SCENARIOS[selected]
    st.markdown(f"<div class='card'><b>{selected}</b><br>{info['description']}<br><br>"
                f"<span class='badge'>Priority: {info['priority']}</span> <span class='badge'>Focus: {info['focus']}</span></div>",
                unsafe_allow_html=True)
    if btn("⚡ ADAPT TRAFFIC NETWORK", key="adapt_btn"):
        set_scenario(selected)
        with st.spinner("Re-optimizing network for the selected scenario..."):
            run_network_optimization()
        st.success("Traffic model updated and network baseline optimized.")
        TR = st.session_state.traffic
    st.markdown("### Directional demand transformation")
    show_df(pd.DataFrame([info["factors"]], index=[selected]))
    st.markdown("### Current traffic (vehicles / min)")
    show_df(scenario_traffic_df())
    st.markdown("### Direction mix")
    show_df(preprocess_traffic(st.session_state.traffic))

elif page == "Hotspot Detection":
    st.title("🔥 Congestion Hotspot Detection")
    st.caption("Simulated road sensors write one reading per junction per minute into SQLite. "
               "A sliding window + SQL GROUP BY finds the busiest junctions (top-k), cross-checked with a heap.")
    ensure_history(60)
    c1, c2, c3 = st.columns(3)
    window = c1.slider("Sliding window (minutes)", 5, 60, 15)
    k = c2.slider("Top-k hotspots", 1, 4, 3)
    c3.markdown(metric_card("SIMULATED CLOCK", sim_clock(st.session_state.sim_minute)), unsafe_allow_html=True)
    b1, b2, b3 = st.columns(3)
    if btn("⏩ Advance +5 min", key="adv5", container=b1):
        generate_readings(5)
    if btn("⏩ Advance +30 min", key="adv30", container=b2):
        generate_readings(30)
    spike_j = b3.selectbox("Surge junction", JUNCTIONS, index=2, label_visibility="collapsed")
    if btn(f"🚧 Simulate 10-min surge at {spike_j}", key="surge"):
        generate_readings(10, spike=spike_j)
    df, since = hotspot_table(window, k)
    st.markdown(f"### Top-{k} hotspots · minutes {since}–{st.session_state.sim_minute - 1}")
    show_df(df)
    heap_top = heapq.nlargest(k, ((r["avg_load"], r["junction"]) for _, r in df.iterrows()))
    st.caption(f"Heap cross-check (heapq.nlargest): {', '.join(j for _, j in heap_top)} — "
               f"{'matches SQL ranking ✅' if [j for _, j in heap_top] == list(df['junction']) else 'differs from SQL ❌'}")
    st.code(HOTSPOT_SQL, language="sql")
    if not df.empty:
        worst = df.iloc[0]
        st.markdown(f"<div class='status {'crit' if 'Critical' in worst['severity'] or 'High' in worst['severity'] else ''}'>"
                    f"Busiest junction: <b>{worst['junction']}</b> — {worst['avg_load']} veh/min ({worst['utilisation %']}% of capacity) · {worst['severity']}</div>",
                    unsafe_allow_html=True)
    hist = pd.read_sql_query("SELECT minute, junction, total FROM readings WHERE minute >= ?", get_db(),
                             params=(max(0, st.session_state.sim_minute - 90),))
    st.markdown("### Load per junction (last 90 min)")
    st.line_chart(hist.pivot(index="minute", columns="junction", values="total"))
    st.markdown("### 🔁 Close the loop: sensor-driven adaptation")
    st.write("Feed the sliding-window averages into the network and re-optimise the signals — this makes the controller sensor-adaptive, not just scenario-adaptive.")
    if btn("📡 ADAPT SIGNALS TO LIVE SENSOR WINDOW", key="sensor_adapt", primary=True):
        live = sensor_window_traffic(window)
        if live:
            set_traffic(live, "Live Sensors", "Sensor-driven adaptation")
            run_network_optimization()
            st.success("Network traffic replaced by sensor averages and signals re-optimised. See Signal Plan / Metrics.")
        else:
            st.error("Not enough sensor data yet.")

elif page == "Traffic Prediction":
    st.title("📈 Traffic Prediction")
    st.caption("Holt linear-trend smoothing on the sensor history, backtested against a naive 'same as last minute' baseline.")
    ensure_history(60)
    c1, c2 = st.columns(2)
    horizon = c1.slider("Forecast horizon (minutes)", 5, 30, 15, step=5)
    junction = c2.selectbox("Junction to plot", JUNCTIONS)
    rows, forecasts = [], {}
    for j in JUNCTIONS:
        s = junction_series(j, 120)
        y = s["total"].tolist()
        one, fc = holt(y, horizon=horizon)
        n = min(30, len(y) - 1)
        mae_h = float(np.mean([abs(y[t] - one[t]) for t in range(len(y) - n, len(y))]))
        mae_p = float(np.mean([abs(y[t] - y[t - 1]) for t in range(len(y) - n, len(y))]))
        forecasts[j] = fc
        now = float(np.mean(y[-5:]))
        rows.append({"Junction": j, "Now (5-min avg)": round(now, 1), f"Forecast +{horizon} min": round(fc[-1], 1),
                     "Trend": "📈 rising" if fc[-1] > now * 1.03 else "📉 falling" if fc[-1] < now * 0.97 else "➡ steady",
                     "Holt MAE": round(mae_h, 2), "Persistence MAE": round(mae_p, 2),
                     "Better model": "Holt" if mae_h < mae_p else "Persistence"})
    show_df(pd.DataFrame(rows))
    s = junction_series(junction, 60)
    _, fc = holt(junction_series(junction, 120)["total"].tolist(), horizon=horizon)
    last = int(s["minute"].iloc[-1])
    chart = pd.DataFrame({"Actual": s.set_index("minute")["total"]})
    chart = pd.concat([chart, pd.DataFrame({"Forecast": fc}, index=range(last + 1, last + 1 + horizon))])
    st.line_chart(chart)
    st.caption("MAE = mean absolute error of one-step-ahead predictions over the last 30 minutes (lower is better). "
               "The table reports honestly which model wins on this data.")
    st.markdown("### 🔮 Proactive control")
    st.write("Pre-optimise the signals for the *predicted* demand instead of waiting for congestion to appear.")
    if btn("🔮 OPTIMISE SIGNALS FOR FORECAST TRAFFIC", key="fc_apply", primary=True):
        since = st.session_state.sim_minute - 10
        mix = pd.read_sql_query("SELECT junction, SUM(north) n, SUM(south) s, SUM(east) e, SUM(west) w FROM readings WHERE minute >= ? GROUP BY junction",
                                get_db(), params=(since,)).set_index("junction")
        new = {}
        for j in JUNCTIONS:
            tot = float(mix.loc[j].sum()) or 1.0
            new[j] = {d: int(round(forecasts[j][-1] * mix.loc[j][c] / tot)) for d, c in zip(DIRECTIONS, ["n", "s", "e", "w"])}
        set_traffic(new, f"Forecast +{horizon}min", "Forecast-driven optimisation")
        run_network_optimization()
        st.success("Forecast demand loaded and signals re-optimised. Check Signal Plan / Metrics.")

elif page == "Route & Incident Planner":
    st.title("🗺️ Route & Incident Planner")
    st.caption("Dijkstra on the live road graph. Edge cost = congested travel time (BPR) + expected signal wait. Blocking a road removes the edge and re-routes.")
    plan = get_plan()
    labels = [road_label(e) for e in ROADS]
    c1, c2 = st.columns(2)
    st.session_state.od_origin = c1.selectbox("Origin", JUNCTIONS, index=JUNCTIONS.index(st.session_state.od_origin))
    st.session_state.od_dest = c2.selectbox("Destination", JUNCTIONS, index=JUNCTIONS.index(st.session_state.od_dest))

    def _sync_blocked():
        st.session_state.blocked_labels = list(st.session_state._blocked_widget)

    st.multiselect("🚧 Blocked roads (incidents)", labels, default=st.session_state.blocked_labels,
                   key="_blocked_widget", on_change=_sync_blocked)
    p1, p2 = st.columns(2)
    if btn("💥 Simulate accident on J1–J4 (main shortcut)", key="acc_btn", container=p1):
        st.session_state.blocked_labels = ["J1–J4"]
        set_scenario("Road Incident")
        log_event("Incident", "J1–J4 blocked")
        st.session_state.pop("_blocked_widget", None)
        st.rerun()
    if btn("✅ Clear all incidents", key="clear_inc", container=p2):
        st.session_state.blocked_labels = []
        st.session_state.pop("_blocked_widget", None)
        log_event("Incidents cleared", "no blocked roads")
        st.rerun()
    o, d = st.session_state.od_origin, st.session_state.od_dest
    if o == d:
        st.warning("Choose different origin and destination.")
    else:
        TR = st.session_state.traffic
        blk = blocked_edges()
        free = dijkstra(TR, o, d, plan, free_flow=True)
        before = dijkstra(TR, o, d, plan, [])
        now = dijkstra(TR, o, d, plan, blk)
        if now is None:
            st.error("🚫 No route exists — every path is blocked.")
        else:
            delta = now["total"] - before["total"]
            m = st.columns(4)
            m[0].metric("Free-flow ETA", f"{free['total']:.0f} s")
            m[1].metric("Live-traffic ETA (no incident)", f"{before['total']:.0f} s")
            m[2].metric("ETA now", f"{now['total']:.0f} s", delta=f"{delta:+.0f} s" if blk else None, delta_color="inverse")
            m[3].metric("Route", " → ".join(now["path"]))
            st.markdown(network_svg(plan=plan, path=now["path"], blocked=blk,
                                    label=f"Fastest route {o} → {d} · {now['total']:.0f}s · {now['km']:.1f} km"), unsafe_allow_html=True)
            show_df(pd.DataFrame([{"From": l["from"], "To": l["to"], "Signal wait (s)": round(l["wait"], 1),
                                   "Travel (s)": round(l["travel"], 1), "Arrive (s)": round(l["arrive"], 1), "km": l["km"]} for l in now["legs"]]))
            if blk and now["path"] != before["path"]:
                st.info(f"Rerouted: {' → '.join(before['path'])} became {' → '.join(now['path'])} (+{delta:.0f} s).")
            elif blk:
                st.success("The blocked road is not on the fastest route; no change needed.")

elif page == "Emergency Green Corridor":
    st.title("🚑 Emergency Green Line Corridor")
    active = st.session_state.emergency_active
    if not active:
        c1, c2 = st.columns(2)
        st.session_state.em_origin = c1.selectbox("Origin (incident site)", JUNCTIONS, index=JUNCTIONS.index(st.session_state.em_origin))
        st.session_state.em_dest = c2.selectbox("Destination (hospital)", JUNCTIONS, index=JUNCTIONS.index(st.session_state.em_dest))
    if st.session_state.em_origin == st.session_state.em_dest:
        st.warning("Origin and destination must differ.")
        st.stop()
    normal, prio = emergency_routes()
    if prio is None:
        st.error("🚫 No route available — all roads to the destination are blocked. Clear incidents in Route & Incident Planner.")
        st.stop()
    TR = st.session_state.traffic
    windows = corridor_windows(prio, TR)
    total = prio["total"]
    saved = normal["total"] - total
    st.markdown(f"<div class='status {'crit' if active else ''}'>{'🚨 <b>CRITICAL PRIORITY ACTIVE</b> — green line pre-empting signals along ' if active else 'Corridor is <b>INACTIVE</b>. Planned route: '}"
                f"<b>{' → '.join(prio['path'])}</b> ({prio['km']:.1f} km). Route found with Dijkstra"
                f"{' around blocked roads' if blocked_edges() else ''}.</div>", unsafe_allow_html=True)
    if st.session_state.emergency_completed and not active:
        st.success("✅ Previous emergency completed; normal signal plans restored.")
    m = st.columns(4)
    m[0].metric("Normal ETA", f"{normal['total']:.0f} s")
    m[1].metric("Green-line ETA", f"{total:.0f} s")
    m[2].metric("Time saved", f"{saved:.0f} s", delta=f"{reduction_pct(normal['total'], total):.0f}%")
    m[3].metric("Cross vehicles held", f"{sum(w['cross_held'] for w in windows):.0f}")
    svg_ph, prog_ph = st.empty(), st.empty()
    weights = [l["arrive"] - (l["depart"] - l["wait"]) for l in prio["legs"]]

    def draw():
        p = st.session_state.ambulance_progress
        t_now = p / 100.0 * total
        lbl = ("🚨 GREEN LINE · " if active else "Green line preview · ") + " → ".join(prio["path"])
        svg_ph.markdown(network_svg(plan=get_plan(), path=prio["path"], weights=weights, blocked=blocked_edges(),
                                    progress=p if (active or p) else 0, corridor=True,
                                    green_nodes=active_nodes(windows, t_now) if active else (), label=lbl),
                        unsafe_allow_html=True)
        prog_ph.progress(min(1.0, p / 100.0), text=f"Ambulance progress {p}%  ·  t = {t_now:.0f}s of {total:.0f}s")

    if not active:
        if btn("🚨 ACTIVATE GREEN LINE CORRIDOR", key="em_activate", primary=True):
            st.session_state.emergency_active = True
            st.session_state.emergency_completed = False
            st.session_state.ambulance_progress = 0
            log_event("Emergency corridor activated", f"{' → '.join(prio['path'])}; cross traffic held at {len(windows)} junctions")
            st.rerun()
    else:
        b1, b2, b3, b4 = st.columns(4)
        adv = btn("▶ Advance +20%", key="em_advance", container=b1)
        run = btn("🎬 Run full trip", key="em_run", container=b2)
        rst = btn("↺ Reset trip", key="em_reset", container=b3)
        done = btn("⏹ COMPLETE EMERGENCY", key="em_complete", container=b4)
        if adv:
            st.session_state.ambulance_progress = min(100, st.session_state.ambulance_progress + 20)
        if rst:
            st.session_state.ambulance_progress = 0
        if run:
            start = 0 if st.session_state.ambulance_progress >= 100 else st.session_state.ambulance_progress
            for pv in range(start, 101, 4):
                st.session_state.ambulance_progress = pv
                draw()
                time.sleep(0.08)
            st.session_state.ambulance_progress = 100
        if done:
            st.session_state.emergency_active = False
            st.session_state.emergency_completed = True
            st.session_state.ambulance_progress = 0
            log_event("Emergency completed", "Corridor released; scenario and plans untouched")
            st.rerun()
    draw()
    t_now = st.session_state.ambulance_progress / 100.0 * total
    if active and st.session_state.ambulance_progress >= 100:
        st.success("🏥 Ambulance has arrived. Press COMPLETE EMERGENCY to release the corridor.")
    st.markdown("### 🟢 Signal pre-emption schedule (time-space view)")
    st.markdown(corridor_timeline_html(windows, total, t_now if active else None), unsafe_allow_html=True)
    st.caption(f"Each junction is forced green for the ambulance's axis {PREEMPT_LEAD}s before it arrives and {PREEMPT_CLEAR}s after it passes. Red line = ambulance clock.")
    show_df(junction_status_rows(windows, t_now if active else -999))
    held_vs = sum(w["cross_vs"] for w in windows)
    st.markdown(f"**Trade-off:** holding cross traffic costs ≈ **{held_vs:.0f} vehicle-seconds** of extra delay across the corridor, "
                f"versus **{saved:.0f} s** saved for a critical patient.")
    with st.expander("Route legs (Dijkstra output)"):
        show_df(pd.DataFrame([{"From": l["from"], "To": l["to"], "Travel (s)": round(l["travel"], 1), "Arrive (s)": round(l["arrive"], 1)} for l in prio["legs"]]))

elif page == "Bottleneck Analysis":
    st.title("🧱 Bottleneck Analysis (Max-Flow / Min-Cut)")
    st.caption("Edmonds-Karp finds the maximum vehicles/min that can travel between two junctions; the min-cut is the set of roads that limit it.")
    c1, c2 = st.columns(2)
    st.session_state.bn_origin = c1.selectbox("Source", JUNCTIONS, index=JUNCTIONS.index(st.session_state.bn_origin))
    st.session_state.bn_dest = c2.selectbox("Sink", JUNCTIONS, index=JUNCTIONS.index(st.session_state.bn_dest))
    s_, t_ = st.session_state.bn_origin, st.session_state.bn_dest
    if s_ == t_:
        st.warning("Source and sink must differ.")
    else:
        blk = blocked_edges()
        caps = road_capacities(blk)
        flow, cut, _ = max_flow(caps, s_, t_)
        m = st.columns(3)
        m[0].metric("Max throughput", f"{flow:.0f} veh/min")
        m[1].metric("Bottleneck roads", ", ".join(road_label(e) for e in cut) or "none (disconnected)")
        m[2].metric("Blocked roads", len(blk))
        st.markdown(network_svg(plan=st.session_state.network_plan, blocked=blk, cut=cut,
                                label=f"Purple = min-cut bottleneck · max flow {s_}→{t_} = {flow:.0f} veh/min"), unsafe_allow_html=True)
        st.markdown("### 🏗️ What-if: widen a road by 25 %")
        rows = []
        for e in caps:
            f2, _, _ = max_flow(road_capacities(blk, {e: 0.25}), s_, t_)
            rows.append({"Road": road_label(e), "Current capacity": ROADS[e]["cap"], "New capacity": round(ROADS[e]["cap"] * 1.25),
                         "Max flow after": round(f2, 1), "Gain (veh/min)": round(f2 - flow, 1), "On min-cut": "Yes" if e in cut else "No"})
        res = pd.DataFrame(rows).sort_values("Gain (veh/min)", ascending=False)
        show_df(res)
        if not res.empty and res.iloc[0]["Gain (veh/min)"] > 0:
            st.success(f"Best investment: widen **{res.iloc[0]['Road']}** (+{res.iloc[0]['Gain (veh/min)']} veh/min). "
                       "Only roads on the min-cut can increase total throughput.")
        else:
            st.info("No single road widening increases throughput between these junctions.")

elif page == "Adaptive Control":
    st.title("🧠 Adaptive Control")
    if st.session_state.emergency_active:
        normal, prio = emergency_routes()
        if prio:
            st.error(f"🚨 EMERGENCY GREEN LINE ACTIVE on {' → '.join(prio['path'])}. Signals on the corridor are pre-empted; other junctions follow the optimised plan.")
            show_df(junction_status_rows(corridor_windows(prio, TR), st.session_state.ambulance_progress / 100 * prio["total"]))
    current = st.session_state.scenario
    if current in ADAPTIVE_SCENARIOS:
        info = ADAPTIVE_SCENARIOS[current]
        priority, focus, description = info["priority"], info["focus"], info["description"]
    else:
        priority, focus, description = current, "User / sensor-supplied J1–J4 demand", "Traffic loaded from a dataset, live sensors or forecast."
    st.markdown(f"<div class='card'><b>Active scenario:</b> {current}<br><b>Priority class:</b> {priority}<br><b>Demand focus:</b> {focus}<br>{description}</div>", unsafe_allow_html=True)
    st.markdown("### Adaptive decision logic")
    st.write("Demand comes from a scenario, a dataset, live sensors or a forecast. The controller re-solves the 4-junction network with a non-linear delay model, "
             "so green splits move continuously with demand — they are outputs of the optimizer, not fixed values.")
    b1, b2 = st.columns(2)
    if btn("⚡ OPTIMIZE CURRENT TRAFFIC", key="ac_opt", primary=True, container=b1):
        with st.spinner("Optimizing..."):
            run_network_optimization()
        st.rerun()
    if btn("📡 ADAPT TO LIVE SENSORS (last 15 min)", key="ac_sensor", container=b2):
        ensure_history(60)
        live = sensor_window_traffic(15)
        if live:
            set_traffic(live, "Live Sensors", "Sensor-driven adaptation")
            run_network_optimization()
            st.rerun()
    if st.session_state.network_plan:
        plan = st.session_state.network_plan
        rows = [{"Junction": j, "Selected Plan": p, "N-S Green (s)": PLANS[p], "E-W Green (s)": CYCLE - PLANS[p], "Demand": vehicles(TR[j])} for j, p in plan.items()]
        st.markdown("### Adapted signal plan")
        show_df(pd.DataFrame(rows))
        base, opt = network_kpis(TR, BASELINE_PLAN), network_kpis(TR, plan)
        m = st.columns(3)
        m[0].metric("Fixed-time waiting", f"{base['waiting']:,.0f}")
        m[1].metric("Adaptive waiting", f"{opt['waiting']:,.0f}")
        m[2].metric("Reduction", f"{reduction_pct(base['waiting'], opt['waiting']):.1f}%")
    else:
        st.info("Press OPTIMIZE to generate the adaptive plan.")

elif page == "Network Optimizer":
    st.title("🌐 Network Optimizer")
    st.caption("Classical search evaluates all 4⁴ = 256 plan combinations with road-flow-weighted coordination. "
               "A 16-variable QUBO can be built for inspection; 16-qubit statevector QAOA is not run because it would block the dashboard.")
    if st.session_state.emergency_active:
        st.warning("🚨 Emergency corridor active — pre-emption overrides these plans only on corridor junctions.")
    if btn("⚡ OPTIMIZE NETWORK", key="no_opt", primary=True):
        with st.spinner("Searching 256 network plan combinations..."):
            run_network_optimization()
        st.success("✅ Network optimization completed.")
    if st.session_state.network_plan:
        plan = st.session_state.network_plan
        rows = []
        for j, p in plan.items():
            mm = simulate_plan(TR[j], p)
            rows.append({"Junction": j, "Plan": p, "N-S Green (s)": PLANS[p], "E-W Green (s)": CYCLE - PLANS[p],
                         "Waiting": round(mm["Waiting Time"], 1), "Queue": round(mm["Queue Length"], 1)})
        show_df(pd.DataFrame(rows))
        a, b, c = st.columns(3)
        a.metric("Network objective", f"{st.session_state.network_score:,.1f}")
        b.metric("Scenario", st.session_state.scenario)
        c.metric("Coordination penalty", f"{coordination_total(plan, TR):,.1f}")
        st.markdown(network_svg(plan=plan, blocked=blocked_edges()), unsafe_allow_html=True)
    else:
        st.info("Click OPTIMIZE NETWORK once. The result is shared with Signal Plan, Simulator and Metrics.")
    if btn("🔬 BUILD 16-VARIABLE NETWORK QUBO", key="no_qubo"):
        with st.spinner("Building QUBO formulation only..."):
            text, nvars, _ = build_network_qubo(TR)
        st.success(f"Network QUBO built: {nvars} binary variables.")
        st.code(text, language="text")

elif page == "Junction Control":
    st.title("🎛️ Junction Control")
    j = st.selectbox("Select junction", JUNCTIONS)
    d = TR[j]
    ver = st.session_state.traffic_version
    cols = st.columns(4)
    vals = {dirn: cols[i].number_input(dirn, 0, 500, int(d[dirn]), key=f"jc_{ver}_{j}_{dirn}") for i, dirn in enumerate(DIRECTIONS)}
    runq = st.checkbox("Also validate with QAOA", value=False)
    if btn("⚡ OPTIMIZE JUNCTION", key="jc_opt", primary=True):
        new = copy.deepcopy(TR)
        new[j] = {k_: int(v_) for k_, v_ in vals.items()}
        set_traffic(new, "Manual Edit", f"Junction {j} edited")
        classical = classical_solution(vals)[0]
        msg = f"Classical best plan: {classical}"
        if runq:
            q = qaoa_solution(vals)
            st.session_state.qaoa_results[j] = q
            msg += f" | QAOA: {q.get('Plan')} | Match: {'YES' if classical == q.get('Plan') else 'NO / unavailable'}"
        st.success(msg + " — network plan invalidated; re-run Network Optimizer.")
    st.markdown("### Plan comparison (for the values above)")
    show_df(pd.DataFrame([{"Plan": p, **simulate_plan(vals, p)} for p in PLANS]))

elif page == "Signal Plan":
    st.title("🚦 Signal Plan")
    if st.session_state.emergency_active:
        normal, prio = emergency_routes()
        if prio:
            st.error("🚨 EMERGENCY SIGNAL MODE — corridor junctions are pre-empted for the ambulance (green line).")
            show_df(junction_status_rows(corridor_windows(prio, TR), st.session_state.ambulance_progress / 100 * prio["total"]))
    plan = get_plan()
    st.success("✅ Current optimized signal plan.")
    for j, p in plan.items():
        st.markdown(f"### {j} — {p}")
        st.markdown(signal_bar(p), unsafe_allow_html=True)
        st.caption(f"North/South green: {PLANS[p]} s | East/West green: {CYCLE - PLANS[p]} s | Cycle: {CYCLE} s")

elif page == "Traffic Simulator":
    st.title("🚗 Traffic Simulator")
    plan = get_plan()
    rows = []
    for j, p in plan.items():
        mm = simulate_plan(TR[j], p)
        rows.append({"Junction": j, "Plan": p, "Waiting": round(mm["Waiting Time"], 1), "Queue": round(mm["Queue Length"], 1),
                     "Travel Indicator": round(mm["Travel Time"], 1), "Stops/Vehicle": round(mm["Stops / Vehicle"], 2)})
    df = pd.DataFrame(rows)
    show_df(df)
    st.bar_chart(df.set_index("Junction")[["Waiting", "Queue"]])
    if st.session_state.emergency_active:
        normal, prio = emergency_routes()
        if prio:
            w = corridor_windows(prio, TR)
            p = st.session_state.ambulance_progress
            st.error(f"🚨 Ambulance on {' → '.join(prio['path'])} — progress {p}% (advance it on the Emergency page).")
            st.markdown(network_svg(plan=plan, path=prio["path"], weights=[l['arrive'] - (l['depart'] - l['wait']) for l in prio["legs"]],
                                    blocked=blocked_edges(), progress=p, corridor=True,
                                    green_nodes=active_nodes(w, p / 100 * prio["total"])), unsafe_allow_html=True)
    else:
        st.markdown(network_svg(plan=plan, blocked=blocked_edges()), unsafe_allow_html=True)
    st.caption("Traffic, travel, queue and stops are prototype simulation indicators, not calibrated real-city measurements.")

elif page == "QUBO Lab":
    st.title("🧮 QUBO Lab")
    j = st.selectbox("Junction", JUNCTIONS)
    d = TR[j]
    show_df(pd.DataFrame([d], index=[j]))
    rows = [{"Plan": p, "N-S Green": PLANS[p], "E-W Green": CYCLE - PLANS[p], "Waiting Cost": round(traffic_cost(d, p), 1),
             "Queue Proxy": round(queue_indicator(d, p), 2), "Objective": round(simulate_plan(d, p)["Objective / Cost"], 2)} for p in PLANS]
    show_df(pd.DataFrame(rows))
    st.markdown("### One-hot QUBO (objective scaled so the largest coefficient is 10)")
    text, _ = build_qubo(d)
    st.code(text, language="text")
    if not QISKIT_OK:
        st.caption("Shown from the built-in formulation (Qiskit not loaded).")
    st.success(f"Classical minimum-objective plan for {j}: {classical_solution(d)[0]}")

elif page == "QAOA Lab":
    st.title("⚛️ QAOA Lab")
    st.markdown("<div class='card'><b>Quantum optimization experiment</b><br>4 binary plan variables → exactly-one constraint → QUBO → QAOA (reps=1) → statevector simulation.</div>", unsafe_allow_html=True)
    quantum_backend_note()
    j = st.selectbox("Junction for QAOA", JUNCTIONS)
    if btn("⚛️ RUN QAOA", key="qaoa_one", primary=True):
        with st.spinner("Running QAOA..."):
            r = qaoa_solution(TR[j])
        st.session_state.qaoa_results[j] = r
        log_event("Single-junction QAOA", {j: r.get("Plan")})
    if j in st.session_state.qaoa_results:
        r = st.session_state.qaoa_results[j]
        if r.get("Plan"):
            a, b, c, dc = st.columns(4)
            a.metric("QAOA Plan", r["Plan"]); b.metric("N-S Green", f"{r['NS_Green']}s"); c.metric("E-W Green", f"{r['EW_Green']}s"); dc.metric("Objective", f"{r['Cost']:,}")
            classical = classical_solution(TR[j])[0]
            st.success(f"Classical plan: {classical} | QAOA plan: {r['Plan']} | Match: {'YES' if classical == r['Plan'] else 'NO'} | Backend: {r.get('Backend')}")
            if r.get("Note"):
                st.caption(r["Note"])
        else:
            st.error(r.get("Error", "QAOA unavailable"))
    if btn("⚛️ RUN QAOA FOR ALL JUNCTIONS", key="qaoa_all"):
        with st.spinner("Running four small QAOA experiments..."):
            results = run_qaoa_all()
        show_df(pd.DataFrame([{"Junction": jj, "Classical": classical_solution(TR[jj])[0], "QAOA": results[jj].get("Plan"),
                               "Backend": results[jj].get("Backend"), "Status": results[jj].get("Status")} for jj in JUNCTIONS]))

elif page == "Quantum Validation":
    st.title("⚛️ Quantum Validation")
    st.markdown("<div class='card'><b>Why keep the quantum layer?</b><br>QAOA is run as a quantum optimization experiment on each 4-variable junction QUBO. "
                "It is not kept because it is faster than the classical baseline; matching the classical decision is the validation result. "
                "The QUBO/QAOA layer is the extension point for larger constrained optimisation problems.</div>", unsafe_allow_html=True)
    st.warning("Current result: no quantum advantage is claimed. A statevector simulator is used for reproducible research validation.")
    quantum_backend_note()
    if btn("⚛️ RUN QUANTUM VALIDATION", key="qv_run", primary=True):
        with st.spinner("Running four small QAOA experiments..."):
            run_qaoa_all()
    if st.session_state.qaoa_results:
        rows = []
        for j in JUNCTIONS:
            cp = classical_solution(TR[j])[0]
            res = st.session_state.qaoa_results.get(j, {})
            qp = res.get("Plan")
            rows.append({"Junction": j, "Classical Plan": cp, "QAOA Plan": qp, "Match": "YES" if qp and cp == qp else "NO / unavailable",
                         "Backend": res.get("Backend", "-")})
        show_df(pd.DataFrame(rows))
    else:
        st.info("Run Quantum Validation to execute the QAOA experiment.")

elif page == "Metrics":
    st.title("📈 Metrics")
    plan = get_plan()
    base, opt = network_kpis(TR, BASELINE_PLAN), network_kpis(TR, plan)
    m = st.columns(4)
    m[0].metric("Total waiting (veh·s)", f"{opt['waiting']:,.0f}", delta=f"{-reduction_pct(base['waiting'], opt['waiting']):.1f}% vs fixed-time", delta_color="inverse")
    m[1].metric("Avg delay (s/veh)", f"{opt['avg_delay']:.1f}", delta=f"{opt['avg_delay'] - base['avg_delay']:+.1f}", delta_color="inverse")
    m[2].metric("Queue proxy (veh)", f"{opt['queue']:.0f}", delta=f"{opt['queue'] - base['queue']:+.0f}", delta_color="inverse")
    m[3].metric("Stops / vehicle", f"{opt['stops']:.2f}", delta=f"{opt['stops'] - base['stops']:+.2f}", delta_color="inverse")
    rows = []
    for j in JUNCTIONS:
        mo, mb = simulate_plan(TR[j], plan[j]), simulate_plan(TR[j], BASELINE_PLAN[j])
        rows.append({"Junction": j, "Plan": plan[j], "Waiting (optimized)": round(mo["Waiting Time"], 1), "Waiting (fixed 30/30)": round(mb["Waiting Time"], 1),
                     "Queue": round(mo["Queue Length"], 1), "Travel": round(mo["Travel Time"], 1), "Stops / Vehicle": round(mo["Stops / Vehicle"], 2)})
    df = pd.DataFrame(rows)
    show_df(df)
    st.markdown("### Waiting time: fixed-time vs optimized")
    st.bar_chart(df.set_index("Junction")[["Waiting (fixed 30/30)", "Waiting (optimized)"]])
    st.caption("Optimized totals include a coordination trade-off between neighbouring junctions, so an individual junction can be slightly worse than fixed-time.")

elif page == "Dataset Validation":
    st.title("📁 Dataset Validation")
    st.write("Accepted columns: Junction, North, South, East, West (vehicles per minute)")
    sample = pd.DataFrame([{"Junction": j, **BASE_TRAFFIC[j]} for j in JUNCTIONS])
    st.download_button("📥 Download working sample dataset", sample.to_csv(index=False).encode("utf-8"), "traffic_sample_working.csv", "text/csv")
    uploaded = st.file_uploader("Upload traffic CSV", type=["csv"], key="traffic_csv")
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            required = ["Junction", "North", "South", "East", "West"]
            df.columns = [str(c).strip() for c in df.columns]
            missing = [x for x in required if x not in df.columns]
            if missing:
                st.error(f"Missing columns: {missing}")
            else:
                df["Junction"] = df["Junction"].astype(str).str.strip()
                for col in DIRECTIONS:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                if df[DIRECTIONS].isna().any().any():
                    st.error("North/South/East/West must contain only numeric values.")
                elif (df[DIRECTIONS] < 0).any().any():
                    st.error("Traffic counts cannot be negative.")
                elif set(df["Junction"]) != set(JUNCTIONS) or len(df) != 4:
                    st.error("The CSV must contain exactly J1, J2, J3 and J4, one row each.")
                    show_df(df)
                else:
                    st.success("✅ Dataset validated successfully.")
                    df = df[["Junction"] + DIRECTIONS]
                    show_df(df)
                    if btn("📥 USE THIS DATASET", key="ds_use", primary=True):
                        new = {row["Junction"]: {d: int(row[d]) for d in DIRECTIONS} for _, row in df.iterrows()}
                        set_traffic(new, "Custom Dataset", "Dataset loaded")
                        st.success("Dataset loaded. Open Network Optimizer and run it.")
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
    else:
        st.markdown("### Built-in working dataset")
        show_df(sample)
        if btn("Use built-in dataset", key="ds_builtin"):
            set_traffic({j: dict(d) for j, d in BASE_TRAFFIC.items()}, "Normal", "Built-in dataset restored")
            st.success("Built-in dataset restored.")

elif page == "Adaptive Experiments":
    st.title("🧪 Adaptive Experiments")
    st.markdown("Research question: <b>How much does adaptive optimisation reduce waiting versus fixed-time signals, and does QUBO-QAOA reproduce the classical decisions?</b>", unsafe_allow_html=True)
    st.info("⚡ The sweep never touches your current traffic or uploaded dataset; it works on copies. If a custom dataset or live-sensor traffic is active it is added as an extra row.")
    runq = st.checkbox("Add one QAOA spot-check per scenario (highest-demand junction, compared with that junction's local optimum)", value=False)
    c1, c2 = st.columns([2, 1])
    if btn("▶ RUN ADAPTIVE EXPERIMENT", key="exp_run", primary=True, container=c1):
        with st.spinner("Running scenario sweep..."):
            adaptive_experiment(runq)
        st.success("Adaptive experiment completed.")
    if btn("🗑 CLEAR EXPERIMENT", key="exp_clear", container=c2):
        st.session_state.adaptive_results = None
        st.rerun()
    if st.session_state.adaptive_results is not None:
        df = st.session_state.adaptive_results
        show_df(df)
        st.download_button("📥 Download results CSV", df.to_csv(index=False).encode("utf-8"), "adaptive_experiment_results.csv", "text/csv")
        st.markdown("### Waiting time: fixed-time vs optimized")
        st.bar_chart(df.set_index("Scenario")[["Fixed-time Waiting", "Optimized Waiting"]])

elif page == "Action Log":
    st.title("🧾 Action Log")
    if st.session_state.experiment_log:
        show_df(pd.DataFrame(st.session_state.experiment_log))
        if btn("🗑 Clear log", key="log_clear"):
            st.session_state.experiment_log = []
            st.rerun()
    else:
        st.info("No actions logged yet.")