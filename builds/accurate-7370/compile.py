"""Compile the rules into exact network weights. No training.

The network is a feed-forward stack of sparse blocks with skip connections,
run once per byte. Input x = [mode one-hot | registers | matcher state | byte one-hot].
  B1 (ReLU)   byte class c_k = sum of its bytes; base mode = relu(1 - other modes);
              matcher node = sum of the edges into it (only where that is cheaper)
  B2 (ReLU)   AND units: p = relu(mode_q + c_k - 1) (or relu(mode_q - other classes) when
              the cell is large); word category f_w = sum of matcher nodes (or the edges
              into them) that end a word of category w
  B3 (ReLU)   AND units: relu(p + register_v [+ f_w] - (arity - 1))
  B4 (ReLU)   next mode, register deltas, label and event one-hots, and the word
              matcher: edge = relu(source + byte - 1), where the source is a node unit
              (sum of incoming edges) or the incoming edges themselves; the state carries
              the edge that fired, not a node one-hot. Every B4 output is 0 or 1.
Backward cell, per byte, on one-hot register r:
  r' = event + relu(r - any_event) + sum over map events m, classes c of relu(r_c + m - 1) e_{M_m(c)}
Head: class = label + relu(wordlabel + r - 1)

The mode, the registers (context, f-string depth and quote, quote kind, triple,
f-prefix) and the label are one-hot with the most common value encoded as all zeros, so
nothing fires in the common case. A unit that depends on register values is derived by
Mobius inversion of the rule table over the register lattice: it only ever conditions on
non-base values.
"""
import struct
from collections import Counter

import numpy as np

import spec

NF = 11
ORDER, ROWS = spec.build_table()
IDS = {st: i for i, st in enumerate(ORDER)}

# --- 1. factor each rule state into (mode, registers) ----------------------------
FIELDS = {"code": ("ctx", "fr"), "word": ("ctx", "fr"), "at": ("ctx", "fr"),
          "bang": ("fr",), "bang2": ("fr",), "dot": ("fr",), "cmt": ("fr",),
          "int": ("fr",), "hex": ("fr",), "frac": ("fr",), "exp0": ("fr",), "exp": ("fr",), "numend": ("fr",),
          "q1": ("q", "f", "fr"), "q2": ("q", "f", "fr"),
          "str": ("q", "tr", "close", "f", "fr"), "fopen": ("q", "tr", "fr"), "esc": ("q", "tr", "f", "fr"),
          "escn": ("q", "tr", "f", "fr"), "nname": ("q", "tr", "f", "fr")}
INMODE = {"close"}  # fields that stay part of the mode name
# The f-string context fr = None | (quote, triple, depth, outer) is factored into three registers:
# fd (bracket depth, one-hot over 0..3, all zero outside an f-string expression), fq and
# ftr (the enclosing string's quote and triple), so units condition on single bits.
BASE = {"ctx": "none", "fd": None, "fq": 34, "ftr": 0, "od": None, "oq": 34, "otr": 0, "q": 34, "tr": 0, "f": False}
REGS = tuple(BASE)
BASE_MODE = "code"
BASE_LABEL = spec.TEXT


def factor(st):
    name, regs = st[0], {}
    for fld, v in zip(FIELDS[st[0]], st[1:]):
        if fld in INMODE:
            name = f"{name}{v}"
        elif fld == "fr":
            if v is not None:
                regs["fq"], regs["ftr"], regs["fd"] = v[:3]
                if v[3] is not None:  # the field around this one: nested f-strings
                    regs["oq"], regs["otr"], regs["od"] = v[3]
        else:
            regs[fld] = v
    return name, regs


modes, states_of = {}, {}
rvals = {x: set() for x in REGS}
for st in ORDER:
    q, regs = factor(st)
    modes.setdefault(q, len(modes))
    states_of.setdefault(q, []).append(st)
    for x, v in regs.items():
        rvals[x].add(v)
M = len(modes)
midx = {}  # mode -> input column offset; the base mode has none
for q in modes:
    if q != BASE_MODE:
        midx[q] = len(midx)
ridx = {}
for x in REGS:
    for v in sorted(rvals[x] - {BASE[x]}, key=str):
        ridx[(x, v)] = len(ridx)
R = len(ridx)

# --- 2. byte classes: bytes the rules never tell apart -----------------------------
sig = {}
bclass = [0] * 256
for b in range(256):
    key = tuple(ROWS[s][f][b] for s in range(len(ORDER)) for f in range(NF))
    bclass[b] = sig.setdefault(key, len(sig))
C = len(sig)
member = {}
for b, k in enumerate(bclass):
    member.setdefault(k, b)

# --- 3. minimise the word matcher: merge trie nodes with identical futures ---------
T, TERM = spec.TRIE, spec.TERM
cls = {n: TERM.get(n, spec.F_NONE) for n in range(len(T))}
while True:
    key = {n: (cls[n], tuple(sorted((b, cls[c]) for b, c in T[n].items()))) for n in range(len(T))}
    new_ids = {}
    nxt = {n: new_ids.setdefault(key[n], len(new_ids)) for n in range(len(T))}
    if len(new_ids) == len(set(cls.values())):
        break
    cls = nxt
order = sorted(set(cls.values()), key=lambda c: c != cls[0])  # root first
node_id = {c: i for i, c in enumerate(order)}
N = len(order)
EDGES = sorted({(node_id[cls[n]], b, node_id[cls[c]]) for n in range(len(T)) for b, c in T[n].items()})
NTERM = {}
for n, w in TERM.items():
    NTERM[node_id[cls[n]]] = w

# --- 3b. matcher state = the edge that fired last. An edge unit is an AND of its byte(s)
# and its source; the source is either a materialised node unit (sum of its incoming
# edges) or, when that is cheaper, the incoming edges themselves. Edges with the same
# source and target share one unit, since their bytes are mutually exclusive.
EBYTES = {}
for i, b, c in EDGES:
    EBYTES.setdefault((i, c), []).append(b)
EUNITS = sorted(EBYTES)
IN_OF, OUT_OF = {}, {}
for i, c in EUNITS:
    IN_OF.setdefault(c, []).append((i, c))
    OUT_OF.setdefault(i, []).append((i, c))
MAT = set()  # no node units: a state-derived hop before the edges costs more time than weights
SLOT = {("node", 0): 0}
for e in EUNITS:
    SLOT[("edge",) + e] = len(SLOT)
NS = len(SLOT)


def node_cols(n):
    """Columns whose sum is the one-hot "matcher is at node n"."""
    if n == 0:
        return [oN + SLOT[("node", 0)]]
    if n in MAT:
        return [gidx(B1, NODE_ROW[n])]
    return [oN + SLOT[("edge",) + e] for e in IN_OF.get(n, [])]


# --- layout -----------------------------------------------------------------------
oM, oR, oN = 0, len(midx), len(midx) + R
NSTATE = oN + NS
oB = NSTATE
NIN = oB + 256
LAB, EV = 10, 9 + len(spec.EV_MAP)
oL, oE = NSTATE, NSTATE + LAB
NOUT = oE + EV


class Block:
    def __init__(self, relu):
        self.relu, self.rows, self.bias, self.entries = relu, 0, [], []

    def unit(self, inputs, bias=0.0):
        """inputs: list of (global column, weight)."""
        i = self.rows
        self.rows += 1
        self.bias.append(bias)
        for j, w in inputs:
            self.entries.append((i, j, w))
        return i


B1, B2, B3, B4 = Block(1), Block(1), Block(1), Block(1)
blocks = [B1, B2, B3, B4]
base_of = {}


def begin(blk):
    """Fix the block's place in the global index space: inputs, then blocks in order.
    Called once the previous blocks are complete."""
    base_of[id(blk)] = NIN + sum(b.rows for b in blocks[:blocks.index(blk)])


def gidx(blk, i):
    return base_of[id(blk)] + i


# --- B1: byte classes, base mode, materialised matcher nodes ----------------------
begin(B1)
CLS_ROW = {}
for k in range(C):
    CLS_ROW[k] = B1.unit([(oB + b, 1.0) for b in range(256) if bclass[b] == k])
WORD_CLS = sorted({bclass[b] for b in spec.WORDCH})  # classes never mix word and non-word bytes
MODE0 = B1.unit([(oM + i, -1.0) for i in range(len(midx))], 1.0)
NODE_ROW = {}
for n in sorted(MAT):
    NODE_ROW[n] = B1.unit([(oN + SLOT[("edge",) + e], 1.0) for e in IN_OF[n]])
begin(B2)
FLAG_ROW = {}
for w in range(1, NF):
    FLAG_ROW[w] = B2.unit([(j, 1.0) for i, t in sorted(NTERM.items()) if t == w for j in node_cols(i)])


def mode_col(q):
    return gidx(B1, MODE0) if q == BASE_MODE else oM + midx[q]


# --- the rule table on factored states --------------------------------------------


def outcome(st, f, k):
    """Output vector for (state, word category, byte class): {output row: weight}."""
    n, lab, ev = ROWS[IDS[st]][f][member[k]]
    q2, regs2 = factor(ORDER[n])
    _, regs = factor(st)
    vec = Counter()
    if q2 != BASE_MODE:
        vec[oM + midx[q2]] = 1.0
    if lab != BASE_LABEL:
        vec[oL + lab] = 1.0
    if ev >= 0:
        vec[oE + ev] = 1.0
    for x in REGS:
        old, new = regs.get(x, BASE[x]), regs2.get(x, BASE[x])
        if old != new:
            if new != BASE[x]:
                vec[oR + ridx[(x, new)]] += 1.0
            if old != BASE[x]:
                vec[oR + ridx[(x, old)]] -= 1.0
    return tuple(sorted((r, v) for r, v in vec.items() if v))


def nonbase(st, f):
    _, regs = factor(st)
    t = {(x, v) for x, v in regs.items() if v != BASE[x]}
    if f != spec.F_NONE:
        t.add(("flag", f))
    return frozenset(t)


def mobius(F):
    """F: {set of non-base conditions: vector}. Returns corrections g with
    F[T] = sum of g[U] over U subset of T, using only conditions that occur."""
    g = {}
    for Tset in sorted(F, key=len):
        pred = Counter()
        for U, gu in g.items():
            if U < Tset:
                pred.update(dict(gu))
        d = Counter(dict(F[Tset]))
        d.subtract(pred)
        d = tuple(sorted((r, v) for r, v in d.items() if v))
        if d:
            g[Tset] = d
    return g


def cond_col(c):
    x, v = c
    return gidx(B2, FLAG_ROW[v]) if x == "flag" else oR + ridx[(x, v)]


# --- cells: for each mode, the byte classes it treats alike; Mobius corrections ----
CELLS, PLAN = {}, {}
for q in modes:
    cells = {}
    for k in range(C):
        s = tuple(outcome(st, f, k) for st in states_of[q] for f in range(NF))
        cells.setdefault(s, []).append(k)
    CELLS[q] = cells
    for s, ks in cells.items():
        F = {}
        for st in states_of[q]:
            for f in range(NF):
                F[nonbase(st, f)] = outcome(st, f, ks[0])
        PLAN[(q, tuple(ks))] = mobius(F)

# --- B2: mode AND units p = relu(mode + class - 1), only for cells that do something --
P_ROW = {}
for q in modes:
    cells = CELLS[q]
    for s, ks in cells.items():
        if not PLAN[(q, tuple(ks))]:
            continue
        # p is a global column: the mode itself when it has one cell; otherwise an AND
        # unit over the cell's classes, or over the complement when that is smaller:
        # relu(mode - sum of other classes) is the same function.
        if len(cells) == 1:
            P_ROW[(q, tuple(ks))] = mode_col(q)
        elif len(ks) <= C - len(ks):
            P_ROW[(q, tuple(ks))] = gidx(B2, B2.unit([(mode_col(q), 1.0)] + [(gidx(B1, CLS_ROW[k]), 1.0) for k in ks], -1.0))
        else:
            P_ROW[(q, tuple(ks))] = gidx(B2, B2.unit([(mode_col(q), 1.0)] + [(gidx(B1, CLS_ROW[k]), -1.0) for k in range(C) if k not in ks]))
begin(B3)

# --- B3: register-conditioned corrections, shared between cells with the same one --
cond_units = {}  # (conditions, correction vector) -> list of p columns feeding it
for (q, ks), g in PLAN.items():
    if not g:
        continue
    p = P_ROW[(q, ks)]
    for Tset, gv in g.items():
        if not Tset:
            for r, v in gv:
                B4.entries.append((r, p, v))
        else:
            cond_units.setdefault((tuple(sorted(Tset, key=str)), gv), []).append(p)
for (conds, gv), ps in cond_units.items():
    u = B3.unit([(p, 1.0) for p in ps] + [(cond_col(c), 1.0) for c in conds], -float(len(conds)))
    for r, v in gv:
        B4.entries.append((r, gidx(B3, u), v))
begin(B4)

# --- B4: outputs. Registers pass through unless a unit changes them. Matcher: edge units
# (AND of source and byte); the root is on after a non-word byte.
for i in range(R):
    B4.entries.append((oR + i, oR + i, 1.0))
B4.rows = NOUT
B4.bias = [0.0] * NOUT
B4.bias[oN + 0] = 1.0
for k in WORD_CLS:
    B4.entries.append((oN + 0, gidx(B1, CLS_ROW[k]), -1.0))
for i, c in EUNITS:
    r = oN + SLOT[("edge", i, c)]
    B4.bias[r] = -1.0
    for j in node_cols(i):
        B4.entries.append((r, j, 1.0))
    for b in EBYTES[(i, c)]:
        B4.entries.append((r, oB + b, 1.0))


# --- pack -------------------------------------------------------------------------
def dedupe(entries):
    acc = {}
    for i, j, v in entries:
        acc[(i, j)] = acc.get((i, j), 0.0) + v
    return [(i, j, v) for (i, j), v in acc.items() if v != 0.0]


def pack(blk, rbase):
    """Column-compressed over the global columns this block reads."""
    ent = sorted(dedupe(blk.entries), key=lambda e: (e[1], e[0]))
    lo = min(j for _, j, _ in ent)
    hi = max(j for _, j, _ in ent) + 1
    ptr = np.zeros(hi - lo + 1, dtype=np.int32)
    for _, j, _ in ent:
        ptr[j - lo + 1] += 1
    ptr = np.cumsum(ptr).astype(np.int32)
    idx = np.array([e[0] for e in ent], dtype=np.int32)
    val = np.array([e[2] for e in ent], dtype=np.float32)
    head = struct.pack("<6i", blk.rows, lo, hi, len(ent), blk.relu, rbase)
    return len(ent), head + ptr.tobytes() + idx.tobytes() + val.tobytes() + np.asarray(blk.bias, dtype=np.float32).tobytes()


start_mode, start_regs = factor(spec.START)
init = ([oM + midx[start_mode]] if start_mode != BASE_MODE else []) \
    + [oR + ridx[(x, v)] for x, v in start_regs.items() if v != BASE[x]] + [oN + 0]
BOUND = 1  # every activation in this network is 0 or 1
header = struct.pack("<10i", NIN, NSTATE, len(blocks), len(init), oL, LAB, oE, EV, oB, BOUND) + np.array(init, dtype=np.int32).tobytes() \
    + np.array(spec.EV_MAP, dtype=np.int32).tobytes()
blob = header
nnz = 0
per = []
rbase = NIN
for blk in blocks:
    assert base_of[id(blk)] == rbase
    n, packed = pack(blk, rbase)
    nnz += n
    per.append(n)
    blob += packed
    rbase += blk.rows
open("pylexnet_weights.bin", "wb").write(blob)
print(f"modes {M}, registers {R}, byte classes {C}, matcher nodes {N} ({len(EDGES)} edges, {len(EUNITS)} edge units, {len(MAT)} node units)")
print(f"AND units: mode {B2.rows - (NF - 1)}, conditioned {B3.rows}, matcher {len(EUNITS)}")
print(f"non-zero weights {nnz} (blocks {per}), rows {[b.rows for b in blocks]}")
print(f"weights file {len(blob) / 1024:.0f} KiB")
