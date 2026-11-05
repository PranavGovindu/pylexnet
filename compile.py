"""pylexnet 'hand' lane: a Python syntax-highlighter network designed unit by unit, lexnet style.

No automaton is compiled. Every unit below does one designed job: ~60 booleans for strings, comments,
numbers, words and contexts, a length counter, two hash-gate registers, a sorted threshold family on the
word hash, and one "distance to the nearest vocabulary word of category c" unit per category.
Activation is clip(x, 0, hi) everywhere.

Two vocabulary matchers, chosen by --matcher:
  add  (default) additive retrieval: per-position value tables (positions 0..P-1 plus a constant each, a tail
       table, three last-byte tables) solved so that every vocabulary word's sum of values is its class target
       t*2^30 (t = 1 keyword, 2 def, 3 class, 4 builtin, 5 magic), rounded to integers (window +-64). A one-hot
       position chain gates the values into an accumulator; three delayed registers add the last bytes' values at
       the word's end; nine threshold units read the class windows. ~1 number per word. Not exact: a non-vocabulary
       word whose sum lands in a window is a false positive (fpcheck.py measures them on the fit split; pinning
       words to 0 via zero_words.txt is possible when the system stays solvable).
  hash exact: a polynomial hash h over the word, a sorted tied threshold family ge_k = clip1(h - T_k + 1) (the rank
       of h), and per category E_c = clip1(h - sum_k ge_k * D_c[k]) with D_c telescoping to the largest category-c
       threshold <= h, so E_c == 0 exactly when h is a category-c word. 2 numbers per word.

Writes pylexnet_weights.bin (PYLX02) and pylexnet_kernel.h (architecture only, with a structure hash).
usage: compile.py [--matcher add|hash] [--vocab N] [--addP P] [--sim TEXT]   N = builtin/magic words kept, by fit frequency.
"""
import keyword
import re
import struct
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------- byte features (the embedding table)
FEATS = ["digit", "nl", "dq", "sq", "bs", "hash", "dot", "at", "lbr", "rbr",
         "obr", "cbr", "fcch", "code", "pf", "po", "eE", "pm"]
MATCHER = "add"        # "add": additive retrieval (IDEAS.md idea 1, default). "hash": sorted threshold family on a polynomial hash
ADD_P, ADD_L = 7, 3    # additive matcher: value tables (+ a constant) for positions 0..P-1, a tail table, 3 last-byte tables (L is fixed at 3)
ADD_EPS = 64           # window half-width around each class target
ADD_T = 2 ** 30        # class targets are t * ADD_T, t = 1 keyword, 2 def, 3 class, 4 builtin, 5 magic
if "--matcher" in sys.argv:
    MATCHER = sys.argv[sys.argv.index("--matcher") + 1]
if "--addP" in sys.argv:
    ADD_P = int(sys.argv[sys.argv.index("--addP") + 1])
if "--addL" in sys.argv:
    ADD_L = int(sys.argv[sys.argv.index("--addL") + 1])
if MATCHER == "add":
    FEATS += [f"v{p}" for p in range(ADD_P)] + ["vt"] + [f"e{j}" for j in range(1, ADD_L + 1)]
NF = len(FEATS)
FI = {f: i for i, f in enumerate(FEATS)}
ADD_TABLES = {}        # feature name -> {code: value}, filled by solve_additive


def code_of(b):
    """Word-character code for the hash: a-z 1..26, A-Z 27..52, _ 53, digits 54, bytes >= 127 55, else 0."""
    if 97 <= b <= 122:
        return b - 96
    if 65 <= b <= 90:
        return b - 64 + 26
    if b == 95:
        return 53
    if 48 <= b <= 57:
        return 54
    if b >= 127:
        return 55
    return 0


BASE = 56          # hash base: one more than the largest code
WIN = 7            # exact polynomial over the first WIN chars; later chars add code + 8*position
HMAX = 2 ** 41     # ceiling of the hash registers (> BASE**WIN + tail)
KG = 2 ** 42       # gate constant for the hash units: larger than any hash value


def emb_table():
    E = np.zeros((128, NF))
    for b in range(128):
        r = E[b]
        c = code_of(b)
        isalpha = (65 <= b <= 90) or (97 <= b <= 122) or b == 95 or b == 127
        isdigit = 48 <= b <= 57
        r[FI["digit"]] = isdigit
        r[FI["nl"]] = b == 10
        r[FI["dq"]] = b == 34
        r[FI["sq"]] = b == 39
        r[FI["bs"]] = b == 92
        r[FI["hash"]] = b == 35
        r[FI["dot"]] = b == 46
        r[FI["at"]] = b == 64
        r[FI["lbr"]] = b == 123
        r[FI["rbr"]] = b == 125
        r[FI["obr"]] = b in (40, 91, 123)
        r[FI["cbr"]] = b in (41, 93, 125)
        r[FI["fcch"]] = b in (125, 58, 33)       # } : !  end a replacement field at depth 0
        r[FI["code"]] = c
        r[FI["pf"]] = b in (102, 70)             # f F
        r[FI["eE"]] = b in (101, 69)
        r[FI["pm"]] = b in (43, 45)
        r[FI["po"]] = b in (114, 82, 98, 66, 117, 85)   # r R b B u U
        for f, tab in ADD_TABLES.items():
            r[FI[f]] = tab.get(c, 0) if c else 0
    return E


def word_feats(w):
    """Additive matcher: the (table, code) terms of a word."""
    cs = [code_of(b) for b in w.encode()]
    out = {}
    for i, c in enumerate(cs):
        k = (f"v{i}" if i < ADD_P else "vt", c)
        out[k] = out.get(k, 0) + 1
        k = (k[0], 0)                       # the position's constant (the gate unit's bias)
        out[k] = out.get(k, 0) + 1
    for j in range(1, ADD_L + 1):
        if len(cs) >= j:
            k = (f"e{j}", cs[-j])
            out[k] = out.get(k, 0) + 1
    return out


def solve_additive(vocab):
    """Integer value tables such that every vocabulary word's sum lands on its class target (within ADD_EPS).
    Orthogonal matching pursuit picks a sparse support, least squares solves it, rounding keeps it integer."""
    words = sorted(vocab)
    tgt = {"KW": 1, "BI": 4, "MAG": 5}
    zero = [w for w in (HERE / "zero_words.txt").read_text().split() if w not in vocab] if (HERE / "zero_words.txt").exists() else []
    T = np.array([ADD_T * (2 if w == "def" else 3 if w == "class" else tgt[vocab[w]]) for w in words] + [0.0] * len(zero))
    words = words + zero          # frequent non-vocabulary words pinned to 0 (chosen on the fit split, see zero_words.txt)
    cols = {}
    rows = [word_feats(w) for w in words]
    for r in rows:
        for k in r:
            cols.setdefault(k, len(cols))
    A = np.zeros((len(words), len(cols)))
    for i, r in enumerate(rows):
        for k, n in r.items():
            A[i, cols[k]] = n
    S, x = [], np.zeros(len(cols))
    for _ in range(len(cols)):
        res = T - A @ x
        if np.abs(res).max() < 1e-3:
            break
        corr = A.T @ res
        corr[S] = 0
        j = int(np.argmax(np.abs(corr)))
        if corr[j] == 0:
            break
        S.append(j)
        xs, *_ = np.linalg.lstsq(A[:, S], T, rcond=None)
        x = np.zeros(len(cols)); x[S] = xs
    x = np.round(x)
    res = A @ x - T
    unsolved = int((np.abs(res) > ADD_EPS).sum())
    for (f, c), j in cols.items():
        if x[j] != 0:
            ADD_TABLES.setdefault(f, {})[c] = int(x[j])
    nz = int((x != 0).sum())
    vmax = max(abs(v) for tab in ADD_TABLES.values() for v in tab.values())
    print(f"additive matcher: {len(words) - len(zero)} words + {len(zero)} zero-pinned, {len(cols)} candidate values, support {nz}, max |value| {vmax}, "
          f"unsolved {unsolved}, max residual {np.abs(res).max():g}")
    assert unsolved == 0, "additive system not solvable with these features"
    return vmax


def word_hash(w):
    h = 0
    for i, ch in enumerate(w.encode()):
        c = code_of(ch)
        h = BASE * h + c if i < WIN else h + c + 8 * i
    return h


# ---------------------------------------------------------------- vocabulary
import pygments.lexers.python as _pl
_src = Path(_pl.__file__).read_text()
_b = _src.index("'builtins': [")
_a = _src.index("'magicfuncs': [")
_v = _src.index("'magicvars': [")
BUILTINS = set(re.findall(r"'(\w+)'", _src[_b:_a])) - {"builtins", "exceptions"} | {"self", "cls", "Ellipsis", "NotImplemented"}
MAGIC = set(re.findall(r"'(__\w+__)'", _src[_a:_v])) - BUILTINS
KEYWORDS = set(keyword.kwlist)
# frequency order measured once on accuracy2's fit split (vocab_prio.txt); words below its cut-off come last, alphabetically
PRIO = [w for w, _ in eval("[" + (HERE / "vocab_prio.txt").read_text() + "]")]


def pick_vocab(limit):
    """Keywords always; builtins and magic words by fit-corpus frequency up to `limit`."""
    ranked = [w for w in PRIO if w in BUILTINS or w in MAGIC]
    ranked += sorted((BUILTINS | MAGIC) - set(ranked))
    chosen = ranked if limit is None else ranked[:limit]
    return {w: ("BI" if w in BUILTINS else "MAG") for w in chosen} | {w: "KW" for w in KEYWORDS}


# ---------------------------------------------------------------- the network, unit by unit
class Unit:
    def __init__(self, name, ins, b=0.0, hi=1.0, guard=None, fam=None):
        self.name, self.ins, self.b, self.hi, self.guard, self.fam = name, ins, float(b), float(hi), guard, fam


UNITS = []


def u(name, b=0.0, hi=1.0, guard=None, fam=None, **ins):
    """ins: ref -> weight; refs written as x_feat (byte feature), p_name (previous state) or a unit name."""
    d = {}
    for k, w in ins.items():
        ref = "x." + k[2:] if k.startswith("x_") else "p." + k[2:] if k.startswith("p_") else "u." + k
        d[ref] = float(w)
    UNITS.append(Unit(name, d, b, hi, guard, fam))
    return name


STATE = ["S", "sq1", "T", "F", "E", "c", "fo", "FS", "FD", "oq1", "oT", "C", "N", "ee", "W", "d", "pf", "fp",
         "cD", "cC", "cAt", "atp", "cDot"]
STATE += ["h1"] if MATCHER == "hash" else [f"pos{p}" for p in range(ADD_P)] + ["tail", "A", "e1", "e2", "e2d", "e3", "e3d", "e3dd"]
STATE_UNIT = {s: s + "_new" for s in STATE}
for _p in range(ADD_P):
    STATE_UNIT[f"pos{_p}"] = f"pos{_p}"
STATE_UNIT["tail"] = "tail"
CATS = ["KW", "BI", "MAG"]


def build(vocab):
    UNITS.clear()
    # ---- layer 0: from the byte and the previous state only
    u("wordch", guard=["x.code"], x_code=1)                                           # word character (any code > 0)
    u("qmd", guard=["x.dq"], x_dq=1, p_sq1=-1, p_E=-1)             # unescaped matching " in a "-string
    u("qms", guard=["x.sq"], x_sq=1, p_sq1=1, p_E=-1, b=-1)        # unescaped matching ' in a '-string
    u("c2", guard=["p.c"], p_c=1, b=-1)                             # quote run >= 2
    if MATCHER == "hash":
        u("lt7", p_d=-1, b=WIN)                                     # fewer than WIN chars of the word seen
    u("wend", guard=["p.W"], p_W=1, x_code=-1)                      # a word ended just before this byte
    u("fieldopen", guard=["p.fo"], p_fo=1, x_lbr=-1)                # { was a field start: this byte is code
    u("pf_new", guard=["x.po", "x.pf"], x_po=3, x_pf=3, p_pf=2, p_W=-1, p_d=-1, b=-2)   # word so far could be a string prefix
    # ---- strings. S_eff: this byte is inside a string (p.S unless "" just closed or a field just opened)
    u("S_eff", guard=["p.S"], p_S=4, fieldopen=-4, p_T=1, qmd=1, qms=1, c2=-1, b=-3)
    u("fp_new", guard=["x.pf", "p.fp"], x_pf=1, p_fp=1, wordch=1, b=-1)   # word contains f
    u("sclose", guard=["u.S_eff"], S_eff=2, p_T=-1, qmd=1, qms=1, p_c=-1, b=-2)
    u("tclose", guard=["u.S_eff"], S_eff=3, p_T=1, qmd=1, qms=1, p_c=1, b=-6)
    u("tripleopen", guard=["u.S_eff"], S_eff=3, p_T=-1, qmd=1, qms=1, p_c=1, b=-5)
    u("nlclose", guard=["u.S_eff"], S_eff=1, p_T=-1, x_nl=1, p_E=-1, b=-1)
    u("E_new", guard=["u.S_eff"], S_eff=1, x_bs=1, p_E=-1, b=-1)
    u("fo_new", guard=["u.S_eff"], S_eff=1, p_F=1, x_lbr=1, p_E=-1, p_fo=-1, b=-2)
    u("C_new", guard=["x.hash", "p.C"], x_hash=1, S_eff=-1, p_C=1, x_nl=-1)
    u("open", guard=["x.dq", "x.sq"], x_dq=1, x_sq=1, S_eff=-1, p_C=-1, p_N=-1)
    u("open_f", guard=["x.dq", "x.sq"], x_dq=1, x_sq=1, S_eff=-1, p_C=-1, p_N=-1, p_pf=1, p_fp=1, b=-2)
    u("open_qs", guard=["x.sq"], x_dq=1, x_sq=2, S_eff=-1, p_C=-1, p_N=-1, b=-1)
    u("fieldclose", guard=["p.FS"], p_FS=1, S_eff=-1, p_C=-1, p_FD=-1, x_fcch=1, b=-1)
    u("inc", guard=["p.FS"], p_FS=1, S_eff=-1, p_C=-1, x_obr=1, b=-1)
    u("dec", guard=["p.FS"], p_FS=4, S_eff=-4, p_C=-4, x_cbr=1, p_FD=1, b=-5)
    u("S_new", guard=["u.S_eff", "u.open", "u.fieldclose"], S_eff=1, sclose=-1, tclose=-1, nlclose=-1, open=1, fieldclose=1)
    u("c_new", hi=3, guard=["u.S_eff", "u.open", "u.open_qs"], p_c=1, S_eff=10, qmd=5, qms=5, tripleopen=-5, tclose=-5, sclose=-5,
      open=10, open_qs=5, b=-14)
    u("T_new", guard=["p.T", "u.tripleopen", "u.fieldclose"], p_T=2, tripleopen=2, tclose=-2, open=-2, fieldclose=1, p_oT=1, b=-1)
    u("F_new", guard=["p.F", "u.open_f", "u.fieldclose"], p_F=1, open_f=2, open=-1, sclose=-1, tclose=-1, nlclose=-1, fieldclose=1)
    u("sq1_new", guard=["p.sq1", "u.open_qs", "u.S_eff", "u.fieldclose"], p_sq1=2, open_qs=4, sclose=-2, tclose=-2, nlclose=-2,
      p_S=-2, S_eff=2, fieldclose=1, p_oq1=1, b=-1)     # quote type; cleared when the string dies (unless a nested one opens); restored from oq1
    u("FS_new", guard=["p.FS", "u.fieldopen"], p_FS=1, fieldopen=1, fieldclose=-1)
    u("FD_new", hi=3, guard=["p.FD", "u.inc"], p_FD=1, inc=1, dec=-1)
    u("oq1_new", guard=["p.oq1", "u.fieldopen"], p_oq1=2, fieldopen=1, p_sq1=1, fieldclose=-2, b=-1)   # outer quote type, saved at a field's start
    u("oT_new", guard=["p.oT", "u.fieldopen"], p_oT=2, fieldopen=1, p_T=1, fieldclose=-2, b=-1)
    u("lab_STR", guard=["u.S_eff", "u.open", "u.fieldclose"], S_eff=1, nlclose=-1, open=1, fieldclose=1)
    # ---- numbers and words
    u("N_new", guard=["x.digit", "p.N"], x_digit=6, p_W=-3, p_N=5, wordch=2, x_dot=2, x_pm=1, p_ee=1, S_eff=-2, p_C=-2, b=-6)
    u("ee_new", guard=["u.N_new"], x_eE=1, N_new=1, b=-1)                    # an exponent e: a sign may follow
    u("W_new", guard=["u.wordch", "p.W"], wordch=2, x_digit=-1, p_W=1, p_N=-2, S_eff=-2, p_C=-2, b=-1)
    u("d_new", hi=30, guard=["u.W_new"], p_d=1, W_new=40, b=-39)
    if MATCHER == "hash":
        u("gA", hi=HMAX, guard=["p.W"], p_h1=1, p_W=KG, wordch=KG, lt7=KG, b=-3 * KG)       # old hash, still in the window
        u("gB", hi=HMAX, guard=["p.W"], p_h1=1, p_W=KG, wordch=KG, lt7=-KG, b=-2 * KG)      # old hash, past the window
        u("cA", hi=HMAX, guard=["u.W_new"], x_code=1, W_new=KG, lt7=KG, b=-2 * KG)          # this char's code, in window
        u("cB", hi=HMAX, guard=["u.W_new"], x_code=1, p_d=8, W_new=KG, lt7=-KG, b=-KG)      # past the window: code + 8*position
        u("h1_new", hi=HMAX, gA=BASE, gB=1, cA=1, cB=1)
    else:
        build_additive(vocab)
    if MATCHER == "add":
        u("is_def", guard=["u.wend"], wend=1, ge2=1, le2=1, b=-2)
        u("is_class", guard=["u.wend"], wend=1, ge3=1, le3=1, b=-2)
        u("cD_new", guard=["p.cD", "u.is_def"], p_cD=1, is_def=2, wend=-1, x_nl=-1)
        u("cC_new", guard=["p.cC", "u.is_class"], p_cC=1, is_class=2, wend=-1, x_nl=-1)
        u("atp_new", guard=["x.at"], x_at=1, S_eff=-1, p_C=-1)
        u("cAt_new", guard=["p.cAt", "p.atp"], p_cAt=1, p_atp=1, wordch=1, wend=-1, b=-1)
        u("cDot_new", guard=["x.dot", "p.cDot"], x_dot=4, S_eff=-4, p_C=-4, p_N=-4, p_cDot=2, wordch=1, wend=-1, b=-2)
        u("ev_DEC", guard=["u.wend"], wend=1, p_cAt=1, b=-1)
        u("ev_CL", guard=["u.wend"], wend=1, p_cC=1, p_cAt=-1, b=-1)
        # keyword targets 1, 2 (def), 3 (class): a value in one of the three windows scores 4 on the sorted sum, 3 otherwise
        u("ev_KW", guard=["u.wend"], wend=3, ge1=1, le1=1, ge2=1, le2=1, ge3=1, le3=1, p_cAt=-1, p_cC=-1, p_cD=-1, b=-6)
        u("ev_FN", guard=["u.wend"], wend=3, ge5=1, le5=1, p_cD=2, p_cAt=-2, p_cC=-2, b=-4)   # after def, or a magic method name
        u("ev_BI", guard=["u.wend"], wend=1, ge4=1, le4=1, p_cAt=-1, p_cC=-1, p_cD=-1, p_cDot=-1, b=-2)
        u("ev_STR", guard=["u.wend"], wend=1, open=1, p_pf=1, b=-2)
        u("lab_W", guard=["u.W_new", "u.atp_new"], W_new=1, atp_new=1)
        return [], []
    # ---- vocabulary: sorted thresholds (one tied family), one distance unit per category
    byhash = {}
    for w, cat in sorted(vocab.items()):
        byhash.setdefault(word_hash(w), set()).add(cat)
    for h, cats in byhash.items():
        assert len(cats) == 1, ("hash collision across categories", h, cats)
    hashes = sorted(byhash)
    cat_of = [next(iter(byhash[h])) for h in hashes]
    for k, h in enumerate(hashes):
        u(f"ge_{k}", p_h1=1, b=1 - h, fam="ge")
    for cat in CATS:
        prev = 0
        deltas = {}
        for k, h in enumerate(hashes):
            if cat_of[k] == cat:
                deltas[f"ge_{k}"] = -(h - prev)
                prev = h
        u(f"E_{cat}", p_h1=1, **deltas)       # 0 exactly when h is a category word, else 1
    kdef, kclass = hashes.index(word_hash("def")), hashes.index(word_hash("class"))
    nxt = lambda k: {f"ge_{k + 1}": -1} if k + 1 < len(hashes) else {}
    u("is_def", guard=["u.wend"], wend=1, E_KW=-1, b=-1, **{f"ge_{kdef}": 1}, **nxt(kdef))
    u("is_class", guard=["u.wend"], wend=1, E_KW=-1, b=-1, **{f"ge_{kclass}": 1}, **nxt(kclass))
    # ---- contexts: after def, after class, after @, after a dot
    u("cD_new", guard=["p.cD", "u.is_def"], p_cD=1, is_def=2, wend=-1, x_nl=-1)
    u("cC_new", guard=["p.cC", "u.is_class"], p_cC=1, is_class=2, wend=-1, x_nl=-1)
    u("atp_new", guard=["x.at"], x_at=1, S_eff=-1, p_C=-1)
    u("cAt_new", guard=["p.cAt", "p.atp"], p_cAt=1, p_atp=1, wordch=1, wend=-1, b=-1)
    u("cDot_new", guard=["x.dot", "p.cDot"], x_dot=4, S_eff=-4, p_C=-4, p_N=-4, p_cDot=2, wordch=1, wend=-1, b=-2)
    # ---- events at a word's end (read by the backward pass) and the word label
    u("ev_DEC", guard=["u.wend"], wend=1, p_cAt=1, b=-1)
    u("ev_CL", guard=["u.wend"], wend=1, p_cC=1, p_cAt=-1, b=-1)
    u("ev_KW", guard=["u.wend"], wend=1, E_KW=-1, p_cAt=-1, p_cC=-1, p_cD=-1)
    u("ev_FN", guard=["u.wend"], wend=2, E_MAG=-1, p_cD=1, p_cAt=-1, p_cC=-1, b=-1)       # after def, or a magic method name
    u("ev_BI", guard=["u.wend"], wend=1, E_BI=-1, p_cAt=-1, p_cC=-1, p_cD=-1, p_cDot=-1)
    u("ev_STR", guard=["u.wend"], wend=1, open=1, p_pf=1, b=-2)                   # the word was a string prefix
    u("lab_W", guard=["u.W_new", "u.atp_new"], W_new=1, atp_new=1)
    return hashes, cat_of


def build_additive(vocab):
    """Additive retrieval: A accumulates shifted position values over the word, three delayed last-byte values are added
    at the end, and the class is read from windows around the targets."""
    vmax = solve_additive(vocab)
    C = 2 * vmax                  # shift: every gated value (+ its position constant) is >= 0; the readout subtracts C per byte
    M = 4 * C + 4                 # gate constant
    AMAX = 2 ** 45; K = 2 ** 46
    u("pos0", guard=["u.W_new"], W_new=1, p_W=-1)                                   # first byte of a word
    for p in range(1, ADD_P):
        u(f"pos{p}", guard=[f"p.pos{p - 1}"], **{f"p_pos{p - 1}": 1}, W_new=1, b=-1)
    u("tail", guard=[f"p.pos{ADD_P - 1}", "p.tail"], **{f"p_pos{ADD_P - 1}": 1}, p_tail=1, W_new=1, b=-1)
    for p in range(ADD_P):
        u(f"a{p}", hi=2 * C, guard=[f"u.pos{p}"], **{f"x_v{p}": 1, f"pos{p}": M}, b=C - M + ADD_TABLES.get(f"v{p}", {}).get(0, 0))
    u("at_", hi=2 * C, guard=["u.tail"], x_vt=1, tail=M, b=C - M + ADD_TABLES.get("vt", {}).get(0, 0))
    u("A_new", hi=AMAX, guard=["u.W_new"], p_A=1, W_new=K, b=-K, **{f"a{p}": 1 for p in range(ADD_P)}, at_=1)
    for j in range(1, ADD_L + 1):
        u(f"e{j}_new", hi=2 * C, guard=["u.W_new"], **{f"x_e{j}": 1}, W_new=M, b=C - M)
    u("e2d_new", hi=2 * C, guard=["p.e2"], p_e2=1)
    u("e3d_new", hi=2 * C, guard=["p.e3"], p_e3=1)
    u("e3dd_new", hi=2 * C, guard=["p.e3d"], p_e3d=1)
    u("m3", hi=3, guard=["p.d"], p_d=1)                                               # min(length, 3): last-byte terms present
    u("SE", hi=AMAX, p_A=1, p_e1=1, p_e2d=1, p_e3dd=1, p_d=-C, m3=-C)   # the word's value sum
    for t in (1, 2, 3, 4, 5):
        u(f"ge{t}", SE=1, b=1 - (t * ADD_T - ADD_EPS))
        u(f"le{t}", SE=-1, b=1 + (t * ADD_T + ADD_EPS))


# Kernel blocks: a unit is computed inside the first block whose condition (any listed atom non-zero) provably
# covers its guard, following guard chains down to byte features and previous state. Everything else is unconditional.
BLOCKS = [("vocab", ["u.wend"]),
          ("str", ["p.S", "p.fo", "p.FS", "p.FD", "p.c", "p.T", "p.F", "p.sq1", "p.oq1", "p.oT", "x.dq", "x.sq"]),
          ("word", ["p.W", "x.code", "x.dot", "x.at", "p.atp", "p.cD", "p.cC", "p.cAt", "p.cDot", "x.po", "x.pf", "p.fp", "p.d"]
           + ([] if MATCHER == "hash" else [f"p.pos{p}" for p in range(ADD_P)] + ["p.tail", "p.e2", "p.e3", "p.e3d"])),
          ("num", ["x.digit", "p.N", "p.ee"]),
          ("cmt", ["x.hash", "p.C"])]

BITS = ["lab_STR", "C_new", "N_new", "lab_W", "ev_KW", "ev_BI", "ev_FN", "ev_CL", "ev_DEC", "ev_STR"]
EV = ["KW", "BI", "FN", "CL", "DEC", "STR"]
CLASSES = ["text", "keyword", "builtin", "string", "comment", "number", "func", "class", "decorator"]
BSTATE = ["u.reg_" + e for e in EV] + ["b.ev_" + e for e in EV]
# class id read out by one integer unit: direct labels, or the carried word class on word bytes
HEAD = {"b.lab_STR": 3, "b.C_new": 4, "b.N_new": 5, "u.reg_KW": 1, "u.reg_BI": 2, "u.reg_STR": 3, "u.reg_FN": 6, "u.reg_CL": 7, "u.reg_DEC": 8}


# ---------------------------------------------------------------- layering and matrices
def layer_units(units):
    """Layers by dependency on same-step units (guards count); a family gets a layer of its own (tied weights)."""
    byname = {x.name: x for x in units}
    lvl = {}

    def L(n):
        if n in lvl:
            return lvl[n]
        x = byname[n]
        deps = [r for r in x.ins if r.startswith("u.")] + [g for g in (x.guard or []) if g.startswith("u.")]
        lvl[n] = 1 + max([L(r[2:]) for r in deps], default=-1)
        return lvl[n]
    for x in units:
        L(x.name)
    layers = []
    for l in range(1 + max(lvl.values())):
        xs = [x for x in units if lvl[x.name] == l]
        for fam in sorted({x.fam for x in xs if x.fam}):
            layers.append([x for x in xs if x.fam == fam])
        rest = [x for x in xs if not x.fam]
        if rest:
            layers.append(rest)
    return layers


def check_guards(units, emb):
    """Same proof as the C loader: with the guard inputs at 0, bias plus every other positive term at its ceiling must be <= 0."""
    hi = {"u." + x.name: x.hi for x in units}
    for s_, un in STATE_UNIT.items():
        if "u." + un in hi:  # matcher-specific state units exist only in that matcher's build
            hi["p." + s_] = hi["u." + un]
    for f in FEATS:
        hi["x." + f] = float(emb[:, FI[f]].max())
    bad = []
    for x in units:
        if not x.guard or getattr(x, "inherited", False):
            continue
        bound = x.b + sum(w * hi[r] for r, w in x.ins.items() if w > 0 and r not in x.guard)
        if bound > 0:
            bad.append((x.name, bound))
    assert not bad, f"guards not provable: {bad}"


def inherit_guards(units):
    """A unit that is not an output and whose consumers all share one guard set inherits it (it is dead otherwise)."""
    cons = {x.name: [] for x in units}
    for x in units:
        for r in x.ins:
            if r.startswith("u."):
                cons[r[2:]].append(x)
    outputs = set(STATE_UNIT.values()) | set(BITS)
    changed = True
    while changed:
        changed = False
        for x in units:
            if x.guard is not None or x.name in outputs or not cons[x.name]:
                continue
            gs = {tuple(c.guard) if c.guard is not None else None for c in cons[x.name]}
            if len(gs) == 1 and None not in gs:
                x.guard = list(gs.pop())
                x.inherited = True
                changed = True


def build_cell(layers, inputs, outputs_needed):
    """Layered units -> (W, b, hi, skip, names, tied) over a growing vector, lexnet style.
    A tied layer stores one weight row and one ceiling for all its rows (a threshold family)."""
    vec = list(inputs)
    mats = []
    later_needs = []
    for l, xs in enumerate(layers):
        need = set(outputs_needed)
        for ys in layers[l + 1:]:
            for y in ys:
                need |= set(y.ins)
        later_needs.append(need)
    for l, xs in enumerate(layers):
        idx = {n: i for i, n in enumerate(vec)}
        tied = all(x.fam for x in xs)
        if tied:
            assert all(x.ins == xs[0].ins and x.hi == xs[0].hi for x in xs)
        W = np.zeros((1 if tied else len(xs), len(vec)))
        b = np.zeros(len(xs)); hi = np.zeros(1 if tied else len(xs))
        for r, x in enumerate(xs):
            if not tied or r == 0:
                for ref, w in x.ins.items():
                    W[r, idx[ref]] = w
                hi[r] = x.hi
            b[r] = x.b
        skip = [i for i, n in enumerate(vec) if n in later_needs[l]]
        mats.append((W, b, hi, np.array(skip, dtype=np.int64), [x.name for x in xs], tied))
        vec = ["u." + x.name for x in xs] + [vec[i] for i in skip]
    return mats, vec


# ---------------------------------------------------------------- weights file
def write_weights(path, emb, fwd, fvec, bwd, bvec, head_W, head_b, h0):
    out = bytearray(b"PYLX02\0\0")
    out += struct.pack("<6q", NF, len(STATE), len(BITS), len(BSTATE), len(CLASSES), head_W.shape[1])
    out += emb.astype("<f8").tobytes()
    for cell in (fwd, bwd):
        out += struct.pack("<q", len(cell))
        for W, b, hi, skip, names, tied in cell:
            out += struct.pack("<4q", len(names), W.shape[1], len(skip), int(tied))
            out += W.astype("<f8").tobytes() + b.astype("<f8").tobytes() + hi.astype("<f8").tobytes() + skip.astype("<q").tobytes()
    f_state = [fvec.index("u." + STATE_UNIT[s]) for s in STATE]
    f_out = [fvec.index("u." + bname) for bname in BITS]
    b_state = [bvec.index(n) for n in BSTATE]
    out += np.array(f_state, dtype="<q").tobytes() + np.array(f_out, dtype="<q").tobytes() + np.array(b_state, dtype="<q").tobytes()
    out += h0.astype("<f8").tobytes() + head_W.astype("<f8").tobytes() + head_b.astype("<f8").tobytes()
    Path(path).write_bytes(out)
    return len(out)


def count(emb, fwd, bwd, head_W, head_b, h0):
    """Every non-zero number the network uses: embedding, weights, biases, ceilings, head, initial state."""
    nz = lambda a: int((np.asarray(a) != 0).sum())
    tied = nz(emb) + nz(head_W) + nz(head_b) + nz(h0) + 1    # +1: the head ceiling
    untied = tied
    for cell in (fwd, bwd):
        for W, b, hi, skip, names, t in cell:
            tied += nz(W) + nz(b) + nz(hi)
            untied += (nz(W) + nz(hi)) * (len(names) if t else 1) + nz(b)
    return tied, untied


# ---------------------------------------------------------------- kernel codegen
def struct_hash(fwd, bwd, f_state, f_out, b_state, head_W, head_cols):
    """FNV-1a over shapes, ceilings, skips, sparsity masks and index maps; identical to the C loader's."""
    h = 0xCBF29CE484222325

    def by(bs):
        nonlocal h
        for c in bs:
            h ^= c; h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF

    def i64(x):
        by(struct.pack("<q", x))
    i64(NF); i64(len(STATE)); i64(len(BITS)); i64(len(BSTATE)); i64(len(CLASSES)); i64(head_cols)
    for cell in (fwd, bwd):
        i64(len(cell))
        for W, b, hi, skip, names, tied in cell:
            i64(len(names)); i64(W.shape[1]); i64(len(skip)); i64(int(tied))
            by(hi.astype("<f8").tobytes())
            for s in skip:
                i64(int(s))
            by(bytes((W != 0).astype(np.uint8).ravel()))
    for i in f_state: i64(i)
    for i in f_out: i64(i)
    for i in b_state: i64(i)
    by(bytes((head_W != 0).astype(np.uint8).ravel()))
    return h


def cfmt(v):
    return repr(float(v)) if v != int(v) else f"{int(v)}.0"


def tree(terms):
    """Balanced sum: every value is an exact integer, so the order does not change the result but halves the latency."""
    while len(terms) > 1:
        terms = [f"({terms[i]} + {terms[i + 1]})" if i + 1 < len(terms) else terms[i] for i in range(0, len(terms), 2)]
    return terms[0]


_FVEC = {}


def fvec_index(cell, l, ref):
    return _FVEC[(id(cell), l)][ref]


def gen_kernel(path, fwd, fvec, bwd, bvec, head_W, head_b, head_hi, shash, units, bunits):
    coef, bias, guards, families, prefixes = [], [], [], [], []
    lines = []
    A = lines.append
    A("/* Generated by compile.py: the network's ARCHITECTURE as C. No weight values live here: pylexnet.c")
    A(" * loads them, checks the structure hash and the guard inequalities, and gathers kc[]/kb[] by K_COEF/K_BIAS. */")
    A("#include <stdint.h>")
    A("#include <string.h>")
    A(f"#define K_STRUCT_HASH 0x{shash:016X}ULL")
    A(f"#define K_N_FEATS {NF}")
    A(f"#define K_N_CLASSES {len(CLASSES)}")
    A(f"#define K_N_BITS {len(BITS)}")
    A("#ifndef CLIP")
    A("#define CLIP(x, hi) clipn((x), (hi))")
    A("static inline double clipn(double x, double hi) { x = x < 0.0 ? 0.0 : x; return x > hi ? hi : x; }")
    A("#endif")
    A("/* A tied, sorted threshold family is a step vector: unit k is 1 iff kb[k] + w*h >= 1, true for k < rank.")
    A(" * The loader verifies integrality, tying and monotonicity, so a binary search gives the same values. */")
    A("static inline int k_rank(const double *kb, int n, double w, double h) {")
    A("    int lo = 0, hi = n;")
    A("    while (lo < hi) { int mid = (lo + hi) >> 1; if (kb[mid] + w * h >= 1.0) lo = mid + 1; else hi = mid; }")
    A("    return lo;")
    A("}")
    A("typedef struct { double " + ", ".join(STATE) + "; } FState;")
    A("typedef struct { double " + ", ".join(n.replace(".", "_") for n in BSTATE) + "; } BState;")
    A("static inline void k_init_state(FState *s, const double *h0) { memcpy(s, h0, sizeof *s); }")
    # ---- forward
    A("static inline __attribute__((always_inline)) uint16_t k_fwd_step(const double *restrict kc, const double *restrict kb, const double *restrict kp, const double *restrict x,")
    A("                                  const FState *restrict p, FState *restrict s) {")
    byname = {x.name: x for x in units}
    famidx, fams = {}, {}
    for x in units:
        if x.fam:
            famidx[x.name] = len(fams.setdefault(x.fam, []))
            fams[x.fam].append(x)

    def ref(r):
        if r.startswith("x."):
            return f"x[{FI[r[2:]]}]"
        if r.startswith("p."):
            return f"p->{r[2:]}"
        x = byname[r[2:]]
        return f"(double)({famidx[x.name]} < r_{x.fam})" if x.fam else f"u_{x.name}"
    scalars = [x.name for x in units if not x.fam]
    A("    double " + ", ".join(f"u_{n} = 0.0" for n in scalars) + ";")
    for fam in fams:
        A(f"    int r_{fam} = 0;")

    def covered(atom, cond, seen=()):
        if atom in cond:
            return True
        if not atom.startswith("u.") or atom in seen:
            return False
        g = byname[atom[2:]].guard
        return g is not None and all(covered(a, cond, seen + (atom,)) for a in g)
    blk = {}
    for x in units:
        blk[x.name] = None
        if x.guard:
            for bname, cond in BLOCKS:
                if all(covered(a, cond) for a in x.guard):
                    blk[x.name] = bname
                    break
    cond_of = dict(BLOCKS)
    layer_of = {}
    for l, (W, b, hi, skip, names, tied) in enumerate(fwd):
        for row, n in enumerate(names):
            layer_of[n] = (l, row)

    def deps(x):
        d = {r[2:] for r in x.ins if r.startswith("u.")}
        if blk[x.name]:
            d |= {a[2:] for a in cond_of[blk[x.name]] if a.startswith("u.")}
        return d
    remaining = {x.name for x in units}
    done = set()
    cur = None
    nseg = 0
    while remaining:
        ready = [byname[n] for n in remaining if deps(byname[n]) <= done]
        assert ready, "cyclic dependency"
        bybl = {}
        for x in ready:
            bybl.setdefault(blk[x.name], []).append(x)
        if cur not in bybl:
            cur = max(bybl, key=lambda k: (len(bybl[k]), k is None))
            if cur is not None:
                A("    if ((" + " + ".join(ref(a) for a in cond_of[cur]) + ") != 0.0) {   /* all activations are >= 0 */")
                nseg += 1
        batch = sorted(bybl[cur], key=lambda x: (layer_of[x.name], x.fam or "", famidx.get(x.name, 0)))
        indent = "        " if cur is not None else "    "
        i = 0
        while i < len(batch):
            x = batch[i]
            l, row = layer_of[x.name]
            if x.fam:
                j = i
                while j < len(batch) and batch[j].fam == x.fam and famidx[batch[j].name] == famidx[x.name] + (j - i):
                    j += 1
                run = batch[i:j]
                assert len(run) == len(fams[x.fam]) and famidx[x.name] == 0, "a family is emitted whole"
                (inref, _), = run[0].ins.items()
                cb, bb = len(coef), len(bias)
                coef.append((0, l, 0, fvec_index(fwd, l, inref)))      # tied: one weight for the family
                for y in run:
                    bias.append((0,) + layer_of[y.name])
                families.append((0, l, bb, len(run), cb))
                A(f"{indent}r_{x.fam} = k_rank(kb + {bb}, {len(run)}, kc[{cb}], {ref(inref)});")
                i = j
                continue
            terms = [f"kb[{len(bias)}]"]
            bias.append((0, l, row))
            famins = {}
            for inref, w in x.ins.items():
                y = byname[inref[2:]] if inref.startswith("u.") else None
                if y is not None and y.fam:
                    famins.setdefault(y.fam, []).append(inref)
            for inref, w in x.ins.items():
                y = byname[inref[2:]] if inref.startswith("u.") else None
                if y is not None and y.fam and len(famins[y.fam]) >= 4:
                    if inref == famins[y.fam][0]:
                        # all of this family's weights into the unit, as a prefix sum indexed by the rank (built at load)
                        cols = [fvec_index(fwd, l, "u." + z.name) for z in fams[y.fam]]
                        assert cols == list(range(cols[0], cols[0] + len(cols))), "family columns are contiguous"
                        pb = sum(n + 1 for _, _, _, _, n in prefixes)
                        prefixes.append((0, l, row, cols[0], len(cols)))
                        terms.append(f"kp[{pb} + r_{y.fam}]")
                    continue
                terms.append(f"kc[{len(coef)}] * {ref(inref)}")
                coef.append((0, l, row, fvec_index(fwd, l, inref)))
            A(f"{indent}u_{x.name} = CLIP({tree(terms)}, {cfmt(x.hi)});")
            if x.guard and not getattr(x, "inherited", False):
                guards.append((0, l, row, [fvec_index(fwd, l, gr) for gr in x.guard]))
            i += 1
        for x in batch:
            done.add(x.name); remaining.discard(x.name)
        nxt = [n for n in remaining if blk[n] == cur and deps(byname[n]) <= done]
        if not nxt and cur is not None:
            A("    }")
            cur = None
    print(f"kernel: {nseg} block segments; units per block " + str({b: sum(1 for x in units if blk[x.name] == b) for b in [None] + [b for b, _ in BLOCKS]}))
    for sname in STATE:
        A(f"    s->{sname} = u_{STATE_UNIT[sname]};")
    A("    return (uint16_t)(" + " | ".join(f"((uint16_t)(u_{bn} != 0.0) << {i})" for i, bn in enumerate(BITS)) + ");")
    A("}")
    # ---- backward: carried word classes, then the integer class id
    A("static inline __attribute__((always_inline)) int k_bwd_step(const double *restrict kc, const double *restrict kb, uint16_t bits,")
    A("                             const BState *restrict r, BState *restrict s) {")
    for i, bn in enumerate(BITS):
        A(f"    const double f_{bn} = (double)((bits >> {i}) & 1);")

    def bref(rf):
        if rf.startswith("b."):
            return "f_" + rf[2:]
        if rf.startswith("r."):
            return "r->" + rf[2:].replace(".", "_")
        return "u_" + rf[2:]
    for l, (W, b, hi, skip, names, tied) in enumerate(bwd):
        for row, n in enumerate(names):
            x = bunits[n]
            terms = [f"kb[{len(bias)}]"]
            bias.append((1, l, row))
            for inref, w in x.ins.items():
                terms.append(f"kc[{len(coef)}] * {bref(inref)}")
                coef.append((1, l, row, fvec_index(bwd, l, inref)))
            A(f"    const double u_{n} = CLIP({tree(terms)}, {cfmt(x.hi)});")
    for n in BSTATE:
        A(f"    s->{n.replace('.', '_')} = {bref(n) if not n.startswith('u.') else 'u_' + n[2:]};")
    terms = [f"kb[{len(bias)}]"]
    bias.append((2, 0, 0))
    for c in np.nonzero(head_W[0])[0]:
        terms.append(f"kc[{len(coef)}] * {bref(bvec[c])}")
        coef.append((2, 0, 0, int(c)))
    A(f"    return (int)CLIP({tree(terms)}, {cfmt(head_hi)});")
    A("}")
    A(f"#define K_N_COEF {len(coef)}")
    A(f"#define K_N_BIAS {len(bias)}")
    A(f"#define K_N_GUARD {len(guards)}")
    A(f"#define K_N_FAMILY {len(families)}")
    A(f"#define K_N_PREFIX {len(prefixes)}")
    A(f"#define K_PREFIX_LEN {sum(n + 1 for *_, n in prefixes)}")
    A("/* sorted threshold families: {cell, layer, first bias index, rows, coef index of the tied weight} */")
    A("static const int32_t K_FAMILY[K_N_FAMILY + 1][5] = {" + "".join("{%d,%d,%d,%d,%d}," % f for f in families) + "{0,0,0,0,0}};")
    A("/* prefix sums of a unit's weights over a family's columns: {cell, layer, row, first column, rows} */")
    A("static const int32_t K_PREFIX[K_N_PREFIX + 1][5] = {" + "".join("{%d,%d,%d,%d,%d}," % f for f in prefixes) + "{0,0,0,0,0}};")
    A("#define K_MAX_GUARD 4")
    A("static const int32_t K_COEF[K_N_COEF][4] = {" + ",".join("{%d,%d,%d,%d}" % c for c in coef) + "};")
    A("static const int32_t K_BIAS[K_N_BIAS][3] = {" + ",".join("{%d,%d,%d}" % c for c in bias) + "};")
    A("/* units whose block is skipped when the listed input columns are all zero; the loader proves each one */")
    A("static const int32_t K_GUARD[K_N_GUARD][3 + K_MAX_GUARD] = {" +
      ",".join("{%d,%d,%d,%s}" % (c, l, r, ",".join(str(v) for v in (cols + [-1] * 4)[:4])) for c, l, r, cols in guards) + "};")
    Path(path).write_text("\n".join(lines) + "\n")
    return len(coef), len(bias), len(guards)


# ---------------------------------------------------------------- simulator (debugging: a third evaluation of the same matrices)
def run_layers(v, layers):
    for W, b, hi, skip, names, tied in layers:
        pre = (W @ v)
        if tied:
            pre = np.full(len(names), pre[0])
        new = np.clip(pre + b, 0, hi if not tied else hi[0])
        v = np.concatenate([new, v[skip]])
    return v


def simulate(text, emb, fwd, fvec, f_state, f_out, bwd, b_state, head_W, head_b, head_hi, h0):
    data = text.encode() + b"\n"
    st = h0.copy()
    allbits = []
    for t, byte in enumerate(data):
        v = run_layers(np.concatenate([emb[byte if byte < 128 else 127], st]), fwd)
        st = v[f_state]
        bits = [int(v[i] != 0) for i in f_out]
        allbits.append(bits)
        on = [n[2:] + ("" if v[i] == 1 else f"={v[i]:g}") for i, n in enumerate(fvec)
              if n.startswith("u.") and v[i] != 0 and not n.startswith("u.ge_")]
        print(f"{t:3} {chr(byte)!r:6} bits={''.join(map(str, bits))} " + " ".join(on))
    r = np.zeros(len(b_state))
    out = []
    for t in range(len(data) - 1, -1, -1):
        v = run_layers(np.concatenate([np.array(allbits[t], dtype=float), r]), bwd)
        r = v[b_state]
        out.append(int(np.clip(head_W @ v + head_b, 0, head_hi)[0]))
    print("classes:", "".join(str(c) for c in out[::-1][:-1]))


# ---------------------------------------------------------------- main
class BUnit:
    def __init__(self, name, ins, b=0.0, hi=1.0):
        self.name, self.ins, self.b, self.hi = name, ins, b, hi


def main():
    limit = None
    if "--vocab" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--vocab") + 1])
    vocab = pick_vocab(limit)
    hashes, cat_of = build(vocab)
    check_guards(UNITS, emb_table())
    inherit_guards(UNITS)
    layers = layer_units(UNITS)
    inputs = ["x." + f for f in FEATS] + ["p." + s for s in STATE]
    needed = {"u." + STATE_UNIT[s] for s in STATE} | {"u." + b for b in BITS}
    fwd, fvec = build_cell(layers, inputs, needed)
    vec = list(inputs)
    for l, (W, b, hi, skip, names, tied) in enumerate(fwd):
        _FVEC[(id(fwd), l)] = {n: i for i, n in enumerate(vec)}
        vec = ["u." + n for n in names] + [vec[i] for i in skip]
    # backward cell: reg_X(i) = clip1(ev_X(i+1) + reg_X(i+1) + lab_W(i) - 1): set at a word's end, carried over
    # the word's bytes, zero elsewhere. Carries this byte's event bits for the next step.
    bunits = {}
    for e in EV:
        bunits["reg_" + e] = BUnit("reg_" + e, {"r.b.ev_" + e: 1.0, "r.u.reg_" + e: 1.0, "b.lab_W": 1.0}, b=-1.0)
    binputs = ["b." + bn for bn in BITS] + ["r." + n for n in BSTATE]
    bidx = {n: i for i, n in enumerate(binputs)}
    Wb = np.zeros((len(EV), len(binputs))); bb = -np.ones(len(EV)); hb = np.ones(len(EV))
    for r, e in enumerate(EV):
        for ref, w in bunits["reg_" + e].ins.items():
            Wb[r, bidx[ref]] = w
    bskip = [bidx["b." + bn] for bn in BITS]
    bwd = [(Wb, bb, hb, np.array(bskip, dtype=np.int64), ["reg_" + e for e in EV], False)]
    _FVEC[(id(bwd), 0)] = bidx
    bvec_out = ["u.reg_" + e for e in EV] + ["b." + bn for bn in BITS]
    hcol = {n: i for i, n in enumerate(bvec_out)}
    head_W = np.zeros((1, len(bvec_out))); head_b = np.zeros(1); head_hi = 8.0
    for n, w in HEAD.items():
        head_W[0, hcol[n]] = w
    emb = emb_table()
    h0 = np.zeros(len(STATE))
    size = write_weights(HERE / "pylexnet_weights.bin", emb, fwd, fvec, bwd, bvec_out, head_W, head_b, h0)
    f_state = [fvec.index("u." + STATE_UNIT[s]) for s in STATE]
    f_out = [fvec.index("u." + bn) for bn in BITS]
    b_state = [bvec_out.index(n) for n in BSTATE]
    shash = struct_hash(fwd, bwd, f_state, f_out, b_state, head_W, head_W.shape[1])
    ncoef, nbias, nguard = gen_kernel(HERE / "pylexnet_kernel.h", fwd, fvec, bwd, bvec_out, head_W, head_b, head_hi, shash, UNITS, bunits)
    if "--sim" in sys.argv:
        simulate(sys.argv[sys.argv.index("--sim") + 1], emb, fwd, fvec, f_state, f_out, bwd, b_state, head_W, head_b, head_hi, h0)
    tied, untied = count(emb, fwd, bwd, head_W, head_b, h0)
    nlog = sum(1 for x in UNITS if not x.fam and not x.name.startswith("E_"))
    vocab_numbers = 2 * len(hashes) if MATCHER == "hash" else sum(len(t) for t in ADD_TABLES.values())
    print(f"units {len(UNITS)} forward in {len(fwd)} layers ({nlog} logic), state {len(STATE)}, bits {len(BITS)}, vocabulary {len(vocab)} words "
          f"({sum(1 for c in vocab.values() if c == 'KW')} keywords, {sum(1 for c in vocab.values() if c == 'BI')} builtins, "
          f"{sum(1 for c in vocab.values() if c == 'MAG')} magic) = {len(hashes)} thresholds; kernel coef {ncoef} bias {nbias} guards {nguard}")
    print(f"non-zero weights {tied} (every number: embedding {int((emb != 0).sum())}, vocabulary {vocab_numbers}; "
          f"{untied} if the threshold family's weight and ceiling were not tied)")
    print(f"weights file {size // 1024} KiB")


if __name__ == "__main__":
    main()
