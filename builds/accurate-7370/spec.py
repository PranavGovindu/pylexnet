"""Python lexical rules as one finite automaton plus a one-register backward pass.

The automaton is what gets compiled into network weights. Nothing here is learned.

Forward pass, per byte: next state, a direct label, and an event. An event says
"the word that just ended has this class". Backward pass: carry that class back
over the word's bytes.
"""
import builtins
import keyword
import re
from pathlib import Path

import pygments.lexers.python as _pl

TEXT, KEYWORD, BUILTIN, STRING, COMMENT, NUMBER, FUNC, CLASS, DECOR = range(9)
NAMES = ["text", "keyword", "builtin", "string", "comment", "number", "func", "class", "decorator"]
WORD = 9  # label meaning "resolve from the backward pass"
# Events 0..8 set the backward register. Events 9, 10 instead map it: they fire at the first byte of a word
# and turn that word's class, once it is known, into the class of whatever came before it.
EV_MATCH, EV_YIELD = 9, 10
EV_MAP = [[TEXT if c == KEYWORD else KEYWORD for c in range(9)],  # match/case: a keyword next means it was a plain name
          [KEYWORD if c == KEYWORD else TEXT for c in range(9)]]  # the space in `yield from` belongs to the keyword

KEYWORDS = set(keyword.kwlist)
# Pygments' word lists, so both sides share the vocabulary.
_src = Path(_pl.__file__).read_text()
_b = _src.index("'builtins': [")
_a = _src.index("'magicfuncs': [")
BUILTINS = set(re.findall(r"'(\w+)'", _src[_b:_a])) - {"builtins", "exceptions"} | {"self", "cls", "Ellipsis", "NotImplemented"}
MAGIC = set(re.findall(r"'(__\w+__)'", _src[_a:_src.index("'magicvars': [")])) - BUILTINS
PLAIN_PRE = {"r", "u", "b", "br", "rb"}
F_PRE = {"f", "fr", "rf"}


def _cases(words):
    out = set()
    for p in words:
        for m in range(1 << len(p)):
            out.add("".join(c.upper() if m >> i & 1 else c for i, c in enumerate(p)))
    return out


PLAIN_PRE, F_PRE = _cases(PLAIN_PRE), _cases(F_PRE)

ALPHA = set(b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_") | set(range(128, 256))
DIGIT = set(b"0123456789")
WORDCH = ALPHA | DIGIT
NL, BS, HASH, DOT, AT = 10, 92, 35, 46, 64
QUOTES = (34, 39)
SPACE = {32, 9, 12}


# Word categories the trie sub-network reports. One small integer per word.
F_NONE, F_KW, F_BI, F_MAGIC, F_PRE_S, F_PRE_F, F_DEF, F_CLASS, F_IMP, F_SOFT, F_YIELD = range(11)


def build_trie():
    nodes = [{}]
    term = {}
    groups = ((F_KW, KEYWORDS), (F_BI, BUILTINS), (F_MAGIC, MAGIC), (F_PRE_S, PLAIN_PRE), (F_PRE_F, F_PRE),
              (F_DEF, {"def"}), (F_CLASS, {"class"}), (F_IMP, {"import", "from"}), (F_SOFT, {"match", "case"}), (F_YIELD, {"yield"}))
    for flag, words in groups:
        for w in words:
            n = 0
            for ch in w.encode():
                if ch not in nodes[n]:
                    nodes.append({})
                    nodes[n][ch] = len(nodes) - 1
                n = nodes[n][ch]
            term[n] = flag
    return nodes, term


TRIE, TERM = build_trie()


def word_class(flag, ctx):
    if ctx == "us":
        return KEYWORD
    if ctx == "dec":
        return DECOR
    if ctx == "def":
        return FUNC
    if ctx == "class":
        return CLASS
    if flag in (F_KW, F_DEF, F_CLASS, F_IMP, F_YIELD):
        return KEYWORD
    if flag == F_MAGIC:
        return FUNC
    if ctx in ("dot", "imp"):
        return TEXT
    if flag == F_BI:
        return BUILTIN
    return TEXT


def ctx_after_word(flag, ctx):
    if ctx != "dot":
        if flag == F_DEF:
            return "def"
        if flag == F_CLASS:
            return "class"
        if flag == F_YIELD:
            return "yield"
        if flag == F_IMP:  # `from a import b` turns it off again; so does `yield from`
            return "none" if ctx in ("imp", "yield") else "imp"
    return ctx if ctx in STICKY else "none"


STICKY = ("imp", "soft")  # contexts that last to the end of the line
# After match/case these mean an ordinary name.
NOT_SOFT = b":,;=^&|@~)]}"
BOL = "numend"  # line-start mode. Takes over the kind name compile.py already lists, now that numbers don't use it.


def code(ctx, fr):
    return ("code", ctx, fr)


def line_start(fr):
    return (BOL, fr) if fr is None else code("none", fr)


def after_string(fr):
    return code("none", fr)


def back(fr):
    """Leave a replacement field: back to the string that owns it, inside any field that holds that string."""
    return ("str", fr[0], fr[1], 0, True, fr[3] + (None,) if fr[3] else None)


def step(state, b, flag):
    """(next state, label for this byte, event for the word just ended)."""
    kind = state[0]
    fr = state[-1]  # f-string context: None or (quote, triple, bracket depth, outer field or None)
    if kind == "code":
        ctx = state[1]
        if ctx == "ysp":  # `yield ` then: only `from` keeps the space in the keyword
            if b == 102:
                return ("word", "yield", fr), WORD, EV_YIELD
            nxt, lab, _ = step(code("none", fr), b, flag)
            return nxt, lab, TEXT
        if ctx == "feq":  # `=` in a field at depth 0: only `}`, `:` or `!r` after it make it the debug specifier
            if b in SPACE:
                return state, WORD, None
            if b == 33:
                return ("bang", fr), WORD, None
            if b in b"}:":
                return step(("bang2", fr), b, flag)
            nxt, lab, _ = step(code("none", fr), b, flag)
            return nxt, lab, TEXT
        if ctx == "sq":  # line-start match/case, class still open: the next byte decides
            if b == 32 or b == 9:
                return state, TEXT, None
            nxt, lab, _ = step(code("none" if b in NOT_SOFT else "soft", fr), b, flag)
            # True/False/None start with capitals: lower-case keywords are the ones that undo a match/case
            return nxt, lab, TEXT if b in NOT_SOFT else EV_MATCH if 97 <= b <= 122 else KEYWORD
        if fr is not None:
            q, triple, depth, out = fr
            if b in b"([{":
                return code("none", (q, triple, min(depth + 1, 3), out)), TEXT, None
            if b in b")]" and depth:
                return code("none", (q, triple, depth - 1, out)), TEXT, None
            if b == 125:  # }
                if depth:
                    return code("none", (q, triple, depth - 1, out)), TEXT, None
                return back(fr), STRING, None
            if b == 33 and depth == 0:  # !r conversion, unless it turns out to be !=
                return ("bang", fr), WORD, None
            if b == 61 and depth == 0:  # = may be the debug specifier
                return code("feq", fr), WORD, None
            if b == 58 and depth == 0:  # : starts the format spec, which is string text
                return back(fr), STRING, None
        if b == NL:
            return line_start(fr), TEXT, None
        if ctx == "yield" and b == 32:
            return code("ysp", fr), WORD, None
        if b in SPACE or b == BS:
            return state, TEXT, None
        if b == HASH:
            return ("cmt", fr), COMMENT, None
        if b in QUOTES:
            return ("q1", b, False, fr), STRING, None
        if b in DIGIT:
            return ("int", fr), NUMBER, None
        if b == DOT:
            if ctx == "imp":  # dotted module name
                return state, TEXT, None
            return ("dot", fr), WORD, None
        if b == AT:
            return ("at", "none", fr), WORD, None
        if ctx == "soft" and b == 95:  # `_` in a match/case line
            return ("word", "us", fr), WORD, None
        if b in ALPHA:
            return ("word", ctx, fr), WORD, None
        return code(ctx if ctx in STICKY else "none", fr), TEXT, None
    if kind == BOL:  # indentation costs no register; only m and c can start a soft keyword
        if b in SPACE:
            return state, TEXT, None
        if b in b"mc":
            return ("word", "bol", fr), WORD, None
        return step(code("none", fr), b, flag)
    if kind == "bang":
        if b in b"sraf":
            return ("bang2", fr), WORD, None
        nxt, lab, _ = step(code("none", fr), b, flag)
        return nxt, lab, TEXT
    if kind == "bang2":
        if b in b"}:":
            return back(fr), STRING, STRING
        nxt, lab, _ = step(code("none", fr), b, flag)
        return nxt, lab, TEXT
    if kind == "dot":  # the dot's class waits for this byte: .5 or an attribute
        if b in DIGIT:
            return ("frac", fr), NUMBER, NUMBER
        nxt, lab, _ = step(code("dot", fr), b, flag)
        return nxt, lab, TEXT
    if kind == "at":
        if b in ALPHA:
            return ("word", "dec", fr), WORD, None
        nxt, lab, _ = step(code("none", fr), b, flag)
        return nxt, lab, TEXT
    if kind == "word":
        ctx = state[1]
        if b in WORDCH:
            return (("word", "none", fr) if ctx == "us" else state), WORD, None
        if b in QUOTES and flag in (F_PRE_S, F_PRE_F) and ctx not in ("dec", "def", "class"):
            return ("q1", b, flag == F_PRE_F, fr), STRING, STRING
        if ctx == "bol" and flag == F_SOFT:
            return step(code("sq", fr), b, F_NONE)
        if b == DOT and flag not in (F_KW, F_DEF, F_CLASS, F_IMP, F_YIELD) and ctx != "imp":
            return code("dot", fr), TEXT, word_class(flag, ctx)  # after a name a dot is always an attribute, no need to wait
        nxt, lab, _ = step(code(ctx_after_word(flag, ctx), fr), b, F_NONE)
        return nxt, lab, word_class(flag, ctx)
    if kind == "cmt":
        if b == NL:
            return line_start(fr), TEXT, None
        return state, COMMENT, None
    if kind == "q1":
        q, f = state[1], state[2]
        if b == q:
            return ("q2", q, f, fr), STRING, None
        return step(("str", q, 0, 0, f, fr), b, flag)
    if kind == "q2":
        q, f = state[1], state[2]
        if b == q:
            return ("str", q, 1, 0, f, fr), STRING, None
        return step(after_string(fr), b, flag)
    if kind == "str":
        q, triple, close, f = state[1], state[2], state[3], state[4]
        if b == BS:
            return ("esc", q, triple, f, fr), STRING, None
        if f and b == 123:  # {
            return ("fopen", q, triple, fr), STRING, None
        if triple:
            if b == q:
                if close == 2:
                    return after_string(fr), STRING, None
                return ("str", q, 1, close + 1, f, fr), STRING, None
            return ("str", q, 1, 0, f, fr), STRING, None
        if b == q:
            return after_string(fr), STRING, None
        if b == NL:
            return line_start(fr), TEXT, None
        return state, STRING, None
    if kind == "fopen":  # just saw { in an f-string: {{ is a literal brace
        q, triple = state[1], state[2]
        if b == 123:
            return ("str", q, triple, 0, True, fr), STRING, None
        return step(code("none", (q, triple, 0, fr[:3] if fr else None)), b, flag)
    if kind == "esc":
        q, triple, f = state[1], state[2], state[3]
        if f and b == 123:  # `\{` escapes nothing: the brace still opens a field
            return ("fopen", q, triple, fr), STRING, None
        if f and b == 78:  # \N{NAME} is one character, not a field
            return ("escn", q, triple, f, fr), STRING, None
        return ("str", q, triple, 0, f, fr), STRING, None
    if kind == "escn":
        if b == 123:
            return ("nname", state[1], state[2], state[3], fr), STRING, None
        return step(("str", state[1], state[2], 0, state[3], fr), b, flag)
    if kind == "nname":
        if b == 125:
            return ("str", state[1], state[2], 0, state[3], fr), STRING, None
        if b == state[1] or b == NL:
            return step(("str", state[1], state[2], 0, state[3], fr), b, flag)
        return state, STRING, None
    if kind == "int":
        if b in DIGIT or b == 95:
            return state, NUMBER, None
        if b in b"xXoObB":
            return ("hex", fr), NUMBER, None
        if b == DOT:
            return ("frac", fr), NUMBER, None
        if b in b"eE":
            return ("exp0", fr), NUMBER, None
        return step(code("none", fr), b, flag)
    if kind == "hex":
        if b in DIGIT or b in b"abcdefABCDEF_":
            return state, NUMBER, None
        return step(code("none", fr), b, flag)
    if kind == "frac":
        if b in DIGIT or b == 95:
            return state, NUMBER, None
        if b in b"eE":
            return ("exp0", fr), NUMBER, None
        return step(code("none", fr), b, flag)
    if kind == "exp0":
        if b in b"+-" or b in DIGIT:
            return ("exp", fr), NUMBER, None
        return step(code("none", fr), b, flag)
    if kind == "exp":
        if b in DIGIT or b == 95:
            return state, NUMBER, None
        if b in b"jJ":  # imaginary; ends the literal
            return code("none", fr), NUMBER, None
        return step(code("none", fr), b, flag)
    raise ValueError(state)


START = line_start(None)


def build_table():
    """Enumerate reachable mode states. rows[state][flag][byte] = (next, label, event)."""
    ids = {START: 0}
    order = [START]
    rows = []
    cells = {}
    i = 0
    while i < len(order):
        s = order[i]
        table = []
        for flag in range(11 if s[0] == "word" else 1):  # only a word's end reads the word category
            row = []
            for b in range(256):
                n, lab, ev = step(s, b, flag)
                if n not in ids:
                    ids[n] = len(order)
                    order.append(n)
                t = (ids[n], lab, -1 if ev is None else ev)
                row.append(cells.setdefault(t, t))
            table.append(row)
        rows.append(table if s[0] == "word" else table * 11)
        i += 1
    return order, rows


def run_table(data, rows):
    """Reference execution: trie, mode machine, then backward pass."""
    n = len(data)
    lab = [0] * n
    ev = [-1] * (n + 1)
    s = 0
    node = 0  # trie position; None once the word has left the trie
    for i, b in enumerate(data):
        flag = TERM.get(node, F_NONE) if node is not None else F_NONE
        s, lab[i], ev[i] = rows[s][flag][b]
        if b in WORDCH:
            node = TRIE[node].get(b) if node is not None else None
        else:
            node = 0
    flag = TERM.get(node, F_NONE) if node is not None else F_NONE
    ev[n] = rows[s][flag][NL][2]
    out = bytearray(n)
    reg = TEXT
    for i in range(n - 1, -1, -1):
        if ev[i + 1] >= 0:
            reg = ev[i + 1] if ev[i + 1] < EV_MATCH else EV_MAP[ev[i + 1] - EV_MATCH][reg]
        out[i] = reg if lab[i] == WORD else lab[i]
    return out
