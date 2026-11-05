"""Score ./pylexnet against Pygments' PythonLexer, per byte, on the dev split of the stdlib. PC only, one process.
Pygments labels are cached in dev_cache/ after the first run. usage: score.py [dev] [report file]"""
import hashlib
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path("/usr/lib/python3.12")
HERE = Path(__file__).resolve().parent
CACHE = HERE / "dev_cache"
NAMES = ["text", "keyword", "builtin", "string", "comment", "number", "func", "class", "decorator"]


def files(split):
    out = []
    for p in sorted(ROOT.rglob("*.py")):
        if "site-packages" in p.parts or "dist-packages" in p.parts or "lib2to3" in p.parts:
            continue
        if p.stat().st_size > 300_000:
            continue
        h = hashlib.sha256(str(p).encode()).digest()[0]
        if (h < 128) == (split == "dev"):
            out.append(p)
    return out


def oracle(text):
    from pygments.lexers import PythonLexer
    from pygments.token import Token as T
    lex = PythonLexer(stripnl=False, ensurenl=False, stripall=False)
    out = bytearray()
    for _, tok, val in lex.get_tokens_unprocessed(text):
        if tok in T.Comment: c = 4
        elif tok in T.Literal.String: c = 3
        elif tok in T.Literal.Number: c = 5
        elif tok in T.Keyword or tok in T.Operator.Word: c = 1
        elif tok in T.Name.Decorator: c = 8
        elif tok in T.Name.Function: c = 6
        elif tok in T.Name.Class: c = 7
        elif tok in T.Name.Builtin or tok in T.Name.Exception: c = 2
        else: c = 0
        out += bytes([c]) * len(val.encode("utf-8", "surrogateescape"))
    return out


def build_cache(split):
    CACHE.mkdir(exist_ok=True)
    idx, data, gold = [], bytearray(), bytearray()
    for p in files(split):
        try:
            text = p.read_text("utf-8")
        except UnicodeDecodeError:
            continue
        text = text.replace("\r\n", "\n")
        d = text.encode()
        g = oracle(text)
        if len(g) != len(d):
            continue
        idx.append(f"{p}\t{len(data)}\t{len(d)}\n")
        data += d; gold += g
    (CACHE / f"{split}_data.bin").write_bytes(data)
    (CACHE / f"{split}_gold.bin").write_bytes(gold)
    (CACHE / f"{split}_index.tsv").write_text("".join(idx))


def main():
    split = sys.argv[1] if len(sys.argv) > 1 else "dev"
    rep = open(sys.argv[2], "w") if len(sys.argv) > 2 else None
    if not (CACHE / f"{split}_index.tsv").exists():
        build_cache(split)
    data = np.memmap(CACHE / f"{split}_data.bin", dtype=np.uint8, mode="r")
    gold = np.memmap(CACHE / f"{split}_gold.bin", dtype=np.uint8, mode="r")
    conf = np.zeros((9, 9), dtype=np.int64)
    exact = nfiles = 0
    bytext = Counter(); sample = {}
    for line in open(CACHE / f"{split}_index.tsv"):
        path, off, n = line.rstrip("\n").split("\t")
        off, n = int(off), int(n)
        d = data[off:off + n].tobytes(); g = gold[off:off + n]
        out = subprocess.run([str(HERE.parent / "pylexnet"), "--classes"], input=d, capture_output=True).stdout
        p = np.frombuffer(out, dtype=np.uint8)
        assert len(p) == n, (path, len(p), n)
        nfiles += 1
        np.add.at(conf, (g, p), 1)
        bad = np.nonzero(g != p)[0]
        if not len(bad):
            exact += 1
            continue
        s = e = int(bad[0]); runs = []
        for i in map(int, bad[1:]):
            if i == e + 1 and g[i] == g[e] and p[i] == p[e]:
                e = i
            else:
                runs.append((s, e)); s = e = i
        runs.append((s, e))
        for s, e in runs:
            k = (f"{NAMES[g[s]]}>{NAMES[p[s]]}", d[s:e + 1][:18].decode("utf-8", "replace"))
            bytext[k] += e - s + 1
            if k not in sample:
                a = d.rfind(b"\n", 0, s) + 1; z = d.find(b"\n", e); z = n if z < 0 else z
                sample[k] = (Path(path).name, d[a:z][:110].decode("utf-8", "replace"))
    tot = conf.sum()
    print(f"{split}: {nfiles} files, {tot / 1e6:.1f} MB")
    print(f"byte accuracy {np.trace(conf) / tot:.5f}, exact files {exact}/{nfiles}, wrong bytes {tot - np.trace(conf)}")
    for k in range(9):
        pr = conf[k, k] / max(1, conf[:, k].sum()); rc = conf[k, k] / max(1, conf[k].sum())
        print(f"  {NAMES[k]:10} P {pr:.5f} R {rc:.5f} n {conf[k].sum()}")
    off = conf.copy(); np.fill_diagonal(off, 0)
    print("top confusions:", [(f"{NAMES[i // 9]}>{NAMES[i % 9]}", int(off.ravel()[i])) for i in np.argsort(off.ravel())[::-1][:8] if off.ravel()[i]])
    for k, v in bytext.most_common(25 if rep is None else 80):
        print(f"{v:7} {k[0]} {k[1]!r} [{sample[k][0]}] {sample[k][1]!r}", file=rep or sys.stdout)


if __name__ == "__main__":
    main()
