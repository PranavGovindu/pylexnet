"""Score the compiled network on the second corpus against cached Pygments labels. PC only, one process.
usage: corpus_score.py <fit|check> [report file]. Prints a one-line summary; ranked mismatches go to the report."""
import re, subprocess, sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

NAMES = ["text", "keyword", "builtin", "string", "comment", "number", "func", "class", "decorator"]
C = Path(__file__).resolve().parent / "corpus_cache"
split = sys.argv[1] if len(sys.argv) > 1 else "fit"
rep = open(sys.argv[2], "w") if len(sys.argv) > 2 else None
data = np.memmap(C / "data.bin", dtype=np.uint8, mode="r")
gold = np.memmap(C / "gold.bin", dtype=np.uint8, mode="r")
tot = wrong = exact = nfiles = 0
conf = Counter()
bytext = Counter()
byprev = Counter()
sample = {}
pkgs = defaultdict(lambda: [0, 0])
wordre = re.compile(rb"[A-Za-z_][A-Za-z_0-9]*|[^\sA-Za-z_0-9]")
for line in open(C / "index.tsv"):
    sp, pkg, rel, off, n = line.rstrip("\n").split("\t")
    if sp != split:
        continue
    off, n = int(off), int(n)
    d = data[off:off + n].tobytes()
    g = gold[off:off + n]
    out = subprocess.run(["./pylexnet", "--classes"], input=d, capture_output=True).stdout
    p = np.frombuffer(out, dtype=np.uint8)
    assert len(p) == n, rel
    nfiles += 1
    tot += n
    pkgs[pkg][1] += n
    bad = np.nonzero(g != p)[0]
    if not len(bad):
        exact += 1
        continue
    wrong += len(bad)
    pkgs[pkg][0] += len(bad)
    s = e = int(bad[0])
    runs = []
    for i in map(int, bad[1:]):
        if i == e + 1 and g[i] == g[e] and p[i] == p[e]:
            e = i
        else:
            runs.append((s, e)); s = e = i
    runs.append((s, e))
    for s, e in runs:
        k = f"{NAMES[g[s]]}>{NAMES[p[s]]}"
        m = e - s + 1
        conf[k] += m
        a = d.rfind(b"\n", 0, s) + 1
        z = d.find(b"\n", e)
        z = n if z < 0 else z
        t = re.sub(rb"\d", b"0", d[s:e + 1][:18]).decode("utf-8", "replace")
        prev = wordre.findall(d[a:s])[-1:] or [b"^"]
        k1 = (k, t)
        k2 = (k, prev[0].decode("utf-8", "replace"))
        bytext[k1] += m
        byprev[k2] += m
        if k1 not in sample:
            sample[k1] = (rel, d[a:z][:110].decode("utf-8", "replace"))
        if k2 not in sample:
            sample[k2] = (rel, d[a:z][:110].decode("utf-8", "replace"))
print(f"{split}: files={nfiles} MB={tot/1e6:.1f} wrong={wrong} acc={1-wrong/tot:.6f} exact={exact}/{nfiles}")
if rep:
    print(f"{split}: files={nfiles} MB={tot/1e6:.1f} wrong={wrong} acc={1-wrong/tot:.6f} exact={exact}/{nfiles}", file=rep)
    print("confusions", conf.most_common(30), file=rep)
    print("by text", file=rep)
    for k, v in bytext.most_common(70):
        print(f"{v:7} {k[0]} {k[1]!r} [{sample[k][0].split('/')[0]}] {sample[k][1]!r}", file=rep)
    print("by previous token", file=rep)
    for k, v in byprev.most_common(50):
        print(f"{v:7} {k[0]} after {k[1]!r} {sample[k][1]!r}", file=rep)
    print("by package", file=rep)
    for k, (w, n) in sorted(pkgs.items(), key=lambda kv: -kv[1][0]):
        print(f"{w:7} {n/1e6:6.1f}MB {1-w/n:.6f} {k}", file=rep)
