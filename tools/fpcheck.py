"""PC tool: false positives of the additive matcher over the fit corpus's words. usage: fpcheck.py [--matcher add]
Prints words (not in the vocabulary) whose value sum falls in a class window, with their counts."""
import re, sys
from collections import Counter
from pathlib import Path
import numpy as np
import compile as c
vocab = c.pick_vocab(None)
c.solve_additive(vocab)
C = Path("corpus_cache")
data = np.memmap(C / "data.bin", dtype=np.uint8, mode="r")
gold = np.memmap(C / "gold.bin", dtype=np.uint8, mode="r")
wre = re.compile(rb"[A-Za-z_\x80-\xff][A-Za-z_0-9\x80-\xff]*")
cnt = Counter(); nbytes = Counter()
for line in open(C / "index.tsv"):
    sp, pkg, rel, off, n = line.rstrip("\n").split("\t")
    if sp != "fit":
        continue
    off, n = int(off), int(n)
    d = data[off:off + n].tobytes(); g = gold[off:off + n]
    for m in wre.finditer(d):
        if g[m.start()] == 0:        # labelled text by Pygments: a hit here is a false positive
            w = m.group().decode("latin1")
            cnt[w] += 1; nbytes[w] += m.end() - m.start()
tot = fp = fpb = 0
hits = Counter(); hitb = Counter()
for w, n in cnt.items():
    if w in vocab:
        continue
    tot += n
    v = sum(c.ADD_TABLES.get(f, {}).get(code, 0) * k for (f, code), k in c.word_feats(w).items())
    for t in (1, 2, 3, 4, 5):
        if abs(v - t * c.ADD_T) <= c.ADD_EPS:
            hits[w] += n; hitb[w] += nbytes[w]; fp += n; fpb += nbytes[w]
print(f"non-vocabulary text words: {tot} occurrences, {len(cnt)} distinct; false positives: {fp} occurrences, {fpb} bytes ({fpb / 74787051:.6f} of corpus)")
for w, n in hits.most_common(40):
    print(f"  {w:30} {n:7} occ {hitb[w]:8} B")
