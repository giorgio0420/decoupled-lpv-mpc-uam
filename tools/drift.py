"""What drifts, as opposed to what oscillates.

The amplitudes are flat for 35 s and then the run departs, so the trigger is not
a growing oscillation. Something else is moving slowly underneath. This prints
the mean of every column per window: a monotone column is the candidate.
"""

import sys

import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/wrench_hover.csv"
window = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0

d = np.genfromtxt(path, delimiter=",", names=True)
cols = [c for c in d.dtype.names if c != "t"]

edges = np.arange(d["t"][0], d["t"][-1], window)
blocks = []
for a in edges:
    m = (d["t"] >= a) & (d["t"] < a + window)
    if m.sum() < 10:
        continue
    blocks.append((a, {c: float(np.mean(d[c][m])) for c in cols}))

# Rank columns by how monotone their per-window mean is over the healthy stretch,
# which is what a slow drift looks like and an oscillation does not.
healthy = [b for b in blocks if b[0] < 40.0]
scores = []
for c in cols:
    series = np.array([b[1][c] for b in healthy])
    if series.size < 3 or np.allclose(series, series[0]):
        continue
    steps = np.diff(series)
    monotone = abs(np.sum(np.sign(steps))) / len(steps)
    span = series[-1] - series[0]
    scores.append((monotone, abs(span), c, series))

scores.sort(reverse=True)
print(f"{path}: medie per finestra da {window:.0f} s, "
      f"ordinate per quanto sono monotone fino a t = 40 s")
print()
for monotone, span, c, series in scores[:8]:
    print(f"{c:12s} monotonia {monotone:4.2f}  variazione {span:9.4f}   "
          + " ".join(f"{v:8.3f}" for v in series))
