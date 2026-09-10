"""What grows, and how fast.

The allocation delivers what was asked while the run is healthy, so the departure
is not a saturation artefact. Something oscillates and the oscillation grows. This
measures the amplitude of each candidate over successive windows: a constant ratio
between windows is an exponential mode and its rate identifies it.
"""

import sys

import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/wrench_hover.csv"
window = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0

d = np.genfromtxt(path, delimiter=",", names=True)
err = np.hypot(d["x"] - d["ref_x"], d["y"] - d["ref_y"])
bad = np.nonzero(err > 1.0)[0]
split = d["t"][bad[0]] if bad.size else d["t"][-1]

signals = {
    "pitch": d["pitch"],
    "pitch_cmd": d["pitch_cmd"],
    "tau_y": d["tau_y"],
    "q_meas": d["q_meas"],
    "q1_arm": d["q0"],
    "q3_arm": d["q2"],
    "fbar_x": d["fbar_x"],
    "err_pos": err,
}

edges = np.arange(d["t"][0], d["t"][-1], window)
print(f"{path}, partenza a t = {split:.1f} s. "
      f"Ampiezza (sd) per finestre da {window:.0f} s.")
print()
print(f"{'finestra':>12} " + " ".join(f"{k:>10s}" for k in signals))
rows = []
for a in edges:
    m = (d["t"] >= a) & (d["t"] < a + window)
    if m.sum() < 10:
        continue
    row = [np.std(v[m]) for v in signals.values()]
    rows.append((a, row))
    mark = "  <- partenza" if a <= split < a + window else ""
    print(f"{a:6.0f}-{a + window:5.0f} "
          + " ".join(f"{v:10.4f}" for v in row) + mark)

print()
print("Rapporto fra finestre consecutive (>1 = cresce):")
print(f"{'finestra':>12} " + " ".join(f"{k:>10s}" for k in signals))
for (a0, r0), (a1, r1) in zip(rows, rows[1:]):
    if a1 > split:
        break
    print(f"{a0:6.0f}-{a1:5.0f} "
          + " ".join(f"{(b / a if a > 1e-9 else np.nan):10.2f}"
                     for a, b in zip(r0, r1)))
