"""Compare the joint accelerations Gazebo integrates with the ones the model predicts.

Reads the plugin's [ARMDYN] lines and, for each sample, feeds the same (q, qd, tau)
into `joint_accelerations`. The run is a free fall with the rotors off, so the base
accelerates at g and the arm is weightless in the base frame: f_z = 0, omega = 0.
"""

import re
import sys

import numpy as np

from uam_control.newton_euler import joint_accelerations

TRIPLE = r"\(([^)]*)\)"
LINE = re.compile(
    r"tau=" + TRIPLE + r" q=" + TRIPLE + r" qd=" + TRIPLE + r" qdd=" + TRIPLE + r" dt=(\S+)"
)


def triple(text):
    return np.array([float(v) for v in text.split(",")])


rows = []
for line in open(sys.argv[1] if len(sys.argv) > 1 else "/tmp/armdyn.txt"):
    m = LINE.search(line)
    if m:
        rows.append(
            (triple(m.group(1)), triple(m.group(2)), triple(m.group(3)), triple(m.group(4)))
        )

print(f"samples read: {len(rows)}")

# The window is the first few tens of milliseconds after the torque comes on, and
# the reason is not the velocity limit. Free fall stops being a good stand-in for
# `f_z = 0` as soon as the arm's own reaction starts moving the base, and the
# joint accelerations decay by two orders of magnitude within the first 100 ms,
# so late samples compare two small numbers whose difference is all systematics.
# Sample period is 1 ms.
WINDOW_MS = int(sys.argv[2]) if len(sys.argv) > 2 else 20

usable = [
    (tau, q, qd, qdd)
    for tau, q, qd, qdd in rows[:WINDOW_MS]
    if np.all(np.abs(qd) < 0.5) and np.all(np.abs(q) < 3.0) and np.any(np.abs(qdd) > 1e-9)
]
print(f"samples in the first {WINDOW_MS} ms with |qd| < 0.5, |q| < 3.0: {len(usable)}")

if not usable:
    raise SystemExit("no usable window -- widen the burst or lower the torque")

zero = np.zeros(3)
print()
print("      observed qdd                 predicted qdd               ratio (per joint)")
ratios = []
for tau, q, qd, qdd in usable[:: max(1, len(usable) // 12)]:
    pred = joint_accelerations(q, qd, tau, zero, zero, 0.0)
    r = np.where(np.abs(pred) > 1e-6, qdd / np.where(np.abs(pred) > 1e-6, pred, 1.0), np.nan)
    ratios.append(r)
    print(
        f"  [{qdd[0]:8.3f} {qdd[1]:8.3f} {qdd[2]:8.3f}]"
        f"   [{pred[0]:8.3f} {pred[1]:8.3f} {pred[2]:8.3f}]"
        f"   [{r[0]:6.3f} {r[1]:6.3f} {r[2]:6.3f}]"
    )

allr = []
for tau, q, qd, qdd in usable:
    pred = joint_accelerations(q, qd, tau, zero, zero, 0.0)
    keep = np.abs(pred) > 1e-3
    allr.extend((qdd[keep] / pred[keep]).tolist())

allr = np.array(allr)
print()
print(f"ratio over all usable samples: median {np.median(allr):.4f}   "
      f"mean {allr.mean():.4f}   sd {allr.std():.4f}   n {allr.size}")
