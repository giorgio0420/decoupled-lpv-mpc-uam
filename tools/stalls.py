"""Do the stalls the user sees in the window line up with the loop losing control?

The node's timer runs on the wall clock. A stall does not skip late callbacks, it
queues them, and rclpy then runs them back to back: each one advances the tube's
nominal state by a whole dt while no simulated time passes between them. The
nominal runs ahead of the real state, the tube error inflates, and K @ e comes out
as a burst of torque. That is the mechanism the observation describes -- lag, then
the vehicle starts spinning.

This reads the two clocks logged per tick and marks where either one jumps.
"""

import sys

import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/TIME_1.csv"
d = np.genfromtxt(path, delimiter=",", names=True)

wall = d["wall"]
sim = d["sim"]
dt = 0.025

dwall = np.diff(wall)
dsim = np.diff(sim)
err = np.hypot(d["x"] - d["ref_x"], d["y"] - d["ref_y"])

print(f"{path}: {d.size} tick")
print(f"  passo di orologio   mediana {np.median(dwall) * 1000:6.1f} ms"
      f"   p99 {np.percentile(dwall, 99) * 1000:7.1f} ms"
      f"   massimo {dwall.max() * 1000:8.1f} ms")
print(f"  passo simulato      mediana {np.median(dsim) * 1000:6.1f} ms"
      f"   p99 {np.percentile(dsim, 99) * 1000:7.1f} ms"
      f"   massimo {dsim.max() * 1000:8.1f} ms")
print(f"  budget per tick     {dt * 1000:.0f} ms")
print()

# A burst: several ticks executed with almost no wall time between them, which is
# rclpy draining a queue that built up during a stall.
burst = dwall < 0.3 * dt
stall = dwall > 3.0 * dt
print(f"  tick eseguiti a raffica (< {0.3 * dt * 1000:.0f} ms l uno dall altro):"
      f" {burst.sum()} su {dwall.size}  ({100 * burst.mean():.1f}%)")
print(f"  tick preceduti da uno stallo (> {3 * dt * 1000:.0f} ms):"
      f" {stall.sum()}  ({100 * stall.mean():.1f}%)")
print()

idx = np.nonzero(stall)[0]
if idx.size:
    print("      t   stallo   errore prima   errore 40 tick dopo   crescita")
    for i in idx[:12]:
        after = min(i + 40, d.size - 1)
        before, later = err[i], err[after]
        print(f"{d['t'][i]:7.2f} {dwall[i] * 1000:8.1f} ms {before:14.4f} m"
              f" {later:21.4f} m {later / max(before, 1e-6):10.1f}x")
else:
    print("  nessuno stallo oltre la soglia in questa corsa")
