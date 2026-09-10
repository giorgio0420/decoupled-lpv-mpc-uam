"""What the vehicle does when nothing is asked of it.

The arm is off and the trajectory is a point, so anything moving here is the
vehicle's own loops. Prints the statistics that separate a slow drift with
occasional spikes from continuous chatter, then a window of consecutive ticks.
"""

import sys

import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/trace_hover_noarm.csv"
start = int(sys.argv[2]) if len(sys.argv) > 2 else 400

d = np.genfromtxt(path, delimiter=",", names=True)
dt = np.median(np.diff(d["t"]))
od = np.abs(np.diff(d["q_meas"]) / dt)
tau, fx = d["tau_y"], d["force_x"]
flips = np.sum(np.diff(np.sign(tau)) != 0)

print(f"{path}, {d.size} ticks")
print(f"  |omega_dot|  median {np.median(od):6.2f}   p95 {np.percentile(od, 95):6.1f} rad/s^2")
print(f"  pitch        sd {d['pitch'].std():6.3f} rad     cmd sd {d['pitch_cmd'].std():6.3f} rad")
print(f"  force_x      sd {fx.std():6.2f} N      at the limit {100 * np.mean(np.abs(fx) > 19.9):5.1f}%")
print(f"  tau_y        sd {tau.std():6.2f} N m    at the limit {100 * np.mean(np.abs(tau) > 6.9):5.1f}%")
print(f"  tau_y sign change every {d.size / max(flips, 1):.1f} ticks"
      f" = {2 * d.size / max(flips, 1) * dt:.3f} s period")
print(f"  x            drift {d['x'].min():+.2f} to {d['x'].max():+.2f} m")
print()
print("     t        x       vx    pitch  pitch_cmd  force_x    tau_y")
for r in d[start : start + 24]:
    print("%7.3f %8.2f %8.2f %8.4f %10.4f %8.2f %8.2f"
          % (r["t"], r["x"], r["vx"], r["pitch"], r["pitch_cmd"], r["force_x"], r["tau_y"]))
