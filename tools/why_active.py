"""Why is the attitude loop working at hover, with nothing commanded to move?

Walks the cascade backwards from the torque to whatever is driving it. At hover
with the arm off, a correct loop should sit near zero everywhere; whatever is not
near zero is the source, and its period says which stage it came from.
"""

import sys

import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/H_noarm.csv"
d = np.genfromtxt(path, delimiter=",", names=True)
d = d[d["t"] > 16.0]
dt = float(np.median(np.diff(d["t"])))

err_x = d["x"] - d["ref_x"]
err_y = d["y"] - d["ref_y"]


def period(v):
    """Dominant period from the mean interval between zero crossings."""
    c = np.nonzero(np.diff(np.sign(v - v.mean())))[0]
    return 2 * len(v) * dt / max(len(c), 1)


print(f"{path}, {d['t'][-1] - d['t'][0]:.0f} s dopo il transitorio, braccio spento")
print()
print(f"{'grandezza':22s} {'media':>10s} {'sd':>10s} {'picco':>10s} {'periodo':>9s}")
for name, v, unit in [
    ("errore x", err_x, "m"),
    ("errore y", err_y, "m"),
    ("velocita vx", d["vx"], "m/s"),
    ("domanda force_x", d["force_x"], "N"),
    ("beccheggio comandato", d["pitch_cmd"], "rad"),
    ("beccheggio misurato", d["pitch"], "rad"),
    ("velocita ang. comandata", d["q_cmd"], "rad/s"),
    ("velocita ang. misurata", d["q_meas"], "rad/s"),
    ("coppia tau_y", d["tau_y"], "N m"),
    ("reazione fbar_x", d["fbar_x"], "N"),
]:
    print(f"{name:22s} {v.mean():+10.4f} {v.std():10.4f} "
          f"{np.abs(v).max():10.4f} {period(v):8.3f}s   {unit}")

print()
# Is the force demand explained by the position and velocity error, or is it
# arriving from somewhere else? A PD on those two should reproduce it.
A = np.column_stack([err_x, d["vx"], np.ones(d.size)])
coef, *_ = np.linalg.lstsq(A, d["force_x"], rcond=None)
fit = A @ coef
resid = d["force_x"] - fit
print("force_x spiegata da un PD su errore e velocita:")
print(f"  kp {coef[0]:+.2f} N/m   kd {coef[1]:+.2f} N s/m   offset {coef[2]:+.2f} N")
print(f"  varianza spiegata {100 * (1 - resid.var() / d['force_x'].var()):.1f}%")
print(f"  residuo sd {resid.std():.2f} N contro un segnale di sd {d['force_x'].std():.2f} N")
