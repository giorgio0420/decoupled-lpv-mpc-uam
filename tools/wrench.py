"""How much of the commanded wrench the rotors actually produce.

`fit` scales the torque down until every rotor stays inside [0, max^2], and the
final sqrt-clip can take more. The realised wrench is reconstructed from the
speeds that were published, using the same allocation matrix the command was
built from, so any difference here is the fitting and the clipping alone.

Split before and after the departure, because the interesting question is whether
the deficit is a consequence of the divergence or already there while the run
still looks good.
"""

import sys

import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/wrench_hover.csv"
d = np.genfromtxt(path, delimiter=",", names=True)

err = np.hypot(d["x"] - d["ref_x"], d["y"] - d["ref_y"])
bad = np.nonzero(err > 1.0)[0]
split = d["t"][bad[0]] if bad.size else d["t"][-1]
print(f"{path}, {d['t'][-1]:.1f} s. Partenza a t = {split:.1f} s.")
print()


def block(w, label):
    if w.size < 2:
        print(f"{label}: troppo corto")
        return
    for name, cmd, real, unit in [
        ("tau_y", w["tau_y"], w["tau_y_real"], "N m"),
        ("tau_x", w["tau_x"], w["tau_x_real"], "N m"),
        ("thrust", w["thrust"], w["thrust_real"], "N"),
    ]:
        deficit = real - cmd
        big = np.abs(cmd) > 0.1
        ratio = real[big] / cmd[big] if big.any() else np.array([np.nan])
        print(
            f"  {name:7s} comandata sd {cmd.std():6.2f}  realizzata sd {real.std():6.2f}"
            f"   deficit medio {deficit.mean():+7.3f} {unit}"
            f"   |deficit| p95 {np.percentile(np.abs(deficit), 95):6.2f}"
            f"   rapporto mediano {np.median(ratio):5.2f}"
        )
    print(f"  sfit      mediana {np.median(w['sfit']):.3f}   minimo {w['sfit'].min():.3f}"
          f"   sotto 0.9 sul {100 * np.mean(w['sfit'] < 0.9):.1f}% dei tick")
    print(f"  omega_min a zero sul {100 * np.mean(w['omega_min'] < 1.0):.1f}% dei tick")


print("PRIMA della partenza, mentre la corsa e' ancora buona:")
block(d[d["t"] < split], "prima")
print()
print("DOPO:")
block(d[d["t"] >= split], "dopo")
