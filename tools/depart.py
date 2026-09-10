"""When does a run actually leave, as opposed to lagging behind a fast reference?

A fixed error threshold cannot tell the two apart: Section V's path moves at
1.3 m/s, so a metre of lag is tracking, while the same metre on a hover is a
departure. Departure is detected instead as the error growing past the size of
the trajectory itself and never coming back.
"""

import sys

import numpy as np


def analyse(path):
    d = np.genfromtxt(path, delimiter=",", names=True)
    err = np.hypot(d["x"] - d["ref_x"], d["y"] - d["ref_y"])
    # Scale of the reference itself: half its peak-to-peak span, floored so a
    # stationary reference still gets a sensible bar.
    scale = max(0.5 * max(d["ref_x"].ptp(), d["ref_y"].ptp()), 0.5)
    bar = 2.0 * scale

    gone = err > bar
    depart = None
    for i in np.nonzero(gone)[0]:
        if gone[i:].all():  # never recovers
            depart = d["t"][i]
            break

    healthy = d[d["t"] < depart] if depart else d
    healthy = healthy[healthy["t"] > 16.0]
    eh = err[: healthy.size] if depart is None else err[(d["t"] > 16.0) & (d["t"] < depart)]
    if healthy.size < 10:
        return None
    qe = np.abs(np.stack([healthy[f"q{i}"] - healthy[f"qref{i}"] for i in range(3)])).max(axis=0)
    return {
        "span": d["t"][-1],
        "scale": scale,
        "bar": bar,
        "depart": depart,
        "good": (depart if depart else d["t"][-1]) - 16.0,
        "err_mean": float(eh.mean()),
        "err_max": float(eh.max()),
        "q_err": float(np.median(qe)),
        "q_ref": float(healthy["qref0"].max()),
    }


print(f"{'corsa':22s} {'corsa':>7s} {'soglia':>7s} {'fuga':>8s} {'buono':>7s}"
      f" {'err med':>8s} {'err max':>8s} {'err giunti':>11s} {'rif giunti':>11s}")
for arg in sys.argv[1:]:
    label, path = arg.split("=", 1)
    r = analyse(path)
    if r is None:
        print(f"{label:22s}  troppo corta")
        continue
    dep = f"{r['depart']:.0f} s" if r["depart"] else "mai"
    print(f"{label:22s} {r['span']:6.0f}s {r['bar']:6.2f}m {dep:>8s} {r['good']:6.0f}s"
          f" {r['err_mean']:7.3f}m {r['err_max']:7.3f}m {r['q_err']:10.3f}r"
          f" {r['q_ref']:10.3f}r")
