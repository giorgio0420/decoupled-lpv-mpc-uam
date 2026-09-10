"""One row per run: vehicle drift, how hard it rotates, how hard the arm fights.

`omega_dot` is the quantity the arm-alone bench showed to be the fatal one, so it
is measured here rather than assumed: the numerical derivative of the pitch rate
the vehicle actually reached. The arm holds against about 0.6 N m per rad/s^2 at
joint 1, so the number to compare it against is the joint torque limit / 0.6.
"""

import re
import sys

import numpy as np

JOINTS = re.compile(r" q=\[([^\]]*)\].*qref=\[([^\]]*)\]")


def joint_error(log_path):
    errs = []
    for line in open(log_path):
        m = JOINTS.search(line)
        if not m:
            continue
        q = np.array([float(v) for v in m.group(1).split()])
        ref = np.array([float(v) for v in m.group(2).split()])
        errs.append(np.abs(q - ref).max())
    return np.median(errs) if errs else float("nan")


print(f"{'run':16s} {'x range':>19s} {'|omega_dot| p95':>16s} {'|qdd_arm| med':>14s}"
      f" {'fbar_x sd':>10s} {'thrust':>16s} {'|q-qref| med':>13s}")
for arg in sys.argv[1:]:
    label, trace, log = arg.split("=")
    d = np.genfromtxt(trace, delimiter=",", names=True)
    dt = np.median(np.diff(d["t"]))
    omega_dot = np.abs(np.diff(d["q_meas"]) / dt)
    print(
        f"{label:16s} [{d['x'].min():8.2f},{d['x'].max():8.2f}]"
        f" {np.percentile(omega_dot, 95):16.1f}"
        f" {np.median(np.abs(d['qdd0'])):14.1f}"
        f" {d['fbar_x'].std():10.2f}"
        f" [{d['thrust'].min():5.1f},{d['thrust'].max():6.1f}]"
        f" {joint_error(log):13.3f}"
    )
