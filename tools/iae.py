"""IAE over a Gazebo run, per difficulty level.

The paper reports the integral of the absolute tracking error. Both references
and both measurements are in the trace at the control rate, so this needs no
other input. The transient is excluded: the vehicle spawns 5 m up and eases onto
the trajectory over EASE_TIME, and integrating that in would measure the ease-in
rather than the tracking.
"""

import sys

import numpy as np

SETTLE = 14.0  # [s] trajectory_start 8.0 + EASE_TIME 6.0


def iae(path):
    d = np.genfromtxt(path, delimiter=",", names=True)
    d = d[d["t"] >= SETTLE]
    if d.size < 2:
        return None
    dt = np.diff(d["t"])
    dt = np.append(dt, dt[-1])

    pos = np.stack([d["x"], d["y"], d["z"]])
    ref = np.stack([d["ref_x"], d["ref_y"], d["ref_z"]])
    q = np.stack([d["q0"], d["q1"], d["q2"]])
    qref = np.stack([d["qref0"], d["qref1"], d["qref2"]])

    result = {
        "span": d["t"][-1] - d["t"][0],
        "pos": np.sum(np.abs(pos - ref) * dt, axis=1),
        "joint": np.sum(np.abs(q - qref) * dt, axis=1),
        "pos_norm": np.sum(np.linalg.norm(pos - ref, axis=0) * dt),
        "joint_norm": np.sum(np.linalg.norm(q - qref, axis=0) * dt),
        "thrust": (d["thrust"].min(), d["thrust"].max()),
    }
    # `roll` was added later; older traces do not have it, so attitude IAE
    # falls back to pitch alone rather than failing the whole read.
    if "roll" in d.dtype.names:
        result["phi"] = np.sum(np.abs(d["roll"] - d["roll_cmd"]) * dt)
    else:
        result["phi"] = None
    result["theta"] = np.sum(np.abs(d["pitch"] - d["pitch_cmd"]) * dt)
    return result


print(f"IAE, transitorio escluso (t >= {SETTLE:.0f} s)")
print()
print(f"{'livello':10s} {'durata':>7s} {'IAE x':>8s} {'IAE y':>8s} {'IAE z':>8s}"
      f" {'phi':>7s} {'theta':>7s} |"
      f" {'IAE q1':>7s} {'IAE q2':>7s} {'IAE q3':>7s}")
for arg in sys.argv[1:]:
    label, path = arg.split("=", 1)
    r = iae(path)
    if r is None:
        print(f"{label:10s}  corsa troppo corta")
        continue
    phi_str = f"{r['phi']:7.3f}" if r["phi"] is not None else "    n/a"
    print(
        f"{label:10s} {r['span']:6.1f}s {r['pos'][0]:8.2f} {r['pos'][1]:8.2f}"
        f" {r['pos'][2]:8.2f} {phi_str} {r['theta']:7.3f} |"
        f" {r['joint'][0]:7.3f} {r['joint'][1]:7.3f} {r['joint'][2]:7.3f}"
    )

print()
print("Unita': m*s per x/y/z, rad*s per phi/theta/giunti.")
print("(vecchie tracce senza colonna 'roll': phi = n/a, non un errore)")
