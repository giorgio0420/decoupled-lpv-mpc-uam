"""Full report from one or more Gazebo trace.csv runs: time-evolution plots
for every state, the 3-D trajectory, Table II's simulation parameters, and
Table III's IAE performance index -- everything this project measures, laid
out the way the paper's own Figs. 4-7 and Tables II-III do.

    python3 tools/report.py mpc=/tmp/run_mpc.csv ertf=/tmp/run_ertf.csv
    python3 tools/report.py run=/tmp/trace.csv --out report

Each `label=path` pair is one run; all runs are overlaid on the same axes so
schemes compare directly. `--out prefix` writes `prefix_timeseries.png` and
`prefix_trajectory3d.png` instead of just showing them.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

SETTLE = 14.0  # [s] trajectory_start 8.0 + EASE_TIME 6.0, same as tools/iae.py

COLOURS = {
    0: "tab:red", 1: "tab:blue", 2: "tab:green", 3: "tab:orange", 4: "tab:purple",
}


def load(path):
    d = np.genfromtxt(path, delimiter=",", names=True)
    return d[d["t"] >= SETTLE]


def has(d, name):
    return name in d.dtype.names


def iae_table(runs):
    """Table III: IAE per axis, one row per run."""
    rows = []
    for label, d in runs.items():
        if d.size < 2:
            rows.append((label, None))
            continue
        dt = np.diff(d["t"])
        dt = np.append(dt, dt[-1])
        row = {
            "span": d["t"][-1] - d["t"][0],
            "x": np.sum(np.abs(d["x"] - d["ref_x"]) * dt),
            "y": np.sum(np.abs(d["y"] - d["ref_y"]) * dt),
            "z": np.sum(np.abs(d["z"] - d["ref_z"]) * dt),
            "theta": np.sum(np.abs(d["pitch"] - d["pitch_cmd"]) * dt),
            "phi": np.sum(np.abs(d["roll"] - d["roll_cmd"]) * dt) if has(d, "roll") else None,
            "psi": np.sum(np.abs(d["yaw"] - d["yaw_cmd"]) * dt) if has(d, "yaw") else None,
            "q1": np.sum(np.abs(d["q0"] - d["qref0"]) * dt),
            "q2": np.sum(np.abs(d["q1"] - d["qref1"]) * dt),
            "q3": np.sum(np.abs(d["q2"] - d["qref2"]) * dt),
        }
        rows.append((label, row))
    return rows


def print_table_ii():
    from uam_control.params import CONTROL, LIMITS

    def q(gain, size):
        return f"{gain:g} * I_{size}"

    print("TABLE II -- Controller's Parameters (this run's config)")
    print("-" * 60)
    print(f"{'N_zeta':10}{CONTROL.N_zeta:<10}{'N_eta':10}{CONTROL.N_eta:<10}"
          f"{'N_gamma':10}{CONTROL.N_gamma:<10}")
    print(f"{'dt_zeta':10}{CONTROL.dt_zeta:<10g}{'dt_eta':10}{CONTROL.dt_eta:<10g}"
          f"{'dt_gamma':10}{CONTROL.dt_gamma:<10g}")
    print(f"{'Q_zeta':10}{q(CONTROL.Q_zeta_gain, 12):<16}"
          f"{'Q_eta':10}{q(CONTROL.Q_eta_gain, 3):<16}"
          f"{'Q_gamma':10}{q(CONTROL.Q_gamma_gain, 6)}")
    print(f"{'R_zeta':10}{q(CONTROL.R_zeta_gain, 3):<16}"
          f"{'R_eta':10}{q(CONTROL.R_eta_gain, 3):<16}"
          f"{'R_gamma':10}{q(CONTROL.R_gamma_gain, 3)}")
    print(f"{'P_eta':10}{q(CONTROL.P_eta_gain, 3):<16}"
          f"{'P_gamma':10}{q(CONTROL.P_gamma_gain, 3)}")
    print(f"{'torque':10}{LIMITS.torque:<10g}{'torque_yaw':12}{LIMITS.torque_yaw:<10g}")
    print()


def print_table_iii(runs):
    print(f"TABLE III -- IAE performance index (transitorio escluso, t >= {SETTLE:.0f} s)")
    print("-" * 100)
    header = f"{'run':10}{'durata':>8}{'x':>8}{'y':>8}{'z':>8}{'phi':>8}{'theta':>8}{'psi':>8}{'q1':>8}{'q2':>8}{'q3':>8}"
    print(header)
    for label, row in iae_table(runs):
        if row is None:
            print(f"{label:10}  corsa troppo corta")
            continue

        def f(key):
            v = row[key]
            return f"{v:8.3f}" if v is not None else f"{'n/a':>8}"

        print(f"{label:10}{row['span']:8.1f}" + "".join(
            f(k) for k in ("x", "y", "z", "phi", "theta", "psi", "q1", "q2", "q3")))
    print()
    print("Unita': m*s per x/y/z, rad*s per phi/theta/psi/q1/q2/q3.")
    print()


def disturbance_windows(d):
    """Contiguous [start, end] pairs where dist_on is nonzero, from the trace
    itself rather than a hardcoded schedule -- correct regardless of which
    scenario or timing offset produced this particular run."""
    if not has(d, "dist_on") or d.size < 2:
        return []
    on = d["dist_on"] > 0.5
    windows = []
    start = None
    for i, flag in enumerate(on):
        if flag and start is None:
            start = d["t"][i]
        elif not flag and start is not None:
            windows.append((start, d["t"][i]))
            start = None
    if start is not None:
        windows.append((start, d["t"][-1]))
    return windows


def robust_ylim(reference, measured, pad_factor=1.5):
    """Y-limits sized to the reference, not the measurement.

    A run that diverges (or is hit hard enough by a disturbance to not
    recover) can reach values orders of magnitude past anything the
    reference ever asks for; scaling the axis to include it squashes the
    well-behaved part -- exactly the part a disturbance response needs to be
    readable in -- into a sliver at one edge. The reference stays sane
    regardless of how badly the measurement fails, so it sets the window;
    the divergence is left to run off the top or bottom of the plot, which
    is the point.
    """
    lo, hi = float(np.min(reference)), float(np.max(reference))
    span = hi - lo
    if span < 1e-6:
        # A flat or near-flat reference (yaw held at zero, a slow z ramp)
        # still needs a real window: pad from the measurement's own spread
        # instead of a zero-width one.
        span = max(np.std(measured), 0.05)
    pad = pad_factor * span
    return lo - pad, hi + pad


def plot_timeseries(runs, out=None):
    import matplotlib
    if out:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 3, figsize=(16, 11))
    panels = [
        ("x", "ref_x", "x (m)"), ("y", "ref_y", "y (m)"), ("z", "ref_z", "z (m)"),
        ("roll", "roll_cmd", "phi (rad)"), ("pitch", "pitch_cmd", "theta (rad)"),
        ("yaw", "yaw_cmd", "psi (rad)"),
        ("q0", "qref0", "q1 (rad)"), ("q1", "qref1", "q2 (rad)"), ("q2", "qref2", "q3 (rad)"),
    ]
    for ax, (meas, ref, ylabel) in zip(axes.flat, panels):
        ref_drawn = False
        span_drawn = False
        ylim_ref = None
        for i, (label, d) in enumerate(runs.items()):
            if d.size < 2 or not has(d, meas):
                continue
            colour = COLOURS.get(i % len(COLOURS), None)
            ax.plot(d["t"], d[meas], color=colour, lw=1.1, label=label)
            if not ref_drawn and has(d, ref):
                ax.plot(d["t"], d[ref], "k--", lw=1.0, alpha=0.6, label="RT" if i == 0 else None)
                ref_drawn = True
                ylim_ref = (d[ref], d[meas])
            for w_start, w_end in disturbance_windows(d):
                # A 0.25 s window barely shows as a span on a 30 s axis, so
                # mark it as a line at its centre instead -- visible, and
                # still exactly where the impulse hit.
                ax.axvline((w_start + w_end) / 2, color="goldenrod", lw=1.4,
                          ls=":", label="disturbance" if not span_drawn else None)
                span_drawn = True
        if ylim_ref is not None:
            ax.set_ylim(*robust_ylim(*ylim_ref))
        ax.set_xlabel("time (s)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("State evolution: measurement vs reference (RT)")
    fig.tight_layout()
    if out:
        path = f"{out}_timeseries.png"
        fig.savefig(path, dpi=130)
        print(f"scritto {path}")
    else:
        plt.show()


def plot_trajectory3d(runs, out=None):
    import matplotlib
    if out:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ref_drawn = False
    for i, (label, d) in enumerate(runs.items()):
        if d.size < 2:
            continue
        colour = COLOURS.get(i % len(COLOURS), None)
        ax.plot(d["x"], d["y"], d["z"], color=colour, lw=1.3, label=label)
        ax.scatter(d["x"][0], d["y"][0], d["z"][0], color=colour, marker="o", s=70,
                  edgecolor="black", zorder=5,
                  label=f"{label} start" if len(runs) > 1 else "start")
        ax.scatter(d["x"][-1], d["y"][-1], d["z"][-1], color=colour, marker="s", s=50,
                  edgecolor="black", zorder=5,
                  label=f"{label} end" if len(runs) > 1 else "end")
        if not ref_drawn:
            ax.plot(d["ref_x"], d["ref_y"], d["ref_z"], "g--", lw=1.2, label="RT")
            ref_drawn = True
        for j, (w_start, w_end) in enumerate(disturbance_windows(d)):
            idx = np.argmin(np.abs(d["t"] - (w_start + w_end) / 2))
            ax.scatter(d["x"][idx], d["y"][idx], d["z"][idx], color="goldenrod",
                      marker="*", s=180, edgecolor="black", zorder=6,
                      label="disturbance" if j == 0 and i == 0 else None)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title("3-D trajectory: drone vs reference")
    ax.legend(fontsize=8)
    fig.tight_layout()
    if out:
        path = f"{out}_trajectory3d.png"
        fig.savefig(path, dpi=130)
        print(f"scritto {path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", help="label=path.csv pairs")
    parser.add_argument("--out", default=None, help="prefix for saved PNGs; omit to show interactively")
    args = parser.parse_args()

    runs = {}
    for arg in args.runs:
        label, path = arg.split("=", 1)
        if not Path(path).exists():
            print(f"manca il file: {path}", file=sys.stderr)
            sys.exit(1)
        runs[label] = load(path)

    print_table_ii()
    print_table_iii(runs)
    plot_timeseries(runs, args.out)
    plot_trajectory3d(runs, args.out)


if __name__ == "__main__":
    main()
