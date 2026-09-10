"""Table III: LPV-MPC against the ERTF baseline, on Section V's three scenarios.

The paper's headline result. It reports IAE, which is what this integrates, but
IAE is an integral over a duration and the two schemes do not fly for identical
simulated times -- so a longer run scores worse on IAE for no reason of merit.
Both are therefore truncated to the shorter of the two before integrating, and
the mean error is reported beside it as a duration-free number.

Errors are also given as a fraction of the trajectory amplitude, because that is
the form the paper's claim takes: 2-8 % for LPV-MPC against 20-30 % for ERTF.

    python3 table3.py [results_dir]
"""
import sys
from pathlib import Path

import numpy as np

EASE = 6.0
SCENARIOS = ('nominal', 'fast', 'disturbance')


def load(path):
    d = np.genfromtxt(path, delimiter=',', names=True)
    return d[d['t'] > EASE]


def metrics(d, until):
    m = d['t'] <= until
    d = d[m]
    t = d['t']
    dt = np.gradient(t)
    error = np.vstack([d['x'] - d['ref_x'],
                       d['y'] - d['ref_y'],
                       d['z'] - d['ref_z']])
    joint_error = np.vstack([d[f'q{k}'] - d[f'qref{k}'] for k in range(3)])
    reference = np.vstack([d['ref_x'], d['ref_y']])
    amplitude = max((r.max() - r.min()) / 2 for r in reference)
    joint_amplitude = np.mean(
        [(d[f'qref{k}'].max() - d[f'qref{k}'].min()) / 2 for k in range(3)])
    return dict(
        duration=float(t[-1] - t[0]),
        iae=(np.abs(error) * dt).sum(axis=1),
        iae_joint=(np.abs(joint_error) * dt).sum(axis=1),
        mean=float(np.linalg.norm(error, axis=0).mean()),
        mean_joint=float(np.abs(joint_error).mean()),
        amplitude=float(amplitude),
        joint_amplitude=float(joint_amplitude),
    )


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else 'results')
    rows = []
    for scenario in SCENARIOS:
        traces = {}
        for scheme in ('mpc', 'ertf'):
            path = root / f'trace_{scheme}_{scenario}.csv'
            if not path.exists():
                print(f'manca {path}')
                return 1
            traces[scheme] = load(path)
        # Same window for both, so IAE compares like with like.
        until = min(t['t'][-1] for t in traces.values())
        for scheme in ('mpc', 'ertf'):
            rows.append((scenario, scheme, metrics(traces[scheme], until)))

    print('Tabella III -- LPV-MPC contro ERTF, finestra comune, transitorio escluso\n')
    header = ('scena         schema   durata    IAE pos   err medio   % amp  |'
              '  IAE giunti  err giunti   % amp')
    print(header)
    print('-' * len(header))
    for scenario, scheme, m in rows:
        print('%-13s %-7s %6.1fs %9.2f %10.3f %7.1f%%  | %10.3f %11.4f %7.1f%%'
              % (scenario, scheme.upper(), m['duration'], m['iae'].sum(),
                 m['mean'], 100 * m['mean'] / m['amplitude'],
                 m['iae_joint'].sum(), m['mean_joint'],
                 100 * m['mean_joint'] / m['joint_amplitude']))
    print('\nIAE in m*s per la posizione e rad*s per i giunti.')
    print('Il paper dichiara 2-8 % per LPV-MPC e 20-30 % per ERTF.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
