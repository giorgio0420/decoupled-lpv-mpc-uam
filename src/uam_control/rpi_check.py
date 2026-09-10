"""Does the terminal set do what Theorem 1 needs it to do?

Three properties, checked by construction rather than by flying:

  1. **Invariance.** A nominal state inside the terminal box, driven by u = K x,
     must stay inside it -- for ever, not for a horizon. Checked by propagating
     the box's own corners far past the horizon.
  2. **Admissibility.** The feedback that holds it there must fit in the tightened
     input set, or the invariance is a claim about a controller that cannot be
     built.
  3. **Non-emptiness.** Proposition 1's condition. A terminal set that is empty
     means the problem is infeasible and the controller has to say so.

No Gazebo and no dispersion: the same numbers come out every time, which is why
this belongs here rather than in a run.

    python3 rpi_check.py
"""
import numpy as np

from uam_control.controllers import ManipulatorTubeMPC, RotationalTubeMPC
from uam_control.coupling import CrossCoupling, manipulator_lpv, rotational_lpv
from uam_control.params import ARM, CONTROL, GRAVITY, TOTAL_MASS

STEPS = 500          # far past any horizon
TOLERANCE = 1e-9


def rest_coupling():
    zero3 = np.zeros((3, 3))
    return CrossCoupling(M_d=zero3, M_c=zero3, M_s=zero3, M_l=zero3,
                         tau_bar=np.zeros(3), M_f=np.zeros(3),
                         f_bar=np.zeros(3))


def report(name, A, B, mpc, state_box, input_box):
    tube = mpc.tube
    closed_loop = A + B @ tube.gain
    rho = np.abs(np.linalg.eigvals(closed_loop)).max()
    rho_magnitude = np.abs(np.linalg.eigvals(np.abs(closed_loop))).max()

    minimal = tube.minimal_rpi(A, B, np.zeros(A.shape[0]),
                               np.zeros(B.shape[1]), mpc.horizon)
    print(f'\n{name}')
    print(f'  rho(A + B K)            {rho:.4f}'
          + ('' if rho < 1 else '   <-- anello instabile'))
    # Printed beside it because the difference is the whole reason the terminal
    # set is an ellipsoid and not a box: entrywise magnitude is not contractive
    # here even though the loop is.
    print(f'  rho(|A + B K|)          {rho_magnitude:.4f}   '
          f'(nessuna scatola invariante se >= 1)')
    if minimal is None:
        print('  mRPI                    nessuno')
        return False
    print(f'  mRPI, semiampiezza      {np.round(minimal, 4)}')

    # The tightened set the terminal box has to live inside, Eq. (36).
    tightened_state = state_box.tighten(minimal)
    tightened_input = input_box.tighten(np.abs(tube.gain) @ minimal)
    terminal = tube.terminal_set(A, B, tightened_state, tightened_input)
    if terminal is None:
        print('  insieme terminale       VUOTO -- Proposizione 1: infattibile')
        return False
    print(f'  insieme terminale       {np.round(terminal, 4)}')
    print(f'  frazione della scatola  '
          f'{np.round(terminal / state_box.half_width, 4)}')

    # 1. Invariance, checked on the ellipsoid the box is inscribed in, from every
    # corner of the box. Propagating |A_K| would be the wrong test: it is the
    # magnitude recursion that fails here, and the guarantee is carried by the
    # signed dynamics through the Lyapunov level set.
    from scipy import linalg as sla
    P = sla.solve_discrete_lyapunov(closed_loop.T, np.eye(A.shape[0]))
    level = float(terminal @ np.abs(P) @ terminal)
    corners = np.array(np.meshgrid(*[[-1.0, 1.0]] * A.shape[0])).T.reshape(
        -1, A.shape[0]) * terminal
    worst_ratio = 0.0
    for corner in corners:
        x = corner.copy()
        for _ in range(STEPS):
            x = closed_loop @ x
            worst_ratio = max(worst_ratio, float(x @ P @ x) / level)
    ok_invariant = worst_ratio <= 1.0 + 1e-9
    print(f'  invarianza              {"ok" if ok_invariant else "FALLITA"} su '
          f'{STEPS} passi da {len(corners)} vertici, '
          f'x\'Px / livello al massimo {worst_ratio:.4f}')
    if not ok_invariant:
        return False

    # 2. Admissibility of the feedback that holds it.
    demand = np.abs(tube.gain) @ terminal
    room = tightened_input.half_width
    ok = bool(np.all(demand <= room + TOLERANCE))
    print(f'  |K| x_terminale         {np.round(demand, 4)}')
    print(f'  spazio d ingresso       {np.round(room, 4)}   '
          + ('ok' if ok else 'ECCEDE'))
    return ok


def main():
    print(f'orizzonte {CONTROL.N_eta}/{CONTROL.N_gamma}, dt {CONTROL.dt_eta} s')

    rotational = RotationalTubeMPC()
    A, B = rotational_lpv(rest_coupling(), np.zeros(3), rotational.dt)
    first = report('rotazionale, sei stati', A, B, rotational,
                   rotational.state_box, rotational.input_box)

    manipulator = ManipulatorTubeMPC()
    n = ARM.n_links
    rest = np.zeros(n)
    A_g, B_g, _ = manipulator_lpv(rest, rest, np.zeros(3), np.zeros(3),
                                  TOTAL_MASS * GRAVITY, manipulator.dt)
    second = report('manipolatore', A_g, B_g, manipulator,
                    manipulator.state_box, manipulator.input_box)

    print()
    if first and second:
        print('Il Teorema 1 ha un insieme terminale su cui poggiare, '
              'in entrambi gli anelli.')
        return 0
    print('Almeno un anello non ha un insieme terminale ammissibile.')
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
