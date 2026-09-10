"""The gate on the generated closed form: it must agree with the recursion.

`derive_dynamics.py` ran our own Newton-Euler passes symbolically, so the
expressions should be the same function the numeric code computes. "Should" is
not enough to swap them into a control loop, so this checks every generated
quantity against the recursion on random states, including states with the
vehicle rotating and accelerating -- which is where the coupling terms live and
where a transcription slip would hide.

Run: python3 test_generated.py
"""

import numpy as np

from uam_control import generated_dynamics as gen
from uam_control.coupling import decompose
from uam_control.newton_euler import mass_matrix, newton_euler
from uam_control.params import ARM, TOTAL_MASS, GRAVITY

RNG = np.random.default_rng(20260815)
TRIALS = 200
TOL = 1e-10


def sample():
    """A random state, with the vehicle genuinely rotating and accelerating."""
    return dict(
        q=RNG.uniform(-2.0, 2.0, 3),
        qd=RNG.uniform(-3.0, 3.0, 3),
        qdd=RNG.uniform(-20.0, 20.0, 3),
        omega=RNG.uniform(-2.0, 2.0, 3),
        omega_dot=RNG.uniform(-30.0, 30.0, 3),
        f_z=RNG.uniform(5.0, 120.0),
    )


def call(fn, s):
    return fn(*s['q'], *s['qd'], *s['qdd'], *s['omega'], *s['omega_dot'], s['f_z'])


def report(name, worst):
    status = 'ok  ' if worst < TOL else 'FALLITO'
    print(f'  {status} {name:16s} scarto massimo {worst:.3e}')
    return worst < TOL


def main():
    print(f'{TRIALS} stati casuali, tolleranza {TOL:.0e}')
    worst = {k: 0.0 for k in
             ('tau_arm', 'M', 'bias = C qd + G + resto', 'M_d', 'M_c', 'M_s',
              'M_l', 'tau_bar', 'M_f', 'f_bar')}

    for _ in range(TRIALS):
        s = sample()
        ref = newton_euler(s['q'], s['qd'], s['qdd'], s['omega'],
                           s['omega_dot'], s['f_z'])

        worst['tau_arm'] = max(worst['tau_arm'], np.abs(
            call(gen.get_tau_arm, s).ravel() - ref.joint_torques).max())

        worst['M'] = max(worst['M'], np.abs(
            call(gen.get_M, s) - mass_matrix(s['q'])).max())

        # The bias is what the recursion returns at zero joint acceleration; the
        # closed form splits it into Coriolis, gravity and a residual that carries
        # the vehicle's own rotation and the joint damping.
        bias_ref = newton_euler(s['q'], s['qd'], np.zeros(3), s['omega'],
                                s['omega_dot'], s['f_z']).joint_torques
        bias_gen = (call(gen.get_C, s) @ s['qd']
                    + call(gen.get_G, s).ravel()
                    + call(gen.get_bias_residual, s).ravel())
        worst['bias = C qd + G + resto'] = max(
            worst['bias = C qd + G + resto'], np.abs(bias_gen - bias_ref).max())

        c = decompose(s['q'], s['qd'], s['qdd'], s['omega'], s['omega_dot'],
                      s['f_z'])
        # decompose returns the reaction on the vehicle, which is the negation of
        # what the inward pass hands back; the generated code follows the pass.
        for name, ours, theirs in (
            ('M_d', -call(gen.get_M_d, s), c.M_d),
            ('M_c', -call(gen.get_M_c, s), c.M_c),
            ('M_s', -call(gen.get_M_s, s), c.M_s),
            ('M_l', -call(gen.get_M_l, s), c.M_l),
            ('tau_bar', -call(gen.get_tau_bar, s).ravel(), c.tau_bar),
            ('M_f', -call(gen.get_M_f, s).ravel(), c.M_f),
            ('f_bar', -call(gen.get_f_bar, s).ravel(), c.f_bar),
        ):
            worst[name] = max(worst[name],
                              np.abs(np.asarray(ours).ravel()
                                     - np.asarray(theirs).ravel()).max())

    ok = all(report(k, v) for k, v in worst.items())
    print()
    if ok:
        print('La forma chiusa coincide con la ricorsione. Si puo sostituire.')
    else:
        print('NON coincide. Non sostituire: una delle due e sbagliata.')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
