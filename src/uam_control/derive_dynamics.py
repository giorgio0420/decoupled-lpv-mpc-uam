"""Closed form for the arm dynamics and the cross-coupling, by SymPy.

Why this exists
---------------
The manipulator's LPV model costs 24 ms of a 43 ms control tick, and not because
the dynamics are hard: a three-link planar arm has an M, C and G that fit on a
page. The cost is that we never wrote them. We have `newton_euler`, a recursion
that returns torques given (q, qd, qdd), and the LPV builders recover the
matrices from it by *probing* -- 43 full dynamics evaluations per tick, at
0.55 ms each in situ. That is numerical differentiation standing in for algebra,
and it is about a thousand times more expensive than evaluating the algebra.

The reference paper says it factorised the decomposition with SymPy. This does
the same, with one difference that matters: rather than re-deriving the model
from a Lagrangian -- which would re-make every modelling choice and could differ
from what the rest of the project has verified -- it runs *our own recursion*
symbolically. The generated expressions are then the same function the numeric
code computes, by construction, not by agreement.

Kept faithful to `newton_euler.py`, term for term:

  * frame {0} of the arm, with x_0 = -z_B, y_0 = -x_B, z_0 = y_B
  * each joint about its frame's z, each link along its frame's x (Craig)
  * link inertia diag(axial, transverse, transverse), not just point masses
  * base pseudo-acceleration R @ [0, 0, f_z / TOTAL_MASS], total mass and not
    the vehicle's alone
  * omega and omega_dot carried *through* the outward pass, as Eq. (6) has them,
    not only into the reaction
  * viscous joint damping, which Eq. (6)-(7) do not carry

Run it to write `generated_dynamics.py`. `test_generated.py` then checks every
generated function against the recursion on random states, which is the gate:
the closed form is accepted because it agrees to 1e-12, not because it looks
right.
"""

import sympy as sp

from uam_control.params import ARM, TOTAL_MASS

N = ARM.n_links

# --- symbols ---------------------------------------------------------------
q = sp.symbols(f'q1:{N + 1}')
qd = sp.symbols(f'dq1:{N + 1}')
qdd = sp.symbols(f'ddq1:{N + 1}')
p_u, q_u, r_u = sp.symbols('p_uav q_uav r_uav')
dp_u, dq_u, dr_u = sp.symbols('dp_uav dq_uav dr_uav')
f_z = sp.Symbol('f_z')

omega = sp.Matrix([p_u, q_u, r_u])
omega_dot = sp.Matrix([dp_u, dq_u, dr_u])

# --- constants, taken from params so there is one source ------------------
R_BODY_TO_ARM = sp.Matrix([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
R_ARM_TO_BODY = R_BODY_TO_ARM.T
Z = sp.Matrix([0, 0, 1])

mass = sp.nsimplify(ARM.mass, rational=True)
length = sp.nsimplify(ARM.length, rational=True)
damping = sp.nsimplify(ARM.damping, rational=True)
inertia = sp.diag(
    sp.nsimplify(ARM.inertia_axial, rational=True),
    sp.nsimplify(ARM.inertia_transverse, rational=True),
    sp.nsimplify(ARM.inertia_transverse, rational=True),
)
com = sp.Matrix([length / 2, 0, 0])
offset = sp.Matrix([length, 0, 0])
total_mass = sp.nsimplify(TOTAL_MASS, rational=True)


def rot_z(angle):
    c, s = sp.cos(angle), sp.sin(angle)
    return sp.Matrix([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def recursion():
    """The same outward and inward passes as `newton_euler`, symbolically."""
    w = R_BODY_TO_ARM * omega
    wd = R_BODY_TO_ARM * omega_dot
    vd = R_BODY_TO_ARM * sp.Matrix([0, 0, f_z / total_mass])

    offsets = [sp.zeros(3, 1)] + [offset] * (N - 1)
    rotations = [rot_z(q[i]).T for i in range(N)]

    forces, moments = [], []
    for i in range(N):
        R, pos = rotations[i], offsets[i]
        vd = R * (wd.cross(pos) + w.cross(w.cross(pos)) + vd)
        wd = R * wd + (R * w).cross(qd[i] * Z) + qdd[i] * Z
        w = R * w + qd[i] * Z
        vd_com = wd.cross(com) + w.cross(w.cross(com)) + vd
        forces.append(mass * vd_com)
        moments.append(inertia * wd + w.cross(inertia * w))

    f_next = sp.zeros(3, 1)
    tau_next = sp.zeros(3, 1)
    joint = [0] * N
    for i in range(N - 1, -1, -1):
        R_back = rotations[i + 1].T if i + 1 < N else sp.eye(3)
        p_next = offsets[i + 1] if i + 1 < N else offset
        f_i = R_back * f_next + forces[i]
        tau_i = (moments[i] + R_back * tau_next + com.cross(forces[i])
                 + p_next.cross(R_back * f_next))
        joint[i] = (tau_i.T * Z)[0, 0]
        f_next, tau_next = f_i, tau_i

    tau_arm = sp.Matrix([joint[i] + damping * qd[i] for i in range(N)])
    return tau_arm, R_ARM_TO_BODY * f_next, R_ARM_TO_BODY * tau_next


print('running the recursion symbolically...')
tau_arm, base_force, base_torque = recursion()

# --- arm side: M, C, G ----------------------------------------------------
# The recursion is affine in qdd, so M is its Jacobian and the bias is what is
# left at qdd = 0. C and G split the bias by whether it survives qd = 0.
print('  M ...')
M = tau_arm.jacobian(sp.Matrix(qdd))
print('  bias, G ...')
bias = tau_arm.subs({a: 0 for a in qdd})
G = bias.subs({v: 0 for v in qd})
print('  C ...')
# C qd = bias - G exactly, and the left side is linear in qd by construction of
# the recursion, so the Jacobian of (bias - G) with respect to qd would double
# the quadratic terms. Christoffel form off M keeps C qd = bias - G while
# leaving C itself well defined.
C = sp.zeros(N, N)
for i in range(N):
    for j in range(N):
        C[i, j] = sum(
            sp.Rational(1, 2) * (sp.diff(M[i, j], q[k]) + sp.diff(M[i, k], q[j])
                                 - sp.diff(M[k, j], q[i])) * qd[k]
            for k in range(N)
        )
# Whatever the Christoffel form does not reproduce -- the Coriolis coupling with
# the vehicle's own rotation, plus the joint damping -- is carried as a residual
# so that C qd + G + residual is the bias exactly.
print('  residual ...')
bias_residual = sp.simplify(bias - C * sp.Matrix(qd) - G)

# --- vehicle side: Eq. (10) and (11) -------------------------------------
print('  M_d, tau_bar ...')
M_d = base_torque.jacobian(omega_dot)
tau_no_dw = base_torque - M_d * omega_dot
tau_bar = tau_no_dw.subs({p_u: 0, q_u: 0, r_u: 0})

print('  M_c, M_s, M_l ...')
rest = tau_no_dw - tau_bar
M_s = sp.zeros(3, 3)
M_l = sp.zeros(3, 3)
for j, s in enumerate((p_u, q_u, r_u)):
    M_s[:, j] = sp.diff(rest, s, 2) / 2
    M_l[:, j] = sp.diff(rest, s).subs({p_u: 0, q_u: 0, r_u: 0})
M_c = sp.zeros(3, 3)
for j, (a, b) in enumerate(((p_u, q_u), (q_u, r_u), (p_u, r_u))):
    M_c[:, j] = sp.diff(sp.diff(rest, a), b)

print('  M_f, f_bar ...')
M_f = base_force.jacobian(sp.Matrix([f_z]))
f_bar = base_force.subs({f_z: 0})

# --- code generation ------------------------------------------------------
ARGS = ('q1, q2, q3, dq1=0.0, dq2=0.0, dq3=0.0, ddq1=0.0, ddq2=0.0, ddq3=0.0, '
        'p_uav=0.0, q_uav=0.0, r_uav=0.0, dp_uav=0.0, dq_uav=0.0, dr_uav=0.0, '
        'f_z=0.0')


def emit(name, expr):
    """One numpy function per quantity, with common subexpressions factored.

    The matrix is written out element by element rather than handed to `pycode`
    whole: on a Matrix, `pycode` emits `ImmutableDenseMatrix([[...]])`, which is
    not valid without importing SymPy into the generated module -- and importing
    SymPy at controller run time would defeat the point of generating this.
    """
    replacements, reduced = sp.cse(expr, optimizations='basic')
    lines = [f'def get_{name}({ARGS}):']
    for symbol, value in replacements:
        lines.append(f'    {symbol} = {sp.pycode(value)}')
    matrix = reduced[0]
    rows = [
        '        [' + ', '.join(sp.pycode(matrix[i, j])
                                for j in range(matrix.cols)) + ']'
        for i in range(matrix.rows)
    ]
    lines.append('    return np.array([\n' + ',\n'.join(rows) + '\n    ], dtype=float)')
    return '\n'.join(lines).replace('math.', 'np.') + '\n'


# --- the LPV triple, defined exactly as manipulator_lpv defines it --------
# Its C is not the Christoffel form: it is whatever satisfies C qd = bias - G,
# which also carries the Coriolis coupling with the vehicle's rotation and the
# joint damping. Reproducing that definition rather than a textbook one is what
# makes the swap exact instead of merely equivalent.
print('  C_lpv, Q_lpv, T_lpv ...')
vp = bias - G                      # = C qd, linear + quadratic in qd
basis = [sp.Matrix([1 if i == j else 0 for i in range(N)]) for j in range(N)]


def at(expr, rates):
    return expr.subs({qd[i]: rates[i] for i in range(N)})


C_linear = sp.zeros(N, N)
for j in range(N):
    C_linear[:, j] = (at(vp, basis[j]) - at(vp, -basis[j])) / 2

# The even part in qd is the quadratic one; half its gradient is the matrix that
# reproduces it, exactly, because it is a quadratic form.
quadratic = (vp + at(vp, [-v for v in qd])) / 2
C_quadratic = sp.zeros(N, N)
for j in range(N):
    C_quadratic[:, j] = sp.diff(quadratic, qd[j]) / 2

C_lpv = C_linear + C_quadratic
Q_lpv = G.jacobian(sp.Matrix(q))
T_lpv = Q_lpv * sp.Matrix(q) - G

pieces = [
    ('M', M), ('C', C), ('G', G), ('bias_residual', bias_residual),
    ('tau_arm', tau_arm),
    ('C_lpv', C_lpv), ('Q_lpv', Q_lpv), ('T_lpv', T_lpv),
    ('M_d', M_d), ('M_c', M_c), ('M_s', M_s), ('M_l', M_l),
    ('tau_bar', tau_bar), ('M_f', M_f), ('f_bar', f_bar),
]
print('generating code...')
out = ['"""Generated by derive_dynamics.py. Do not edit; regenerate."""',
       'import numpy as np', '']
for name, expr in pieces:
    print(f'  {name}')
    out.append(emit(name, expr))

path = 'uam_control/generated_dynamics.py'
with open(path, 'w') as handle:
    handle.write('\n'.join(out))
print(f'written to {path}')
