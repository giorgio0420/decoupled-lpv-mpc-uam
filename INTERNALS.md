# Internals: the defects, the method, and the traps

Companion to `README.md`, which covers the theory and how to run things. This
covers everything else — what was actually wrong, how it was found, what was
exonerated so nobody re-investigates it, and the ways this rig lies.

---

## 1. Every defect closed, with the measurement that closed it

### 1.1 The vehicle spawned into the floor

`gz_args` carried `-r`, so the world ran from launch. The launch takes about six
seconds. The vehicle engaged at **z = +0.990 m** with the arm resting on the floor
and curled to `[0, -pi, 3.03]`, and from there no thrust recovers it: the joints
are on a contact, the reaction terms read a pinned plant, and the rotors fight the
ground.

This single defect invalidated *every* attitude measurement taken before it. The
whole "the vehicle will not hold attitude" investigation — five sessions of it —
was measuring an aircraft on the ground.

Spawning higher cannot fix it and it is worth knowing why: six seconds of free
fall is 176 m. The world has to be paused. It now is, and the spawn is at 10 m
only for margin afterwards.

The user spotted this from the Gazebo window, not from a trace. Worth remembering
as a class: some faults are obvious visually and invisible in the CSV.

### 1.2 Odometry reports a point 2 m below the aircraft

`/uav/odom` publishes the link origin. The XACRO builds the whole airframe 2 m
above it — rotors at z = 2.021, inertial origin at 2.000 — confirmed by the
plugin's own diagnostic (`link origin = 0 0 5`, `link CoM = 1e-06 -0 7`).

Regulating that point makes the horizontal loop non-minimum-phase. Tilt by
`+theta` and a point 2 m below the axis swings to `-2 theta` while the thrust tilts
toward `+x`: to move right, the measurement must first go left. Verified open loop
with `tau_y = +0.3 N m`, the match is exact:

```
t = 0.011 s   pitch +0.000091   x -0.000181   -2 pitch = -0.000181
t = 0.101 s   pitch +0.008264   x -0.016443   -2 pitch = -0.016528
```

| | link origin | centre of mass |
|---|---|---|
| correlation pitch ↔ x | **−0.997** | **+0.962** |
| `vx` standard deviation | 3.76 m/s | 0.33 |
| `vx` peak | 8.01 m/s | 0.87 |
| oscillation period | 0.479 s | 1.675 |

The crossover is `omega = sqrt(g/2) = 2.2 rad/s`, which explains why slow
trajectories tracked and the paper's did not. No gain, weight, torque limit or tilt
cap ever helped, because on a non-minimum-phase response the gain only sets how
fast it diverges.

### 1.3 Three unbounded predictions

`f_bar` reached **29 140 N** on a 54 N aircraft, the thrust demand 80 095 N against
135 N of rotor authority, and `predicted_body_rate_dot` 7 322 rad/s². Each is a
model extrapolating past anything the plant can do, with no loop being told.

All three bounded. The joint-acceleration bound went through two wrong forms first:
`joint_rate / dt`, which is not a physical quantity and silently multiplied by 2.5
when the rate went 40 → 100 Hz, then the correct `|M^-1|(|tau_max| + |bias|)` —
which turned out to be *more* permissive, 224 to 4130 rad/s² depending on pose,
because the distal links have tiny inertia. So the ±131 N of `f_bar` seen later is
physically legitimate.

### 1.4 The clocks

Documented in `README.md` section 5.1. Four separate patches preceded the real fix
and each was treating a symptom: measuring `dt` from odometry stamps, re-seeding
the tube after a stall, clipping the period into a range, raising the stall
threshold. The cause was a controller on the wall clock and physics on its own.

### 1.5 Compute

The tick cost 42.93 ms. The manipulator's LPV model alone was 24.90 ms of it, and
not because the dynamics are hard — a three-link planar arm's `M`, `C`, `G` fit on
a page — but because they had never been written down. The LPV builders recovered
them from the recursion by *probing*: 43 full dynamics evaluations per tick.

Numerical differentiation standing in for algebra, about a thousand times more
expensive than evaluating the algebra. `derive_dynamics.py` fixed it:

| | before | after |
|---|---|---|
| tick | 42.93 ms | 22.94 |
| `decompose` | 8.06 ms | 0.66 |
| manipulator | 24.90 ms | 11.32 |
| samples per run | 585 | 3463 |

### 1.6 The arm clamp was itself the hammer

Used only when `arm_enabled=false`, to hold the arm still so the run measures the
vehicle carrying a rigid mass. It had one gain for three joints whose inertias run
0.163, 0.048, 0.006 kg m². A discrete damper needs `KD dt / I < 2`; the flat
`KD = 2.0` gave `[0.123, 0.415, 3.323]` — divergent at the wrist.

Measured, it drove all three joints from `[0.12, -0.17, -0.71]` to
`[-1.571, 3.142, 2.534]` **inside one 10 ms sample**, and the stop impact put
7 rad/s of body rate into the vehicle. The run that produced that trace was the one
meant to prove the vehicle could not hold attitude.

Now one bandwidth with the gains taken from the mass matrix, so `2 omega dt` is 0.8
at every joint and at any `dt`.

### 1.7 The manipulator's reference had no future

`README.md` section 5.2.

### 1.8 The attitude command annihilated itself

`README.md` section 5.3.

### 1.9 The circle was anchored to its y extreme

`README.md` section 5.4.

### 1.10 Smaller ones

- `controllers.py` called `max_joint_acceleration` with an undefined name, so every
  Gazebo run for a whole session crashed on the first tick. The measurement
  reported for that change had come from a standalone probe, not a run.
- The trace CSV header and data disagreed in column count three times, because
  `sed` and inline-Python replacements failed silently. Use the editing tools.
- `set -u` in the run script broke ROS setup with `AMENT_TRACE_SETUP_FILES: unbound
  variable`. Twice.
- `pkill -9 -f "ign gazebo"` matched the script's own command line and killed the
  shell before it reached Gazebo. Every failed run left a server alive; four had
  piled up, all publishing `/uav/odom`, and the controller was reading a blend of
  four worlds.
- `sp.pycode` on a SymPy Matrix emits `ImmutableDenseMatrix`, which is not valid
  without importing SymPy into the generated module. Elements are written
  individually.
- Adding a second `ign model --list` call to a guard delayed the controller by
  several seconds — enough for the arm to fold before the loop closed. Reuse the
  list the wait loop already fetched.

---

## 2. Exonerated. Do not re-investigate these

Each was measured innocent. The measurement is given so it does not have to be
repeated.

| suspect | verdict |
|---|---|
| the airframe, `k_f`, masses, spin directions | Six rotors at hover speed, nothing closed: z 5.000 → 4.986 over 11 s, `a_z` = −0.001 m/s², attitude under 0.024 rad. |
| the cross-coupled decomposition | `UAM_NO_COUPLING=1` forces every reaction term to zero. The vehicle still tipped. Innocent of the tipping. |
| the arm model vs Gazebo | Ratio 1.03 in free fall, on exactly the quantity the model feeds forward. |
| `attitude_reference` | Exact, checked symbolically. |
| the rotational loop's tracking | Tracks at every `R_eta`; `tau_y = K(q_cmd − q_meas)` reproduced to three decimals from the trace. |
| the altitude loop | IAE z is 0.4–1.0 m s in every scenario, best of the three axes. |
| the allocation | 0.05 N m of error until divergence; `sfit` = 1.000 in the current configuration. |
| the wrench frame | `worldPose.Rot().RotateVector` is body→world, confirmed by measurement, not by reading. |
| all twelve Table II parameters | Five combinations × three runs. Not the discriminant. |
| the translational sign | Verified offline while the vehicle flew the other way — which is what pointed at the lever arm. |
| the tilt cap | Never the active constraint: peak demand 0.445 rad on `fast` against a 0.6 cap, zero samples at the cap in any scenario. |
| x/y asymmetry as a controller fault | It was the reference construction. Ratio 0.19 before, 1.11 after, same controller. |

Two things that were tried and made matters worse, recorded so they are not tried
again:

- **Headless Gazebo** (`-s`). The reasoning was sound — the GUI runs on software
  OpenGL and is a plausible source of the stalls — but it loads fast enough that
  the world advanced five metres of free fall before the controller saw a sample,
  and the stalls did not go away: one per 32 s either way, with the attitude
  excursion three times larger.
- **Low-passing `arm_acceleration`.** It cut `f_bar`'s standard deviation from 13.8
  to 2.5 N and made the vehicle worse at every setting: `|omega_dot|` p95 went from
  20.4 to 41.7 rad/s² as the filter tightened. That signal is feedforward for a push
  the arm really delivers; smoothing it makes the rate loop chase the push after
  the fact. Zeroing it instead is worse still.

---

## 3. Nine ways this rig lied

Collected because each cost hours and each will recur.

1. **A single run is not a measurement.** Identical configurations departed anywhere
   between 3 and 19 seconds. One configuration that looked good at ±0.34 m was
   disproved by three re-runs of itself. This alone accounts for most of the
   twenty-four wrong conclusions.
2. **Stale Gazebo servers.** Four at once, all publishing `/uav/odom`.
3. **A divergence threshold is scenario-dependent.** A fixed 1.5 m flagged normal
   tracking lag on a 1.3 m/s trajectory; 0.05 m fired on the spawn transient at
   0.4 s in every configuration. `depart.py` now uses a scale-aware criterion.
4. **`pitch` from `arcsin` folds.** A body spinning past π/2 reads as pitch
   bouncing near ±1.55 rad with an apparent 6.8 Hz oscillation. The real signal
   was `q_meas` at 85 rad/s. Never diagnose a tumble from Euler angles.
5. **The clock ratio can be 1.000 while the loop is wrong.** It was, at
   `dt = 0.025`, and the system still diverged. Real-time factor is not
   synchronisation.
6. **A CSV time column that is `ticks * dt` is wrong** from the first late tick
   onward, and every IAE integrated against it is wrong with it. Use the simulated
   stamp.
7. **`q = 0` is not necessarily the built pose.** Joint 2 was captured at −3.142
   while the joint later read +3.142: 2π of error for zero physical displacement,
   enough to pin every joint at full torque. Wrap into the half-turn around the
   measurement.
8. **A model that agrees offline can be starved online.** The manipulator MPC
   follows its reference to 3 % on the bench and to nothing in flight, at the same
   torque budget. The difference was the reference's shape, not the arm.
9. **Free fall is the only clean way to measure arm dynamics here.** With the
   vehicle held or hovering, the rotor plugin, the contacts and the controller all
   contribute. In free fall only gravity does, and that is how the 1.03 ratio in
   section 2 was obtained.

---

## 4. Method rules earned the hard way

- **Three runs minimum** for any Gazebo comparison, and prefer a bench with no
  dispersion when the question can be asked there: `cascade.py` for stability,
  `arm_paper.py` for the arm, `test_generated.py` for the model.
- **Isolate before tuning.** `UAM_ATTITUDE_ONLY` and `UAM_NO_COUPLING` exist
  because a failure with all three loops closed looks like a failure of all three.
- **Bound every prediction that leaves a controller.** Three did not, and each
  reached values worth hundreds of times the plant's authority.
- **Derive, do not re-derive.** `derive_dynamics.py` runs the project's own
  recursion symbolically instead of writing a fresh Lagrangian, so the closed form
  is the same function by construction. It is accepted because it agrees to
  3.6e-15, not because it looks right.
- **A margin written as a constant stops being a margin** the moment the authority
  it guards changes. Both the tube input margin and the clamp gains are now
  fractions.
- **Record the sweeps that failed.** `params.py` carries several, including one
  explicitly marked void because the rig it was taken on had the vehicle on the
  ground. A deleted negative result gets rediscovered.

---

## 5. What the tube machinery actually contributes

Worth stating plainly, because it is easy to credit the wrong thing for the recent
improvement.

The Minkowski sums and differences — Eq. (35) and (36), `Tube.reachable_sets` and
`BoxConstraint.tighten` — were **already implemented and unchanged** through all
four of the fixes in `README.md` section 5. Only one tube-related number changed:
the manipulator's input margin became `0.2 * joint_torque` instead of a constant
1.0 N m, and on the rigid-base bench that moved the arm's error from 23 % to 12 %
at a 2.5 N m budget and left the 1.5 N m case unchanged. Real, and small.

The factor of five on `fast`, the collapse of the run-to-run dispersion, and the
manipulator's correlation going from zero to 0.98 came from timing, a reference
shape, an attitude parameterisation and a trajectory offset. None is a robust-MPC
mechanism.

Where the tube construction *did* prove itself is narrower and worth keeping: with
the gain frozen at the design point instead of re-solved per step, the closed loop
`|A + BK|` had a row sum of 39.44 with the arm folded against 1.000 for the gain
solved at that pose. The reachable set then swallowed the input box, the tightened
set came out empty at the first horizon step, and the joints sat at their bound
permanently. So the scheduled re-solve of Eq. (32) is load-bearing; the set
arithmetic around it has not yet been shown to earn its cost, and showing that
would need the ERTF comparison in `README.md` section 7.

Two parts of the paper's robustness argument are absent and should not be claimed:
the terminal RPI set of Theorem 1, where we have only a Riccati terminal cost, and
Proposition 1's infeasibility declaration, which the 75 % tightening cap replaces
by continuing with a quarter of the box.

---

## 6. Commit history as a record

Each commit carries the measurement that motivated it, negative results included —
which on this project are worth more than the positive ones. Tags mark the points
worth returning to.

```
000ef99  tre-scene              the circle centred; three scenarios under a metre
49be67a  sincronia              the simulator drives the controller; dispersion gone
2bc0223                         the translational MPC measured: worse, and why
94f0fa5  manipolatore-insegue   horizon reference; geometric attitude error
0b16b0a  punto-buono-cascata    cascade separated 1.5 / 4.0 / 16.5
de64ca2                         world paused until engage; 20 Hz instead of 100
590e361                         physical bound on joint acceleration
f726414                         dt of the model matches the simulator's
905e81e                         closed form: tick 43 → 23 ms
ccde426                         R_eta 3.0 does not fix it; f_bar is not the cause
2bf874a                         Table II is not the discriminant
8cd2a08                         Table II at 100 Hz not executable: a tick costs 20.17 ms
26545c0                         position at the centre of mass, not the link origin
```

```bash
git checkout tre-scene      # the current working point
```
