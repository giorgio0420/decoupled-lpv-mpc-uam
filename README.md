# Tube-based LPV-MPC for an aerial manipulator

A ROS 2 / Gazebo reproduction of

> M. Eskandarpour, A. Soltanshah, K. Gupta, M. Mehrandezh, *Decoupled Dynamic
> Modeling by Decomposing the Cross-Coupled Dynamics and Tube-Based LPV-MPC
> Control Scheme for Aerial Manipulation*, IEEE Transactions on Aerospace and
> Electronic Systems, **61**(5), October 2025.

A hexarotor carrying a three-link arm. The arm's reaction on the aircraft is
derived in closed form as a function of both subsystems' states, and each
subsystem is controlled by its own tube-based MPC on a linear
parameter-varying model.

The paper validates in MATLAB. This validates in Gazebo Fortress, which turned
out to be a different problem: most of the work below was not control design but
finding out that the simulated aircraft was on the ground, or that the loop was
closing at a period nobody had measured.

---

## 1. What runs today

Three trajectories at three difficulties, the vehicle and the arm both tracking,
with the arm's reaction fed forward into the attitude loop.

### Table III — LPV-MPC against the ERTF baseline

| scenario | scheme | position error | % of amplitude | joint error | % of amplitude |
|---|---|---|---|---|---|
| `nominal` | **LPV-MPC** | 0.123 m | **4.9 %** | 0.0153 rad | **4.8 %** |
| `nominal` | ERTF | 0.353 m | 14.1 % | 0.4749 rad | 149.3 % |
| `fast` | **LPV-MPC** | 0.462 m | **9.2 %** | 0.0230 rad | **4.3 %** |
| `fast` | ERTF | 0.652 m | 13.0 % | 0.5581 rad | 105.3 % |
| `disturbance` | **LPV-MPC** | 0.118 m | **4.7 %** | 0.0155 rad | **4.9 %** |
| `disturbance` | ERTF | 0.348 m | 13.9 % | 0.4741 rad | 149.0 % |

The paper claims 2–8 % for LPV-MPC and 20–30 % for ERTF. Measured past the 6 s
ease-in blend, arm enabled, two runs per cell, both schemes truncated to the same
window before integrating so a longer run cannot score worse on IAE for no reason
of merit. `python3 src/uam_control/table3.py results` regenerates it from a
directory of collected traces (`trace_<mpc|ertf>_<scenario>.csv`).

**How to read this table honestly.** The joint column is clean: a factor of 22 to
31, with ERTF's error exceeding the amplitude of its own reference, which means it
does not track at all. The position column is *partly contaminated*, because the
outer position loop is shared between the two schemes — tuning it better improved
both, and improved ERTF more, to the point where our ERTF now sits *below* the
band the paper attributes to it. The 2.9× ratio on position should be read knowing
that.

The cleanest result in the project is not in this table. It is the ablation: same
controller, same tuning, `UAM_NO_COUPLING=1` zeroing the reaction terms of
Eq. (10)–(11). Joint error goes from 0.011 to 0.570 rad and the correlation from
1.00 to 0.00 — a factor of fifty, with nothing else changed. That is the paper's
central claim put on trial rather than assumed.

Attitude stays level: peak pitch 0.080 rad on `nominal`, 0.167 on `disturbance`,
0.318 on `fast`. The allocation never has to give up torque (`sfit` = 1.000), so
the wrench the rotors deliver is the wrench the controllers asked for. Runs repeat
to the third decimal, which is itself a result — section 5 explains why.

**Not reproduced**: the terminal RPI set of Theorem 1 is implemented and verified
but destroys tracking in flight, and Proposition 1 exists only as a diagnostic.
`fast` still diverges about one run in four for reasons not yet localised. And the
airframe itself is ours, not the paper's — see section 9. Details in section 7.

---

## 2. Building from nothing

### 2.1 What you need

| | version used here |
|---|---|
| OS | Ubuntu 22.04 (WSL2 is fine) |
| ROS 2 | Humble |
| Gazebo | Fortress — `ign gazebo` 6.18.0 |
| Python | 3.10, with numpy 1.21, scipy 1.8, sympy 1.9 |

```bash
sudo apt update
sudo apt install -y ros-humble-desktop ros-humble-ros-gz \
  ros-humble-ros-gz-sim ros-humble-ros-gz-bridge \
  ros-humble-robot-state-publisher ros-humble-joint-state-publisher \
  ros-humble-xacro python3-colcon-common-extensions \
  python3-numpy python3-scipy python3-sympy python3-pytest
```

On WSL2 without a GPU, Gazebo's GUI needs software rendering; `phase3.sh` already
exports `LIBGL_ALWAYS_SOFTWARE=1`.

### 2.2 Clone and build

```bash
git clone <the private repo URL> ~/uam_ws
cd ~/uam_ws
source /opt/ros/humble/setup.bash
colcon build
source install/setup.bash
```

`colcon build` builds three packages:

- `hexacopter_sim` — the XACRO model, the launch file, the ROS–Gazebo bridge
  parameters
- `hexacopter_control` — the C++ Gazebo system plugin that turns rotor speeds
  into forces and torques, plus an allocation node
- `uam_control` — everything else: the dynamics, the three controllers, the
  trajectories, the flight node

The plugin has to be findable by Gazebo, which is why the run script exports
`IGN_GAZEBO_SYSTEM_PLUGIN_PATH`. `phase3.sh` locates the workspace from its own
path (`WS_DIR`), so it runs unmodified from wherever you cloned it.

### 2.3 Check the model before flying it

```bash
source /opt/ros/humble/setup.bash && source install/setup.bash
python3 -m pytest src/uam_control/test -q
```

33 tests. They check the Newton–Euler recursion against analytic cases, the
cross-coupled decomposition, the LPV construction and the MPC's constraint
handling. If these fail, nothing downstream is worth running.

Two further gates, both worth running once after a fresh clone:

```bash
python3 src/uam_control/test_generated.py     # closed form vs the recursion, 200 random states
python3 src/uam_control/check_swap.py         # the fast path vs the probing path
```

The first must agree to about 1e-15, the second to 1e-7. The closed form is
accepted because it agrees, not because it looks right.

---

## 3. Running a simulation

```bash
cd ~/uam_ws
UAM_SCENARIO=nominal bash phase3.sh 95 true
```

Two positional arguments: the wall-clock timeout in seconds, and whether the arm
is enabled. `95 true` gives about 40 s of simulated time with the arm under its
own MPC. `95 false` clamps the arm at the hanging pose instead, which is the
configuration for judging the aircraft alone.

What the script does, in order — and every step of it is there because of a
failure it prevents:

1. Kills any surviving Gazebo server, then refuses to continue unless exactly one
   `hexacopter_robot` answers. Four stale servers were once publishing `/uav/odom`
   at the same time and the controller was reading a blend of four worlds.
2. Launches the world **paused**. The launch takes about six seconds, and six
   seconds of free fall is 176 m — with the world running, the vehicle engaged at
   0.99 m with its arm curled against the floor, and every "it will not hold
   attitude" measurement was taken from there.
3. Starts the controller, which drives the arm straight and latches the starting
   pose only once every joint is within 0.05 rad of hanging.
4. Releases the pause.

Outputs:

- `/tmp/trace.csv` — 32 columns at the full control rate: both references beside
  both measurements, the commanded and realised wrench, and the per-tick cost.
  Every number in this README comes out of this file.
- `/tmp/ctrl.log` — the controller's own log
- `/tmp/phase3.log` — the launch, the plugin's geometry diagnostics

Scenarios: `hover` (nothing moves — the baseline for telling a control fault from
a tracking error), `nominal`, `fast`, `disturbance`.

### 3.0 Seeing what it is doing

The commanded pose is drawn in the world: a **green sphere** at the reference and
a **blue trail** of where the reference has been. Until that existed nothing on
screen said where the vehicle was *supposed* to be, and a run holding station to
12 cm looked exactly like one drifting ten metres.

It is drawn from inside the rotor plugin rather than from the controller, because
the marker service belongs to Ignition and not to ROS: a Python node would have to
shell out to `ign service` once per tick, which costs more than the whole control
loop. `hover_node` publishes the point on `/uav/reference` and the plugin issues
one in-process request per fifty physics steps, which is 20 Hz on screen.

### 3.1 Environment switches

| variable | default | what it does |
|---|---|---|
| `UAM_SCENARIO` | `nominal` | which of the four references to fly |
| `UAM_GUI` | `1` | `0` starts the server alone; attach the window later with `ign gazebo -g` |
| `UAM_USE_ERTF` | `0` | fly Section V's baseline instead of the two tube controllers |
| `UAM_POS_BW` | `2.0` | position-loop bandwidth [rad/s] |
| `UAM_ATT_GAIN` | `4.0` | attitude error → commanded body rate |
| `UAM_JOINT_TAU` | `4.0` | joint torque budget [N m] |
| `UAM_N_ETA` | `4` | rotational horizon |
| `UAM_TERMINAL_SET` | `0` | impose Theorem 1's terminal constraint |
| `UAM_ATTITUDE_ONLY` | `0` | bypass the position loop, hold level at hover thrust |
| `UAM_NO_COUPLING` | `0` | force every reaction term to zero |
| `UAM_TRANS_MPC` | `0` | use the paper's translational MPC instead of the PD |
| `UAM_ERTF_KP`, `UAM_ERTF_KD` | `30.0`, `0.05` | the baseline's attitude PID |

**`UAM_GUI=0` is worth using for every measurement.** Gazebo's GUI has no working
hardware path under WSL — the d3d12 GL translation raises
`Ogre::UnimplementedException` and Gazebo aborts — so it falls to the software
rasteriser, which measured **393.8 % of an 8-core machine** while the control loop
got 50 %. Detached, the server drops to 87.5 %, the controllers' profile goes from
31.6 ms to 17.5, and the same 95 s of wall clock buys 82 s of simulated time
instead of 33. The window is not lost: `ign gazebo -g` attaches to a running
server at any moment, including mid-run, which is how the videos are made.

The last two are diagnostic and both earned their place. `UAM_NO_COUPLING=1`
proved the decomposition was innocent of the tipping; `UAM_ATTITUDE_ONLY=1`
separated a rotational fault from a cascade fault, which no measurement with both
loops closed could do.

### 3.2 Reading the results

```bash
python3 tools/iae.py run=/tmp/trace.csv          # the paper's metric: IAE per axis and per joint
python3 src/uam_control/table3.py results/       # section 1's table, from a directory of collected traces
```

`tools/iae.py` takes several `label=path` pairs at once, which is how a
multi-run comparison should be read:

```bash
python3 tools/iae.py run1=/tmp/run_1.csv run2=/tmp/run_2.csv run3=/tmp/run_3.csv
```

### 3.3 Benches that need no simulator

```bash
python3 src/uam_control/arm_paper.py nominal 0.0    # the arm against Section V's reference, base rigid
python3 src/uam_control/arm_paper.py nominal 40.0   # same, with 40 rad/s^2 of vehicle rotation injected
python3 src/uam_control/cascade.py                  # closed-loop eigenvalues, no dispersion
python3 src/uam_control/rpi_check.py                # the terminal set: invariance, admissibility, non-emptiness
```

`arm_paper.py` is the most useful of the four. It holds the base perfectly still
and asks the same `ManipulatorTubeMPC` that flies in Gazebo to follow the same
reference, so an arm fault and a coupling fault stop being confusable. It is how
the manipulator's real problem was found.

---

## 4. The theory, and what of it is implemented

### 4.1 The cross-coupled decomposition — Eq. (10)–(11)

The paper's first contribution. Rather than treating the arm's reaction on the
aircraft as an unknown disturbance to be estimated, it is written in closed form
as a function of both subsystems' states:

```
tau_rea = M_d w_dot + M_c [pq, qr, pr] + M_s [p^2, q^2, r^2] + M_l [p, q, r] + tau_bar
f_rea   = M_f f_z + f_bar
```

Every coefficient depends on the manipulator configuration `(Theta, Theta_dot,
Theta_ddot)`. `M_d` is the arm's contribution to the effective inertia, so the
aircraft flies with `I_t' = I_t - M_d`, and the residuals enter the rotational
model as part of the input rather than as an additive unknown.

**Implemented**, in `coupling.py`, two ways that agree:

- a *probing* path that recovers each matrix from `newton_euler.py` by finite
  differences — 43 full dynamics evaluations per tick, 0.55 ms each in situ
- a *closed-form* path in `generated_dynamics.py`

The closed form was not re-derived from a Lagrangian. `derive_dynamics.py` runs
*the project's own recursion* through SymPy, so the generated expressions are the
same function the numeric code computes by construction rather than by agreement,
and `test_generated.py` confirms it on 200 random states to 3.6e-15. That took the
tick from 42.9 ms to 22.9 ms, `decompose` alone from 8.06 ms to 0.66.

### 4.2 The LPV representation — Eq. (39)–(42)

Both nonlinear subsystems are rewritten affine in a scheduling vector: `[p, q, r]`
for the rotational dynamics, `[p, q, r, q1, q2, q3]` for the manipulator. For the
arm,

```
P1 = -M^-1 C     P2 = -M^-1 Q     P3 = M^-1     u = tau + T
```

with `M`, `C`, `Q`, `T` rebuilt at every step from the measured configuration and
then discretised by zero-order hold. The input is `tau + T`, not `tau` — only
`tau` is actuated, so the admissible input set is the actuator box *shifted by the
residual*, not a box centred on zero. With the residual reaching several times the
actuator authority, that distinction decides whether the optimiser is asked for
something the hardware can produce.

**Implemented**, `coupling.manipulator_lpv` and `controllers.rotational_lpv`.

### 4.3 Tube-based robust MPC — Eq. (22)–(38)

The parameters vary, so the nominal model is wrong by a bounded amount. The tube
construction bounds where the true trajectory can be relative to the nominal one
and shrinks the constraints by that much:

```
R(i|k) = A_K R(i-1|k) (+) W(i|k)         reachable set, Eq. (35), (+) = Minkowski sum
X~(i)  = X (-) R(i|k)                    tightened set,  Eq. (36), (-) = Minkowski difference
u      = K e + u~                        Eq. (32), tube feedback plus nominal command
```

**Implemented**, `mpc.Tube.reachable_sets`, `BoxConstraint.tighten`,
`Tube.confine`. Both MPCs re-solve `K` at the current operating point every step,
which is what Eq. (32) asks for: freezing it at the design point was measured
giving a closed-loop row sum of 39.44 with the arm folded, against 1.000 for the
gain solved at that pose — the reachable set then swallowed the input box, the
tightened set came out empty, and the joints sat at their bound permanently.

Two deviations, both deliberate and both recorded:

- **The tightening is capped** at 75 % of each half-width. Uncapped, an over-wide
  tube collapses the set onto its centre and the controller produces no command
  at all while still reporting a solved QP. The paper's Proposition 1 would
  declare infeasibility instead. Ours keeps a quarter of the box and continues,
  which turns a silent failure into visible saturation — but does not reproduce
  recursive feasibility.
- **The input margin is a fraction of the authority**, `0.2 * joint_torque`, not a
  constant. It was a constant 1.0 N m against a 0.5 N m budget: twice the entire
  box it was tightening.

**Not implemented**: the terminal RPI set. We have `riccati_terminal_cost`, a
quadratic terminal *penalty*, where Theorem 1's asymptotic-stability argument
rests on a terminal *set* that is robustly positively invariant. The guarantee
does not transfer.

### 4.4 The cascade

| stage | paper | here |
|---|---|---|
| translational | constrained linear MPC, Eq. (13)–(18) | PD, bandwidth 1.5 rad/s |
| rotational | tube LPV-MPC, Eq. (22)–(38) | as the paper |
| manipulator | tube LPV-MPC, Eq. (39)–(48) | as the paper |

`TranslationalMPC` implements the paper's stage and is wired in behind
`UAM_TRANS_MPC=1`, but it is **worse** — 3.5 to 135 m against the PD's 0.2 — and
it fails specifically on altitude, IAE z going from 1.4 to 377. That points at the
gravity bookkeeping between Eq. (13) and Eq. (14), where the optimiser works in
net force and the interface expects `u_z + m g`, or at integral windup in the
augmented state of Eq. (16), which has no anti-windup and an eight-step horizon.
It is kept with its numbers so the next attempt starts from the altitude channel.

### 4.5 What has no counterpart in the paper

The paper simulates; this flies a model. Three stages exist only here:

- **Control allocation.** The wrench has to become six rotor speeds. `pinv` alone
  returns a minimum-norm solution free to go negative, and clipping that at zero
  silently delivers a different wrench. It showed as rotors alternating 1-3-5 /
  2-4-6 straight to zero, because yaw comes from rotor drag and `k_m/k_f` is 0.02
  here — a negligible yaw demand asks for an enormous alternating split. So:
  thrust on all six equally, which is always feasible, then as much of roll and
  pitch as keeps every rotor in range, then whatever yaw still fits. Direction is
  preserved, magnitude gives way, and the realised wrench is logged next to the
  commanded one.
- **Measurement at the centre of mass.** `/uav/odom` publishes the link origin,
  which the XACRO puts 2 m below the aircraft. Regulating that point makes the
  horizontal loop non-minimum-phase: tilt by `+theta` and a point 2 m below the
  axis swings to `-2 theta` while the thrust tilts toward `+x`. Correlation
  between pitch and measured x is **-0.997** at the link origin and **+0.962** at
  the centre of mass. No gain, weight or torque limit ever helped, because on a
  non-minimum-phase response the gain only sets how fast it diverges.
- **Geometric attitude error.** See section 5.

---

## 5. The four things that made it work

None of them was a control-design change. Presented in the order the effect size
runs, largest first.

### 5.1 The simulator drives the controller

The loop used to run on a ROS timer, the physics on its own clock, and the two
agreed only by luck. Now the tick fires from the odometry callback, decimated on
the *simulated* timestamp, so the control period is exact in simulated time by
construction. The schedule is absolute rather than incremental — the next due time
advances by whole periods from the previous due time — so one late tick does not
drag the rest, and a jump larger than one period resynchronises instead of firing
a catch-up burst.

This is what `sysCall_actuation` gives a CoppeliaSim script: the simulator calls
the controller, one step at a time.

|  | ROS timer | simulator-driven |
|---|---|---|
| error, three runs | 1.18 / 1.22 / 1.35 m | 0.281 / 0.302 / 0.309 m |
| IAE x / y | 20 – 38 | 2.6 – 3.3 |
| peak pitch | 0.25 – 0.60 rad | 0.079 rad |
| joint correlation | 0.57 – 0.88 | 0.86 / 0.91 / 0.98 |
| `sfit` min | 0.63 – 0.93 | 1.000 |

The **dispersion** matters more than the mean. Three independent runs agreeing to
within 10 %, where the same configuration used to spread 1.18 to 1.80 with
divergences on top. That scatter was wide enough to swallow every effect this
project tried to measure, and it invalidated a whole session of single-run
conclusions. It was the clocks.

Every timing patch that preceded it — measuring dt from odometry stamps, re-seeding
the tube after a stall, clipping the period — was treating a symptom.

### 5.2 The manipulator was given a reference with a future

`hover_node` built `tile(concat([pose, zeros]))`: one pose repeated across the
horizon with a **zero rate target**. So the optimiser was told to be at 0.33 rad
*and* at rest, at every one of the four horizon steps, against a reference really
moving at up to 0.83 rad/s. It split the difference.

| | before | after |
|---|---|---|
| joint correlation | −0.14 / −0.14 / +0.68 | **0.84 / 0.74 / 0.88** |
| joint error | 0.32 / 0.46 / 0.58 rad | 0.12 / 0.12 / 0.11 |

The correlation was **zero at every torque budget tried**, while the same
controller on the rigid-base bench follows the same waveform to 3 % of its
amplitude. The joints were moving because the vehicle shook them. A tiled constant
also discards the preview, which is the only thing an MPC has over a PD on a
reference that reverses every 5 s.

### 5.3 The attitude loop stopped giving up past a radian

`rate_target = B_T_I @ (gain * euler_error)` looks like a coordinate change and is
really an attitude-dependent gain. Its pitch row is `[0, cos(phi),
sin(phi) cos(theta)]`, so the commanded pitch rate carries a factor of
**cos(roll)**. Near level that factor is 1 and nothing shows; when roll passes
π/2 it is zero and the pitch command vanishes however large the error.

Measured at roll = π/2 with 0.3 rad of pitch error, the old expression asks for
**−0.0001 rad/s**; the geometric error asks for −0.59.

In the run that crashed: pitch 1.41 rad, body rate 8.55 rad/s, commanded torque
between −0.04 and +0.03 N m for half a second, all the way from 3.4 m down to the
floor. The vehicle was not unrecoverable — the controller had stopped asking. The
arm reaching its stop, which every failure had been blamed on, happens *after* the
ground impact.

Replaced with `0.5 vee(R_d^T R - R^T R_d)`: smooth at every attitude short of
exactly inverted, no small-angle assumption, already in the body frame. At level
attitude it agrees with the old expression to within 2 %.

### 5.4 The circle is centred where the vehicle is

Section V's path is `x = a sin(wt)`, `y = a cos(wt)` — a circle, symmetric in
amplitude and frequency. Referring it to the vehicle by subtracting `paper(0)` put
the start at the circle's **y extreme**, so commanded y ran from the origin to
`origin - 2a` while x ran from `-a` to `+a` about it. Y travelled twice the
distance from the start that x did — 10 m against 5 on `fast`.

| `fast` | before | after |
|---|---|---|
| \|ex\| / \|ey\| | 1.193 / 6.351 → ratio **0.19** | 0.457 / 0.412 → ratio **1.11** |
| error mean | 3.42 / 3.45 / 6.67 m | 0.675 / 0.679 m |

A factor of five on the hardest scenario, from a trajectory with no asymmetry in
it at all.

### 5.5 What the cross-coupled decomposition is worth

The paper's central claim, tested directly rather than assumed. Same 4.0 N m
budget, same `nominal` scenario, two runs each, with `UAM_NO_COUPLING=1` — which
forces every term of Eq. (10) and (11) to zero — as the only difference:

| | with coupling | without |
|---|---|---|
| vehicle error | **0.226 m** | 0.667 m |
| joint error | **0.011 rad** | 0.570 rad |
| joint correlation | **1.00 / 0.99 / 1.00** | 0.00 / 0.35 / 0.92 |
| peak pitch | **0.076 rad** | 0.159 rad |

Fifty times better at the joints, three times at the vehicle. This is the same
shape of result the paper reports for LPV-MPC against ERTF, obtained by removing
the decomposition from our own controller rather than by implementing a second
one — so it is evidence that Eq. (10)–(11) earn their place, not yet the paper's
comparison. That still needs `ErtfBaseline`; see section 7.

Note also which parts of the scheme this does *not* validate. The Minkowski set
arithmetic was unchanged and identical in both columns.

---

## 6. The configuration, and why each number is what it is

```
dt_eta = dt_gamma = dt_zeta = 0.05 s     20 Hz, not Table II's 100
N_eta = N_gamma = 4, N_zeta = 8          Table II
Q_eta = 15, Q_gamma = 100, R_eta = 0.3, R_gamma = 0.01
position bandwidth  1.5 rad/s
attitude gain       4.0
rate loop           K = 2.97 N m per rad/s, K/I = 16.5 rad/s
joint torque        4.0 N m   (actuator effort 6.0 in the XACRO)
vehicle torque      7.0 N m
tilt cap            0.6 rad
spawn               -z 10.0, world paused until the loop engages
```

**Why 20 Hz and not Table II's 100.** A tick costs 19 to 45 ms, median 25. The
rate loop's eigenvalues with the one sample of actuation delay the implementation
really has:

| dt | 10 ms | 20 | 25 | 30 | 40 |
|---|---|---|---|---|---|
| ρ | 0.979 | 0.957 | 0.947 | 0.987 | **1.140** |

Divergent inside the spread the node produces. At 50 ms the budget exceeds the
cost with margin, and the LQR gain is re-solved at whichever dt is in force, so
20 Hz is the same design procedure at a slower rate rather than a detuning.
Table II's 100 Hz needs a tick under about 8 ms — a compiled controller, not this
one.

**Why the cascade is 1.5 / 4.0 / 16.5 rad/s.** Each loop several times faster than
the one commanding it. Swept on `nominal`:

```
bandwidth   0.8   1.2   1.5   2.0        attitude gain   4.0   6.0    9.0    12.0
error      2.10  1.57  0.99  1.10        error          1.31  0.99   3.83  216.4
```

6.0 is the best single number and the wrong choice: nine runs at 6.0 came out
bimodal, six tracking to 0.9–1.2 m and three leaving at 11, 64 and 95 m. 6.0
against a 0.3 rad error asks 1.8 rad/s, the rate loop delivers about 30 rad/s²,
and past roughly 4 rad/s² the shoulder cannot hold itself up. 4.0 is a third worse
when both work and reliable when 6.0 is not.

**Why the joint budget is 4.0 N m.** On the rigid-base bench more is strictly
better:

| budget | 0.5 | 1.0 | 1.5 | 2.5 | 4.0 |
|---|---|---|---|---|---|
| arm error vs amplitude | 73 % | 37 % | 47 % | 12 % | **3 %** |

and it was held at 1.5 for a while on a measurement that said 4.0 threw the
vehicle. That measurement came off the rig *before* the loop was driven by the
simulator and before the attitude error was geometric — a rig whose runs diverged
for reasons that had nothing to do with the arm. Re-measured, 4.0 costs the
vehicle nothing (0.207 m on `nominal` either way, peak pitch 0.076 rad either way)
and takes the joints from 28 % of their reference amplitude to 4 %.

It also improved the vehicle on `fast`, 18.1 % → 13.0 %. An arm that follows its
reference delivers a reaction the decomposition can predict; a starved arm
delivers whatever the vehicle's motion shakes out of it. More torque from a
tracking arm is easier to carry than less from a flailing one.

The arm's own tracking is also limited by the vehicle's aggressiveness,
independently of budget:

| vehicle ω̇ | 0 | 2 | 5 | 20 | 40 rad/s² |
|---|---|---|---|---|---|
| arm error | 3 % | 3 % | 7 % | 11 % | **73 %** |

---

## 7. What to do next

Ordered by what each buys for the exposition.

**1. The IAE table against the ERTF baseline.** This is the paper's headline
result — Table III, 2–8 % tracking error for LPV-MPC against 20–30 % for ERTF —
and it is the one number a reader will look for. `ErtfBaseline` is already written
in `controllers.py` and `hover_node` does not use it; the calling convention is at
`simulate.py:203`. About ten lines, and the runs are now reproducible enough for
the comparison to mean something, which they were not two days ago.

Done

**2. Put the attitude in a constraint set.** The rotational MPC's state is the body
rate alone, three states, so the attitude belongs to no set and Eq. (36) cannot
touch it. `LIMITS.tilt` caps the *commanded* tilt at 0.6 rad and never binds —
measured demand peaks at 0.445 on `fast`, zero samples at the cap — while the
*achieved* attitude overshoots the command by about 70 % and once reached 1.5 rad
unopposed. A six-state rotational model `[angles, rates]` with the tilt in the box
is what the paper's Eq. (22) actually describes, and it is the honest fix for the
last failure mode still visible.

**3. The terminal RPI set.** Theorem 1's guarantee needs it. Computing a maximal
RPI set for the tightened system is a standard offline iteration and would make
the stability claim transferable rather than merely plausible.

**4. Show the reference in Gazebo.** Nothing in the window says where the vehicle
is *supposed* to be, so a watcher cannot tell tracking from wandering. A marker
along the commanded path, published from `hover_node`, would make the runs
readable without the CSV.

**5. `fast` is still three times worse than `nominal`**, 13.0 % of amplitude
against 8.3 %, and it is the only scenario outside the paper's band. The position
bandwidth was swept against `nominal` and never against `fast`, where the required
tilt is five times larger — so the sweep that chose 1.5 rad/s has never been asked
the question `fast` poses. This is the cheapest of the remaining gaps.

**6. Recover yaw control.** `torque[2]` is forced to zero. Yaw is passively stable
here and never wandered past 0.07 rad, so it costs nothing today, but it has to
come back before any yaw-holding manoeuvre. The suspect is the tube feedback
swamping the nominal command once the nominal state has drifted: a 0.05 rad yaw
error needs 0.03 N m and the optimiser asked for thirty times that.

**7. Gazebo stalls.** The simulated clock still jumps 450 to 960 ms occasionally.
The synchronisation makes the controller handle them correctly rather than being
corrupted by them, so this is now a nuisance rather than a fault — but a
controller written as a Gazebo system plugin, called from the physics loop the way
`hexacopter_rotor_plugin` already is, would remove the last of it. Headless was
tried and made things worse, and section 4 of `INTERNALS.md` records why so it is
not tried again.

---

## 8. Repository layout

```
phase3.sh                     run a scenario in Gazebo (section 3)

src/uam_control/
  hover_node.py               the flight node: cascade, allocation, tracing
  uam_control/
    params.py                 Table I and II, with every failed sweep recorded
    newton_euler.py           the recursion, Craig's convention
    coupling.py               Eq. (10)-(11), probing and closed-form paths
    generated_dynamics.py     generated -- do not hand-edit
    controllers.py            the three MPCs, allocation, geometric attitude, ERTF
    mpc.py                    QP, LQR, Riccati, tube sets, box constraints
    trajectory.py             Section V's four scenarios
    plant.py, simulate.py     the offline plant and driver
    control_node.py           ROS 2 node wrapper (rclpy)
  derive_dynamics.py          SymPy: runs our recursion symbolically, writes the closed form
  test_generated.py           the gate: closed form vs recursion, 1e-15
  check_swap.py               the gate: fast path vs probing path, 1e-7
  arm_paper.py, arm_only.py   the arm alone, base rigid
  cascade.py                  closed-loop eigenvalues
  rpi_check.py                the terminal RPI set: invariance, admissibility, non-emptiness
  table3.py                   regenerates section 1's table from collected traces
  test/test_model.py          33 tests
  package.xml, setup.py, resource/   ament_python package

src/hexacopter_sim/           XACRO model, launch file, ROS-Gazebo bridge parameters
src/hexacopter_control/       the Gazebo rotor plugin and the allocation node (C++)

tools/iae.py                  the paper's metric: IAE per axis and per joint

README.md        this file
INTERNALS.md      the defects, the method, and the traps
```

---

## 9. A warning about the numbers in this repository

Twenty-four conclusions in this project's history were wrong, and most were wrong
the same way: a single Gazebo run compared against another single run, on a
process whose run-to-run dispersion was wider than the effect being measured. One
configuration that looked good at ±0.34 m was disproved by three re-runs of
itself.

The dispersion is gone as of section 5.1 — runs now repeat to the third decimal —
but the habit it should leave behind is: **three runs minimum, and prefer a bench
with no dispersion at all** (`cascade.py`, `arm_paper.py`, `test_generated.py`)
whenever the question can be asked there instead.

Comments in the source that record a sweep or a measurement are load-bearing.
Several of them exist to stop a plausible-looking change from being made again,
and one of them (`params.py`, the joint-torque sweep) is explicitly marked void
because the rig it was taken on had the vehicle sitting on the ground.
