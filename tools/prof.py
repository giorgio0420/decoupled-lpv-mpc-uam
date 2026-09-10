"""How much of the tick-to-tick interval is the controller, and how much is not.

The controller measured 20.17 ms in isolation while the node runs at 49.7 ms per
tick. Either the controller costs more in situ, or most of the interval is spent
outside it -- CPU contention with Gazebo, ROS callback overhead, waiting. This
compares the time spent inside tick() against the interval between ticks, which
separates the two without guessing.
"""

import sys

import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/PROF_1.csv"
d = np.genfromtxt(path, delimiter=",", names=True)

tm = d["tick_ms"]
gap = np.diff(d["wall"]) * 1000.0

print(f"{path}: {d.size} tick, budget 25.0 ms")
print(f"  dentro tick()      mediana {np.median(tm):6.2f} ms"
      f"   p95 {np.percentile(tm, 95):7.2f}   massimo {tm.max():8.1f}")
print(f"  intervallo         mediana {np.median(gap):6.2f} ms"
      f"   p95 {np.percentile(gap, 95):7.2f}   massimo {gap.max():8.1f}")
print()
share = np.median(tm) / np.median(gap)
print(f"  il controllore occupa il {100 * share:.1f}% dell intervallo")
print(f"  il resto, {100 * (1 - share):.1f}%, e fuori dal tick")
print()
if share > 0.7:
    print("  Il collo di bottiglia e il controllore: ottimizzarlo paga.")
else:
    print("  Il collo di bottiglia NON e il controllore. Il tempo se ne va")
    print("  fuori dal tick: contesa di CPU con Gazebo, o attesa. Ridurre il")
    print("  costo del controllore non accorcia l intervallo -- va ridotta la")
    print("  domanda totale di CPU, oppure il nodo va disaccoppiato dal")
    print("  simulatore con use_sim_time.")
