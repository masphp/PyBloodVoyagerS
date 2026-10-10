# PyBloodVoyagerS (PyBVS)

**PyBloodVoyagerS (PyBVS)** is a Python-based Blood-Voyager-style Monte Carlo simulator for generating nanobot circulation trajectories through a configurable vascular-network topology.

The simulator preserves the concept of vascular-network traversal while replacing the legacy artificial stream-based motion model with an event-driven circulation model based on:

* 3-D vascular geometry
* vessel diameter
* supplied cross-sectional mean velocity
* continuous Poiseuille laminar flow
* area-uniform cross-sectional sampling
* diameter-based branch selection
* an effective arteriole–capillary–venule organ model
* exponential cardiac residence times
* topology-driven circulation cycles

PyBVS is intended to generate reproducible synthetic circulation trajectories. It is a computational abstraction and is **not** a full CFD or physiologically exact cardiovascular model.

## Repository contents

The repository contains:

```text
PyBloodVoyagerS.py
Pvasculature.csv
PcsvNanoSec.csv
Topology_visualization.png
README.md
```

`PyBloodVoyagerS.py` is the simulator.

`Pvasculature.csv` is the vascular topology input adopted from vasculature.csv of BloodVoyagerS project: <https://github.com/RegineWendt/blood-voyager-s>

`PcsvNanoSec.csv` is the primary Monte Carlo output generated from:

```bash
python PyBloodVoyagerS.py --input Pvasculature.csv --num-cycles 100000 --seed 12345
```

`Topology_visualization.png` is an example 3-D topology visualization generated with:

```bash
python PyBloodVoyagerS.py --input Pvasculature.csv --num-cycles 1000 --seed 12345 --plot-topology
```

## Installation

The minimum dependency required for simulation is NumPy:

```bash
pip install numpy
```

For the optional 3-D topology visualization:

```bash
pip install numpy matplotlib
```

Matplotlib is only required when the `--plot-topology` option is used.

## Running the simulator

A standard 100,000-cycle simulation can be run with:

```bash
python PyBloodVoyagerS.py \
    --input Pvasculature.csv \
    --num-cycles 100000 \
    --seed 12345 \
    --min-velocity 0.001 \
    --topology-tolerance 1e-6
```

A smaller diagnostic run is:

```bash
python PyBloodVoyagerS.py \
    --input Pvasculature.csv \
    --num-cycles 1000 \
    --seed 12345 \
    --verbose
```

To generate the 3-D topology visualization:

```bash
python PyBloodVoyagerS.py \
    --input Pvasculature.csv \
    --num-cycles 1000 \
    --seed 12345 \
    --plot-topology
```

To stop on the first invalid or aborted realization:

```bash
python PyBloodVoyagerS.py \
    --input Pvasculature.csv \
    --num-cycles 1000 \
    --seed 12345 \
    --strict-validation
```

The random seed is configurable. Running the same topology with the same simulation parameters and seed produces a reproducible Monte Carlo realization sequence.

## Input topology

The input file is `Pvasculature.csv` adopted from vasculature.csv of BloodVoyagerS project <https://github.com/RegineWendt/blood-voyager-s> by replacing Type column with the SubType column and adding Diameter(cm) and Mean Velocity (cm/s) columns for SubTypes 1 to 9.
The Velocities are taken from Table 2.1 (Values of various physical quantities for the systemic circulation) of:
Hoskins, P. R., Lawford, P. V., & Doyle, B. J. (Eds.). (2017). *Cardiovascular Biomechanics*. Springer. <https://doi.org/10.1007/978-3-319-46407-7>

Each topology record contains:

```text
ID
Name
SubType
StartX
StartY
StartZ
StopX
StopY
StopZ
Diameter(cm)
Mean Velocity (cm/s)
```

Coordinates are in centimeters.

Diameter is in centimeters.

Mean velocity is in centimeters per second.

Time is in seconds.

The simulator does **not** assume that IDs are sequential. IDs can start at any value and can contain gaps. All topology operations are based on the actual CSV IDs.

The following subtype definitions are used:

| SubType | Meaning      |
| ------: | ------------ |
|       0 | Organ        |
|       1 | Aorta        |
|       2 | Large artery |
|       3 | Small artery |
|       4 | Arteriole    |
|       5 | Capillaries  |
|       6 | Venules      |
|       7 | Small veins  |
|       8 | Large veins  |
|       9 | Venae cava   |
|      10 | Left heart   |
|      11 | Right heart  |

The actual left-heart and right-heart object IDs are discovered automatically from `SubType=10` and `SubType=11`.

The simulator therefore does not assume that the left heart has ID 10 or that the right heart has ID 11. Those numbers identify subtypes, not object IDs.

## Automatic topology construction

No vessel-to-vessel connections are hard-coded.

For two topology objects A and B, B is considered a downstream candidate of A when:

```text
distance(A.Stop, B.Start) < topology_tolerance
```

where the default topology tolerance is:

```text
1e-6 cm
```

This tolerance avoids requiring exact floating-point equality.

The topology therefore supports:

* one downstream object
* multiple downstream objects
* branches
* loops
* cycles
* dead ends
* non-sequential object IDs
* future changes in the number of topology objects

No topology ID such as `14`, `17`, or `65` is hard-coded into the simulator.

## Organ model

An organ is **not** modeled as a 4-cm vessel.

The geometric distance between the organ's CSV `Start` and `Stop` coordinates is retained as diagnostic geometry but is not used to calculate organ residence time.

Instead, each organ is represented by an effective three-section microvascular model:

```text
arteriole → capillary → venule
```

The model parameters are:

| Section   | Diameter | Length | Mean velocity |
| --------- | -------: | -----: | ------------: |
| Arteriole | 0.002 cm | 0.3 cm |      0.3 cm/s |
| Capillary | 0.001 cm | 0.1 cm |     0.04 cm/s |
| Venule    | 0.004 cm | 0.3 cm |     0.03 cm/s |

Each section uses the same Poiseuille formulation as a normal vessel.

A new area-uniform radial position is independently sampled for each section.

## Example circulation

Suppose:

```text
Left heart ID  = 57
Right heart ID = 83
```

and a circulation realization follows:

```text
57 → 1 → 3 → 20 → 15 → 31 → 45 → 83
```

where object `15` is an organ.

The completed cycle contains:

```text
57
1
3
20
15
31
45
```

The destination heart `83` is detected but not entered.

The primary expanded traversal units are:

```text
57
1
3
20
15:arteriole
15:capillary
15:venule
31
45
```

Therefore:

```text
Number_of_Passed_Objects = 9
```

and:

```text
Residence_Times_s =
[
    T57,
    T1,
    T3,
    T20,
    T15_arteriole,
    T15_capillary,
    T15_venule,
    T31,
    T45
]
```

The total circulation time is:

```text
T57
+ T1
+ T3
+ T20
+ T15_arteriole
+ T15_capillary
+ T15_venule
+ T31
+ T45
```

The residence time of heart `83` is not included.

## Primary output

The main output file is:

```text
PcsvNanoSec.csv
```

The primary output contains one row for each generated circulation realization.

Important fields include:

| Field                           | Description                                                                                                         |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `Nanobot_ID`                    | Unique circulation realization ID                                                                                   |
| `Cycle_Number`                  | Sequence number of the generated cycle                                                                              |
| `Organ_ID`                      | Organ visited by a valid cycle; multiple encountered organs are preserved as JSON for invalid multiple-organ cycles |
| `Starting_Heart_ID`             | Actual CSV object ID of the starting heart                                                                          |
| `Expected_Destination_Heart_ID` | Opposite heart expected to terminate the current cycle                                                              |
| `Destination_Heart_ID`          | Populated when the expected destination heart is actually selected                                                  |
| `Total_Circulation_Time_s`      | Sum of all recorded residence times, including the starting heart and excluding the destination heart               |
| `Number_of_Passed_Objects`      | Expanded modeled-object count; an organ contributes three units                                                     |
| `Residence_Times_s`             | JSON array containing the residence times in traversal order                                                        |
| `Cycle_Status`                  | `COMPLETED`, `INVALID`, or `ABORTED`                                                                                |
| `Termination_Reason`            | Explicit completion, invalidity, or abortion reason                                                                 |
| `Passed_Object_IDs`             | Actual CSV-object path                                                                                              |
| `Expanded_Passed_Object_Units`  | One-to-one correspondence with `Residence_Times_s`                                                                  |

The residence-time sequence is stored as JSON inside the CSV cell so that trajectories of different lengths can be represented without creating dynamically numbered columns.

## 3-D topology visualization

The simulator provides an optional topology visualization.

Generate it with:

```bash
python PyBloodVoyagerS.py \
    --input Pvasculature.csv \
    --num-cycles 1000 \
    --seed 12345 \
    --plot-topology
```

The visualization uses the three-dimensional start and stop coordinates from the topology file and distinguishes different vascular subtypes.

The visualization is optional and is not required for Monte Carlo simulation.

## Legacy model exclusions

PyBloodVoyagerS does **not** implement the legacy artificial Blood-Voyager 21-stream model.

In particular, it does not use:

* 21 artificial velocity streams
* 100%, 95%, 90%, etc. stream-velocity classes
* artificial stream offsets
* random stream switching inside organs
* hard-coded vessel-ID routing
* hard-coded artery or vein velocities
* the legacy 4-cm organ residence-time calculation
* fixed 1-s movement time steps
* legacy 2-D movement special cases

Instead, the current simulator uses continuous laminar-flow calculations, topology-derived routing, explicit 3-D geometry, and an effective three-section organ model.

## Reproducible example

To reproduce the main dataset:

```bash
python PyBloodVoyagerS.py \
    --input Pvasculature.csv \
    --num-cycles 100000 \
    --seed 12345
```

To reproduce a diagnostic run:

```bash
python PyBloodVoyagerS.py \
    --input Pvasculature.csv \
    --num-cycles 1000 \
    --seed 12345 \
    --verbose
```

To reproduce the topology visualization:

```bash
python PyBloodVoyagerS.py \
    --input Pvasculature.csv \
    --num-cycles 1000 \
    --seed 12345 \
    --plot-topology
```
