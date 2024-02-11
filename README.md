# PF-DECOMP

This is a small sample showcase applying (basic) ADMM to a LOPF + transmission expansion problem (Benders did not work as well).

Data is loosely based on the JAO static grid model for Germany, with a lot of dummy/synthetic data.
Default: 1500 grid assets (either: "thermal", "storage", "res"), with 839 lines and 452 nodes (each with a unique demand profile).
Lines are expandable with random "new capacity" limits.
p.u. conversion may be wrong, but since it's synthetic anyways I did not check more thoroughly.

It applies a temporal and spatial decomposition:
1. PF: A LOPF model for each timestep, using net positions for each node.
2. DP: A dispatch model for each node, across all timesteps.

ADMM consensus then matches:
1. The grid expansion across all "PF" models.
2. The net positions of each node's DP with the respective net positions of that node from all "PF" models.

This results in `452 + T` models (with `T` the number of timesteps), which are fairly fast to solve (note: analytical solutions would speed that up immensely).
Assuming full parallelization (which may be unrealistic!), the max. time spent on any of those models is assumed as time of the overall iteration (which, again, may be overly optimistic).
So, just consider this a super simplistic proof of concept.

> Note: Delegating the grid expansion decision to a higher-level Benders may be beneficial ...

## How to run
```julia
(pf-decomp) pkg> activate .
(pf-decomp) pkg> instantiate
```

Then, either directly call the ADMM using
```bash
user@pc:~/pf-decomp$ julia --project=. src/admm.jl 8
```
which runs the model for `8` timesteps (change that!), resulting in a nice print of each iteration, or 

run the automated scaling test using
```bash
user@pc:~/pf-decomp$ julia --project=. test.jl
```
which creates `results.md`.

## Results
Running this on an
- Intel(R) Core(TM) i7-8700K CPU @ 3.70GHz (unfortunately in thermal throttling),
- with 2x8 GB DDR4 RAM @ 2133 MT/s, under
- Julia 1.10.0 (2023-12-25), and
- Ubuntu 23.10, using
- an academic license of Gurobi `v10.0.1rc0 (linux64)` (interfaced by `Gurobi.jl` `v1.2.1`)

results in

|T|delta_rel|time_parallel|time_original|
|:---|:---|:---|:---|
|12.0|0.012|1.606|0.436|
|24.0|0.013|2.103|1.488|
|48.0|0.003|2.423|6.946|
|72.0|0.001|2.561|3.399|
|96.0|0.004|2.85|5.791|
|120.0|0.007|3.283|8.131|
|144.0|0.008|3.524|11.606|
|168.0|0.009|4.65|13.392|
|336.0|0.01|6.008|23.291|

with
- `delta_rel`: relative absolute gap, after 50 iterations, to the true optimal objective
- `time_parallel`: total time spent, after 50 iterations, assuming theoretical (unrealistic & ideal) full parallelization
- `time_original`: total time that the monolithic solve took

`delta_rel` shows that 50 iterations seem to be enough to achieve a good rel. gap (which however may not say anything about primal variable convergence).
