# PULPO Test Suite

## Environment setup

PULPO supports two Brightway generations, and the test suite must be run
against **both** before a release. Two dedicated, permanent test venvs exist
so the main development venv (`.venv`) never needs re-tuning:

- **`.venv-bw25`** — bw25 stack (bw2data ≥ 4, bw2calc ≥ 2)
- **`.venv-bw2`** — bw2 stack (bw2data 3.6.6, bw2calc 1.8.2, numpy < 2)

Both include the **`uncertainty`** extra (SALib, stats_arrays, seaborn,
matplotlib) so the most complete dependency set — including potential cross
effects between optional and core packages — is always exercised. Each venv
needs `pytest` and an **editable** install of pulpo (`-e`), so tests always
run against the working tree instead of a stale site-packages copy.

Create / recreate them with [uv](https://docs.astral.sh/uv/) from the repo root:

```powershell
# bw25
uv venv .venv-bw25 --python 3.12
uv pip install -p .venv-bw25 -e ".[bw25,uncertainty]" pytest

# bw2 (bw2 requires Python <= 3.12)
uv venv .venv-bw2 --python 3.12
uv pip install -p .venv-bw2 -e ".[bw2,uncertainty]" pytest
```

The `bw2` and `bw25` extras are mutually exclusive (declared in
`pyproject.toml`), hence the separate venvs.

**Sanity check** — before trusting any test run, confirm which stack you are
on and that pulpo resolves to the repo:

```powershell
.\.venv-bw25\Scripts\python.exe -c "import bw2data, pulpo; print(bw2data.__version__, pulpo.__file__)"
```

(Swap in `.venv-bw2` to check the other env.) The pulpo path must point into
this repo, not into `site-packages`.

## Running the tests

No manual data preparation is needed: importing `test_functions.py` builds
the sample Brightway project (biosphere, LCIA methods, background/foreground
databases) automatically — this is why **collection alone takes ~8 seconds**
(mostly library imports; the batched DB builds are ~2 s).
The project name adapts to the stack (`sample_project` vs
`sample_project_bw25` via `pulpo.utils.utils.is_bw25()`).

Against **bw25**:

```powershell
.\.venv-bw25\Scripts\python.exe -m pytest tests -v -rA --durations=10
```

Against **bw2** (the bw25-only module `test_bw25_uncertainty.py` skips
itself automatically):

```powershell
.\.venv-bw2\Scripts\python.exe -m pytest tests -v -rA --durations=10
```

Excluding the NEOS test even if `NEOS_EMAIL` is set (stay offline / avoid
the remote solver queue):

```powershell
# bw25, without NEOS
.\.venv-bw25\Scripts\python.exe -m pytest tests -v -rA --durations=10 --deselect tests/test_functions.py::TestPULPO::test_neos_solver
```

```powershell
# bw2, without NEOS
.\.venv-bw2\Scripts\python.exe -m pytest tests -v -rA --durations=10 --deselect tests/test_functions.py::TestPULPO::test_neos_solver
```

Flag summary:

- `-v` — one line per test as it runs
- `-rA` — final report listing every test's outcome, including skip reasons
  and full tracebacks for failures
- `--durations=10` — the ten slowest tests

## Parallel execution (evaluated, not worthwhile)

`pytest-xdist` was evaluated (July 2026): no speedup, because every worker
re-pays the import-time collection cost (library imports + DB builds). The
suite was instead sped up serially — batched `Database.write()` calls in the
sample-DB builders and `n_jobs=1` in `test_monte_carlo` (a joblib pool spawn
costs far more than its 10 tiny LP solves) — to ~12 s (bw25) / ~6 s (bw2).
The suite is xdist-safe if ever needed — each worker gets its own temp
Brightway directory via `conftest.py`.

## Environment-gated tests

These skip themselves with an explanatory message when the requirement is
missing:

| Test | Requirement |
|---|---|
| `test_gurobi_solver` | `gurobipy` importable and licensed |
| `test_gams_solver` | `GAMS_PULPO` env var pointing to the GAMS installation |
| `test_neos_solver` | `NEOS_EMAIL` env var set (submits jobs to the remote NEOS server) |
| `test_bw25_uncertainty.py` (whole module) | bw25 stack (bw2data ≥ 4) |

All remaining tests use the bundled HiGHS solver and run offline.

## Files

- `test_functions.py` — main suite: parser (`TestParser`), optimization and
  solvers (`TestPULPO`), result extraction/saving (`TestSaver`)
- `test_time.py` — time-extension result extraction and saving
- `test_bw25_uncertainty.py` — bw25 uncertainty-parameter extraction; can
  also be run directly as a script (see its module docstring)
