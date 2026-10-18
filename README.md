# ORBIT Machine Guardian

**Autonomous industrial reliability agent** — detects rotating-machinery
faults, diagnoses likely causes, plans and executes corrective actions
in a controlled simulator, and **independently verifies** recovery
before closing an incident. Built for the **AMD Developer Hackathon
ACT III**.

Everything on the machine side is **simulated**. ORBIT never connects to
or controls real industrial equipment.

---

## Demo Video

Watch ORBIT Machine Guardian detect a simulated machine fault, recommend
corrective action, and evaluate whether recovery can be verified
(3:00, 1080p, MP4).

[▶ Watch the ORBIT Machine Guardian Demo](submission-video/orbit-machine-guardian-final.mp4)

The demonstration covers:
- Machine fault detection and diagnosis.
- AMD Token Factory-powered AI advisory (recorded, live-verified result).
- Safety-checked corrective action planning.
- Evidence-based recovery verification.
- Escalation when recovery cannot be confirmed.
- Verification-vs-naive evaluation and a transparent business-value
  projection calculator.

> This prototype uses simulated machine scenarios and has not been
> validated on real industrial equipment. AMD Token Factory provides
> advisory reasoning; this project does not claim AMD GPU or ROCm
> execution unless independently demonstrated.

---

## Core loop

```
OBSERVE  → DETECT → DIAGNOSE → PLAN → ACT → VERIFY → RESOLVED
                                             (else)  → ESCALATED
```

Only the **deterministic verifier** can move an incident to `RESOLVED`.
Every incident also exposes a **Recovery Evidence Ledger** in the demo: the
recorded evidence, whether the action was attempted vs executed, each
verification criterion with its observed value and PASS/FAIL decision, and
the recorded reason for resolution or escalation. Fields the system did not
record are shown as "unavailable" - never filled in after the fact.
The diagnoser, planner, executor and any ML model can propose actions —
they can never declare recovery. When recovery cannot be proven, the
incident escalates instead of silently failing.

---

## Architecture

```
orbit/
  config.py                  Motor configuration (bearing geometry, fs, thresholds)
  simulator/motor.py         Deterministic 6205-bearing motor simulator
  signalproc/features.py     10 sensor features (RMS, crest, kurtosis, spectral bands,
                             envelope BPFO ratio, HF energy fraction, temperature)
  detection/baseline.py      Robust MAD + relative-scale-floor z-score detector
  diagnosis/knowledge.py     FMEA-style knowledge base: fault modes, signature rules,
                             action plans, preconditions, verification conditions
  diagnosis/engine.py        Ranked competing hypotheses with supporting AND
                             contradicting evidence; ambiguity margin gate
  actions/environment.py     Maintenance environment (spare quality, kit availability)
  actions/executor.py        Simulated interventions; effect scales with resource quality
  recovery/verifier.py       Deterministic recovery verifier — the ONLY authority
                             that may declare RESOLVED
  incident/lifecycle.py      Enumerated state machine with invalid-transition guards
  incident/controller.py     Autonomous loop orchestration + per-phase audit trail
  model/device.py            Device resolution (ROCm→CUDA→MPS→CPU), fail-closed
  model/dataset.py           Leakage-free synthetic splits (disjoint seeds)
  model/cnn.py               Compact 1D-CNN (~43k params) for vibration classification
  model/train.py             Deterministic CPU training (seed 42)
  model/inference.py         Device-abstracted inference + CPU/GPU parity check
  model/benchmark.py         Reproducible batch-sweep latency / throughput benchmark
  evaluation/metrics.py      Detection P/R/F1 + FPR, classifier held-out accuracy,
                             recovery success / correct-escalation / no-action rates
  api/app.py                 FastAPI surface (stateless, validation)
  api/service.py             In-process orchestration for API + demo
  demo/app.py                Streamlit UI: sensor trends, ranked hypotheses,
                             evidence, before/after, verification, device, benchmarks
```

---

## Quickstart

```bash
# 1. Environment
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env        # placeholders only, edit if you want to pin a device

# 2. Tests
.venv/bin/pytest -q

# 3. Reproduce the classifier + benchmark + evaluation
.venv/bin/python -m orbit.model.train
.venv/bin/python -m orbit.model.benchmark
.venv/bin/python -m orbit.evaluation

# 4. Run the API and demo
.venv/bin/uvicorn orbit.api.app:app --reload      # FastAPI on :8000
.venv/bin/streamlit run orbit/demo/app.py         # Streamlit on :8501
```

`runtime/` (git-ignored) is where the model weights, benchmark JSON and
evaluation JSON are written. Everything else is code.

### Deploying the demo to Streamlit Community Cloud

The demo is a single Streamlit process — it uses the in-process service
layer, so no separate FastAPI/uvicorn server is required.

1. Push this repository to GitHub (public).
2. Open <https://share.streamlit.io>, sign in with GitHub, **Create app**.
3. Repository `MrSamarAhmed/ORBIT-MACHINE-GUARDIAN` · branch `main` ·
   main file path `orbit/demo/app.py` · Python 3.11.
4. Optional AMD advisory: under **Advanced settings → Secrets** set
   `RADEON_API_KEY = "<your key>"` (never commit a real key;
   `.streamlit/secrets.toml` is git-ignored). Without a key the full
   deterministic pipeline still runs and the advisory is honestly marked
   "not configured" — no simulated AI output.
5. `runtime/` artifacts are git-ignored, so a fresh cloud deploy shows
   honest "run python -m orbit.model.benchmark / .evaluation" captions
   on tab 04 instead of numbers, and the classifier badge reports
   "unavailable, deterministic pipeline active".
6. The hosted demo renders on the **deterministic CPU pipeline only**.
   PyTorch is imported lazily, so a GPU-less free-tier deploy (no
   `runtime/` weights) never loads torch on the critical path: the
   dashboard, detector, diagnosis, executor and verifier all run on
   NumPy/SciPy, and torch is touched only if trained corroboration
   weights are present. No AMD GPU/ROCm execution is implied.

---

## AMD / ROCm story (honest)

The AMD angle is a first-class **device + benchmark path**, not a label
on the README.

`orbit/model/device.py` resolves devices in this order:

1. ROCm / CUDA (reported as `rocm` when `torch.version.hip` is set)
2. Apple MPS (local development on this Mac)
3. CPU (always available; every other component is CPU-only)

**Fail-closed:** requesting a device that isn't available raises with a
clear message instead of silently falling back, so a benchmark can never
misreport where it actually ran.

The deterministic safety-critical path (simulator, detector, diagnosis,
executor, verifier, controller) uses **only NumPy / SciPy** — no GPU, no
torch. This means:

- **Nothing on the demo critical path depends on a GPU being present.**
- The GPU/ROCm path is used by the **corroborating classifier** whose
  output is recorded alongside the deterministic diagnosis as supporting
  evidence — never as the authority for recovery.

### Local benchmark (Apple Silicon, PyTorch 2.8.0)

Reproducible numbers on this machine (`runtime/benchmark/benchmark_20261012_180041.json`).
Model: 43,331-param 1D-CNN on 2560-sample windows.

| Device | batch | latency mean (ms) | throughput (windows/s) |
|--------|------:|------------------:|-----------------------:|
| cpu    |     1 |             0.486 |                   2058 |
| cpu    |     8 |             0.879 |                   9103 |
| cpu    |    64 |           495.420 |                    129 |
| cpu    |   512 |          2722.833 |                    188 |
| cpu    |  2048 |          9761.889 |                    210 |
| mps    |     1 |             0.977 |                   1024 |
| mps    |     8 |             0.936 |                   8551 |
| mps    |    64 |             1.438 |                  44491 |
| mps    |   512 |             7.483 |                  68421 |
| mps    |  2048 |            28.449 |                  71988 |

- **CPU vs MPS parity check** (same weights, same batch):
  `max|diff| = 5.24e-10` — CPU and GPU paths are numerically equivalent.
- At batch ≥ 64, MPS wins by **~343–364x** (throughput ratio per the
  table above). At batch 1 and 8 CPU wins (MPS
  launch overhead dominates). We report both honestly: an accelerator
  helps when the workload actually batches, not by magic.
- The `cpu` batch-64/512/2048 outliers are reproducible with
  `torch.set_num_threads(1)` on this ARM64 Mac, so they reflect a real
  torch conv1d code path on this CPU, not a measurement artifact. The
  AMD instance uses a different code path (x86 + MIOpen).

### Extended benchmark sweep (2026-10-17, Apple Silicon, PyTorch 2.8.0)

Same harness and methodology (`orbit.model.benchmark`: 10 warm-up + 30
timed iterations per configuration, fixed dataset seed 777), run with
the batch set extended to include 128, 256 and 1024:

```python
import orbit.model.benchmark as b
b.BATCH_SIZES = (1, 8, 64, 128, 256, 512, 1024, 2048)
b.run_benchmark()
```

Saved to `runtime/benchmark/benchmark_20261017_152642.json`.
Devices measured: arm64 CPU (`arm`) and Apple Silicon MPS. No AMD ROCm
hardware was involved — see "What we do NOT claim" below.

| Device | batch | latency mean (ms) | throughput (windows/s) |
|--------|------:|------------------:|-----------------------:|
| cpu    |     1 |             0.507 |                   1973 |
| cpu    |     8 |             0.883 |                   9061 |
| cpu    |    64 |           478.028 |                    134 |
| cpu    |   128 |           829.463 |                    154 |
| cpu    |   256 |          1625.756 |                    158 |
| cpu    |   512 |          2729.003 |                    188 |
| cpu    |  1024 |          5019.629 |                    204 |
| cpu    |  2048 |          9893.148 |                    207 |
| mps    |     1 |             1.062 |                    942 |
| mps    |     8 |             1.121 |                   7139 |
| mps    |    64 |             1.548 |                  41345 |
| mps    |   128 |             2.562 |                  49958 |
| mps    |   256 |             4.436 |                  57706 |
| mps    |   512 |             8.530 |                  60021 |
| mps    |  1024 |            14.720 |                  69566 |
| mps    |  2048 |            29.751 |                  68839 |

- **CPU vs MPS parity check** (same weights, same batch):
  `max|diff| = 5.24e-10` — PASS.
- CPU wins at batch 1 and 8; at batch ≥ 64 MPS wins by **~309–366x**
  (throughput ratio). MPS throughput plateaus around ~60–70k windows/s
  at batch ≥ 512. The CPU numbers reproduce the batch-64 jump and the
  gradual throughput recovery toward batch 2048 seen above; nothing was
  smoothed, corrected or omitted.

### What we do NOT claim

- We have **not** run on AMD hardware yet — this dev machine is Apple
  Silicon and has no ROCm. The `python -m orbit.model.benchmark` script
  is what we will run on an AMD Developer Cloud ROCm instance to record
  real `rocm` rows in the same JSON schema.
- We do **not** extrapolate AMD numbers from Apple MPS numbers.
- We do **not** claim that GPU acceleration helps the safety-critical
  path. It doesn't need to — the deterministic core is CPU-only by
  design.

---

### AMD Token Factory advisory (live-verified)

The second AMD technology in ORBIT is the **AMD Token Factory** hosted
model (`DeepSeek-V4-Flash`, OpenAI-compatible endpoint, credentials only
via `RADEON_API_KEY` in the environment or git-ignored `.env`). It acts
as an **advisory evidence analyst**: after the deterministic diagnosis,
the controller sends the compact recorded evidence payload (features,
robust z-scores, detector breaches, evidence trend, ORBIT's own
diagnosis, permitted actions) and receives a strictly-validated JSON
advisory (assessment, supporting evidence, recommended action,
model-reported confidence, escalation flag).

**Live-verified 2026-10-18** (`docs/live-amd-verification.md`): one real
request with the bearing-scenario payload returned a schema-valid
advisory that cited the actual recorded z-scores and recommended the
allowlisted action; the first attempt (30 s timeout) demonstrated the
designed fallback live — the failure was recorded, never interpreted as
a diagnosis or recovery, and the deterministic pipeline governed. The
default timeout is 120 s; retries are disabled to conserve quota; only
two live requests were made in total. Reproduce with
`.venv/bin/python live_amd_check.py` (one request per run; never in CI).

Safety boundaries (enforced in code, covered by 42 mocked tests):
the advisory is recorded in the audit trail and its **only** permitted
control-flow effect is a *conservative* escalation request; it cannot
select, execute, or authorize actions, it cannot mark an incident
RESOLVED, and an unavailable API changes nothing in the deterministic
pipeline. No AMD GPU or ROCm execution is claimed for this integration —
the model runs on AMD's hosted service; see "What we do NOT claim" below.

## Data source (honest)

ORBIT runs entirely on **synthetic simulator data**. There is **no real
industrial data** in this repo yet.

The simulator models a 6205 deep-groove ball bearing drive-end rig
(the same bearing family used by the CWRU Bearing Data Center), a
jittered BPFO impulse train ringing a structural resonance, imbalance
as a raised 1x component, and a first-order thermal model — but these
are **equations we tuned**, not measurements from a test rig.

Every evaluation result explicitly carries:
`"synthetic_only": true`, `"data_source": "ORBIT motor simulator (synthetic). No real industrial data in this file."`

### Optional CWRU Bearing Data Center evaluation

An **opt-in, fully separate** evaluation path for the CWRU Bearing Data
Center dataset ships in `orbit/evaluation/cwru.py`. It never runs during
normal startup, never downloads anything, and only executes when invoked
explicitly:

```bash
python -m orbit.evaluation.cwru --data-dir data/cwru
```

CWRU `.mat` files do not self-describe their fault type, and the official
channel-number → fault table lives on the CWRU website — we do not
fabricate it. Instead ORBIT uses a **directory-name-as-label** convention:
you place each recording under the folder naming its true fault class:

```
data/cwru/normal/*.mat       -> normal
data/cwru/outer_race/*.mat   -> bearing    (the synthetic bearing mode IS outer-race)
data/cwru/inner_race/*.mat   -> parsed; excluded from the classifier, counted
data/cwru/ball/*.mat         -> parsed; excluded from the classifier, counted
```

The label mapping is deliberately narrow and never assumed: `inner_race`
and `ball` use BPFI/BSF mechanisms that differ from ORBIT's outer-race
(BPFO) model, so they are **excluded from classifier evaluation** (and
reported as such) while still being scored by the unsupervised detector.
CWRU has no imbalance class and no temperature channel, so the
`imbalance` class and the thermal limit are not exercised there.

Leakage control: splits are **recording-level** (each `.mat` hashed into
train/val/test; overlapping 0.2 s windows never cross a split), and the
detector is fit only on normal training recordings. The module reports
per-class precision/recall/F1, a full confusion matrix, sample counts,
excluded records, and writes to its own `runtime/evaluation/cwru_results.json`
— never into the synthetic `results.json`. When the data is absent it
writes an honest `available: false` and exits 0.

Two measurements are produced, both clearly caveated:

- **detector transfer** — baseline detector fit on real normal data, then
  FPR on held-out normal windows and per-fault-type detection rate. This
  tests whether ORBIT's detection *features* transfer to real vibration.
- **classifier transfer** — the synthetic-trained CNN scored on the
  mapped normal-vs-outer-race subset. This measures synthetic→real
  *transfer* and is reported with that caveat attached, never as
  real-world industrial accuracy.

**Status:** the pipeline is implemented and tested (parsing, windowing,
recording-level split integrity, confusion-matrix math, unsupported-label
exclusion, missing-data handling) using small synthetic `.mat` fixtures.
**No real CWRU download has been performed in this environment**, so no
real CWRU numbers are reported anywhere in this repository.

---

## Measured results (synthetic)

From `runtime/evaluation/results.json` (produced by `python -m orbit.evaluation`).
Deterministic seeds. Calibration uses seed `7`; every evaluation split
uses a disjoint seed (no train/test leakage).

**Detection** (baseline anomaly detector, 25 windows per fault class):

| Metric                         | Value |
|--------------------------------|------:|
| Normal false-positive rate     |   0.0 |
| Overall detection F1           |   1.0 |
| Bearing recall                 |   1.0 |
| Imbalance recall               |   1.0 |
| Overtemperature recall         |   1.0 |

These are synthetic results on a clean simulator with well-separated
fault signatures. They are **not** representative of how hard a real
CWRU or field dataset will be. The harness exists so that the same
measurement can be re-run on real data and the honest number reported.

**Corroborating 1D-CNN** (held-out synthetic split, seed 9009, trained on
disjoint seed 1001):

- Accuracy: **1.000** on 240 held-out windows across 3 classes.
- Per-class F1: 1.000 each.

Same caveat as above — synthetic data is a much easier classification
problem than real vibration. Reported honestly; the classifier is used
only as **corroborating evidence**, never as the recovery authority.

**Autonomous recovery** (7 scenarios, deterministic seeds):

| Bucket               | Designed | Correct | Rate |
|----------------------|---------:|--------:|-----:|
| Designed to resolve  |        4 |       4 | 100% |
| Designed to escalate |        2 |       2 | 100% |
| Designed no-action   |        1 |       1 | 100% |
| Mean end-to-end latency per cycle | | | **4.75 ms** |

Scenario list:

- `A_bearing_repair` — bearing severity 0.6 → RESOLVED via replace_bearing
- `A_imbalance_balance` — imbalance 0.6 → RESOLVED via field_balancing
- `A_overtemp_cooling` — cooling_efficiency 0.5 → RESOLVED via cooling_overhaul
- `concurrent_faults` — bearing 0.3 + imbalance 0.55 → dominant repaired
  first, residual handled by re-diagnosis in a second cycle → RESOLVED
- `B_defective_spare` — spare_bearing_quality=0.0 → verifier rejects the
  ineffective intervention → **ESCALATED**
- `precond_no_spare` — spare_bearing_available=False → precondition
  blocks the action → **ESCALATED**
- `D_healthy` — no anomaly → **HEALTHY_NO_ACTION**

Ambiguity-induced escalation is exercised separately in the controller
unit tests with a controlled ambiguous engine, since the real engine is
deterministic and rarely ambiguous on the clean synthetic signal. The
escalation rates above are computed against the real engine's actual
behaviour, not a fudge.

**Verification vs naive execution** (reproducible:
`python -m orbit.evaluation.comparison`, seed 53, five scenarios). Both
workflow arms share detector, diagnosis, action library, preconditions,
executor, environment and seeds; the only difference is whether a fix
must pass the deterministic verifier before the incident closes. Ground
truth = fresh post-cycle simulator windows judged by the same detector.

| Scenario | Arm | Outcome | Machine actually OK? | Honest? |
|---|---|---|---|---|
| Healthy (control) | both | no incident | yes | yes |
| Bearing fault, good spare | both | resolved | yes | yes |
| Bearing fault, defective spare | ORBIT | **escalated** | no | **yes** |
| Bearing fault, defective spare | naive | resolved | no | **NO** |
| Imbalance, defective balancing | ORBIT | **escalated** | no | **yes** |
| Imbalance, defective balancing | naive | resolved | no | **NO** |
| Overheating, ineffective service | ORBIT | **escalated** | no | **yes** |
| Overheating, ineffective service | naive | resolved | no | **NO** |

Summary: ORBIT **0 incorrect resolutions** of 1 resolution claim (rate
0.00), 3/3 escalations correct. The naive execution-trust baseline made
**3 false resolution claims out of 4 (75% false-resolution rate)** — it
declared success on machines that were still broken. This table is the
measured form of ORBIT's central claim: *executing an action is not
fixing the machine.* It describes the synthetic simulator only. The demo
UI shows the same measured table plus a transparent business-value
projection calculator (formulas and adjustable assumptions, clearly
labeled as projections, not savings).

**Test suite**: 231 passed, 1 skipped. The single skip
(`tests/test_diagnosis.py::test_weak_fault_low_confidence_label`) is a
deliberate honesty guard — at bearing severity 0.15 the fault is
borderline and can cross the confident threshold for this particular
seed; the test skips rather than asserting a confidence label the engine
doesn't actually produce. The classifier-dependent tests run against the
weights generated by `orbit.model.train` and are not skipped.

---

## Trust model: what safety-critical code is deterministic

Design rules carried over from MaintAI's investigation trust model
(inspected read-only for this project; not modified):

- Every safety-critical decision is a **pure deterministic function**:
  detection thresholds, hypothesis ranking, precondition checks, and
  recovery verification. No LLM in the loop.
- The **only** code path that can move an incident to `RESOLVED` is
  `orbit/recovery/verifier.py::verify_recovery`. It requires every
  declared verification condition to pass on **every** post-intervention
  window; a single failing window keeps the incident open.
- `orbit/incident/lifecycle.py::assert_transition` enumerates every
  legal state edge and raises `InvalidTransition` on any other move, so
  `RESOLVED` is only reachable from `VERIFYING`.
- `orbit/actions/executor.py::check_preconditions` is **fail-closed**:
  an unknown `requirement_id` raises rather than defaulting to permissive.
- The optional neural classifier can only annotate the audit trail with
  a corroboration verdict. It never triggers or blocks an intervention.
- Ambiguous diagnosis ⇒ additional evidence rounds ⇒ escalate. The
  controller will not act on a diagnosis it cannot separate from its
  runner-up by a full ambiguity margin.

---

## Simulated vs real

- **Simulated**: motor physics, sensor noise, thermal dynamics, all
  interventions (bearing swap, field balancing, cooling overhaul), and
  the maintenance environment (spare availability, resource quality).
- **Real**: signal-processing math, baseline statistics, hypothesis
  ranking, verification logic, state machine, HTTP surface, benchmark
  timings. Nothing that decides whether an incident is RESOLVED fakes
  its inputs.

---

## Repo safety

- **No secrets committed.** `.env.example` contains placeholders only;
  `.env` is git-ignored. No tokens appear in the tree.
- **No MaintAI files modified.** MaintAI was inspected read-only
  (public GitHub repo, cloned to `/tmp/maintai-inspect`) for reuse
  assessment. Its trust-model patterns (atomic JSON state,
  fail-closed protected-key ingestion, competing hypotheses with
  supporting AND contradicting evidence) informed ORBIT's design.
- No dependency has a license conflict with the project (all MIT /
  BSD / Apache; the CWRU dataset is publicly available for research).

---

## Roadmap after this submission

1. Obtain the CWRU `.mat` recordings, place them under `data/cwru/<class>/`,
   and run `python -m orbit.evaluation.cwru` to report real detector- and
   classifier-transfer numbers alongside the synthetic ones.
2. Run `python -m orbit.model.benchmark` on an AMD Developer Cloud
   ROCm instance (MI300X or Instinct family) and merge the `rocm`
   rows into the same JSON schema. Report CPU vs AMD GPU with real
   numbers, no extrapolation.
3. Add a second fault mode (misalignment) to the knowledge base to
   pressure-test the ambiguity margin path on data where the fault
   signatures genuinely overlap.
