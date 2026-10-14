# ORBIT — Architecture Decision Record

Short, dated notes on the choices that shaped ORBIT and why. Each entry
lists the decision, the alternatives considered, and the reason.

## ADR-001 — Build the safety-critical path on NumPy/SciPy, not the NN
**Date:** 2026-10-12

Detection, diagnosis, precondition checks, and recovery verification are
all deterministic numeric code. A PyTorch model is used only as
*corroborating evidence*.

- **Why:** the recovery decision must be auditable and reproducible; a
  neural net that can be confidently wrong is unsafe as the sole
  authority. Also keeps the whole pipeline runnable on CPU with no
  weights, so a missing GPU never blocks the core workflow.
- **Alternatives:** classifier-first (rejected: model is not the
  recovery authority); LLM-in-the-loop for diagnosis (rejected per brief
  — LLM may interpret/propose but never decides recovery).

## ADR-002 — Synthetic simulator as the primary data source; CWRU deferred
**Date:** 2026-10-12

ORBIT is validated on a deterministic 6205-bearing motor simulator. CWRU
Bearing Data Center integration is documented as a planned next step,
not a completed one.

- **Why:** we never want to present synthetic performance as real. With
  a simulator we control the fault physics exactly, can prove leakage-free
  splits, and can state precisely what is modelled vs measured. The CWRU
  files were not fetched in this session (no dataset was downloaded), so
  we do not claim CWRU results.
- **Alternatives:** fabricate/embellish "CWRU accuracy" (rejected —
  violates the honesty constraint outright); ship without any data path
  (rejected — no way to measure anything).

## ADR-003 — MAD + relative scale floor for detection
**Date:** 2026-10-12

Robust MAD-based z-scores, plus a per-feature relative scale floor
(`rel_scale_floor * |median|`) so ultra-tight healthy features can't fire
false alarms on micro-deviations.

- **Why:** several features (e.g. `spec_1x_amp`, `env_bpfo_ratio`) have
  tiny MAD in the healthy baseline, so a plain MAD-z-score reported
  huge z values on harmless noise and self-false-alarm'd on healthy data.
  The floor was found empirically during M2 testing.
- **Alternatives:** std-based z (less robust to the faulted tails); fixed
  absolute thresholds (fragile across operating conditions).

## ADR-004 — z-saturation + ambiguity margin in diagnosis
**Date:** 2026-10-12

Rule scores use `weight * log1p(min(|z|, 10))`, and a hypothesis is
"confident" only when it beats the runner-up by a full margin (1.0).

- **Why:** raw z across features is incomparable (one feature's healthy
  baseline is 100x tighter than another's), so blended faults stayed
  spuriously "confident". Saturation caps the influence of absurd z, and
  the margin forces the engine to admit ambiguity rather than invent
  certainty. This directly enables the "ambiguous → escalate" scenario.
- **Alternatives:** softmax posterior over modes (would hide the
  supporting/contradicting evidence the audit trail requires).

## ADR-005 — Resource quality scales intervention effectiveness
**Date:** 2026-10-12

The maintenance environment carries quality factors (spare bearing
quality, balancing quality, cooling service quality). An intervention
removes that fraction of the fault severity, so a defective spare is
*executed but ineffective*.

- **Why:** models the realistic Scenario B failure — the action ran
  successfully but the machine did not recover. This is what forces the
  deterministic verifier (not the executor) to be the recovery authority
  and to escalate a false repair.

## ADR-006 — Fail-closed device resolution
**Date:** 2026-10-12

Requesting `cuda`/`rocm`/`mps` that isn't available raises instead of
falling back to CPU.

- **Why:** a benchmark that silently ran on CPU while you asked for the
  GPU would misreport where it ran. Fail-closed keeps the "never
  fabricate GPU measurements" rule enforceable in code, not just policy.

## ADR-007 — Streamlit over the MaintAI Next.js dashboard
**Date:** 2026-10-12

- **Why:** MaintAI's dashboard is investigation-specific (software
  failures, patch/approval gates) and reusing it would mean heavy
  modification for little gain. Streamlit keeps the whole demo in the
  Python dependency set already present and lets the recovery loop be
  shown without a separate frontend build.
- **Reused from MaintAI (conceptually, not as code):** the trust model —
  atomic JSON state, protected/fail-closed ingestion, and the requirement
  that competing hypotheses carry *both* supporting and contradicting
  evidence before a decision. MaintAI source was read-only inspected and
  never modified.

## ADR-008 — Report the CPU/GPU crossover honestly
**Date:** 2026-10-12

The benchmark shows CPU beating MPS at batch 1 and MPS winning ~180–340x
at batch ≥ 64. We publish both.

- **Why:** the brief says never claim acceleration helps unless the
  benchmark shows it. Small-workload overhead is a real result; hiding it
  would overstate the accelerator's value. AMD ROCm rows are pending a
  real ROCm run and are explicitly *not* extrapolated from Apple MPS.

## ADR-009 — CWRU label mapping is narrow and directory-based; never assumed
**Date:** 2026-10-13

The optional CWRU evaluation (`orbit/evaluation/cwru.py`) maps only
`normal -> normal` and `outer_race -> bearing`. `inner_race` and `ball`
are excluded from the classifier (counted and reported), and provenance
comes from a directory-name-as-label convention rather than a fabricated
channel table.

- **Why:** CWRU `.mat` files do not self-describe fault type, and the
  official channel table could not be fetched without network access —
  inventing it would fabricate labels. The synthetic `bearing` class is an
  outer-race (BPFO) model; inner-race (BPFI) and ball (BSF) faults are a
  different mechanism, so mapping them to `bearing` would mislabel the
  evaluation. Exclusion with an explicit reason is the honest choice.
- **Alternatives:** map all CWRU faults to one `bearing` class (rejected —
  masks the mechanism mismatch); ship a built-in channel table (rejected —
  could not verify it without the network).
