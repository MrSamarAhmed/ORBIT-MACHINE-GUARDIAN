# ORBIT Machine Guardian - Demo Plan & Pitch Outline

AMD Developer Hackathon ACT III - Intelligent Industry track.
All interventions are simulated. No real equipment is ever controlled.

## Launch

```bash
.venv/bin/streamlit run orbit/demo/app.py        # demo UI on http://localhost:8501
.venv/bin/uvicorn orbit.api.app:app --reload     # optional REST API on :8000
```

The demo calls `orbit.api.service` in-process, so the API server is **not**
required for the demonstration. No internet access, external calls, or
credentials are needed at runtime.

## Five-minute run of show

| Time | Beat | Preset in the sidebar | What the judge must see |
|------|------|----------------------|-------------------------|
| 0:00-0:30 | Problem: unplanned downtime. Detection alone is not enough - teams act, then *assume* the fix worked. | - | Title + caption; the loop OBSERVE -> DIAGNOSE -> PLAN -> ACT -> VERIFY -> RESOLVE/ESCALATE. |
| 0:30-1:15 | Healthy machine: no unnecessary intervention. | `1. Healthy machine` | Outcome `HEALTHY_NO_ACTION`, no incident opened, no action taken. |
| 1:15-2:30 | Real fault: detection evidence, ranked hypotheses, supporting vs contradicting evidence, selected action + preconditions. | `2. Successful bearing recovery` | DETECT z-score and breached features; DIAGNOSE primary hypothesis; PLAN action with preconditions met; ACT (simulated) effectiveness and state delta. |
| 2:30-3:15 | **The differentiator**: independent verification decides. | `2. Successful bearing recovery` | Timeline row `INDEPENDENT VERIFICATION` reached, criteria table passed, then and only then `RESOLVED`. |
| 3:15-4:15 | **Failed recovery** (essential beat). | `3. Failed recovery` | The action *executes* (defective spare), the fault persists, `envelope_bpfo_normal` still fails verification, incident is `ESCALATED` with the failing criterion quoted. Point at the callout: `ACTION EXECUTED != INCIDENT RESOLVED`. |
| 4:15-4:45 | Conflicting evidence and insufficient evidence - no invented certainty. | `4. Conflicting evidence`, then `6. Insufficient evidence` | Neural classifier disagrees with the deterministic diagnosis; disagreement recorded, verifier still governs. Ambiguous case escalates after 3 evidence rounds instead of guessing a repair. |
| 4:45-5:00 | Business value, measured results, limitations. | - | "Measured results" panel (recovery / escalation / latency), device panel, and the stated limitations. |

Backup beat if time allows: `5. Persistent overheating` - cooling repair executes
but `temp_below_limit` stays failed, so ORBIT escalates.

### Recovery Evidence Ledger

Every incident also exposes a **Recovery Evidence Ledger** (expander below the
verification table): incident id and opening time (UTC, recorded by the
controller when the incident opens), scenario, observed anomalies, diagnosis,
planned action and preconditions, whether the action was attempted vs actually
executed (with the executor's effectiveness reading), every verification
criterion with its expected condition, observed value, and PASS/FAIL decision,
the final status, and the recorded reason for resolution or escalation. Fields
the system did not record are labeled **"unavailable"** - nothing is filled in
after the fact, and an executed action is never presented as a resolution.

## Reproducing every scenario

UI: pick the preset in the sidebar, press **Run recovery cycle**. Each preset
prints its expected outcome and the app then reports whether the live run
matched it (a mismatch is shown as an error, never silently corrected).

Equivalent API calls (server on :8000):

```bash
# 1 healthy
curl -s localhost:8000/cycle -H 'content-type: application/json' -d '{"scenario":"normal","use_classifier":false}'
# 2 verified recovery
curl -s localhost:8000/cycle -H 'content-type: application/json' -d '{"scenario":"bearing","use_classifier":false}'
# 3 failed recovery -> escalation
curl -s localhost:8000/cycle -H 'content-type: application/json' -d '{"scenario":"bearing","environment":{"spare_bearing_quality":0.0},"use_classifier":false}'
# 4 conflicting evidence (classifier on)
curl -s localhost:8000/cycle -H 'content-type: application/json' -d '{"scenario":"overtemperature","use_classifier":true}'
# 5 persistent overheating -> escalation
curl -s localhost:8000/cycle -H 'content-type: application/json' -d '{"scenario":"overtemperature","environment":{"cooling_service_quality":0.0},"use_classifier":false}'
# 6 insufficient evidence -> escalation
curl -s localhost:8000/cycle -H 'content-type: application/json' -d '{"scenario":"bearing","machine_state":{"bearing_severity":0.15,"imbalance_severity":0.15},"use_classifier":false}'
```

Presets and their expected outcomes live in `orbit/demo/scenarios.py` and are
re-verified by `tests/test_demo_scenarios.py` (deterministic, no credentials,
no AMD hardware required). The Recovery Evidence Ledger is built by
`build_incident_ledger` in the same module and covered by `tests/test_ledger.py`.

## Pitch outline

- **Target user & problem**: plant reliability / maintenance engineers. Unplanned
  downtime is expensive, and "we replaced the part" is routinely treated as
  "the machine is fixed" without independent confirmation.
- **Differentiation**: ORBIT closes the loop. An executed action never marks an
  incident resolved - a deterministic verifier re-samples the machine and only
  then allows `RESOLVED`; otherwise it escalates with the failing criterion.
  Every incident carries a full, replayable decision trace.
- **Architecture**: signal features -> robust z-score baseline detector ->
  deterministic hypothesis engine (supporting *and* contradicting evidence,
  confidence margin) -> precondition-checked action plan -> simulated execution
  -> independent verification -> lifecycle state machine with audit trail.
  Optional PyTorch classifier is corroborating evidence only, never authoritative.
- **Demonstrated results** (synthetic simulator, shown live in the UI):
  recovery success rate and correct escalation rate per designed bucket, plus
  mean end-to-end latency - read from `runtime/evaluation/results.json`.
  Denominators are the designed scenario buckets, not an industrial fleet.
- **AMD integration**: device resolution is fail-closed ROCm -> CUDA -> MPS -> CPU
  and the classifier/benchmark run on whatever is genuinely present. On this
  machine only Apple MPS was measured. **No ROCm or AMD GPU numbers exist yet**;
  the integration point is the classifier device layer plus the benchmark
  harness, ready to record real ROCm measurements when hardware access is
  available. Token Factory access was not available, so no external AMD service
  is integrated and none is faked.
- **Business value**: fewer false "fixed" closures, auditable decisions for
  compliance, and a maintenance record that distinguishes executed from verified.
- **Limitations & next validation**: simulator-based evidence, not a real plant;
  the optional CWRU module is a synthetic -> real *transfer* measurement and has
  not been run on real CWRU data yet; no AMD GPU validation; imbalance and
  thermal limits are not exercised by CWRU.
