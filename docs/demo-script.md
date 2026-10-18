# ORBIT demo script (3 minutes)

Every step below runs from the deterministic simulator — the demo never
depends on the live network. The AMD advisory is the only optional live
element and its failure path is itself part of the story.

Launch (from the repository root, after the Quickstart steps):

```bash
.venv/bin/streamlit run orbit/demo/app.py
```

## Scene 1 — The problem (0:00–0:30)

**Say:** "Detecting a fault is not enough. Most pipelines stop at a
recommendation — or worse, they execute an action and assume it worked.
In industry, an intervention that silently fails is an outage waiting to
happen. ORBIT closes the loop: it refuses to call an incident resolved
until independent verification proves the machine recovered."

**Show:** the header tagline `OBSERVE → DIAGNOSE → PLAN → ACT → VERIFY →
RESOLVE OR ESCALATE` and the caption *"Executing an action never implies
resolution — an independent verifier decides."*

## Scene 2 — Healthy machine, no drama (0:30–1:00)

**Do:** sidebar → scenario preset **"1. Healthy machine - no unnecessary
action"** → **Run recovery cycle**.

**Expected on screen:** Outcome `HEALTHY_NO_ACTION`, "No incident was
opened — ORBIT took no action and nothing needed verification."

**Say:** "A reliability agent that invents work is a liability. A healthy
machine produces zero incidents and zero interventions."

## Scene 3 — Verified recovery (1:00–1:45)

**Do:** preset **"2. Successful bearing recovery - independently
verified"** → run. Optionally enable the **AMD advisory** toggle first
(requires `RADEON_API_KEY` in `.env`).

**Expected:** anomaly detected (max |z| and breach count in section 01),
primary diagnosis *bearing outer-race defect* with ranked evidence, plan
*Replace bearing* with preconditions checked, simulated execution, the
verifier table showing **every criterion PASS on every window**, verdict
banner **VERIFICATION PASSED**, final status chip **RESOLVED**.

If the AMD toggle is on and the endpoint answers: an info panel shows the
model's recommendation, explicitly labeled *"a recommendation, not the
executed action and not a verified recovery."* If it times out: a caption
states the advisory was unavailable and *"the deterministic pipeline
governed — an API failure is never interpreted as a diagnosis or
recovery."* Either way the outcome is identical, which is the point.

**Say:** "The plan, the execution and the resolution are three different
events, and ORBIT proves each one separately."

## Scene 4 — Failed recovery, honest escalation (1:45–2:20)

**Do:** preset **"3. Failed recovery - action runs, verification fails,
escalation"** → run.

**Expected:** action executed with effectiveness ~0 (state unchanged),
verifier table with **FAIL** rows listed first, red **FAILED RECOVERY**
banner, final status **ESCALATED** with the recorded reason, and the
metric **Recovery: NOT RECOVERED**.

**Say:** "The spare was defective. The action executed perfectly — and
ORBIT still refuses to claim success. It escalates to a human with the
full evidence trail. This is the behavior that prevents the second
outage."

## Scene 5 — The proof (2:20–2:45)

**Do:** open tab **04 · Evaluation & benchmarks** → *"Why verification
matters — measured on this machine."*

**Expected:** the measured table: same scenarios, two workflow arms —
ORBIT 0 incorrect resolutions out of 1 verified resolution claim,
3/3 correct escalations; naive execution-trust baseline **3 false
resolutions out of 4 claims (75% false-resolution rate)**.

**Say:** "Same detector, same actions, same seeds — the only difference
is verification policy. The naive arm declares success three times on
machines that are still broken. ORBIT never does. Reproducible with
`python -m orbit.evaluation.comparison`."

## Scene 6 — AMD technology + business value (2:45–3:00)

**Show:** same tab → the AMD disclosure (deterministic core is
NumPy/SciPy; classifier corroboration on Apple MPS; AMD contribution is
the hosted **Token Factory DeepSeek-V4-Flash** model, live-verified
2026-10-18 per `docs/live-amd-verification.md` — one real request,
schema-validated, advisory only) → the **business value projection**
expander (formulas and adjustable assumptions; the failed-recovery
fraction defaults to the measured 0.75).

**Say:** "AMD's hosted model strengthens the evidence analysis; the
deterministic controller keeps authority over actions and resolution.
For a plant manager: fewer silent failures means fewer repeat outages —
projected from your own assumptions, never presented as measured
savings."

## Honesty rules for the presenter

- Everything on screen is simulated; the banner says so. Never call it a
  real machine.
- If asked about AMD GPUs: ORBIT calls an AMD-hosted model over an API.
  No AMD GPU or ROCm execution was used; the README states this.
- If the AMD toggle times out live: say so, and point at the recorded
  fallback — the degradation is itself a feature.
- All numbers quoted in this script are produced by commands in the
  repository; re-run them rather than trusting this document.
