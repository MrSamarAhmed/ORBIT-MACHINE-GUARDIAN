# Live AMD Token Factory verification record

**Date:** 2026-10-18
**Endpoint:** `https://developer.amd.com.cn/radeon/api/v1` (AMD Token
Factory, OpenAI-compatible)
**Model:** `DeepSeek-V4-Flash`
**Credential source:** `RADEON_API_KEY` from the git-ignored `.env`
(the key is never printed, logged, committed, or stored in any artifact)

## What was run

`live_amd_check.py` — one full ORBIT recovery cycle on the deterministic
bearing-fault scenario with the real `AmdAdvisor` injected. The cycle
sends **exactly one** chat-completion request (temperature 0, retries
disabled) containing the compact evidence payload ORBIT recorded:
sensor features with robust z-scores, detector breaches, evidence trend,
ORBIT's own diagnosis summary, and the permitted-action list.

## Result

| Attempt | Timeout | Outcome |
|---------|--------:|---------|
| 1 | 30 s | `APITimeoutError` — advisory recorded as `available: false, reason: timeout` |
| 2 | 120 s | **Success** — valid schema-compliant advisory returned |

Attempt 1 doubles as a live demonstration of the designed fallback: the
API timeout was recorded verbatim, was **never** interpreted as a
diagnosis or recovery, and the incident still resolved only through the
deterministic verifier. The default timeout was then raised to 120 s
(`orbit/amd/config.py`), and attempt 2 succeeded within it.

### Sanitized advisory actually returned (attempt 2)

```json
{
  "available": true,
  "provider": "amd_token_factory",
  "model": "DeepSeek-V4-Flash",
  "assessment": "Strong evidence of a bearing outer-race defect: very
                 high vibration RMS, peak, kurtosis, BPFO band amplitude,
                 envelope BPFO ratio, and high-frequency energy fraction.
                 The deterministic diagnosis is confident and consistent
                 with these features. No contradicting trend or
                 temperature issue.",
  "supporting_evidence": [
    "vib_rms 0.5523 with z=337.28 (extreme high)",
    "vib_peak 1.8934 with z=57.13",
    "vib_kurtosis 3.9737 with z=10.48 (impulsive)",
    "spec_bpfo_band_amp 0.0123 with z=12.86",
    "env_bpfo_ratio 32.1537 with z=57.72",
    "hf_energy_frac 0.9379 with z=262.56"
  ],
  "recommended_action": "replace_bearing",
  "action_supported": true,
  "confidence": "high",
  "needs_escalation": false,
  "diagnosis_produced": false,
  "recovery_declared": false,
  "agrees_with_plan": true
}
```

The evidence values above are the real z-scores ORBIT's robust baseline
detector computed for that window — the model reasoned over recorded
evidence, not free text. The incident's final status was `RESOLVED`
because the deterministic verifier passed, not because the model said so.

## What this verifies — and what it does not

**Verified:**
- The live AMD Token Factory endpoint accepts ORBIT's evidence payload
  and returns a schema-valid advisory (strict field validation passed).
- The model cites the actual recorded z-scores and recommends an action
  inside ORBIT's allowlist.
- The safety architecture holds on a live call: the advisory is recorded
  as advice only; `diagnosis_produced` and `recovery_declared` are both
  false; the executed action came from the deterministic knowledge base;
  RESOLVED came only from the deterministic verifier.
- API failure degrades cleanly to the deterministic pipeline with the
  failure recorded in the audit trail.

**Not claimed:**
- No AMD GPU or ROCm execution was used or measured. The AMD model runs
  on AMD's hosted service; this Mac runs ORBIT's deterministic core on
  CPU and its optional classifier on Apple MPS. "AMD technology" here
  means the AMD-hosted Token Factory model, stated as such.
- Latency/reliability statistics: two calls cannot characterize an
  endpoint. The >30 s response time motivates the 120 s default timeout;
  no other performance claim is made.
- Quota: only two live requests were ever made while preparing the
  submission. Automated tests never touch the network (all 42 advisor
  tests are mocked).

## Reproduce

```bash
# requires RADEON_API_KEY in the environment or .env (git-ignored)
.venv/bin/python live_amd_check.py
```

One request per invocation. The script prints a sanitized outcome.
