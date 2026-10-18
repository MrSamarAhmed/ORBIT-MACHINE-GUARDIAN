"""One-shot live AMD Token Factory verification (single request).

WARNING: performs exactly ONE real network request to the AMD Token
Factory endpoint using RADEON_API_KEY from the environment or .env.
Never run this in tests or CI — it consumes API quota. Default timeout
is 120s because the live endpoint answered in >30s when verified
(2026-10-18); override with RADEON_TIMEOUT_S.

Runs one full ORBIT recovery cycle on the bearing scenario with the real
advisor injected — exactly the demo path. That cycle consults the AMD
endpoint once. Prints a sanitized outcome; the API key is never printed
and failure details are redacted by the advisor itself.
"""
import json

from orbit.actions.environment import MaintenanceEnvironment
from orbit.amd.advisor import AmdAdvisor
from orbit.amd.config import AmdConfig
from orbit.diagnosis.engine import DiagnosisEngine
from orbit.detection.baseline import BaselineDetector
from orbit.incident.controller import IncidentController
from orbit.simulator.motor import MotorSimulator
from orbit.config import MotorConfig
from orbit.signalproc.features import extract_features
from tests.conftest import SCENARIOS_STATE

config = MotorConfig()
detector = BaselineDetector()
healthy = MotorSimulator(config, seed=53)
healthy.advance(600.0)
detector.fit([extract_features(w, config) for w in healthy.sample_windows(40)])

sim = MotorSimulator(config, seed=53)
sim.advance(600.0)
sim.set_state(SCENARIOS_STATE["bearing"])

cfg = AmdConfig.from_env()
if cfg is None:
    raise SystemExit("No RADEON_API_KEY configured; live call not attempted.")
print("Endpoint:", json.dumps(cfg.redacted()))

controller = IncidentController(
    simulator=sim,
    detector=detector,
    engine=DiagnosisEngine(),
    environment=MaintenanceEnvironment(),
    config=config,
    evidence_rounds=3,
    amd_advisor=AmdAdvisor(cfg),
)
result = controller.run_cycle()
incident = result.incident
print("final_status:", incident.final_status)
print("action_id:", incident.action_id)
print("verification passed:", incident.verification.passed if incident.verification else None)
print("attempts_used:", incident.attempts_used)
advisory = incident.amd_advisory or {}
print("--- amd advisory (sanitized) ---")
print(json.dumps(advisory, indent=2, default=str)[:2500])
