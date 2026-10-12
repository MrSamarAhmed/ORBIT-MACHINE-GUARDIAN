"""FastAPI surface for ORBIT Machine Guardian.

Run: ``uvicorn orbit.api.app:app --reload``

The API is a thin, stateless wrapper over `orbit.api.service`. It exposes
the deterministic recovery loop, live device info, and the measured
evaluation/benchmark reports. No endpoint claims a result the pipeline
did not actually compute.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from orbit import __version__
from orbit.api import service

app = FastAPI(
    title="ORBIT Machine Guardian",
    version=__version__,
    description="Autonomous industrial reliability agent (simulated interventions only).",
)


class CycleRequest(BaseModel):
    scenario: str = Field(..., description="one of: normal, bearing, imbalance, overtemperature")
    cycles: int = Field(1, ge=1, le=5, description="autonomous cycles to run")
    environment: Optional[Dict[str, Any]] = Field(
        None, description="MaintenanceEnvironment overrides (resource quality, availability)"
    )
    seed: int = Field(6006, ge=0, description="deterministic simulator seed")
    monitor_interval_s: float = Field(60.0, ge=0.0, description="seconds between monitoring checks")
    use_classifier: bool = Field(True, description="attach neural corroboration when weights exist")


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.get("/device")
def device() -> Dict[str, Any]:
    return service.device_summary()


@app.get("/scenarios")
def scenarios() -> Dict[str, Any]:
    return {
        "scenarios": service.scenario_names(),
        "environment_fields": list(service.valid_env_fields()),
        "feature_names": list(service.FEATURE_NAMES),
    }


@app.get("/trends")
def trends(
    scenario: str = Query("bearing"),
    windows: int = Query(60, ge=2, le=500),
    switch_after: int = Query(15, ge=1),
    seed: int = Query(2024, ge=0),
) -> Dict[str, Any]:
    _require_scenario(scenario)
    try:
        return service.sensor_trends(scenario=scenario, windows=windows,
                                     switch_after=switch_after, seed=seed)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/cycle")
def run_cycle(request: CycleRequest) -> Dict[str, Any]:
    _require_scenario(request.scenario)
    try:
        return service.run_recovery_cycle(
            scenario=request.scenario,
            cycles=request.cycles,
            environment=request.environment,
            seed=request.seed,
            monitor_interval_s=request.monitor_interval_s,
            use_classifier=request.use_classifier,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/evaluation")
def evaluation() -> Dict[str, Any]:
    return service.evaluation_results()


@app.get("/benchmark")
def benchmark() -> Dict[str, Any]:
    return service.latest_benchmark()


def _require_scenario(scenario: str) -> None:
    if scenario not in service.scenario_names():
        raise HTTPException(
            status_code=422,
            detail=f"unknown scenario {scenario!r}; options: {service.scenario_names()}",
        )
