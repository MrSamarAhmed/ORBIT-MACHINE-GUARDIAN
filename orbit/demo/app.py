"""Streamlit demo for ORBIT Machine Guardian.

Run: ``streamlit run orbit/demo/app.py``

The recovery loop is meant to be understood without reading source: the
page walks through OBSERVE -> DETECT -> DIAGNOSE -> PLAN -> ACT -> VERIFY
-> RECOVER-OR-ESCALATE for the selected scenario, showing live evidence,
the proposed action, and the deterministic verification verdict.

All interventions are simulated. No real equipment is controlled.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from orbit.api import service

st.set_page_config(page_title="ORBIT Machine Guardian", layout="wide",
                   page_icon="")

st.title("ORBIT Machine Guardian")
st.caption(
    "Autonomous industrial reliability agent. Detects motor faults, diagnoses "
    "likely causes, performs corrective actions in a controlled simulator, and "
    "independently verifies recovery. All interventions are simulated."
)


@st.cache_resource(show_spinner=False)
def _device_panel_data():
    return service.device_summary()


with st.sidebar:
    st.header("Scenario")
    scenario = st.selectbox("Fault scenario", service.scenario_names(), index=1)
    st.number_input("Healthy windows before fault onset", value=15, min_value=1,
                    max_value=40, key="switch_after")
    st.number_input("Trend window count", value=60, min_value=20, max_value=200,
                    key="trend_windows")
    st.header("Autonomous cycle")
    cycles = st.slider("Cycles to run", 1, 3, 1)
    monitor_interval = st.slider("Monitoring interval (s)", 5, 120, 60)
    use_classifier = st.toggle("Neural corroboration", value=True,
                               help="Optional. Requires trained weights. Never authoritative.")
    defective_spare = st.toggle("Simulate defective spare (quality 0)", value=False)
    run = st.button("Run recovery cycle", type="primary")

# --- Execution device panel -------------------------------------------------
dev = _device_panel_data()
env = dev["environment"]
c1, c2, c3 = st.columns(3)
c1.metric("Compute device", env["platform"].upper())
c2.metric("Device name", env["device_name"][:28])
c3.metric("PyTorch", env["torch_version"])
clf = dev["classifier"]
if clf.get("available"):
    st.info(f"Corroborating classifier active on "
            f"{clf['device']['platform']} ({clf['device']['device_name']}).")
else:
    st.caption(f"Neural corroboration unavailable: {clf.get('reason') or clf.get('error')}.")

# --- Sensor trends ----------------------------------------------------------
st.subheader("Sensor trends")
switch_after = st.session_state["switch_after"]
trend = service.sensor_trends(
    scenario=scenario,
    windows=st.session_state["trend_windows"],
    switch_after=switch_after,
)
trends_df = pd.DataFrame(trend["windows"])
left, right = st.columns(2)
left.caption("Vibration RMS & kurtosis")
left.line_chart(trends_df.set_index("t_start_s")[["vib_rms", "vib_kurtosis"]])
right.caption("Envelope BPFO ratio & shaft 1x amplitude")
right.line_chart(trends_df.set_index("t_start_s")[["env_bpfo_ratio", "spec_1x_amp"]])
st.caption(f"Temperature (C) - fault onset at t={trend['windows'][switch_after]['t_start_s']}s")
st.line_chart(trends_df.set_index("t_start_s")[["temperature_c"]])

# --- Incident history -------------------------------------------------------
if "history" not in st.session_state:
    st.session_state["history"] = []

if run:
    env_over = {"spare_bearing_quality": 0.0} if defective_spare else None
    result = service.run_recovery_cycle(
        scenario=scenario,
        cycles=cycles,
        environment=env_over,
        monitor_interval_s=float(monitor_interval),
        use_classifier=use_classifier,
    )
    st.session_state["last_result"] = result
    for cyc in result["cycles"]:
        st.session_state["history"].append({
            "scenario": scenario,
            "outcome": cyc["outcome"],
            "incident_id": (cyc["incident"] or {}).get("incident_id"),
        })

result = st.session_state.get("last_result")
if not result:
    st.info("Pick a scenario and press **Run recovery cycle** to see the loop.")
else:
    st.subheader("Recovery loop result")
    feat_before = pd.Series(result["features_before"], name="before")
    feat_after = pd.Series(result["features_after"], name="after")
    cmp = pd.concat([feat_before, feat_after], axis=1).reset_index()
    cmp.columns = ["feature", "before", "after"]
    m1, m2, m3 = st.columns(3)
    m1.metric("Outcome", result["cycles"][-1]["outcome"])
    m2.metric("Still anomalous?", "YES" if result["still_anomalous"] else "NO")
    m3.metric("Incidents opened", sum(1 for c in result["cycles"] if c["incident"]))
    st.dataframe(cmp, hide_index=True, use_container_width=True)

    for idx, cyc in enumerate(result["cycles"], 1):
        inc = cyc["incident"]
        st.markdown(f"**Cycle {idx}** - anomaly: {cyc['anomaly_detected']} - "
                    f"outcome: `{cyc['outcome']}`")
        if not inc:
            st.success("Machine healthy - no action taken (scenario D).")
            continue

        # DETECT
        detect = inc["phases"][0]["evidence"]
        st.markdown(f"**DETECT** - peak deviation "
                    f"z={detect.get('max_abs_z')}, breaches: {len(detect.get('breaches', []))}")

        # DIAGNOSE + ranked hypotheses
        diag = inc.get("diagnosis") or {}
        st.markdown(f"**DIAGNOSE** - primary: *{diag.get('primary')}* "
                    f"(confident: {diag.get('confident')})")
        if diag.get("hypotheses"):
            hyp_df = pd.DataFrame(diag["hypotheses"])
            st.dataframe(hyp_df, hide_index=True, use_container_width=True)
        if diag.get("ambiguity_note"):
            st.warning(diag["ambiguity_note"])

        # evidence (supporting / contradicting) from ACTION_PLANNED or last phase
        planned = next((p["evidence"] for p in inc["phases"]
                        if p["phase"] == "ACTION_PLANNED"), None)
        if planned:
            e1, e2 = st.columns(2)
            e1.markdown("**Supporting evidence**")
            e1.dataframe(pd.DataFrame(planned.get("supporting_evidence", [])),
                         hide_index=True, use_container_width=True)
            e2.markdown("**Contradicting evidence**")
            e2.dataframe(pd.DataFrame(planned.get("contradicting_evidence", [])),
                         hide_index=True, use_container_width=True)

            # PLAN / ACT
            st.markdown(
                f"**PLAN / ACT** - action: *{planned.get('action')}* - "
                f"preconditions met: {planned.get('preconditions_met')}")
            if planned.get("neural_corroboration"):
                nc = planned["neural_corroboration"]
                st.caption(
                    f"Neural corroboration (device {nc['device']['platform']}): "
                    f"{nc['predicted_class']} - agrees with diagnosis: "
                    f"{nc.get('agrees_with_diagnosis')}")

        # VERIFY
        ver = inc.get("verification")
        if ver:
            st.markdown("**VERIFY** - deterministic conditions")
            st.dataframe(pd.DataFrame(ver["results"]), hide_index=True,
                         use_container_width=True)
            if ver["passed"]:
                st.success("Recovery verified - incident RESOLVED.")
            else:
                st.error("Verification failed - conditions not met.")
        if inc.get("escalation_reason"):
            st.error(f"**ESCALATED**: {inc['escalation_reason']}")

# --- Incident history table -------------------------------------------------
if st.session_state["history"]:
    st.subheader("Incident history")
    st.dataframe(pd.DataFrame(st.session_state["history"]),
                 hide_index=True, use_container_width=True)

# --- Measured benchmarks & evaluation ---------------------------------------
st.subheader("Measured results")
bench = service.latest_benchmark()
if bench.get("available") is False:
    st.caption(f"Benchmark: {bench.get('reason')}")
else:
    rows = []
    for r in bench["results"]:
        for m in r["measurements"]:
            rows.append({
                "device": r["device"]["platform"],
                "batch_size": m["batch_size"],
                "latency_mean_ms": m["latency_mean_ms"],
                "throughput_windows_per_s": m["throughput_windows_per_s"],
            })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    if not bench.get("gpu_available"):
        st.caption("CPU-only environment - no GPU numbers fabricated. AMD ROCm "
                   "measurements will be added from a real ROCm instance.")

ev = service.evaluation_results()
if ev.get("available") is False:
    st.caption(f"Evaluation: {ev.get('reason')}")
else:
    rec = ev["recovery"]
    d1, d2, d3 = st.columns(3)
    d1.metric("Recovery success rate", f"{rec['recovery_success_rate']:.0%}",
              help=f"{rec['resolve_correct']}/{rec['resolve_designed']} designed-to-resolve scenarios")
    d2.metric("Correct escalation rate", f"{rec['correct_escalation_rate']:.0%}",
              help=f"{rec['escalate_correct']}/{rec['escalate_designed']} designed-to-escalate")
    d3.metric("Mean end-to-end latency", f"{rec['mean_end_to_end_latency_ms']} ms")
    st.caption(ev["data_source"] + " - " + ev["detection"].get("note", ""))
