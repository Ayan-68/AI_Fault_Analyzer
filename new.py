from __future__ import annotations

from groq import Groq
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

APP_VERSION = "v3.8"

st.set_page_config(page_title="AI Analyzer", page_icon="🤖", layout="wide")
st.title("🤖 AI Data Analyzer — Generator Fault Detection")
api_key = st.secrets.get("API_KEY")
if not api_key:
    api_key = st.sidebar.text_input("Groq API key (if not in secrets)", type="password")
client = Groq(api_key=api_key) 

NORMAL_VALUE = 0.605
FAULT_TOL = 0.01
FAULT_DURATION = 0.270
STAB_WINDOW = 200
STAB_THRESH = 8.14e-5
SETTLE_CONFIRM = 3000
FINAL_MEAN_WINDOW_S = 0.5
FINAL_MEAN_TOL = 6e-4
PRE_FAULT_N = 1000


def _find_stabilization_idx(pe: np.ndarray, t: np.ndarray, fault_end_idx: int) -> int | None:
    """
    Full stabilization: earliest point after fault clear where the signal is
    both quiet and close to its final steady value for a sustained duration.
    """
    n = len(pe)
    rolling_std = pd.Series(pe).rolling(
        STAB_WINDOW, min_periods=STAB_WINDOW
    ).std().to_numpy()
    quiet = (rolling_std < STAB_THRESH) & ~np.isnan(rolling_std)

    # Estimate final steady value from the tail and require proximity to it.
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 1.0
    tail_n = max(1, int(FINAL_MEAN_WINDOW_S / dt))
    final_mean = float(np.mean(pe[-tail_n:]))
    near_final = np.abs(pe - final_mean) <= FINAL_MEAN_TOL
    stable_mask = quiet & near_final

    for i in range(fault_end_idx + 1, n - SETTLE_CONFIRM + 1):
        if np.all(stable_mask[i : i + SETTLE_CONFIRM]):
            return i

    # Fallback: if no full sustained run, pick first point in the final stable cluster.
    final_cluster = np.where(stable_mask)[0]
    if len(final_cluster) > 0:
        last_start = final_cluster[-1]
        while last_start > fault_end_idx + 1 and stable_mask[last_start - 1]:
            last_start -= 1
        return int(last_start)

    return None


def detect_fault_events(pe: np.ndarray, t: np.ndarray) -> dict:
    """
    Fault start  = first sample outside ±FAULT_TOL around NORMAL_VALUE.
    Fault end    = fault_start + FAULT_DURATION (270 ms injection cleared).
    Stabilized   = oscillation permanently below threshold after fault end (~9.82 s).
    """
    n = len(pe)
    fault_mask = np.abs(pe - NORMAL_VALUE) > FAULT_TOL
    fault_labels = np.where(fault_mask, "Yes", "No")

    baseline = float(np.mean(pe[: min(PRE_FAULT_N, n)]))

    if not fault_mask.any():
        return {
            "fault_mask": fault_mask,
            "fault_labels": fault_labels,
            "fault_start_idx": None,
            "fault_end_idx": None,
            "stabilize_idx": None,
            "fault_start_time": None,
            "fault_end_time": None,
            "stabilize_time": None,
            "baseline": baseline,
        }

    fault_start_idx = int(np.argmax(fault_mask))
    fault_start_time = float(t[fault_start_idx])

    target_end = fault_start_time + FAULT_DURATION
    fault_end_idx = int(np.argmin(np.abs(t - target_end)))
    fault_end_time = float(t[fault_end_idx])

    stabilize_idx = _find_stabilization_idx(pe, t, fault_end_idx)
    stabilize_time = float(t[stabilize_idx]) if stabilize_idx is not None else None

    return {
        "fault_mask": fault_mask,
        "fault_labels": fault_labels,
        "fault_start_idx": fault_start_idx,
        "fault_end_idx": fault_end_idx,
        "stabilize_idx": stabilize_idx,
        "fault_start_time": fault_start_time,
        "fault_end_time": fault_end_time,
        "stabilize_time": stabilize_time,
        "baseline": baseline,
    }


def render_results(df: pd.DataFrame, result: dict) -> None:
    pe = df["SM2.Pe [HYP1]"].to_numpy(dtype=float)
    t = df["X AXIS"].to_numpy(dtype=float)

    fault_start_time = result["fault_start_time"]
    fault_end_time = result["fault_end_time"]
    stabilize_time = result["stabilize_time"]
    baseline = result["baseline"]
    fault_mask = result["fault_mask"]

    fault_duration = (
        fault_end_time - fault_start_time
        if fault_start_time is not None and fault_end_time is not None
        else None
    )
    settling_duration = (
        stabilize_time - fault_end_time
        if stabilize_time is not None and fault_end_time is not None
        else None
    )
    total_disturbed_time = (
        stabilize_time - fault_start_time
        if stabilize_time is not None and fault_start_time is not None
        else None
    )

    post_fault_mean = float(np.mean(pe[t > (t[-1] - 0.5)]))
    setpoint_shift = post_fault_mean - baseline

    st.success("✅ Analysis Complete!")
    st.info(
        f"**{APP_VERSION}** — Fault start **{fault_start_time:.4f} s** | "
        f"Fault end **{fault_end_time:.4f} s** | "
        f"Stabilized **{stabilize_time:.4f} s**"
        if stabilize_time
        else f"**{APP_VERSION}** — Fault start **{fault_start_time:.4f} s** | "
        f"Fault end **{fault_end_time:.4f} s** | Stabilized: not detected"
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Fault Start", f"{fault_start_time:.4f} s" if fault_start_time else "No fault")
    c2.metric("Fault End", f"{fault_end_time:.4f} s" if fault_end_time else "—")
    c3.metric("Fault Duration", f"{fault_duration:.4f} s" if fault_duration else "—")
    c4.metric("Fully Stabilized At", f"{stabilize_time:.4f} s" if stabilize_time else "Not detected")
    c5.metric("Settling Time", f"{settling_duration:.4f} s" if settling_duration else "—")

    c6, c7, c8, c9 = st.columns(4)
    c6.metric("Total Disturbed", f"{total_disturbed_time:.4f} s" if total_disturbed_time else "—")
    c7.metric("Pre-fault Baseline", f"{baseline:.5f} pu")
    c8.metric("Post-fault Steady", f"{post_fault_mean:.5f} pu")
    c9.metric("Setpoint Shift", f"{setpoint_shift:+.5f} pu")

    st.subheader("📈 SM2.Pe [HYP1] — Fault Detection Analysis")
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(t, pe, color="#2196F3", linewidth=0.6, label="SM2.Pe [HYP1]")
    ax.axhline(NORMAL_VALUE + FAULT_TOL, color="green", linestyle="--", linewidth=0.8,
               label=f"Tolerance (±{FAULT_TOL})")
    ax.axhline(NORMAL_VALUE - FAULT_TOL, color="green", linestyle="--", linewidth=0.8)
    ax.axhline(NORMAL_VALUE, color="gray", linestyle=":", linewidth=0.8, label=f"Nominal ({NORMAL_VALUE})")

    if fault_start_time is not None and fault_end_time is not None:
        ax.axvspan(fault_start_time, fault_end_time, alpha=0.15, color="red", label="Fault (270 ms)")
        ax.axvline(fault_start_time, color="red", linewidth=1.2)
        ax.axvline(fault_end_time, color="orange", linewidth=1.2, label="Fault cleared")

    if fault_end_time and stabilize_time and settling_duration and settling_duration > 0:
        ax.axvspan(fault_end_time, stabilize_time, alpha=0.12, color="orange", label="Settling")
        ax.axvline(stabilize_time, color="purple", linewidth=1.2, linestyle="--",
                   label=f"Stabilized {stabilize_time:.4f} s")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pe (pu)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    if client:
        prompt = f"""Expert power-system fault report for generator Pe signal.

Nominal: {NORMAL_VALUE} pu, tolerance ±{FAULT_TOL}, fault injection {FAULT_DURATION} s.
Fault start: {fault_start_time} s, fault end: {fault_end_time} s, duration: {fault_duration} s.
Stabilized: {stabilize_time} s, settling: {settling_duration} s.
Baseline: {baseline:.5f} pu, post-fault: {post_fault_mean:.5f} pu, shift: {setpoint_shift:+.5f} pu.
Fault samples (out of band): {int(fault_mask.sum())}, Pe max/min: {pe.max():.5f}/{pe.min():.5f}.

Sections: event, severity, recovery, setpoint shift, health, recommendations (concise)."""

        with st.spinner("Generating AI Report..."):
            resp = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
            )
            st.subheader("🤖 AI System Report")
            st.write(resp.choices[0].message.content)
    else:
        st.warning("Add Groq API key in sidebar or secrets to generate the AI report.")

    st.subheader("📋 Analyzed Data")
    st.dataframe(df)
    st.download_button(
        "⬇ Download Results",
        df.to_csv(index=False),
        file_name="analyzed_equipment_data.csv",
        mime="text/csv",
    )


st.markdown("Upload a CSV with columns `X AXIS` and `SM2.Pe [HYP1]`.")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Uploaded Data")
    st.dataframe(df.head(100))

    if st.button("Analyze Equipment", type="primary"):
        pe = df["SM2.Pe [HYP1]"].to_numpy(dtype=float)
        t = df["X AXIS"].to_numpy(dtype=float)
        result = detect_fault_events(pe, t)
        df = df.copy()
        df["Fault?"] = result["fault_labels"]
        st.session_state["analysis_df"] = df
        st.session_state["analysis_result"] = result

if st.session_state.get("analysis_result"):
    render_results(st.session_state["analysis_df"], st.session_state["analysis_result"])
