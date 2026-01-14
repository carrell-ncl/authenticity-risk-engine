# streamlit_app.py
# Option 1: Streamlit UI that calls your FastAPI service (no model code in the UI).
#
# Run:
#   pip install streamlit requests pandas numpy matplotlib
#   streamlit run streamlit_app.py
#
# Requires your API running, e.g.:
#   PYTHONPATH=. uvicorn api.audio_det.app:app --host 0.0.0.0 --port 8000

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns 

# ---- Session state init ----
if "single_report" not in st.session_state:
    st.session_state.single_report = None

if "batch_df" not in st.session_state:
    st.session_state.batch_df = None

if "batch_reports" not in st.session_state:
    st.session_state.batch_reports = None

if "uploader_key_single" not in st.session_state:
    st.session_state.uploader_key_single = 0
if "uploader_key_batch" not in st.session_state:
    st.session_state.uploader_key_batch = 0


sns.set(style="darkgrid")


# ---------------------------
# Helpers
# ---------------------------
@dataclass
class ApiConfig:
    base_url: str = "http://localhost:8000"
    score_path: str = "/v1/score"
    health_path: str = "/health"


def api_health(cfg: ApiConfig, timeout_s: float = 3.0) -> Tuple[bool, str]:
    url = cfg.base_url.rstrip("/") + cfg.health_path
    try:
        r = requests.get(url, timeout=timeout_s)
        if r.status_code == 200:
            return True, r.text
        return False, f"HTTP {r.status_code}: {r.text}"
    except Exception as e:
        return False, str(e)


def post_score_file(
    cfg: ApiConfig,
    file_bytes: bytes,
    filename: str,
    threshold: Optional[float] = None,
    timeout_s: float = 120.0,
) -> Dict[str, Any]:
    url = cfg.base_url.rstrip("/") + cfg.score_path
    params = {}
    if threshold is not None:
        params["threshold"] = threshold

    files = {"file": (filename, file_bytes)}
    try:
        r = requests.post(url, params=params, files=files, timeout=timeout_s)
    except Exception as e:
        st.error(f"[ERROR] Exception during POST: {e}")
        return {"ok": False, "error": str(e), "_http_status": None}

    # The API sometimes returns 422 with JSON for ok=False; handle robustly.
    try:
        payload = r.json()
    except Exception:
        st.error(f"[ERROR] Non-JSON response: HTTP {r.status_code}")
        payload = {"ok": False, "error": f"Non-JSON response: HTTP {r.status_code}", "raw": r.text}

    payload["_http_status"] = r.status_code
    return payload


def extract_row(report: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten the report into a table row for batch scoring."""
    row = {
        "filename": report.get("filename"),
        "ok": report.get("ok", False),
        "http_status": report.get("_http_status"),
        "score": report.get("score"),
        "tier": report.get("tier"),
        "decision": report.get("decision"),
        "error": report.get("error"),
    }
    sig = report.get("signals") or {}
    row.update(
        {
            "cnn_only_score": sig.get("cnn_only_score"),
            "calibrated_score": sig.get("calibrated_score"),
            "cnn_median": sig.get("cnn_median"),
            "cnn_max": sig.get("cnn_max"),
            "cnn_var": sig.get("cnn_var"),
            "silence_ratio": sig.get("silence_ratio"),
            "total_seconds": sig.get("total_seconds"),
        }
    )
    return row


def plot_segment_scores(seg_probs: List[float], title: str = "Segment spoof probabilities"):
    if not seg_probs:
        st.info("No segment scores available.")
        return
    x = np.arange(1, len(seg_probs) + 1)
    y = np.array(seg_probs, dtype=float)

    fig = plt.figure()
    plt.plot(x, y, marker="o")
    plt.ylim(0.0, 1.0)
    plt.xticks(x)
    plt.xlabel("Segment index")
    plt.ylabel("P(fake)")
    plt.title(title)
    st.pyplot(fig)
    plt.close(fig)


def plot_score_histogram(df: pd.DataFrame, col: str = "score", title: str = "Score distribution"):
    d = df[df["ok"] & df[col].notna()].copy()
    if d.empty:
        st.info("No scores to plot.")
        return
    fig = plt.figure()
    plt.hist(d[col].astype(float).values, bins=30)
    plt.xlabel(col)
    plt.ylabel("count")
    plt.title(title)
    st.pyplot(fig)
    plt.close(fig)


# ---------------------------
# UI
# ---------------------------
st.set_page_config(page_title="Audio Authenticity Risk (Demo)", layout="wide")

st.title("Audio Authenticity Risk — Demo UI (API-backed)")
st.caption("Upload audio → calls your FastAPI `/v1/score` → shows risk score, signals, and segment scores.")

with st.sidebar:
    st.header("API Settings")
    base_url = st.text_input("API Base URL", value="http://localhost:8000")
    timeout_s = st.number_input("Request timeout (seconds)", min_value=5, max_value=600, value=120, step=5)
    cfg = ApiConfig(base_url=base_url)

    st.divider()
    st.header("Decision Settings")
    threshold = st.slider("Decision threshold (P(fake) ≥ threshold ⇒ fake)", 0.0, 1.0, 0.5, 0.01)

    st.divider()
    if st.button("🧹 Clear / Reset"):
        st.session_state.single_report = None
        st.session_state.batch_df = None
        st.session_state.batch_reports = None

        st.session_state.uploader_key_single += 1
        st.session_state.uploader_key_batch += 1

        st.rerun()




    st.divider()
    st.header("Display")
    show_raw_json = st.checkbox("Show raw JSON response", value=True)
    show_signals = st.checkbox("Show signals block", value=True)
    show_segments_plot = st.checkbox("Plot segment scores", value=True)

# Health check
ok, msg = api_health(cfg)
if ok:
    st.success(f"API reachable: {cfg.base_url}")
else:
    st.error(f"API not reachable: {cfg.base_url}\n\n{msg}")
    st.stop()

tabs = st.tabs(["Single file", "Batch upload", "About / Tips"])

# ---------------------------
# Single file
# ---------------------------
with tabs[0]:
    st.subheader("Single file scoring")

    up = st.file_uploader(
        "Upload an audio file (.wav / .flac recommended)",
        accept_multiple_files=False,
        key=f"single_uploader_{st.session_state.uploader_key_single}",
    )

    colA, colB = st.columns([1, 1])
    with colA:
        run_btn = st.button("Score file", type="primary", disabled=(up is None))
    with colB:
        st.write("")
        st.write("")

    if run_btn and up is not None:
        report = None
        error_msg = None
        try:
            with st.spinner("Scoring…"):
                report = post_score_file(
                    cfg=cfg,
                    file_bytes=up.getvalue(),
                    filename=up.name,
                    threshold=float(threshold),
                    timeout_s=float(timeout_s),
                )
        except Exception as e:
            error_msg = str(e)
            st.error(f"[ERROR] Exception during scoring: {error_msg}")

        if report is not None:
            # Headline
            if report.get("ok"):
                st.success(
                    f"Score: {report.get('score')} | Tier: {report.get('tier')} | Decision: {report.get('decision')} "
                    f"(HTTP {report.get('_http_status')})"
                )
            else:
                st.warning(f"Scoring returned ok=False (HTTP {report.get('_http_status')}).")
                st.error(f"[ERROR] {report.get('error') or report}")

            # Signals
            if show_signals and report.get("signals"):
                st.markdown("### Signals")
                st.json(report["signals"])

            # Segments plot
            segs = report.get("segments")
            if show_segments_plot and isinstance(segs, list):
                st.markdown("### Segment scores")
                plot_segment_scores(segs, title=f"Segment P(fake) — {up.name}")

            # Raw JSON
            if show_raw_json:
                st.markdown("### Full response JSON")
                st.json(report)

            # Download report
            st.download_button(
                "Download JSON report",
                data=json.dumps(report, indent=2).encode("utf-8"),
                file_name=f"{up.name}.score.json",
                mime="application/json",
            )
        elif error_msg:
            st.error(f"[ERROR] No report returned: {error_msg}")

# ---------------------------
# Batch upload
# ---------------------------
with tabs[1]:
    st.subheader("Batch scoring (multi-file upload)")
    st.caption("Upload multiple files. Results will appear in a table; you can sort/filter and download CSV.")

    ups = st.file_uploader(
        "Upload multiple audio files",
        accept_multiple_files=True,
        key=f"batch_uploader_{st.session_state.uploader_key_batch}",
    )

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        run_batch = st.button("Score batch", type="primary", disabled=(not ups))
    with col2:
        max_files = st.number_input("Max files to score", min_value=1, max_value=5000, value=100, step=10)
    with col3:
        st.write("")

    if run_batch and ups:
        selected = ups[: int(max_files)]
        rows = []
        reports = []

        progress = st.progress(0)
        for i, f in enumerate(selected, start=1):
            rep = post_score_file(
                cfg=cfg,
                file_bytes=f.getvalue(),
                filename=f.name,
                threshold=float(threshold),
                timeout_s=float(timeout_s),
            )
            reports.append(rep)
            rows.append(extract_row(rep))
            progress.progress(i / len(selected))

        df = pd.DataFrame(rows)

        st.markdown("### Results table")
        st.dataframe(df, use_container_width=True)

        # Basic aggregates
        ok_df = df[df["ok"]].copy()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Files scored (ok)", int(ok_df.shape[0]))
        c2.metric("High tier", int((ok_df["tier"] == "high").sum()) if "tier" in ok_df else 0)
        c3.metric("Medium tier", int((ok_df["tier"] == "medium").sum()) if "tier" in ok_df else 0)
        c4.metric("Low tier", int((ok_df["tier"] == "low").sum()) if "tier" in ok_df else 0)

        st.markdown("### Score histogram")
        plot_score_histogram(df, col="score", title="Batch score distribution (final score)")

        # Download CSV + JSONL
        st.download_button(
            "Download CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="batch_scores.csv",
            mime="text/csv",
        )

        jsonl = "\n".join(json.dumps(r) for r in reports)
        st.download_button(
            "Download JSONL (all reports)",
            data=jsonl.encode("utf-8"),
            file_name="batch_reports.jsonl",
            mime="application/jsonl",
        )

# ---------------------------
# About / Tips
# ---------------------------
with tabs[2]:
    st.subheader("About / Tips")
    st.markdown(
        """
**How this demo works**
- Streamlit uploads audio to your FastAPI endpoint (`POST /v1/score`) using `multipart/form-data`.
- The API returns a JSON report: `score`, `tier`, `signals`, and `segments`.

**Common gotchas**
- If you see many `cnn_var = 0` and identical segment scores: lots of files may be **shorter than `clip_seconds`** and heavily padded.
- If ASVspoof behaves very differently from your real_or_fake set: that's often **domain shift**. Consider routing / calibrator selection later.

**Recommended next step**
- Add a `/v1/model-info` endpoint to return model config + versions for auditability.
"""
    )
