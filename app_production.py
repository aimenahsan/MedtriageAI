"""
MedTriageAI+ : AI Emergency Operations Center
Production-Ready Stage 1 Application
Orchestrates Whisper -> DistilBERT -> XGBoost KTAS Pipeline

CHANGE LOG (see accompanying notes for the full list): this version wires
voice input, NLP symptom extraction, prediction, and explainability into a
real, working pipeline. Every number displayed in the UI is either a live
user input or something computed on this run -- nothing is hard-coded
demo/marketing text. Search for "REAL:" comments at the points that most
directly replace previously-fabricated content.
"""

import streamlit as st
import pandas as pd
import time
import tempfile
import os
from datetime import datetime
from pathlib import Path
import sys
import json

# ===== SETUP PATHS & IMPORTS =====
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / 'models'))
sys.path.insert(0, str(Path(__file__).parent / 'database'))

# Import database
import importlib.util
spec = importlib.util.spec_from_file_location("database_module", str(Path(__file__).parent / 'database' / 'database.py'))
database_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(database_module)
init_database = database_module.init_database

# Initialize database
init_database()

# Import AI models orchestrator.
# REAL: no more silent mock fallback. A clinical decision-support tool that
# silently returns a made-up "KTAS 3, 88% confidence" when its models fail
# to load is worse than one that visibly refuses to run -- the old fallback
# here made that mistake. If the import fails, the app now shows a clear
# error banner naming the problem instead of quietly faking a result.
ORCHESTRATOR_IMPORT_ERROR = None
try:
    from models.ai_orchestrator import orchestrate_models
except Exception as e:
    ORCHESTRATOR_IMPORT_ERROR = str(e)

    def orchestrate_models(**kwargs):
        raise RuntimeError(
            f"AI orchestrator failed to load ({ORCHESTRATOR_IMPORT_ERROR}). "
            f"Install dependencies with: pip install -r REBUILD_requirements.txt"
        )


def check_model_availability():
    """REAL: actually probe which optional dependencies are importable,
    instead of a hard-coded 'Loaded' caption for every model regardless of
    whether it works."""
    avail = {}
    try:
        import xgboost, shap  # noqa: F401
        avail['xgboost'] = True
    except ImportError:
        avail['xgboost'] = False
    try:
        import transformers, torch  # noqa: F401
        avail['distilbert'] = True
    except ImportError:
        avail['distilbert'] = False
    try:
        import whisper  # noqa: F401
        avail['whisper'] = True
    except ImportError:
        avail['whisper'] = False
    return avail


def transcribe_audio_bytes(audio_bytes):
    """
    REAL: actually calls Whisper. Returns (transcript_text, error_message).
    Whisper/torch are heavy optional dependencies, so this is imported
    lazily here rather than at module load time -- text-only use of the app
    still works fully without them installed.
    """
    try:
        from models.whisper_handler import transcribe_audio_improved
    except Exception as e:
        return None, (
            "Voice transcription needs the 'openai-whisper' and 'torch' packages. "
            f"Install with: pip install -r REBUILD_requirements.txt  (import error: {e})"
        )
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        text = transcribe_audio_improved(tmp_path, noise_reduction_level=1)
        return text, None
    except Exception as e:
        return None, f"Transcription failed: {e}"
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


# ===== PAGE CONFIG =====
st.set_page_config(
    page_title="MedTriageAI+ : AI Emergency Operations Center",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== BRAND COLORS & THEME =====
BRAND_COLORS = {
    "bg_dark": "#081120",
    "card_dark": "#182235",
    "primary": "#00D4FF",
    "accent": "#FF3366",
    "success": "#00E676",
    "warning": "#FFD600",
    "text_main": "#E2EDF8",
    "text_muted": "#708098"
}

# ===== SESSION STATE INITIALIZATION =====
defaults = {
    "page": "Command Center",
    "current_assessment": None,
    "pipeline_active": False,
    "assessment_complete": False,
    "ktas_level": None,
    "confidence": None,
    "explanation": None,
    "model_outputs": None,
    "inference_times": [],
    "voice_transcription_error": None,
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ===== INJECT GLOBAL STYLING =====
st.markdown(f"""
<style>
    .stApp {{
        background-color: {BRAND_COLORS["bg_dark"]};
        color: {BRAND_COLORS["text_main"]};
    }}
    h1, h2, h3 {{
        color: {BRAND_COLORS["primary"]} !important;
        font-family: 'Inter', sans-serif;
    }}
    .metric-card {{
        background-color: {BRAND_COLORS["card_dark"]};
        border-left: 4px solid {BRAND_COLORS["primary"]};
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 15px;
    }}
    .pipeline-stage {{
        background: {BRAND_COLORS["card_dark"]};
        border: 1px solid #24344d;
        padding: 15px;
        border-radius: 8px;
        margin-top: 10px;
    }}
</style>
""", unsafe_allow_html=True)

# ===== SIDEBAR SYSTEM NAVIGATION =====
with st.sidebar:
    st.markdown(f"""
    <div style='text-align: center; padding: 10px 0;'>
        <h2 style='margin: 0; color: {BRAND_COLORS["primary"]}; font-size: 26px;'>🏥 MedTriage<span style='color:#FFF;'>AI+</span></h2>
        <p style='color: {BRAND_COLORS["text_muted"]}; font-size: 12px; letter-spacing: 1px;'>AI EMERGENCY OPERATIONS CENTER</p>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    st.markdown("### 🌐 System Navigation")

    def on_nav_change():
        st.session_state.page = st.session_state.nav_select

    nav_options = ["Command Center", "Patient Assessment", "AI Pipeline", "Assessment Results", "Explainable AI", "System Status"]
    selected = st.selectbox(
        "Navigate System",
        nav_options,
        index=nav_options.index(st.session_state.page),
        key="nav_select",
        on_change=on_nav_change,
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown("### ⚡ Triage Core Load")
    # REAL: these now reflect whether the packages actually import, not a
    # static caption that says "Loaded" unconditionally.
    avail = check_model_availability()
    st.caption(("✅" if avail['whisper'] else "⚠️") + f" Whisper ASR: {'Available' if avail['whisper'] else 'Not installed (voice input disabled)'}")
    st.caption(("✅" if True else "⚠️") + " DistilBERT/Keyword NLP: Available" + ("" if avail['distilbert'] else " (keyword-only -- transformers/torch not installed)"))
    st.caption(("✅" if avail['xgboost'] else "❌") + f" XGBoost + SHAP Engine: {'Online' if avail['xgboost'] else 'Missing dependency'}")

    st.divider()

    st.markdown(f"""
    <div style='padding: 10px; background: {BRAND_COLORS["card_dark"]}; border-radius: 10px; margin-top: 20px;'>
        <p style='font-size: 12px; color: {BRAND_COLORS["text_muted"]}; margin: 0;'>
            <strong>MedTriageAI+ v1.0</strong><br>
            Emergency Operations Center<br>
            Powered by MEDA AI
        </p>
    </div>
    """, unsafe_allow_html=True)

if ORCHESTRATOR_IMPORT_ERROR:
    st.error(
        f"⚠️ AI orchestrator could not be loaded: {ORCHESTRATOR_IMPORT_ERROR}\n\n"
        f"Run `pip install -r REBUILD_requirements.txt` and restart the app. "
        f"Predictions will fail until this is resolved."
    )

# ===== PAGE 1: COMMAND CENTER =====
def page_command_center():
    st.title("📊 Clinical Command Center")
    st.markdown("Real-time telemetry and intake pipelines for machine-assisted emergency sorting.")

    try:
        stats = database_module.get_session_stats()
        active_patients = stats.get("total", 0)
    except Exception:
        active_patients = 0

    confidence = st.session_state.get('confidence')
    confidence_display = f"{confidence*100:.1f}%" if confidence is not None else "—"

    # REAL: average triage time computed from actual pipeline runs this
    # session, not a hard-coded "2.1s". Shows an empty state until at least
    # one assessment has run.
    times = st.session_state.get('inference_times', [])
    avg_time_display = f"{sum(times)/len(times):.2f}s" if times else "—"

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"<div class='metric-card'><h4 style='color:{BRAND_COLORS['text_muted']};margin:0;'>Total Intakes Logged</h4><h1 style='margin:10px 0;'>{active_patients}</h1><span style='color:{BRAND_COLORS['success']}'>● Stored in medtriage.db</span></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='metric-card'><h4 style='color:{BRAND_COLORS['text_muted']};margin:0;'>Avg Pipeline Time</h4><h1 style='margin:10px 0;'>{avg_time_display}</h1><span style='color:{BRAND_COLORS['text_muted']}'>{'This session, ' + str(len(times)) + ' run(s)' if times else 'No runs yet'}</span></div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='metric-card'><h4 style='color:{BRAND_COLORS['text_muted']};margin:0;'>Last Prediction Confidence</h4><h1 style='margin:10px 0;'>{confidence_display}</h1><span style='color:{BRAND_COLORS['primary']}'>Current Session</span></div>", unsafe_allow_html=True)
    with col4:
        engine_ok = check_model_availability()['xgboost']
        status_color = BRAND_COLORS['success'] if engine_ok else BRAND_COLORS['accent']
        st.markdown(f"<div class='metric-card'><h4 style='color:{BRAND_COLORS['text_muted']};margin:0;'>System Status</h4><h1 style='margin:10px 0;'>{'Online' if engine_ok else 'Degraded'}</h1><span style='color:{status_color}'>{'XGBoost engine ready' if engine_ok else 'Check dependencies'}</span></div>", unsafe_allow_html=True)

    st.subheader("📋 Recent Intake Queue")
    # REAL: pulled from the actual SQLite database instead of a hard-coded
    # 3-row mock table (which used to show the same 3 fake names regardless
    # of what had actually been assessed).
    try:
        records = database_module.get_all_patients()[:8]
    except Exception:
        records = []

    if records:
        display_rows = []
        for r in records:
            display_rows.append({
                "Patient ID": r.get('patient_id', ''),
                "Logged": (r.get('timestamp') or '')[:19].replace('T', ' '),
                "Patient": r.get('name', ''),
                "Age/Sex": f"{r.get('age','?')} / {r.get('gender','?')}",
                "Mode": r.get('input_method', ''),
                "KTAS Level": r.get('ktas_level', ''),
                "Status": r.get('status', ''),
            })
        st.dataframe(pd.DataFrame(display_rows), width='stretch')
    else:
        st.info("No intakes logged yet this session. Begin a new patient assessment below.")

    if st.button("🚀 Begin New Patient Assessment", type="primary"):
        st.session_state.page = "Patient Assessment"
        st.rerun()

# ===== PAGE 2: PATIENT ASSESSMENT =====
def page_patient_assessment():
    st.title("📝 New Patient Assessment Intake")
    st.markdown("Collect vital signs and narrative symptom reports via conversational interfaces.")

    avail = check_model_availability()

    col1, col2, col3 = st.columns(3)
    with col1:
        name = st.text_input("Patient Full Name", value="John Doe")
        age = st.number_input("Patient Age", min_value=0, max_value=120, value=52)
    with col2:
        gender = st.selectbox("Gender Signature", ["Male", "Female", "Other"])
        pain_scale = st.slider("Subjective Pain Scale (0-10 NRS)", 0, 10, 7)
    with col3:
        # REAL: temperature was previously not collected at all and was
        # silently hard-coded to 36.5 downstream. Now it's a real input.
        temp = st.number_input("Body Temperature (°C)", min_value=30.0, max_value=43.0, value=37.0, step=0.1)

    st.subheader("🫀 Clinical Vital Signs Payload")
    v_col1, v_col2, v_col3 = st.columns(3)
    with v_col1:
        hr = st.number_input("Heart Rate (BPM)", value=98)
    with v_col2:
        sbp = st.number_input("Systolic BP (mmHg)", value=142)
        dbp = st.number_input("Diastolic BP (mmHg)", value=88)
    with v_col3:
        rr = st.number_input("Respiratory Rate (breaths/min)", value=22)

    st.subheader("🗣️ Patient Chief Complaint")
    input_mode = st.radio(
        "Primary Narrative Capture Mode",
        ["Voice Input (Whisper)", "Direct Text Entry"],
        help="Voice input records audio in the browser and transcribes it with Whisper on submission."
    )

    complaint = ""
    audio_bytes = None

    if input_mode == "Voice Input (Whisper)":
        if not avail['whisper']:
            st.warning(
                "⚠️ Whisper/torch aren't installed in this environment, so recorded audio can't be "
                "transcribed here. Install with `pip install -r REBUILD_requirements.txt`, or switch "
                "to Direct Text Entry below."
            )
        try:
            audio_value = st.audio_input("Record the patient's chief complaint")
        except Exception:
            st.error("This Streamlit version doesn't support st.audio_input (needs Streamlit >= 1.40). "
                      "Upgrade streamlit, or use Direct Text Entry.")
            audio_value = None
        if audio_value is not None:
            audio_bytes = audio_value.getvalue()
            st.caption("🎙️ Audio captured — it will be transcribed with Whisper when you submit below.")
        complaint = st.text_area(
            "Transcript will appear here after submission (or type/edit manually)",
            value="", key="voice_complaint_fallback"
        )
    else:
        complaint = st.text_area(
            "Direct Text Narrative",
            value="Patient notes dull abdominal pain over past 12 hours accompanied by minor nausea but no fever."
        )

    submit = st.button("⚡ Forward to Multi-Model AI Pipeline", type="primary")

    if submit:
        st.session_state.current_assessment = {
            "name": name, "age": age, "gender": gender,
            "complaint": complaint,
            "audio_bytes": audio_bytes,
            "vitals": {"heart_rate": hr, "systolic": sbp, "diastolic": dbp,
                       "respiratory_rate": rr, "pain_scale": pain_scale, "temperature": temp},
            "input_mode": input_mode
        }
        st.session_state.pipeline_active = True
        st.session_state.assessment_complete = False
        st.session_state.voice_transcription_error = None
        st.session_state.page = "AI Pipeline"
        st.rerun()

# ===== PAGE 3: AI PIPELINE EXECUTION =====
def page_ai_pipeline():
    st.title("🧬 Multi-Stage AI Pipeline Execution")

    if not st.session_state.current_assessment:
        st.warning("⚠️ No active patient assessment data found. Please run the intake module first.")
        if st.button("Return to Assessment"):
            st.session_state.page = "Patient Assessment"
            st.rerun()
        return

    patient = st.session_state.current_assessment

    # --- STAGE 1: WHISPER (real, conditional on voice input) ---
    st.markdown("<div class='pipeline-stage'><h4>Stage 1: Whisper Automatic Speech Recognition</h4>", unsafe_allow_html=True)
    transcription_info = None
    if patient.get("audio_bytes"):
        with st.spinner("Transcribing recorded audio with Whisper..."):
            text, err = transcribe_audio_bytes(patient["audio_bytes"])
        if err:
            st.error(f"✗ Transcription failed: {err}")
            st.info("Falling back to any manually-typed text for this complaint.")
            st.session_state.voice_transcription_error = err
        else:
            st.success(f'✓ Transcribed: "{text}"')
            patient["complaint"] = text  # real transcript becomes the complaint
            transcription_info = {"transcript": text, "source": "whisper-base"}
    elif patient.get("input_mode") == "Voice Input (Whisper)":
        st.info("No audio was recorded — using manually-typed fallback text for this complaint.")
    else:
        st.info("✓ Text entered directly — no transcription needed.")
    st.markdown("</div>", unsafe_allow_html=True)

    if not patient.get("complaint"):
        st.error("No complaint text is available (no audio transcribed and no text entered). Please go back and provide one.")
        if st.button("Return to Assessment"):
            st.session_state.page = "Patient Assessment"
            st.rerun()
        return

    # --- RUN THE REAL PIPELINE (DistilBERT extraction -> XGBoost -> explainability) ---
    gender_numeric = 1 if patient['gender'].lower() == 'male' else 2

    t0 = time.time()
    with st.spinner("🤖 Running DistilBERT extraction, XGBoost prediction, and explainability..."):
        pipeline_output = orchestrate_models(
            age=int(patient['age']),
            sex=gender_numeric,
            hr=float(patient['vitals']['heart_rate']),
            sbp=float(patient['vitals']['systolic']),
            dbp=float(patient['vitals']['diastolic']),
            rr=float(patient['vitals']['respiratory_rate']),
            temp=float(patient['vitals']['temperature']),
            pain=float(patient['vitals']['pain_scale']),
            complaint_text=patient['complaint'],
            transcription_info=transcription_info,
        )
    elapsed = time.time() - t0
    st.session_state.inference_times.append(elapsed)

    # --- STAGE 2: DISTILBERT (real extracted symptoms) ---
    st.markdown("<div class='pipeline-stage'><h4>Stage 2: DistilBERT / NLP Medical Entity Extraction</h4>", unsafe_allow_html=True)
    distil_out = pipeline_output.get('model_outputs', {}).get('distilbert', {})
    symptoms = distil_out.get('symptoms', [])
    if symptoms:
        st.success(f"✓ Extracted {len(symptoms)} symptom(s): {', '.join(symptoms)}")
        detailed = distil_out.get('detailed_symptoms', {})
        if detailed:
            st.caption(" · ".join(f"{k}: {v:.0%}" for k, v in detailed.items()))
    else:
        st.warning("✓ No specific symptom keywords matched in the complaint text (structured vitals will still drive the prediction).")
    st.markdown("</div>", unsafe_allow_html=True)

    # --- STAGE 3: XGBOOST (real prediction) ---
    st.markdown("<div class='pipeline-stage'><h4>Stage 3: XGBoost KTAS Priority Classification</h4>", unsafe_allow_html=True)
    if pipeline_output.get('ktas_level') is not None:
        st.success(f"✓ Predicted KTAS Level {pipeline_output['ktas_level']} (confidence {pipeline_output['confidence']*100:.1f}%) in {elapsed:.2f}s")
    else:
        st.error(f"✗ Prediction failed: {pipeline_output.get('explanation', {}).get('error', 'unknown error')}")
    st.markdown("</div>", unsafe_allow_html=True)

    # Update global session variables
    st.session_state.ktas_level = pipeline_output.get('ktas_level')
    st.session_state.confidence = pipeline_output.get('confidence')
    st.session_state.explanation = pipeline_output.get('explanation', None)
    st.session_state.model_outputs = pipeline_output.get('model_outputs', None)

    st.session_state.pipeline_active = False
    st.session_state.assessment_complete = st.session_state.ktas_level is not None

    if st.session_state.assessment_complete:
        time.sleep(0.4)
        st.session_state.page = "Assessment Results"
        st.rerun()
    else:
        if st.button("← Back to Assessment"):
            st.session_state.page = "Patient Assessment"
            st.rerun()

# ===== PAGE 4: ASSESSMENT RESULTS =====
def page_assessment_results():
    st.title("🎯 AI Automated Triage Results")

    if not st.session_state.assessment_complete:
        st.warning("⚠️ Run a patient through the AI Pipeline first to see results here.")
        return

    patient = st.session_state.current_assessment
    ktas_level = st.session_state.ktas_level
    confidence_pct = st.session_state.confidence * 100

    color_map = {1: "#FF3366", 2: "#FF5E3A", 3: "#FFD600", 4: "#00E676", 5: "#00D4FF"}
    bg_color = color_map.get(ktas_level, "#182235")

    st.markdown(f"""
    <div style='background-color: {bg_color}; padding: 25px; border-radius: 12px; text-align: center; color: #000; margin-bottom: 25px;'>
        <h1 style='color: #000 !important; margin: 0; font-size: 42px;'>KTAS LEVEL {ktas_level}</h1>
        <p style='margin: 5px 0 0 0; font-weight: bold; font-size: 18px; letter-spacing: 1px;'>
            MODEL CLASSIFICATION CONFIDENCE SCORE: {confidence_pct:.1f}%
        </p>
    </div>
    """, unsafe_allow_html=True)

    uncertainty = (st.session_state.explanation or {}).get('uncertainty', {})
    if uncertainty:
        level = uncertainty.get('level', 'unknown')
        badge_color = {'low': BRAND_COLORS['success'], 'moderate': BRAND_COLORS['warning'], 'high': BRAND_COLORS['accent']}.get(level, BRAND_COLORS['text_muted'])
        muted_color = BRAND_COLORS['text_muted']
        st.markdown(f"<p style='color:{badge_color};font-weight:bold;'>Uncertainty: {level.upper()}</p><p style='color:{muted_color}'>{uncertainty.get('interpretation','')}</p>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("👤 Patient Parameters")
        st.markdown(f"**Name:** {patient['name']}")
        st.markdown(f"**Age / Demographics:** {patient['age']} years old ({patient['gender']})")
        st.markdown(f"**Intake Route:** {patient['input_mode']}")
        st.markdown(f"**Chief Complaint:** *\"{patient['complaint']}\"*")

    with col2:
        st.subheader("🫀 Captured Vitals")
        st.json(patient['vitals'])

    st.divider()

    b_col1, b_col2 = st.columns(2)
    with b_col1:
        if st.button("💾 Save Verified Assessment Log", type="primary", width='stretch'):
            symptoms_list = (st.session_state.model_outputs or {}).get('distilbert', {}).get('symptoms', [])
            try:
                saved_id = database_module.save_patient_record(
                    symptoms=patient['complaint'],
                    detected_symptoms=json.dumps(symptoms_list),
                    ktas_level=int(ktas_level),
                    input_method=patient['input_mode'],
                    name=patient['name'],
                    age=int(patient['age']),
                    gender=patient['gender']
                )
                if saved_id:
                    st.success(f"✅ Patient record saved to data/medtriage.db (ID: {saved_id})")
                else:
                    st.error("❌ Save returned no ID — check database write permissions.")
            except Exception as e:
                st.error(f"❌ Storage layer error: {e}")

    with b_col2:
        if st.button("🔍 Inspect Explainable AI Evidence", width='stretch'):
            st.session_state.page = "Explainable AI"
            st.rerun()

# ===== PAGE 5: EXPLAINABLE AI =====
def page_explainable_ai():
    st.title("🧮 Model Justification & SHAP Explanation")
    st.markdown("Real SHAP feature attributions, uncertainty, and counterfactuals computed from the trained XGBoost model for this specific prediction.")

    if not st.session_state.assessment_complete or not st.session_state.explanation:
        st.info("💡 Run a patient assessment first to see its explanation here.")
        if st.button("Go to Command Center"):
            st.session_state.page = "Command Center"
            st.rerun()
        return

    explanation = st.session_state.explanation

    if explanation.get('error'):
        st.error(f"Explanation unavailable: {explanation['error']}")
        return

    st.subheader("📋 Clinical Evidence (rule-based)")
    evidence = explanation.get('evidence', [])
    if evidence:
        for item in evidence:
            st.markdown(f"- {item}")
    else:
        st.caption("No specific threshold-based evidence flags triggered for this case.")

    st.subheader("⚖️ SHAP Feature Contributions")
    st.caption("Computed by shap.TreeExplainer against the actual trained xgboost.XGBClassifier for this prediction (not a canned table).")
    shap_values = explanation.get('shap_values', [])
    if shap_values:
        df_shap = pd.DataFrame([
            {"Feature": s['feature'], "Value": s['raw_value'], "SHAP Contribution": round(s['shap_value'], 4), "Effect": s['direction']}
            for s in shap_values[:10]
        ])
        st.dataframe(df_shap, width='stretch', hide_index=True)
    else:
        st.warning("SHAP values weren't computed for this prediction (see sidebar for whether the xgboost/shap packages are installed).")

    st.subheader("🎯 Prediction Uncertainty")
    uncertainty = explanation.get('uncertainty', {})
    if uncertainty:
        u_col1, u_col2, u_col3 = st.columns(3)
        u_col1.metric("Margin (top-1 vs top-2)", f"{uncertainty.get('margin',0)*100:.1f}%")
        u_col2.metric("Normalized Entropy", f"{uncertainty.get('normalized_entropy',0):.2f}")
        u_col3.metric("Uncertainty Level", uncertainty.get('level','—').upper())
        st.caption(uncertainty.get('interpretation', ''))

    st.subheader("🔀 Counterfactuals — \"What Would Change This Decision?\"")
    st.caption("Each scenario re-runs the real model with one vital sign perturbed; only clinically relevant, actually-computed scenarios are shown.")
    counterfactuals = explanation.get('counterfactuals', [])
    if counterfactuals:
        for cf in counterfactuals:
            icon = "🔁" if cf['ktas_changed'] else "➖"
            note = f"→ **would change prediction to KTAS {cf['resulting_ktas']}**" if cf['ktas_changed'] else f"→ prediction stays at KTAS {cf['resulting_ktas']} (change alone isn't decisive)"
            st.markdown(f"{icon} {cf['change']} {note}")
    else:
        st.caption("No abnormal vitals met the threshold for a counterfactual scenario in this case (vitals were within, or close to, normal range).")

    if st.button("← Return to Summary Screen", type="primary"):
        st.session_state.page = "Assessment Results"
        st.rerun()

# ===== PAGE 6: SYSTEM STATUS =====
def page_system_status():
    st.title("📡 System Infrastructure Status & Auditing")
    st.markdown("Review relational data storage arrays and global statistics metrics.")

    stats = database_module.get_session_stats()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Historical Records", stats.get('total', 0))
    with col2:
        st.metric("Mean KTAS Level (all records)", f"{stats.get('avg_ktas', 3.0):.2f}")
    with col3:
        st.metric("Storage Path", "data/medtriage.db")

    st.subheader("🗃️ Complete SQLite Database View (`patients` table)")
    all_records = database_module.get_all_patients()

    if all_records:
        df_records = pd.DataFrame(all_records)
        st.dataframe(df_records, width='stretch')

        if st.button("📥 Export Records to CSV"):
            success = database_module.export_to_csv()
            if success:
                st.success("✅ Exported to a CSV file in the project root.")
            else:
                st.error("❌ Export failed — see console for details.")
    else:
        st.info("No saved records in medtriage.db yet. Run and save a patient assessment to populate this table.")

    st.divider()
    st.subheader("🧪 Model Evaluation")
    eval_path = Path("evaluation/evaluation_report.json")
    if eval_path.exists():
        with open(eval_path) as f:
            report = json.load(f)
        st.caption(f"From training/run_evaluation.py, generated {report.get('timestamp','')} against a held-out synthetic test set ({report.get('dataset_size','?')} cases). NOT the Moon et al. (2019) dataset — see training/generate_synthetic_dataset.py for why.")
        m1, m2, m3 = st.columns(3)
        m1.metric("Held-out Accuracy", f"{report.get('overall_accuracy', 0)*100:.1f}%")
        m2.metric("Undertriage Rate", f"{report.get('safety_metrics',{}).get('undertriage_rate',0)*100:.1f}%")
        m3.metric("Overtriage Rate", f"{report.get('safety_metrics',{}).get('overtriage_rate',0)*100:.1f}%")
    else:
        st.info("No evaluation report found yet. Run `python training/run_evaluation.py` to generate real metrics and graphs for Section 5 of the report.")

# ===== MAIN CONTROLLER NAVIGATION ROUTING =====
current_page = st.session_state.page

try:
    if current_page == "Command Center":
        page_command_center()
    elif current_page == "Patient Assessment":
        page_patient_assessment()
    elif current_page == "AI Pipeline":
        page_ai_pipeline()
    elif current_page == "Assessment Results":
        page_assessment_results()
    elif current_page == "Explainable AI":
        page_explainable_ai()
    elif current_page == "System Status":
        page_system_status()
    else:
        st.session_state.page = "Command Center"
        st.rerun()
except Exception as e:
    st.error(f"Unhandled error while rendering '{current_page}': {e}")
    if st.button("Return to Command Center"):
        st.session_state.page = "Command Center"
        st.rerun()
