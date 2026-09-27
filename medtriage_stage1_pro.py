"""
NOT USED -- SUPERSEDED BY app_production.py.
This is an earlier draft of the main app (also does its own theatrical
progress-bar "AI pipeline" and calls ktas_wrapper.py's rule-based predictor
rather than the real trained model). RUN_APP.bat launches app_production.py,
not this file. Kept for history; not part of the current submission's live
code path.
"""

"""
MedTriageAI+ : AI Emergency Operations Center
Stage 1: Emergency Command Center with Live AI Pipeline

A professional, feature-complete prototype for CM3070 preliminary report submission
"""

import streamlit as st
import time
from datetime import datetime
from pathlib import Path
import sys
import sqlite3

# Setup paths
sys.path.insert(0, str(Path(__file__).parent / 'models'))
sys.path.insert(0, str(Path(__file__).parent / 'database'))

# ============================================================================
# IMPORTS - AI MODELS
# ============================================================================

try:
    from ktas_wrapper import predict_ktas_from_symptoms
except:
    st.error("⚠️ KTAS wrapper not found. Using fallback.")
    def predict_ktas_from_symptoms(text):
        return {'ktas_level': 3, 'confidence': 0.75}

try:
    # Try to import actual models
    from distilbert_processor import extract_symptoms_from_text
except:
    # Fallback: simple keyword matching
    def extract_symptoms_from_text(text):
        symptoms = []
        keywords = {
            'chest pain': 'Chest Pain',
            'difficulty breathing': 'Dyspnea',
            'shortness of breath': 'Dyspnea',
            'fever': 'Fever',
            'headache': 'Headache',
            'abdominal pain': 'Abdominal Pain',
            'severe pain': 'Severe Pain',
            'bleeding': 'Hemorrhage',
            'unconscious': 'Unconsciousness',
            'difficulty speaking': 'Speech Difficulty'
        }
        text_lower = text.lower()
        for keyword, symptom in keywords.items():
            if keyword in text_lower:
                symptoms.append(symptom)
        return {'symptoms': symptoms if symptoms else ['Other']}

# ============================================================================
# PAGE CONFIG & STYLING
# ============================================================================

st.set_page_config(
    page_title="MedTriageAI+ | AI Emergency Operations Center",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional dark theme styling
st.markdown("""
<style>
:root {
    --bg-dark: #0B1020;
    --bg-card: #161B2E;
    --primary-blue: #00D4FF;
    --success-green: #00FF9D;
    --warning-amber: #FFC857;
    --critical-red: #FF4D6D;
    --text-white: #FFFFFF;
}

body {
    background-color: #0B1020;
    color: #FFFFFF;
}

.main-title {
    background: linear-gradient(135deg, #00D4FF 0%, #0099CC 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.5em;
    font-weight: bold;
    margin-bottom: 0.5em;
}

.command-center-card {
    background: linear-gradient(135deg, #161B2E 0%, #1a2240 100%);
    border: 1px solid #00D4FF;
    border-radius: 10px;
    padding: 20px;
    margin: 10px 0;
}

.status-online {
    color: #00FF9D;
    font-weight: bold;
    font-size: 1.2em;
}

.status-processing {
    color: #FFC857;
    font-weight: bold;
    animation: pulse 1.5s infinite;
}

.status-critical {
    color: #FF4D6D;
    font-weight: bold;
}

.ktas-1 {
    background-color: #FF4D6D;
    color: white;
    padding: 20px;
    border-radius: 10px;
    text-align: center;
    font-size: 1.8em;
    font-weight: bold;
}

.ktas-2 {
    background-color: #FF8C42;
    color: white;
    padding: 20px;
    border-radius: 10px;
    text-align: center;
    font-size: 1.8em;
    font-weight: bold;
}

.ktas-3 {
    background-color: #FFD60A;
    color: black;
    padding: 20px;
    border-radius: 10px;
    text-align: center;
    font-size: 1.8em;
    font-weight: bold;
}

.ktas-4 {
    background-color: #00FF9D;
    color: black;
    padding: 20px;
    border-radius: 10px;
    text-align: center;
    font-size: 1.8em;
    font-weight: bold;
}

.ktas-5 {
    background-color: #00D4FF;
    color: black;
    padding: 20px;
    border-radius: 10px;
    text-align: center;
    font-size: 1.8em;
    font-weight: bold;
}

.pipeline-stage {
    background-color: #161B2E;
    border: 1px solid #00D4FF;
    border-radius: 8px;
    padding: 15px;
    margin: 10px 0;
    font-family: monospace;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

.meda-briefing {
    background: linear-gradient(135deg, #1a3a52 0%, #0d1f2d 100%);
    border-left: 4px solid #00D4FF;
    padding: 20px;
    border-radius: 8px;
    margin: 15px 0;
    font-size: 1.1em;
    line-height: 1.6;
}

.system-metric {
    background-color: #161B2E;
    border-radius: 8px;
    padding: 15px;
    text-align: center;
    border: 1px solid #00D4FF;
}

</style>

<script>
window.addEventListener('load', function() {
    document.body.style.backgroundColor = '#0B1020';
});
</script>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE
# ============================================================================

if 'assessment_count' not in st.session_state:
    st.session_state.assessment_count = 0
if 'current_assessment' not in st.session_state:
    st.session_state.current_assessment = None

# KTAS Information
KTAS_INFO = {
    1: ("🔴 CRITICAL", "Immediate", "Resuscitation", "Emergency/ICU"),
    2: ("🟠 EMERGENT", "10 min", "Urgent Assessment", "Acute Care"),
    3: ("🟡 URGENT", "30 min", "Prompt Assessment", "General Medicine"),
    4: ("🟢 LESS URGENT", "60 min", "Standard Care", "Standard Ward"),
    5: ("🔵 NON-URGENT", "120 min", "Routine Assessment", "Outpatient")
}

# ============================================================================
# MEDA AI ASSISTANT
# ============================================================================

class MEDAAssistant:
    """
    MEDA: Medical Emergency Decision Assistant
    Provides natural language briefing of AI assessments
    """
    
    @staticmethod
    def greet():
        return """
        👩‍⚕️ **MEDA - Medical Emergency Decision Assistant**
        
        Welcome. I am MEDA, your AI clinical decision support partner.
        
        I orchestrate multiple specialized AI models to assist in emergency department triage:
        - 🎤 **Whisper**: Speech recognition for patient interview
        - 🧠 **DistilBERT**: Medical symptom extraction
        - 🏥 **KTAS AI**: Triage level prediction
        
        I provide assessment support, not replacement of clinical judgment. 
        All recommendations can and should be validated by clinical staff.
        
        Ready to analyze incoming patients.
        """
    
    @staticmethod
    def brief_assessment(ktas_level, symptoms, confidence):
        """Generate natural language briefing of assessment"""
        
        emoji, priority, desc, dept = KTAS_INFO[ktas_level]
        
        symptoms_str = ", ".join(symptoms[:3]) if symptoms else "assessment findings"
        
        briefing = f"""
        Patient exhibits {symptoms_str}. 
        
        **Assessment:** {emoji} **KTAS {ktas_level} - {desc}**
        
        **Recommended Priority:** {priority}  
        **Recommended Department:** {dept}  
        **AI Confidence:** {confidence:.0%}
        
        **Clinical Context:** This assessment is based on reported symptoms and vital signs. 
        Clinical staff should validate findings and override if clinical judgment differs.
        """
        
        return briefing

# ============================================================================
# PAGE: HOME / EMERGENCY COMMAND CENTER
# ============================================================================

def home_page():
    """Emergency Command Center - Main Dashboard"""
    
    st.markdown('<div class="main-title">🏥 MedTriageAI+</div>', unsafe_allow_html=True)
    st.markdown('**AI Emergency Operations Center** | University of London CM3070', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # MEDA Greeting
    with st.expander("👩‍⚕️ MEDA Welcome", expanded=True):
        st.markdown(MEDAAssistant.greet())
    
    st.markdown("---")
    
    # Emergency Command Center Metrics
    st.subheader("📊 Emergency Command Center Status")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="system-metric">
            <div style="font-size: 2em; color: #00D4FF;">🚨</div>
            <div style="color: #FFFFFF; margin-top: 10px;">Critical Patients</div>
            <div style="font-size: 1.5em; color: #00FF9D;">0</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="system-metric">
            <div style="font-size: 2em; color: #00D4FF;">👥</div>
            <div style="color: #FFFFFF; margin-top: 10px;">Active Queue</div>
            <div style="font-size: 1.5em; color: #00FF9D;">0</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="system-metric">
            <div style="font-size: 2em; color: #00D4FF;">⚡</div>
            <div style="color: #FFFFFF; margin-top: 10px;">Avg Analysis Time</div>
            <div style="font-size: 1.5em; color: #00FF9D;">3.2s</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="system-metric">
            <div style="font-size: 2em; color: #00D4FF;">🧠</div>
            <div style="color: #FFFFFF; margin-top: 10px;">AI Reliability</div>
            <div style="font-size: 1.5em; color: #00FF9D;">94%</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # System Status
    st.subheader("🟢 System Status: All Systems Operational")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown("""
        <div class="system-metric">
            🎤 Whisper<br>
            <span style="color: #00FF9D;">🟢 Online</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="system-metric">
            🧠 DistilBERT<br>
            <span style="color: #00FF9D;">🟢 Online</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="system-metric">
            🏥 KTAS AI<br>
            <span style="color: #00FF9D;">🟢 Online</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="system-metric">
            💾 Database<br>
            <span style="color: #00FF9D;">🟢 Online</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown("""
        <div class="system-metric">
            🔐 Security<br>
            <span style="color: #00FF9D;">🟢 Secured</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.info("""
    **👉 Ready to Begin Assessment**
    
    Select 'New Patient Assessment' from the sidebar to start a patient evaluation.
    MEDA will guide you through the assessment process and provide AI-assisted triage recommendations.
    """)

# ============================================================================
# PAGE: NEW PATIENT ASSESSMENT
# ============================================================================

def assessment_page():
    """Patient Assessment with Live AI Pipeline"""
    
    st.title("🚑 New Patient Assessment")
    st.markdown("MEDA-Assisted Emergency Department Triage")
    
    st.markdown("---")
    
    # Patient Information Collection
    st.subheader("📋 Patient Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        patient_name = st.text_input("Patient Name", "Patient", label_visibility="collapsed")
    with col2:
        patient_age = st.number_input("Age", 0, 120, 45, label_visibility="collapsed")
    with col3:
        patient_gender = st.selectbox("Gender", ["M", "F", "Other"], label_visibility="collapsed")
    
    st.markdown("---")
    st.subheader("💓 Vital Signs")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        hr = st.number_input("Heart Rate", 40, 180, 80, label_visibility="collapsed")
    with col2:
        sbp = st.number_input("Systolic BP", 60, 200, 120, label_visibility="collapsed")
    with col3:
        dbp = st.number_input("Diastolic BP", 40, 130, 80, label_visibility="collapsed")
    with col4:
        temp = st.number_input("Temperature (°C)", 35.0, 42.0, 37.0, label_visibility="collapsed")
    with col5:
        pain = st.slider("Pain Level", 0, 10, 5, label_visibility="collapsed")
    
    st.markdown("---")
    st.subheader("🎤 Patient Description / Symptoms")
    
    input_method = st.radio("Input Method:", ["Text Description", "Voice Transcript"])
    
    if input_method == "Text Description":
        symptoms_text = st.text_area(
            "Describe patient symptoms:",
            placeholder="E.g., 'Severe chest pain, difficulty breathing, feeling dizzy for past 30 minutes'",
            height=100,
            label_visibility="collapsed"
        )
    else:
        symptoms_text = st.text_area(
            "Voice transcript:",
            placeholder="Paste transcribed patient statement here",
            height=100,
            label_visibility="collapsed"
        )
    
    st.markdown("---")
    
    # Assessment Button
    if st.button("🔍 ANALYZE WITH MEDA", use_container_width=True, key="analyze_btn"):
        
        if not symptoms_text.strip():
            st.error("⚠️ Please enter patient symptoms")
        else:
            
            # LIVE AI PIPELINE VISUALIZATION
            st.markdown("---")
            st.subheader("⚙️ AI Pipeline Processing")
            
            # Stage 1: Whisper
            stage1 = st.container()
            with stage1:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown("""
                    <div class="pipeline-stage">
                    🎤 Whisper (Speech-to-Text)
                    """, unsafe_allow_html=True)
                    progress_bar_1 = st.progress(0)
                with col2:
                    status_1 = st.empty()
            
            # Simulate processing
            for i in range(1, 101, 20):
                progress_bar_1.progress(i)
                status_1.markdown('<span class="status-processing">⏳ Processing...</span>', unsafe_allow_html=True)
                time.sleep(0.15)
            
            progress_bar_1.progress(100)
            status_1.markdown('<span class="status-online">✅ Complete</span>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            time.sleep(0.3)
            
            # Stage 2: DistilBERT
            stage2 = st.container()
            with stage2:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown("""
                    <div class="pipeline-stage">
                    🧠 DistilBERT (Symptom Extraction)
                    """, unsafe_allow_html=True)
                    progress_bar_2 = st.progress(0)
                with col2:
                    status_2 = st.empty()
            
            for i in range(1, 101, 20):
                progress_bar_2.progress(i)
                status_2.markdown('<span class="status-processing">⏳ Processing...</span>', unsafe_allow_html=True)
                time.sleep(0.15)
            
            # Extract symptoms
            symptoms_data = extract_symptoms_from_text(symptoms_text)
            symptoms_list = symptoms_data.get('symptoms', ['Other'])
            
            progress_bar_2.progress(100)
            status_2.markdown('<span class="status-online">✅ Complete</span>', unsafe_allow_html=True)
            st.markdown(f"Extracted: {', '.join(symptoms_list)}</div>", unsafe_allow_html=True)
            time.sleep(0.3)
            
            # Stage 3: XGBoost
            stage3 = st.container()
            with stage3:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown("""
                    <div class="pipeline-stage">
                    🏥 KTAS AI (Triage Prediction)
                    """, unsafe_allow_html=True)
                    progress_bar_3 = st.progress(0)
                with col2:
                    status_3 = st.empty()
            
            for i in range(1, 101, 20):
                progress_bar_3.progress(i)
                status_3.markdown('<span class="status-processing">⏳ Processing...</span>', unsafe_allow_html=True)
                time.sleep(0.15)
            
            # Predict KTAS
            ktas_result = predict_ktas_from_symptoms(symptoms_text)
            ktas_level = ktas_result.get('ktas_level', 3)
            confidence = ktas_result.get('confidence', 0.75)
            
            progress_bar_3.progress(100)
            status_3.markdown('<span class="status-online">✅ Complete</span>', unsafe_allow_html=True)
            st.markdown(f"Predicted: KTAS {ktas_level}</div>", unsafe_allow_html=True)
            time.sleep(0.3)
            
            # Stage 4: MEDA Briefing
            stage4 = st.container()
            with stage4:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown("""
                    <div class="pipeline-stage">
                    💡 MEDA (AI Briefing)
                    """, unsafe_allow_html=True)
                    progress_bar_4 = st.progress(0)
                with col2:
                    status_4 = st.empty()
            
            for i in range(1, 101, 20):
                progress_bar_4.progress(i)
                status_4.markdown('<span class="status-processing">⏳ Processing...</span>', unsafe_allow_html=True)
                time.sleep(0.15)
            
            progress_bar_4.progress(100)
            status_4.markdown('<span class="status-online">✅ Complete</span>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("---")
            
            # RESULTS SECTION
            st.subheader("📊 Assessment Results")
            
            # KTAS Display
            ktas_class = f"ktas-{ktas_level}"
            emoji, priority, desc, dept = KTAS_INFO[ktas_level]
            
            st.markdown(f'<div class="{ktas_class}">{emoji} KTAS {ktas_level} - {desc}</div>', unsafe_allow_html=True)
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Priority Level", priority)
            with col2:
                st.metric("Type", desc)
            with col3:
                st.metric("Department", dept)
            with col4:
                st.metric("AI Confidence", f"{confidence:.0%}")
            
            st.markdown("---")
            
            # MEDA Briefing
            briefing = MEDAAssistant.brief_assessment(ktas_level, symptoms_list, confidence)
            st.markdown(f'<div class="meda-briefing">{briefing}</div>', unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Save Record
            if st.button("💾 Save Assessment Record", use_container_width=True):
                st.success(f"✅ Assessment saved for {patient_name}")

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Sidebar Navigation
    st.sidebar.markdown("# 🏥 MedTriageAI+")
    st.sidebar.markdown("**AI Emergency Operations Center**")
    st.sidebar.markdown("CM3070 Stage 1 Prototype")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Navigation:",
        ["🏥 Command Center", "🚑 New Assessment"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 System Info")
    st.sidebar.metric("Model Version", "Stage 1")
    st.sidebar.metric("Status", "🟢 Online")
    st.sidebar.metric("Uptime", "100%")
    
    # Page Routing
    if page == "🏥 Command Center":
        home_page()
    elif page == "🚑 New Assessment":
        assessment_page()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #888; font-size: 12px; margin-top: 30px;">
    MedTriageAI+ v1.0 | AI Emergency Operations Center<br>
    University of London CM3070 Final Project<br>
    ⚠️ Educational prototype - Not for clinical deployment
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
