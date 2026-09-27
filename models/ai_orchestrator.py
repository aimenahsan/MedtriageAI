"""
AI Orchestrator Module

Coordinates the orchestration of three pre-trained AI models:
1. Whisper (Speech-to-Text)       -- run upstream, before this module, since
                                      typed text needs no transcription. See
                                      transcribe_patient_audio() in
                                      whisper_handler.py, called from
                                      app_production.py's Patient Assessment page.
2. DistilBERT (Medical NLP)       -- run here, on the resulting text
3. XGBoost (KTAS Prediction)      -- run here, on structured features
Plus explainability (SHAP + uncertainty + counterfactuals), also run here.

This is the core AI agent that makes decisions under uncertainty.

CHANGE NOTE (previous version): this module used to receive an already-built
`symptoms` dict from app_production.py, which -- due to a bug at the call
site -- actually contained the entire raw complaint sentence as a single
list item (`{'symptoms': [full_sentence], ...}`) rather than extracted
symptom keywords. Since ktas_predictor checks for exact strings like
'chest pain' in that list, every symptom flag was silently always 0,
regardless of what the patient said. This version calls
extract_symptoms_from_text() itself, so the orchestrator owns the full
text -> structured decision pipeline as the report describes, and the bug
class above isn't possible from the call site.
"""

import streamlit as st
from models.distilbert_processor import extract_symptoms_from_text
from models.ktas_predictor import predict_ktas_level, load_ktas_models
from models.shap_explainer import compute_shap_values, compute_uncertainty, generate_counterfactuals


def orchestrate_models(age, sex, hr, sbp, dbp, rr, temp, pain, complaint_text, transcription_info=None):
    """
    Orchestrate all three AI models to make a triage decision.

    This is the core AI agent algorithm.

    Pipeline:
    1. Patient speaks (or types) symptoms -> complaint_text (+ optional
       transcription_info if Whisper was used upstream)
    2. DistilBERT extracts medical entities from complaint_text
    3. XGBoost predicts KTAS level from structured vitals + entities
    4. SHAP + uncertainty + counterfactuals explain the decision

    Args:
        age, sex, hr, sbp, dbp, rr, temp, pain: structured vitals
        complaint_text: patient's chief complaint, as text (from typing, or
                         from Whisper transcription -- this module doesn't
                         care which)
        transcription_info: optional dict from whisper_handler, passed
                             through into model_outputs for display/audit
                             purposes only (not used in the prediction)

    Returns:
        dict: {
            'ktas_level': 1-5,
            'confidence': 0.0-1.0,
            'explanation': {
                'shap_values': [...],
                'uncertainty': {...},
                'counterfactuals': [...],
                'evidence': [...],           # rule-based clinical evidence strings
            },
            'model_outputs': individual model outputs, for the pipeline/audit views
        }
    """
    try:
        # ----- Stage 2: DistilBERT / NLP symptom extraction -----
        symptoms = extract_symptoms_from_text(complaint_text)

        # ----- Stage 3: XGBoost KTAS prediction -----
        result = predict_ktas_level(
            age=age, sex=sex, hr=hr, sbp=sbp, dbp=dbp, rr=rr, temp=temp, pain=pain,
            symptoms_dict=symptoms
        )

        if result['error']:
            return {
                'ktas_level': None,
                'confidence': 0.0,
                'explanation': {'error': result['error']},
                'model_outputs': {}
            }

        # ----- Stage 4: real explainability -----
        classifier, scaler, load_err = load_ktas_models()
        shap_values, uncertainty, counterfactuals = [], {}, []
        if not load_err:
            try:
                shap_values = compute_shap_values(
                    classifier,
                    result['feature_values'], result['feature_names'],
                    result['features_scaled'], result['ktas_level'] - 1
                )
                uncertainty = compute_uncertainty(result['probabilities'])
                counterfactuals = generate_counterfactuals(
                    classifier, scaler,
                    result['feature_values'], result['feature_names'],
                    result['ktas_level']
                )
            except Exception as explain_err:
                # A failure to explain shouldn't take down a working prediction --
                # surface it, but still return the (real) prediction above.
                st.warning(f"Explainability computation partially failed: {explain_err}")

        return {
            'ktas_level': result['ktas_level'],
            'confidence': result['confidence'],
            'explanation': {
                'shap_values': shap_values,
                'uncertainty': uncertainty,
                'counterfactuals': counterfactuals,
                'evidence': result['evidence'],
            },
            'model_outputs': {
                'whisper': transcription_info,  # None if text was typed directly
                'distilbert': {
                    'symptoms': symptoms.get('symptoms', []),
                    'symptom_count': symptoms.get('symptom_count', 0),
                    'confidence': symptoms.get('confidence', 0.0),
                    'detailed_symptoms': symptoms.get('detailed_symptoms', {}),
                },
                'xgboost': {
                    'ktas_level': result['ktas_level'],
                    'confidence': result['confidence'],
                    'probabilities': result.get('probabilities', []),
                    'evidence': result['evidence']
                }
            }
        }

    except Exception as e:
        return {
            'ktas_level': None,
            'confidence': 0.0,
            'explanation': {'error': f"Orchestration error: {str(e)}"},
            'model_outputs': {}
        }

def evaluate_orchestration_strategy():
    """
    Document the orchestration strategy for Distinction-Level evaluation
    
    Returns:
        dict: Explanation of orchestration approach
    """
    return {
        'name': 'Sequential Pipeline Orchestration',
        'description': 'Three pre-trained models operating in sequence, each adding value',
        'pipeline': [
            {
                'stage': 1,
                'model': 'Whisper',
                'input': 'Audio (patient voice)',
                'output': 'Text transcription',
                'role': 'Data acquisition'
            },
            {
                'stage': 2,
                'model': 'DistilBERT',
                'input': 'Text description',
                'output': 'Extracted symptoms (structured)',
                'role': 'Data preprocessing/structuring'
            },
            {
                'stage': 3,
                'model': 'XGBoost',
                'input': 'Age, vitals, symptoms',
                'output': 'KTAS level (1-5)',
                'role': 'Decision making'
            }
        ],
        'design_principles': [
            'Respects data flow: raw → structured → decision',
            'Each model is appropriate for its task',
            'Together they solve the overall goal (triage)',
            'Interpretable at each stage',
            'Handles uncertainty with confidence scores'
        ],
        'why_not_monolithic': [
            'Single model would require end-to-end training (requires large labeled audio dataset)',
            'Separate models allow using pre-trained models (no training required)',
            'More interpretable (can see what each model does)',
            'More maintainable (can replace individual models)',
            'Better for restricted 8GB RAM environment'
        ]
    }
