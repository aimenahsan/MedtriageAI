"""
Model 3: XGBoost Classifier
KTAS (Korean Triage and Acuity Scale) Level Prediction

Trained on: a procedurally-generated, clinically-grounded synthetic dataset
(see training/generate_synthetic_dataset.py). This is NOT the Moon et al.
(2019) dataset -- no real patient-level data ships with this project. If you
obtain the real dataset, retrain with training/train_model.py against it and
the numbers below will update automatically.
Purpose: Predict emergency department triage level (KTAS 1-5)

NOTE ON LABEL ENCODING: xgboost's sklearn API requires 0-indexed contiguous
class labels, so the model is trained on classes [0..4] internally
representing KTAS [1..5]. `predict_ktas_level()` below converts back to the
1-5 scale used everywhere else in the app -- don't call classifier.predict()
directly elsewhere without adding 1.
"""

import json
import pickle
import numpy as np
import streamlit as st
import os

@st.cache_resource
def load_ktas_models():
    """
    Load pre-trained XGBoost classifier and feature scaler

    Returns:
        tuple: (classifier, scaler, error_message)

    See get_real_evaluation_metrics() below for actual measured performance
    (computed by training/run_evaluation.py against a held-out test split,
    not hand-written).
    """
    try:
        # Check if model files exist
        model_path = 'data/classifier_model.pkl'
        scaler_path = 'data/classifier_scaler.pkl'
        
        if not os.path.exists(model_path):
            return None, None, f"Model file not found: {model_path}"
        if not os.path.exists(scaler_path):
            return None, None, f"Scaler file not found: {scaler_path}"
        
        # Load models
        with open(model_path, 'rb') as f:
            classifier = pickle.load(f)
        
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        return classifier, scaler, None
    
    except Exception as e:
        return None, None, str(e)

def predict_ktas_level(age, sex, hr, sbp, dbp, rr, temp, pain, symptoms_dict):
    """
    Predict KTAS level using XGBoost classifier
    
    Args:
        age: Patient age (1-120)
        sex: Gender (1=Male, 2=Female)
        hr: Heart rate (40-200 bpm)
        sbp: Systolic blood pressure (70-250 mmHg)
        dbp: Diastolic blood pressure (40-150 mmHg)
        rr: Respiratory rate (8-50)
        temp: Temperature (35-42°C)
        pain: Pain level (0-10)
        symptoms_dict: Output from DistilBERT processor
    
    Returns:
        dict: {
            'ktas_level': 1-5,
            'confidence': 0.0-1.0,
            'evidence': reasons for prediction,
            'error': error message (if any)
        }
    """
    classifier, scaler, error = load_ktas_models()
    
    if error:
        return {
            'ktas_level': None,
            'confidence': 0.0,
            'evidence': [],
            'error': error
        }
    
    try:
        # Prepare features (21 total, must match training data column order
        # in training/generate_synthetic_dataset.py::FEATURE_NAMES exactly)
        feature_names = ['Age', 'Sex', 'HR', 'SBP', 'DBP', 'RR', 'BT', 'NRS_pain', 'symptom_count',
            'has_chest_pain', 'has_shortness_of_breath', 'has_abdominal_pain', 'has_fever',
            'has_headache', 'has_dizziness', 'has_nausea', 'has_vomiting', 'has_weakness',
            'has_syncope', 'has_confusion', 'has_laceration']

        detected = symptoms_dict.get('symptoms', [])
        feature_values = [
            age,                                              # 0
            sex,                                              # 1
            int(hr),                                          # 2
            int(sbp),                                         # 3
            int(dbp),                                         # 4
            int(rr),                                          # 5
            float(temp),                                      # 6
            int(pain),                                        # 7
            symptoms_dict.get('symptom_count', 0),           # 8
            1 if 'chest pain' in detected else 0,             # 9
            1 if 'shortness of breath' in detected else 0,   # 10
            1 if 'abdominal pain' in detected else 0,         # 11
            1 if 'fever' in detected else 0,                  # 12
            1 if 'headache' in detected else 0,               # 13
            1 if 'dizziness' in detected else 0,              # 14
            1 if 'nausea' in detected else 0,                 # 15
            1 if 'vomiting' in detected else 0,               # 16
            1 if 'weakness' in detected else 0,               # 17
            1 if 'syncope' in detected else 0,                # 18
            1 if 'confusion' in detected else 0,              # 19
            1 if 'laceration' in detected else 0              # 20
        ]

        import pandas as pd
        features_df = pd.DataFrame([feature_values], columns=feature_names)

        # Scale features
        features_scaled = scaler.transform(features_df)

        # Predict. IMPORTANT: xgboost's sklearn API only accepts 0-indexed
        # contiguous class labels, so the model was trained on classes
        # [0..4] representing KTAS [1..5] (see training/train_model.py).
        # raw_prediction is therefore 0-4; add 1 to get the real KTAS level.
        raw_prediction = int(classifier.predict(features_scaled)[0])
        prediction = raw_prediction + 1

        # Get confidence. probabilities[i] is P(class i), i.e. P(KTAS i+1),
        # so index with raw_prediction (0-indexed), not the KTAS level.
        probabilities = classifier.predict_proba(features_scaled)[0]
        confidence = float(probabilities[raw_prediction])

        # Build evidence
        evidence = collect_evidence(age, hr, sbp, dbp, rr, temp, pain, symptoms_dict)

        return {
            'ktas_level': prediction,
            'confidence': confidence,
            'evidence': evidence,
            'error': None,
            'probabilities': probabilities.tolist(),  # index i = P(KTAS i+1)
            'feature_values': feature_values,          # raw (unscaled) features, for explainability
            'feature_names': feature_names,
            'features_scaled': features_scaled[0].tolist(),
        }
    
    except Exception as e:
        return {
            'ktas_level': None,
            'confidence': 0.0,
            'evidence': [],
            'error': str(e)
        }

def collect_evidence(age, hr, sbp, dbp, rr, temp, pain, symptoms_dict):
    """
    Collect evidence supporting the KTAS prediction
    Used for explainability
    
    Returns:
        list: Evidence items
    """
    evidence = []
    
    # Check abnormal vitals
    if hr > 100:
        evidence.append(f"Tachycardia: {hr} bpm (>100)")
    if hr < 60:
        evidence.append(f"Bradycardia: {hr} bpm (<60)")
    
    if sbp > 160 or sbp < 90:
        evidence.append(f"Abnormal systolic BP: {sbp} mmHg")
    
    if temp > 38.5:
        evidence.append(f"High fever: {temp}°C")
    if temp < 36:
        evidence.append(f"Hypothermia: {temp}°C")
    
    if rr > 24:
        evidence.append(f"Tachypnea: {rr} breaths/min")
    
    if pain >= 8:
        evidence.append(f"Severe pain: {pain}/10")
    
    # High-risk symptoms
    high_risk = ['chest pain', 'shortness of breath', 'syncope', 'confusion']
    detected_high_risk = [s for s in high_risk if s in symptoms_dict.get('symptoms', [])]
    if detected_high_risk:
        evidence.append(f"High-risk symptoms: {', '.join(detected_high_risk)}")
    
    # Age-related factors
    if age > 65:
        evidence.append(f"Older patient: {age} years")
    
    return evidence if evidence else ["Routine presentation"]

def evaluate_xgboost_model():
    """
    Evaluation results for the XGBoost model.

    These numbers are read from evaluation/evaluation_report.json, which is
    generated by training/run_evaluation.py from real predictions against a
    held-out synthetic test set (training/test_split.csv) -- not hand-typed.
    Run `python training/run_evaluation.py` (after train_model.py) to
    (re)generate that file. If it doesn't exist yet, this returns a
    'not_yet_evaluated' flag instead of silently guessing numbers.

    Returns:
        dict: Performance metrics and study details
    """
    report_path = os.path.join('evaluation', 'evaluation_report.json')
    if not os.path.exists(report_path):
        return {
            'status': 'not_yet_evaluated',
            'message': 'Run training/run_evaluation.py to generate real metrics.'
        }

    with open(report_path, 'r') as f:
        report = json.load(f)

    return {
        'status': 'ok',
        'model_name': 'XGBoost Classifier (xgboost.XGBClassifier)',
        'training_data': 'Procedurally-generated synthetic KTAS dataset (see training/generate_synthetic_dataset.py) -- NOT the Moon et al. (2019) dataset',
        'dataset_size': report.get('dataset_size', 'unknown'),
        'features': 21,
        'classes': 5,
        'test_accuracy': report.get('overall_accuracy'),
        'cross_validation': report.get('cross_validation'),
        'per_class_accuracy': report.get('per_class_accuracy'),
        'safety_metrics': report.get('safety_metrics'),
        'confidence_calibration': report.get('confidence_calibration'),
        'generated_at': report.get('timestamp'),
        'why_chosen': 'Gradient-boosted trees are a strong, fast, well-understood baseline for structured/tabular clinical data, and xgboost specifically has first-class support for exact SHAP explanations via shap.TreeExplainer.'
    }
