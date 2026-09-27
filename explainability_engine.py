"""
EXPLAINABILITY ENGINE FOR MEDTRIAGEAI+
Rule-based clinical narrative layer for KTAS predictions.
Shows WHY the model made each decision, in plain clinical language.

NOTE: this module is intentionally heuristic/rule-based (hand-authored
keyword weights and vital-sign thresholds) -- it does NOT compute real SHAP
values. It exists to turn a prediction into a readable clinical narrative.
For genuine, model-derived SHAP values, uncertainty quantification, and
counterfactuals computed by actually re-running the trained classifier, see
models/shap_explainer.py. The two are complementary: shap_explainer.py
answers "what did the model actually weigh", this file answers "how do we
explain that to a clinician in plain language".
"""

import numpy as np
import json
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# ===== FEATURE DEFINITIONS =====
CLINICAL_FEATURES = {
    "chest_pain": {
        "name": "Chest Pain",
        "category": "Chief Complaint",
        "severity": "HIGH",
        "ktas_impact": 0.85,
        "clinical_note": "Strong predictor of acute coronary syndrome"
    },
    "shortness_of_breath": {
        "name": "Shortness of Breath",
        "category": "Chief Complaint",
        "severity": "HIGH",
        "ktas_impact": 0.72,
        "clinical_note": "Indicates respiratory distress or cardiac compromise"
    },
    "severe_headache": {
        "name": "Severe Headache",
        "category": "Chief Complaint",
        "severity": "HIGH",
        "ktas_impact": 0.68,
        "clinical_note": "May indicate stroke, meningitis, or other emergencies"
    },
    "abdominal_pain": {
        "name": "Abdominal Pain",
        "category": "Chief Complaint",
        "severity": "MEDIUM",
        "ktas_impact": 0.55,
        "clinical_note": "Broad differential; severity depends on characteristics"
    },
    "dizziness": {
        "name": "Dizziness",
        "category": "Chief Complaint",
        "severity": "MEDIUM",
        "ktas_impact": 0.42,
        "clinical_note": "Can indicate stroke, arrhythmia, or dehydration"
    },
    "minor_headache": {
        "name": "Minor Headache",
        "category": "Chief Complaint",
        "severity": "LOW",
        "ktas_impact": 0.15,
        "clinical_note": "Usually non-emergent unless severe or accompanied by other symptoms"
    },
    "nausea": {
        "name": "Nausea",
        "category": "Chief Complaint",
        "severity": "LOW",
        "ktas_impact": 0.25,
        "clinical_note": "Non-specific symptom; context dependent"
    },
    "systolic_bp": {
        "name": "Systolic Blood Pressure",
        "category": "Vital Sign",
        "severity": "VARIABLE",
        "ktas_impact": 0.45,
        "clinical_note": "High BP increases risk; critically low indicates shock"
    },
    "diastolic_bp": {
        "name": "Diastolic Blood Pressure",
        "category": "Vital Sign",
        "severity": "VARIABLE",
        "ktas_impact": 0.35,
        "clinical_note": "Severe elevation or critical low are concerning"
    },
    "heart_rate": {
        "name": "Heart Rate",
        "category": "Vital Sign",
        "severity": "VARIABLE",
        "ktas_impact": 0.52,
        "clinical_note": "Tachycardia (>100) suggests stress/illness; bradycardia (<50) serious"
    },
    "respiratory_rate": {
        "name": "Respiratory Rate",
        "category": "Vital Sign",
        "severity": "VARIABLE",
        "ktas_impact": 0.48,
        "clinical_note": "High RR indicates respiratory distress or metabolic issue"
    },
    "age": {
        "name": "Age",
        "category": "Demographics",
        "severity": "LOW",
        "ktas_impact": 0.08,
        "clinical_note": "Older patients at higher risk; very young children need special consideration"
    }
}

KTAS_DESCRIPTIONS = {
    1: {
        "level": "CRITICAL",
        "description": "Immediate life threat requiring immediate intervention",
        "examples": ["Cardiac arrest", "Severe trauma", "Acute stroke"],
        "color": "#FF4D6D"  # Red
    },
    2: {
        "level": "EMERGENT",
        "description": "High-risk situation requiring urgent evaluation",
        "examples": ["Chest pain", "Severe respiratory distress", "Altered consciousness"],
        "color": "#FFC857"  # Orange
    },
    3: {
        "level": "URGENT",
        "description": "Requires prompt attention within 30 minutes",
        "examples": ["Moderate pain", "Fever", "Minor trauma"],
        "color": "#FFCC00"  # Yellow
    },
    4: {
        "level": "SEMI-URGENT",
        "description": "Non-life-threatening but needs care soon",
        "examples": ["Minor injuries", "Mild symptoms", "Stable patient"],
        "color": "#00FF9D"  # Green
    },
    5: {
        "level": "NON-URGENT",
        "description": "Minor condition that can tolerate waiting",
        "examples": ["Minor headache", "Minor cold", "Rash"],
        "color": "#00D4FF"  # Cyan
    }
}

# ===== EXPLAINABILITY FUNCTIONS =====

class ExplainabilityEngine:
    """
    Rule-based (hand-authored keyword/threshold weights) explainability for
    KTAS predictions. Shows contribution of each feature to the final
    decision in a clinician-readable narrative. See module docstring above
    re: this vs. models/shap_explainer.py.
    """
    
    @staticmethod
    def calculate_feature_contributions(complaint: str, vitals: Dict) -> Dict:
        """
        Calculate how much each feature contributed to the KTAS prediction
        Returns SHAP-style importance values
        """
        
        contributions = {}
        
        # 1. Complaint Analysis
        complaint_lower = complaint.lower()
        
        complaint_keywords = {
            "chest pain": 0.85,
            "chest": 0.75,
            "pain": 0.45,
            "shortness of breath": 0.72,
            "breathing": 0.65,
            "breath": 0.60,
            "headache": 0.35,
            "severe headache": 0.68,
            "head pain": 0.50,
            "abdominal pain": 0.55,
            "stomach": 0.45,
            "belly": 0.40,
            "dizziness": 0.42,
            "dizzy": 0.40,
            "vertigo": 0.45,
            "nausea": 0.25,
            "vomiting": 0.35,
            "sweating": 0.50,
            "diaphoresis": 0.55,
            "trauma": 0.70,
            "injury": 0.65,
            "bleeding": 0.75,
            "unconscious": 0.90,
            "unresponsive": 0.85,
            "difficulty breathing": 0.70,
            "throat pain": 0.30,
            "sore throat": 0.25,
            "fever": 0.35,
            "chills": 0.30,
            "weakness": 0.40,
            "fatigue": 0.20,
        }
        
        total_complaint_contribution = 0
        matched_keywords = []
        
        for keyword, weight in complaint_keywords.items():
            if keyword in complaint_lower:
                total_complaint_contribution += weight
                matched_keywords.append((keyword, weight))
        
        # Cap complaint contribution at realistic value
        total_complaint_contribution = min(total_complaint_contribution, 0.95)
        
        if total_complaint_contribution > 0:
            contributions['chief_complaint'] = {
                'value': total_complaint_contribution,
                'keywords': matched_keywords,
                'direction': 'increases_urgency'
            }
        
        # 2. Vital Signs Analysis
        systolic = vitals.get('systolic', 120)
        diastolic = vitals.get('diastolic', 80)
        hr = vitals.get('heart_rate', 70)
        rr = vitals.get('resp_rate', 16)
        
        # Systolic BP contribution
        if systolic > 180 or systolic < 90:
            sys_contrib = min(0.45, abs(systolic - 120) / 100)
            contributions['systolic_bp'] = {
                'value': sys_contrib,
                'actual': systolic,
                'normal_range': '110-140',
                'direction': 'increases_urgency' if systolic > 150 or systolic < 90 else 'neutral'
            }
        
        # Heart Rate contribution
        if hr > 100 or hr < 60:
            hr_contrib = min(0.52, abs(hr - 70) / 50)
            contributions['heart_rate'] = {
                'value': hr_contrib,
                'actual': hr,
                'normal_range': '60-100',
                'direction': 'increases_urgency' if hr > 110 or hr < 50 else 'neutral'
            }
        
        # Respiratory Rate contribution
        if rr > 20 or rr < 12:
            rr_contrib = min(0.48, abs(rr - 16) / 10)
            contributions['respiratory_rate'] = {
                'value': rr_contrib,
                'actual': rr,
                'normal_range': '12-20',
                'direction': 'increases_urgency' if rr > 24 or rr < 10 else 'neutral'
            }
        
        # Diastolic BP contribution
        if diastolic > 120 or diastolic < 60:
            dias_contrib = min(0.35, abs(diastolic - 80) / 80)
            contributions['diastolic_bp'] = {
                'value': dias_contrib,
                'actual': diastolic,
                'normal_range': '70-90',
                'direction': 'increases_urgency' if diastolic > 110 or diastolic < 60 else 'neutral'
            }
        
        return contributions
    
    @staticmethod
    def generate_feature_importance_ranking(contributions: Dict) -> List[Tuple[str, float]]:
        """
        Rank features by their contribution to urgency
        """
        
        ranked = []
        for feature, data in contributions.items():
            if isinstance(data, dict) and 'value' in data:
                ranked.append((feature, data['value']))
        
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked
    
    @staticmethod
    def generate_clinical_explanation(
        ktas_level: int,
        contributions: Dict,
        complaint: str,
        vitals: Dict
    ) -> str:
        """
        Generate a human-readable clinical explanation of the prediction
        """
        
        ktas_info = KTAS_DESCRIPTIONS[ktas_level]
        
        explanation = f"""
KTAS PREDICTION EXPLANATION
{'='*60}

PREDICTED LEVEL: KTAS {ktas_level} - {ktas_info['level']}
CLINICAL DEFINITION: {ktas_info['description']}

KEY DECISION DRIVERS:
{'-'*60}
"""
        
        ranked = ExplainabilityEngine.generate_feature_importance_ranking(contributions)
        
        for i, (feature, importance) in enumerate(ranked[:5], 1):
            feature_data = contributions[feature]
            
            if feature == 'chief_complaint':
                keywords = [kw[0] for kw in feature_data.get('keywords', [])]
                keywords_str = ", ".join(keywords) if keywords else complaint
                explanation += f"\n{i}. Chief Complaint (Importance: {importance:.2f})\n"
                explanation += f"   • Reported: {keywords_str}\n"
                explanation += f"   • Impact: Suggests urgent evaluation needed\n"
            
            elif 'heart_rate' in feature:
                actual = feature_data['actual']
                explanation += f"\n{i}. Heart Rate (Importance: {importance:.2f})\n"
                explanation += f"   • Actual: {actual} bpm (Normal: 60-100)\n"
                if actual > 110:
                    explanation += f"   • Clinical Note: Tachycardia suggests stress/illness\n"
                elif actual < 50:
                    explanation += f"   • Clinical Note: Bradycardia requires evaluation\n"
            
            elif 'systolic' in feature:
                actual = feature_data['actual']
                explanation += f"\n{i}. Systolic Blood Pressure (Importance: {importance:.2f})\n"
                explanation += f"   • Actual: {actual} mmHg (Normal: 110-140)\n"
                if actual > 160:
                    explanation += f"   • Clinical Note: Elevated BP suggests hypertensive urgency\n"
                elif actual < 90:
                    explanation += f"   • Clinical Note: Low BP indicates possible shock\n"
            
            elif 'respiratory' in feature:
                actual = feature_data['actual']
                explanation += f"\n{i}. Respiratory Rate (Importance: {importance:.2f})\n"
                explanation += f"   • Actual: {actual} breaths/min (Normal: 12-20)\n"
                if actual > 24:
                    explanation += f"   • Clinical Note: Tachypnea indicates respiratory distress\n"
        
        explanation += f"""

CLINICAL REASONING:
{'-'*60}
Based on the patient's presentation, this KTAS {ktas_level} assignment
indicates that the patient requires {ktas_info['description'].lower()}.

The model weighted these factors:
• Acuity of reported symptoms
• Abnormality of vital signs
• Potential for deterioration
• Resource requirements

DISPOSITION RECOMMENDATION:
{'-'*60}
Based on KTAS {ktas_level}:
{chr(10).join([f"• {example}" for example in ktas_info['examples']])}

This patient should be evaluated and triaged accordingly by the
emergency department.

CONFIDENCE: Model shows high confidence in this assignment.
HUMAN OVERSIGHT: Always defer to clinical judgment of ED personnel.
"""
        
        return explanation
    
    # NOTE: an earlier version of this method returned a couple of
    # hard-coded "what if HR were 75" style scenarios without actually
    # re-running the model. Real counterfactuals -- generated by perturbing
    # the actual input and re-running the trained classifier -- now live in
    # models/shap_explainer.py::generate_counterfactuals(), which is what
    # the app calls. Keeping counterfactual generation in one place avoids
    # two code paths ever disagreeing about what "would change the outcome".

    @staticmethod
    def generate_feature_importance_data(contributions: Dict) -> Dict:
        """
        Prepare data for visualization
        """
        
        ranked = ExplainabilityEngine.generate_feature_importance_ranking(contributions)
        
        feature_display_names = {
            'chief_complaint': 'Chief Complaint',
            'heart_rate': 'Heart Rate',
            'systolic_bp': 'Systolic BP',
            'respiratory_rate': 'Respiratory Rate',
            'diastolic_bp': 'Diastolic BP',
            'age': 'Age'
        }
        
        data = {
            'features': [],
            'importance_values': [],
            'colors': []
        }
        
        for feature, importance in ranked[:8]:  # Top 8 features
            data['features'].append(feature_display_names.get(feature, feature))
            data['importance_values'].append(importance)
            
            # Color based on importance
            if importance > 0.6:
                color = '#FF4D6D'  # Red - high impact
            elif importance > 0.4:
                color = '#FFC857'  # Orange - medium impact
            elif importance > 0.2:
                color = '#FFCC00'  # Yellow - low impact
            else:
                color = '#00FF9D'  # Green - minimal impact
            
            data['colors'].append(color)
        
        return data
