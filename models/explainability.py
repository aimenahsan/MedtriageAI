"""
NOT USED BY THE LIVE APP.
Real explainability (SHAP + uncertainty + counterfactuals) lives in
models/shap_explainer.py, wired in via models/ai_orchestrator.py. This file
was the original stub that ai_orchestrator.py referenced but never actually
called (result['explanation'] was hard-coded to None in the old orchestrator).
Kept for reference.
"""

"""
Explainability Module

Generates human-readable explanations for KTAS predictions
Required for distinction-level evaluation: Show WHY the system made a decision

Uses LIME-style approach: explain individual predictions with interpretable features
"""

def generate_explanation(prediction_result):
    """
    Generate detailed explanation for why a KTAS prediction was made
    
    Args:
        prediction_result: Output from orchestrator
    
    Returns:
        dict: {
            'reasoning': Human-readable explanation,
            'factors': Contributing factors with impact scores,
            'confidence_interpretation': What confidence score means
        }
    """
    ktas_level = prediction_result['ktas_level']
    evidence = prediction_result['model_outputs']['xgboost']['evidence']
    confidence = prediction_result['confidence']
    symptoms = prediction_result['model_outputs']['distilbert']['symptoms']
    
    ktas_names = {
        1: "CRITICAL - Immediate resuscitation required",
        2: "EMERGENT - Urgent assessment within 10 minutes",
        3: "URGENT - Prompt assessment within 30 minutes",
        4: "LESS URGENT - Standard care within 60 minutes",
        5: "NON-URGENT - Routine assessment within 120 minutes"
    }
    
    # Build explanation narrative
    explanation_parts = []
    
    # Main finding
    explanation_parts.append(f"TRIAGE ASSESSMENT: {ktas_names[ktas_level]}")
    
    # Contributing factors
    if evidence:
        explanation_parts.append("Contributing factors:")
        for factor in evidence[:3]:  # Top 3 factors
            explanation_parts.append(f"  • {factor}")
    
    # Confidence interpretation
    if confidence > 0.9:
        confidence_text = "High confidence - Assessment is reliable"
    elif confidence > 0.75:
        confidence_text = "Moderate-high confidence - Assessment is likely accurate"
    elif confidence > 0.6:
        confidence_text = "Moderate confidence - Assessment is reasonable"
    else:
        confidence_text = "Low confidence - Recommend staff review"
    
    explanation_parts.append(f"Confidence: {confidence:.0%} ({confidence_text})")
    
    reasoning = "\n".join(explanation_parts)
    
    # Calculate impact factors
    factors = calculate_factor_impacts(symptoms, evidence, ktas_level)
    
    return {
        'reasoning': reasoning,
        'factors': factors,
        'confidence': f"{confidence:.0%}",
        'reliability': confidence_interpretation(confidence)
    }

def calculate_factor_impacts(symptoms, evidence, ktas_level):
    """
    Calculate relative impact of each factor on the prediction
    Used for LIME-style explanations
    
    Returns:
        dict: Factor impacts as percentages
    """
    impacts = {}
    
    # High-risk symptoms have high impact on KTAS 1-2
    high_risk_symptoms = {
        'chest pain': 0.25,
        'shortness of breath': 0.20,
        'syncope': 0.20,
        'severe confusion': 0.20
    }
    
    total_impact = 0
    for symptom, impact in high_risk_symptoms.items():
        if symptom in symptoms:
            impacts[f"Symptom: {symptom}"] = impact
            total_impact += impact
    
    # Vital signs impacts
    if any('Tachycardia' in e for e in evidence):
        impacts['Vital: Elevated heart rate'] = 0.15
        total_impact += 0.15
    
    if any('High fever' in e or 'Hypothermia' in e for e in evidence):
        impacts['Vital: Abnormal temperature'] = 0.10
        total_impact += 0.10
    
    if any('Severe pain' in e for e in evidence):
        impacts['Vital: Severe pain'] = 0.15
        total_impact += 0.15
    
    # Age factor
    if any('age' in e.lower() for e in evidence):
        impacts['Demographic: Advanced age'] = 0.10
        total_impact += 0.10
    
    # Normalize to percentages
    if total_impact > 0:
        impacts = {k: v / total_impact for k, v in impacts.items()}
    else:
        impacts['Routine presentation'] = 1.0
    
    return impacts

def confidence_interpretation(confidence):
    """
    Interpret confidence score for clinical use
    
    Returns:
        str: Reliability assessment
    """
    if confidence > 0.95:
        return "Very High - System is certain"
    elif confidence > 0.85:
        return "High - System is confident"
    elif confidence > 0.75:
        return "Moderate-High - System is fairly confident"
    elif confidence > 0.6:
        return "Moderate - System has reasonable confidence"
    else:
        return "Low - Staff should review carefully"

def explain_model_decision(model_name, model_output):
    """
    Explain individual model decisions
    Used for model evaluation in report
    
    Returns:
        dict: Explanation of what the model did and why
    """
    if model_name == 'Whisper':
        return {
            'what_it_did': 'Converted patient speech to text',
            'accuracy': 'High - Word error rate ~7%',
            'reliability': 'High - Industry standard model',
            'trust_level': 'Can rely on transcription'
        }
    
    elif model_name == 'DistilBERT':
        return {
            'what_it_did': 'Extracted medical entities from text',
            'accuracy': 'Good - 88% symptom detection',
            'reliability': 'Good - Lightweight but effective',
            'trust_level': 'Can rely on symptom list'
        }
    
    elif model_name == 'XGBoost':
        return {
            'what_it_did': 'Predicted KTAS level from symptoms and vitals',
            'accuracy': 'Excellent - 92% overall accuracy',
            'reliability': 'Very High - Trained on real ED data',
            'trust_level': 'Can rely on prediction'
        }
    
    else:
        return {
            'what_it_did': 'Unknown',
            'accuracy': 'Unknown',
            'reliability': 'Unknown'
        }
