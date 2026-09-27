"""
NOT USED BY THE LIVE APP.
app_production.py uses models/ktas_predictor.py (the real trained
xgboost.XGBClassifier) via models/ai_orchestrator.py. This file is a
simpler, rule-based fallback that was never wired in anywhere. Left in
place for reference / in case you want a zero-dependency fallback path,
but nothing currently imports it.
"""

"""
Simplified KTAS predictor wrapper for voice/manual input
"""

def predict_ktas_from_symptoms(symptoms_text):
    """
    Simple KTAS prediction based on symptoms text only
    Returns a dict with ktas_level
    """
    
    # Keywords for each KTAS level
    critical_keywords = ['unconscious', 'unresponsive', 'not breathing', 'no pulse', 'severe hemorrhage', 'cardiac arrest']
    emergent_keywords = ['chest pain', 'severe difficulty breathing', 'severe burns', 'stroke', 'overdose', 'severe trauma', 'severe abdominal pain']
    urgent_keywords = ['moderate pain', 'difficulty breathing', 'fever', 'vomiting', 'severe headache', 'abdominal pain']
    less_urgent_keywords = ['mild pain', 'cough', 'congestion', 'fatigue', 'rash']
    
    symptoms_lower = symptoms_text.lower()
    
    # Check for critical
    if any(keyword in symptoms_lower for keyword in critical_keywords):
        return {'ktas_level': 1, 'confidence': 0.9}
    
    # Check for emergent
    if any(keyword in symptoms_lower for keyword in emergent_keywords):
        return {'ktas_level': 2, 'confidence': 0.85}
    
    # Check for urgent
    if any(keyword in symptoms_lower for keyword in urgent_keywords):
        return {'ktas_level': 3, 'confidence': 0.80}
    
    # Check for less urgent
    if any(keyword in symptoms_lower for keyword in less_urgent_keywords):
        return {'ktas_level': 4, 'confidence': 0.75}
    
    # Default to non-urgent
    return {'ktas_level': 5, 'confidence': 0.70}
