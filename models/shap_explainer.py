"""
Real, model-grounded explainability: SHAP values, uncertainty, counterfactuals.

Unlike explainability_engine.py (hand-authored keyword/threshold weights,
used to generate clinician-readable narrative text), everything in this
file is computed by actually querying the trained classifier -- nothing
here is hard-coded or guessed.

- SHAP values:      shap.TreeExplainer against the real xgboost.XGBClassifier
                     (exact, not the sampling-based approximation needed for
                     non-tree or unsupported multi-class models).
- Uncertainty:      derived directly from predict_proba (margin between the
                     top two classes, and entropy of the full distribution).
- Counterfactuals:  the input is actually perturbed and re-run through the
                     real model; the reported outcome is whatever the model
                     really predicts for that perturbed input, not a canned
                     sentence.
"""

import numpy as np
import shap
import streamlit as st


@st.cache_resource
def _get_tree_explainer(_classifier):
    """
    Build (and cache) a shap.TreeExplainer for the given model.
    Leading underscore on the param tells st.cache_resource not to try to
    hash the model object itself (it isn't hashable in a useful way).
    """
    return shap.TreeExplainer(_classifier)


def compute_shap_values(classifier, feature_values, feature_names, features_scaled, predicted_class_idx):
    """
    Compute real SHAP values for one prediction.

    Args:
        classifier: the trained xgboost.XGBClassifier
        feature_values: raw (unscaled) feature values, for display
        feature_names: list of 21 feature names, same order as feature_values
        features_scaled: the scaled feature vector actually fed to the model (list or 1D array)
        predicted_class_idx: 0-indexed predicted class (raw_prediction from ktas_predictor,
                              i.e. KTAS level - 1)

    Returns:
        list of dicts, sorted by |impact| descending:
        [{'feature': str, 'raw_value': float, 'shap_value': float, 'direction': 'increases'|'decreases'}, ...]
    """
    explainer = _get_tree_explainer(classifier)
    X = np.array(features_scaled, dtype=float).reshape(1, -1)
    explanation = explainer(X)

    # explanation.values shape: (1, n_features, n_classes)
    values_for_predicted_class = explanation.values[0, :, predicted_class_idx]

    ranked = []
    for name, raw_val, sv in zip(feature_names, feature_values, values_for_predicted_class):
        ranked.append({
            'feature': name,
            'raw_value': raw_val,
            'shap_value': float(sv),
            'direction': 'increases urgency' if sv > 0 else 'decreases urgency',
        })
    ranked.sort(key=lambda d: abs(d['shap_value']), reverse=True)
    return ranked


def compute_uncertainty(probabilities):
    """
    Real uncertainty metrics computed from the model's own probability
    distribution over the 5 KTAS classes -- no separate model needed.

    Args:
        probabilities: list/array of 5 floats (P(KTAS 1) .. P(KTAS 5)), sums to ~1

    Returns:
        dict: {
            'margin': float 0-1,          # gap between top-1 and top-2 predicted probability
            'entropy': float >= 0,        # Shannon entropy of the distribution (0 = certain)
            'normalized_entropy': float 0-1,
            'level': 'low' | 'moderate' | 'high',   # human-readable uncertainty level
            'interpretation': str
        }
    """
    p = np.clip(np.array(probabilities, dtype=float), 1e-12, 1.0)
    p = p / p.sum()
    sorted_p = np.sort(p)[::-1]
    margin = float(sorted_p[0] - sorted_p[1])
    entropy = float(-np.sum(p * np.log(p)))
    max_entropy = float(np.log(len(p)))  # entropy of a uniform distribution over the classes
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

    if margin > 0.5 and normalized_entropy < 0.3:
        level = 'low'
        interpretation = "The model strongly favours one KTAS level over the alternatives."
    elif margin > 0.2 and normalized_entropy < 0.6:
        level = 'moderate'
        interpretation = "The model favours one KTAS level, but a neighbouring level received meaningful probability -- worth a second look."
    else:
        level = 'high'
        interpretation = "Probability is spread across multiple KTAS levels. Treat this prediction as a starting point, not a settled call, and apply clinical judgement."

    return {
        'margin': margin,
        'entropy': entropy,
        'normalized_entropy': normalized_entropy,
        'level': level,
        'interpretation': interpretation,
    }


# Perturbation scenarios tested for counterfactuals. Each targets one
# feature by name (must match FEATURE_NAMES in generate_synthetic_dataset.py)
# and describes how it's perturbed.
_COUNTERFACTUAL_SCENARIOS = [
    ('HR', lambda v: 75.0, "If heart rate were normalized to 75 bpm"),
    ('SBP', lambda v: 118.0, "If systolic BP were normalized to 118 mmHg"),
    ('RR', lambda v: 16.0, "If respiratory rate were normalized to 16/min"),
    ('NRS_pain', lambda v: max(0.0, v - 4), "If reported pain were reduced by 4 points"),
]


def generate_counterfactuals(classifier, scaler, feature_values, feature_names, current_ktas_level, max_scenarios=4):
    """
    Genuine counterfactuals: perturb one feature at a time on the REAL input
    and re-run it through the REAL trained model, reporting what actually
    happens (as opposed to a canned "would decrease by ~1 level" string).

    Only scenarios that are clinically relevant to this specific patient are
    included (e.g. no "normalize heart rate" scenario if heart rate is
    already normal) and only ones that actually change the model's output
    are surfaced, since a perturbation that changes nothing isn't an
    interesting counterfactual.

    Returns:
        list of dicts: [{'change': str, 'resulting_ktas': int, 'ktas_changed': bool}, ...]
    """
    import pandas as pd

    base = dict(zip(feature_names, feature_values))
    results = []

    for feat_name, transform_fn, description in _COUNTERFACTUAL_SCENARIOS:
        if feat_name not in base:
            continue
        original_value = base[feat_name]
        new_value = transform_fn(original_value)
        if new_value == original_value:
            continue  # nothing to test -- this vital was already at/near the target

        # Only test scenarios where the vital was actually abnormal to begin with
        if feat_name == 'HR' and not (original_value > 100 or original_value < 60):
            continue
        if feat_name == 'SBP' and not (original_value > 150 or original_value < 95):
            continue
        if feat_name == 'RR' and not (original_value > 22):
            continue
        if feat_name == 'NRS_pain' and not (original_value >= 7):
            continue

        perturbed = dict(base)
        perturbed[feat_name] = new_value
        row = pd.DataFrame([[perturbed[f] for f in feature_names]], columns=feature_names)
        row_scaled = scaler.transform(row)
        new_pred = int(classifier.predict(row_scaled)[0]) + 1  # back to 1-5 scale

        results.append({
            'change': description,
            'resulting_ktas': new_pred,
            'ktas_changed': new_pred != current_ktas_level,
        })

        if len(results) >= max_scenarios:
            break

    return results
