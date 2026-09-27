"""
Synthetic KTAS Training/Evaluation Dataset Generator
=====================================================

IMPORTANT — READ THIS FIRST:
This generates PROCEDURALLY-GENERATED, CLINICALLY-GROUNDED synthetic patient
cases. It is NOT a copy of, sample from, or reconstruction of the Moon et al.
(2019) Korean ED dataset. No real patient data exists anywhere in this
project (there is no such CSV/file in the codebase) — this script exists
because a model needs *something* to train/evaluate on, and fabricating
accuracy numbers without any underlying data is worse than being explicit
about using synthetic data.

The generation logic below encodes standard, publicly-documented KTAS
clinical criteria (vital sign thresholds for tachycardia/bradycardia,
hypotension, tachypnea, fever; pain scale; and which chief complaints are
associated with which acuity levels) — the same kind of domain knowledge
described in the report's Chapter 2 literature review. Each class's vitals
are sampled from overlapping distributions on purpose: real triage vitals
are not perfectly separable between adjacent KTAS levels, and a synthetic
set that was trivially separable (~100% accuracy) would be far less
defensible than one with realistic, honestly-reported performance.

If a real dataset (Moon et al. 2019, or a UK ED extract) becomes available,
replace this generator's output with that real data and rerun
`train_model.py` — nothing downstream needs to change, since both produce
the same 21-column schema.

Usage:
    python training/generate_synthetic_dataset.py
Produces:
    training/synthetic_ktas_dataset.csv   (all generated cases + labels)
"""

import numpy as np
import pandas as pd
from pathlib import Path

FEATURE_NAMES = [
    'Age', 'Sex', 'HR', 'SBP', 'DBP', 'RR', 'BT', 'NRS_pain', 'symptom_count',
    'has_chest_pain', 'has_shortness_of_breath', 'has_abdominal_pain', 'has_fever',
    'has_headache', 'has_dizziness', 'has_nausea', 'has_vomiting', 'has_weakness',
    'has_syncope', 'has_confusion', 'has_laceration'
]

SYMPTOM_KEYS = [
    'chest_pain', 'shortness_of_breath', 'abdominal_pain', 'fever', 'headache',
    'dizziness', 'nausea', 'vomiting', 'weakness', 'syncope', 'confusion', 'laceration'
]

# Probability each symptom is present, by KTAS level (1=most critical, 5=least).
# High-risk complaints (chest pain, SOB, syncope, confusion) are weighted toward
# levels 1-2; benign complaints (headache, fatigue) weighted toward levels 4-5.
SYMPTOM_WEIGHTS = {
    1: dict(chest_pain=.35, shortness_of_breath=.45, abdominal_pain=.10, fever=.10, headache=.10,
            dizziness=.20, nausea=.15, vomiting=.15, weakness=.25, syncope=.40, confusion=.35, laceration=.15),
    2: dict(chest_pain=.40, shortness_of_breath=.35, abdominal_pain=.20, fever=.15, headache=.20,
            dizziness=.25, nausea=.20, vomiting=.15, weakness=.20, syncope=.20, confusion=.15, laceration=.15),
    3: dict(chest_pain=.15, shortness_of_breath=.15, abdominal_pain=.30, fever=.25, headache=.25,
            dizziness=.15, nausea=.20, vomiting=.15, weakness=.15, syncope=.05, confusion=.05, laceration=.15),
    4: dict(chest_pain=.03, shortness_of_breath=.05, abdominal_pain=.15, fever=.15, headache=.20,
            dizziness=.08, nausea=.10, vomiting=.05, weakness=.10, syncope=.01, confusion=.01, laceration=.15),
    5: dict(chest_pain=.01, shortness_of_breath=.02, abdominal_pain=.08, fever=.08, headache=.15,
            dizziness=.04, nausea=.05, vomiting=.03, weakness=.06, syncope=.00, confusion=.00, laceration=.08),
}

# (mean, std) per level for each vital sign. Distributions intentionally overlap
# between adjacent levels (e.g. KTAS 2 and KTAS 3 heart rate) to avoid an
# artificially/trivially separable dataset.
VITALS_PARAMS = {
    #        HR            SBP            DBP           RR            BT             pain
    1: dict(hr=(136, 18), sbp=(80, 18),  dbp=(52, 12), rr=(32, 5),   bt=(37.7, 1.1), pain=(9.0, 1.3)),
    2: dict(hr=(118, 13), sbp=(98, 15),  dbp=(64, 11), rr=(26, 3),   bt=(37.6, 0.9), pain=(7.7, 1.3)),
    3: dict(hr=(97, 11),  sbp=(124, 15), dbp=(77, 10), rr=(20, 2.5), bt=(37.4, 0.7), pain=(5.7, 1.5)),
    4: dict(hr=(83, 9),   sbp=(124, 12), dbp=(79, 9),  rr=(17, 1.8), bt=(37.0, 0.5), pain=(3.5, 1.4)),
    5: dict(hr=(75, 8),   sbp=(119, 10), dbp=(78, 7),  rr=(15, 1.5), bt=(36.8, 0.4), pain=(1.6, 1.2)),
}

# Roughly mirrors real ED case-mix shape (KTAS 3/4 most common, KTAS 1 rare).
LEVEL_PREVALENCE = {1: 0.065, 2: 0.16, 3: 0.365, 4: 0.29, 5: 0.12}


def _clip(v, lo, hi):
    return max(lo, min(hi, v))


def generate_case(rng, level):
    v = VITALS_PARAMS[level]
    hr = _clip(rng.normal(*v['hr']), 35, 190)
    sbp = _clip(rng.normal(*v['sbp']), 60, 220)
    dbp = _clip(rng.normal(*v['dbp']), 35, 130)
    rr = _clip(rng.normal(*v['rr']), 6, 46)
    bt = _clip(rng.normal(*v['bt']), 34.5, 41.5)
    pain = int(round(_clip(rng.normal(*v['pain']), 0, 10)))
    age = int(_clip(rng.gamma(4.2, 12), 1, 99))
    sex = int(rng.integers(1, 3))  # 1=Male, 2=Female (matches ktas_predictor.py convention)

    flags = {k: (1 if rng.random() < SYMPTOM_WEIGHTS[level][k] else 0) for k in SYMPTOM_KEYS}
    symptom_count = sum(flags.values())

    row = [age, sex, round(hr), round(sbp), round(dbp), round(rr), round(bt, 1), pain, symptom_count]
    row += [flags[k] for k in SYMPTOM_KEYS]
    return row


def generate_dataset(n, seed=2026):
    rng = np.random.default_rng(seed)
    levels = list(LEVEL_PREVALENCE.keys())
    probs = list(LEVEL_PREVALENCE.values())
    rows, labels = [], []
    for _ in range(n):
        level = int(rng.choice(levels, p=probs))
        rows.append(generate_case(rng, level))
        labels.append(level)
    X = pd.DataFrame(rows, columns=FEATURE_NAMES)
    y = pd.Series(labels, name='KTAS_level')
    return X, y


def generate_edge_case_dataset(n, seed=99):
    """
    A second, deliberately harder evaluation set: each case's vitals are
    sampled from a 50/50 BLEND of two adjacent KTAS levels' distributions
    (not just "level lo or level hi" -- an actual blended mean), so the
    resulting cases sit near the clinical boundary between them. The label
    is then assigned to whichever of the two levels' canonical vitals the
    sampled case ends up closer to (Euclidean distance in normalized vital
    space), so ground truth is derived rather than pre-decided.

    This is a genuine robustness/edge-case stress test, evaluated
    separately from the main test set (see training/run_evaluation.py) --
    it's *supposed* to be harder. This is what the report's original
    "Dataset 2: Synthetic Diverse Cases" section described; that section
    just never had an actual dataset or evaluation behind it before.
    """
    rng = np.random.default_rng(seed)
    boundaries = [(1, 2), (2, 3), (3, 4), (4, 5)]
    vital_keys = ['hr', 'sbp', 'dbp', 'rr', 'bt', 'pain']

    rows, labels = [], []
    for _ in range(n):
        lo, hi = boundaries[rng.integers(0, len(boundaries))]
        p_lo, p_hi = VITALS_PARAMS[lo], VITALS_PARAMS[hi]
        blended = {k: ((p_lo[k][0] + p_hi[k][0]) / 2, max(p_lo[k][1], p_hi[k][1])) for k in vital_keys}

        hr = _clip(rng.normal(*blended['hr']), 35, 190)
        sbp = _clip(rng.normal(*blended['sbp']), 60, 220)
        dbp = _clip(rng.normal(*blended['dbp']), 35, 130)
        rr = _clip(rng.normal(*blended['rr']), 6, 46)
        bt = _clip(rng.normal(*blended['bt']), 34.5, 41.5)
        pain = int(round(_clip(rng.normal(*blended['pain']), 0, 10)))
        age = int(_clip(rng.gamma(4.2, 12), 1, 99))
        sex = int(rng.integers(1, 3))

        # Symptom flags: blend the two levels' probabilities per symptom.
        flags = {}
        for k in SYMPTOM_KEYS:
            p = (SYMPTOM_WEIGHTS[lo][k] + SYMPTOM_WEIGHTS[hi][k]) / 2
            flags[k] = 1 if rng.random() < p else 0
        symptom_count = sum(flags.values())

        # Derive ground truth from which level's canonical vitals (z-scored) this case sits closer to.
        def dist_to(level):
            p = VITALS_PARAMS[level]
            vals = dict(hr=hr, sbp=sbp, dbp=dbp, rr=rr, bt=bt, pain=pain)
            return sum(((vals[k] - p[k][0]) / p[k][1]) ** 2 for k in vital_keys)

        label = lo if dist_to(lo) <= dist_to(hi) else hi

        row = [age, sex, round(hr), round(sbp), round(dbp), round(rr), round(bt, 1), pain, symptom_count]
        row += [flags[k] for k in SYMPTOM_KEYS]
        rows.append(row)
        labels.append(label)

    X = pd.DataFrame(rows, columns=FEATURE_NAMES)
    y = pd.Series(labels, name='KTAS_level')
    return X, y


if __name__ == '__main__':
    X, y = generate_dataset(1800)
    out = X.copy()
    out['KTAS_level'] = y
    out_path = Path(__file__).parent / 'synthetic_ktas_dataset.csv'
    out.to_csv(out_path, index=False)
    print(f"Wrote {len(out)} synthetic cases to {out_path}")
    print("Label distribution:")
    print(y.value_counts().sort_index())
