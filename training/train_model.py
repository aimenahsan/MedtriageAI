"""
Trains the real XGBoost KTAS classifier used by the app.

This replaces the model that shipped in data/classifier_model.pkl, which on
inspection was actually a scikit-learn GradientBoostingClassifier despite
being labelled "XGBoost" everywhere in the report and code. This script
fits a genuine xgboost.XGBClassifier so the two agree.

Run from the project root:
    python training/train_model.py

Writes:
    data/classifier_model.pkl        (real xgboost.XGBClassifier)
    data/classifier_scaler.pkl       (StandardScaler, same 21 features/order as before)
    training/test_split.csv          (held-out test set, used by run_evaluation.py —
                                       kept separate from training data so evaluation
                                       numbers are honest, not just training-set fit)
"""

import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, str(Path(__file__).parent))
from generate_synthetic_dataset import generate_dataset, FEATURE_NAMES

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
TRAINING_DIR = Path(__file__).parent

RANDOM_STATE = 42
N_CASES = 1800
TEST_SIZE = 0.23


def main():
    print("=" * 70)
    print("Generating synthetic dataset...")
    X, y = generate_dataset(N_CASES, seed=2026)
    y0 = y - 1  # xgboost needs 0-indexed class labels internally

    X_train, X_test, y_train, y_test, y0_train, y0_test = train_test_split(
        X, y, y0, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train: {len(X_train)} cases | Test (held out): {len(X_test)} cases")

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    print("\nTraining xgboost.XGBClassifier...")
    clf = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        objective='multi:softprob',
        num_class=5,
        eval_metric='mlogloss',
        random_state=RANDOM_STATE,
        n_jobs=4,
    )
    clf.fit(X_train_s, y0_train)

    # --- Honest evaluation on the held-out test set (not used in training) ---
    pred = clf.predict(X_test_s) + 1
    acc = (pred == y_test.values).mean()
    cv_scores = cross_val_score(clf, X_train_s, y0_train, cv=5)

    print("\n" + "=" * 70)
    print(f"HELD-OUT TEST ACCURACY: {acc*100:.1f}%")
    print(f"5-FOLD CV (train set):  {cv_scores.mean()*100:.1f}% +/- {cv_scores.std()*100:.1f}%")
    print("=" * 70)
    print(classification_report(y_test, pred, digits=3))
    print("Confusion matrix (rows=true 1..5, cols=predicted 1..5):")
    print(confusion_matrix(y_test, pred))

    under = int((pred < y_test.values).sum())
    over = int((pred > y_test.values).sum())
    total = len(y_test)
    print(f"\nUndertriage: {under}/{total} ({under/total*100:.1f}%)  "
          f"Overtriage: {over}/{total} ({over/total*100:.1f}%)")

    # --- Save model + scaler (same schema/order as the original pickles) ---
    DATA_DIR.mkdir(exist_ok=True)
    with open(DATA_DIR / 'classifier_model.pkl', 'wb') as f:
        pickle.dump(clf, f)
    with open(DATA_DIR / 'classifier_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print(f"\nSaved model  -> {DATA_DIR / 'classifier_model.pkl'}")
    print(f"Saved scaler -> {DATA_DIR / 'classifier_scaler.pkl'}")

    # --- Save the held-out test set so run_evaluation.py reuses the exact same
    #     cases (rather than silently regenerating a different sample) ---
    test_out = X_test.copy()
    test_out['KTAS_level'] = y_test.values
    test_out.to_csv(TRAINING_DIR / 'test_split.csv', index=False)
    print(f"Saved held-out test split -> {TRAINING_DIR / 'test_split.csv'}")


if __name__ == '__main__':
    main()
