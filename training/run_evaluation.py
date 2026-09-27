"""
Runs a genuine evaluation of the trained model against the held-out test
split (training/test_split.csv, produced by train_model.py -- these cases
were NOT used for training, so this is an honest generalisation estimate,
not a training-set fit).

Every number in evaluation/evaluation_report.json and every graph in
evaluation/*.png is computed from real predictions made in this run. This
directly answers the "[PLACEHOLDER: Evaluation Graphs]" section in the
report (confusion matrix, ROC curves, confidence calibration curve) and
replaces the previously-fabricated Section 5 numbers.

It deliberately reuses REBUILD_evaluation_framework.py's EvaluationFramework
class (that framework was already a real, honest implementation -- it just
had never actually been run against real predictions before).

Usage (after training/train_model.py has been run):
    python training/run_evaluation.py
Writes:
    evaluation/evaluation_report.json
    evaluation/confusion_matrix.png
    evaluation/roc_curves.png
    evaluation/confidence_calibration.png
    evaluation/dataset_comparison.png
"""

import json
import pickle
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'models'))

from generate_synthetic_dataset import FEATURE_NAMES, generate_dataset, generate_edge_case_dataset  # noqa: E402
from REBUILD_evaluation_framework import EvaluationFramework  # noqa: E402 (path set above)

EVAL_DIR = PROJECT_ROOT / 'evaluation'
KTAS_LABELS = [1, 2, 3, 4, 5]
KTAS_COLORS = {1: '#FF3366', 2: '#FF5E3A', 3: '#FFD600', 4: '#00E676', 5: '#00D4FF'}


def load_model():
    with open(PROJECT_ROOT / 'data' / 'classifier_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open(PROJECT_ROOT / 'data' / 'classifier_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler


def run_predictions(model, scaler, X_df):
    X_scaled = scaler.transform(X_df)
    proba = model.predict_proba(X_scaled)          # (n, 5), columns 0..4 = KTAS 1..5
    preds = proba.argmax(axis=1) + 1                # back to KTAS 1-5
    return preds, proba


def plot_confusion_matrix(y_true, y_pred, out_path):
    cm = confusion_matrix(y_true, y_pred, labels=KTAS_LABELS)
    fig, ax = plt.subplots(figsize=(6, 5.2))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    ax.set_xticklabels([f'KTAS {i}' for i in KTAS_LABELS])
    ax.set_yticklabels([f'KTAS {i}' for i in KTAS_LABELS])
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title('Confusion Matrix -- Held-Out Synthetic Test Set')
    for i in range(5):
        for j in range(5):
            color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', color=color, fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_roc_curves(y_true, proba, out_path):
    y_bin = label_binarize(y_true, classes=KTAS_LABELS)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for i, level in enumerate(KTAS_LABELS):
        fpr, tpr, _ = roc_curve(y_bin[:, i], proba[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f'KTAS {level} (AUC={roc_auc:.2f})', color=KTAS_COLORS[level], linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1)
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves -- One-vs-Rest per KTAS Level')
    ax.legend(loc='lower right', fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_confidence_calibration(y_true, y_pred, proba, out_path):
    confidences = proba.max(axis=1)
    correct = (y_pred == y_true).astype(int)
    bins = np.linspace(0, 1, 11)
    bin_idx = np.digitize(confidences, bins) - 1
    bin_idx = np.clip(bin_idx, 0, 9)
    bin_acc, bin_conf, bin_counts = [], [], []
    for b in range(10):
        mask = bin_idx == b
        if mask.sum() > 0:
            bin_acc.append(correct[mask].mean())
            bin_conf.append(confidences[mask].mean())
            bin_counts.append(mask.sum())
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect calibration')
    ax.plot(bin_conf, bin_acc, marker='o', color='#00D4FF', label='Model (this run)')
    for c, a, n in zip(bin_conf, bin_acc, bin_counts):
        ax.annotate(str(n), (c, a), textcoords="offset points", xytext=(0, 6), fontsize=7, color='#708098')
    ax.set_xlabel('Predicted confidence'); ax.set_ylabel('Empirical accuracy')
    ax.set_title('Confidence Calibration Curve\n(point labels = # test cases in that confidence bin)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_dataset_comparison(main_acc, edge_acc, out_path):
    """
    Bar chart comparing accuracy on the two datasets this project actually
    evaluated against. Unlike the report's original draft, this does NOT
    include a "Korean ED (Moon et al. 2019)" bar, because no such data
    exists in this project to evaluate against -- see
    training/generate_synthetic_dataset.py. Add that bar back in yourself
    if/when real data is obtained and evaluated.
    """
    labels = ['Standard Test Set\n(held-out, main distribution)', 'Edge-Case Stress Test\n(boundary-blended, harder)']
    values = [main_acc * 100, edge_acc * 100]
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    bars = ax.bar(labels, values, color=['#00D4FF', '#FF5E3A'], width=0.5)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, v + 1, f'{v:.1f}%', ha='center', fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Evaluation Accuracy by Dataset\n(only datasets this project actually evaluated against)')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    EVAL_DIR.mkdir(exist_ok=True)
    test_split_path = PROJECT_ROOT / 'training' / 'test_split.csv'
    if not test_split_path.exists():
        print("No held-out test split found. Run `python training/train_model.py` first.")
        sys.exit(1)

    test_df = pd.read_csv(test_split_path)
    y_true = test_df['KTAS_level'].values
    X_df = test_df[FEATURE_NAMES]

    model, scaler = load_model()

    t0 = time.time()
    y_pred, proba = run_predictions(model, scaler, X_df)
    elapsed = time.time() - t0
    per_case_ms = (elapsed / len(X_df)) * 1000

    # --- Also genuinely exercise the project's own EvaluationFramework class
    #     (REBUILD_evaluation_framework.py) so it's not just unused code --
    #     this becomes evaluation/evaluation_framework_native_report.json,
    #     a second, independent artifact for anyone who wants to see that
    #     specific class actually running. It formats accuracy as a percent
    #     string, so the merged report.json below uses the numpy-computed
    #     numbers as the single source of truth.
    framework = EvaluationFramework()
    for i in range(len(y_true)):
        framework.add_test_case(patient_data={'row': i}, expected_ktas=int(y_true[i]))
        test_id = framework.test_cases[-1]['test_id']
        framework.record_prediction(test_id, predicted_ktas=int(y_pred[i]), confidence=float(proba[i].max()))
    with open(EVAL_DIR / 'evaluation_framework_native_report.json', 'w') as f:
        json.dump(framework.generate_report(), f, indent=2, default=str)

    overall_accuracy = float((y_pred == y_true).mean())
    safety = {
        'undertriage_rate': float((y_pred < y_true).mean()),
        'overtriage_rate': float((y_pred > y_true).mean()),
        'exact_match_rate': overall_accuracy,
    }
    per_class_accuracy = {}
    for level in KTAS_LABELS:
        mask = y_true == level
        if mask.sum() > 0:
            per_class_accuracy[f'KTAS {level}'] = float((y_pred[mask] == y_true[mask]).mean())

    # --- Second dataset: boundary-blended edge cases (harder by design) ---
    X_edge, y_edge = generate_edge_case_dataset(300, seed=99)
    y_edge_pred, _ = run_predictions(model, scaler, X_edge)
    edge_accuracy = float((y_edge_pred == y_edge.values).mean())
    print(f"Edge-case stress test accuracy: {edge_accuracy*100:.1f}% (n={len(X_edge)}, boundary-blended -- expected to be lower than the main set)")

    report = {
        'timestamp': datetime.now().isoformat(),
        'dataset_size': len(test_df),
        'dataset_description': 'Held-out split of the procedurally-generated synthetic KTAS dataset (training/generate_synthetic_dataset.py). NOT the Moon et al. (2019) dataset.',
        'overall_accuracy': overall_accuracy,
        'cross_validation': None,  # see training/train_model.py console output for the CV score at training time
        'per_class_accuracy': per_class_accuracy,
        'safety_metrics': safety,
        'confidence_calibration': framework.confidence_calibration(),
        'avg_inference_time_ms': per_case_ms,
        'model': type(model).__name__,
        'edge_case_evaluation': {
            'description': 'Boundary-blended stress test (training/generate_synthetic_dataset.py::generate_edge_case_dataset) -- deliberately harder, ground truth derived from distance to each level\'s canonical vitals.',
            'dataset_size': len(X_edge),
            'accuracy': edge_accuracy,
        },
    }

    with open(EVAL_DIR / 'evaluation_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)

    plot_confusion_matrix(y_true, y_pred, EVAL_DIR / 'confusion_matrix.png')
    plot_roc_curves(y_true, proba, EVAL_DIR / 'roc_curves.png')
    plot_confidence_calibration(y_true, y_pred, proba, EVAL_DIR / 'confidence_calibration.png')
    plot_dataset_comparison(overall_accuracy, edge_accuracy, EVAL_DIR / 'dataset_comparison.png')

    print("=" * 70)
    print(f"Held-out test cases: {len(test_df)}")
    print(f"Overall accuracy:    {overall_accuracy*100:.1f}%")
    print(f"Undertriage rate:    {safety['undertriage_rate']*100:.1f}%")
    print(f"Overtriage rate:     {safety['overtriage_rate']*100:.1f}%")
    print(f"Avg inference time:  {per_case_ms:.2f} ms/case")
    print("Per-class accuracy:")
    for k, v in per_class_accuracy.items():
        print(f"  {k}: {v*100:.1f}%")
    print("=" * 70)
    print(f"Wrote {EVAL_DIR / 'evaluation_report.json'}")
    print(f"Wrote 4 graphs to {EVAL_DIR}/")


if __name__ == '__main__':
    main()
