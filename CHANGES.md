# What changed, and why

This documents every substantive change made to close the gap between the
report (Sections 3-6) and the actual code. Organized by area. File paths
are relative to the project root.

## 1. The prediction model was not XGBoost

`data/classifier_model.pkl` was a `sklearn.ensemble.GradientBoostingClassifier`,
not `xgboost.XGBClassifier` as named throughout the report and code
comments. There was also no dataset anywhere in the project to reproduce
the claimed "1,267 real ED patient records (Moon et al., 2019)" /
92.1% accuracy / 8.1% undertriage numbers — and those exact figures were
inconsistent across files (92.0% vs 92.1%, 4.3% vs 8.1% undertriage, 58%
vs 33% safety improvement), which is a strong sign they were placeholder
text from an earlier scaffolding pass (`00_START_HERE.md` /
`DELIVERY_SUMMARY.md`) that never got replaced with real numbers.

**Fixed:**
- `training/generate_synthetic_dataset.py` — a new, clinically-grounded
  synthetic data generator (vital sign thresholds and symptom associations
  per standard KTAS criteria, deliberate overlap between adjacent levels
  so it isn't trivially separable). Clearly documented as NOT the Moon et
  al. dataset.
- `training/train_model.py` — trains a real `xgboost.XGBClassifier`,
  evaluates honestly on a held-out split, saves `data/classifier_model.pkl`
  + `data/classifier_scaler.pkl`. **Real held-out accuracy: 84.8%**
  (5-fold CV 83.7% ± 0.8%), not 92.1%. This is a genuinely good result for
  a 5-class task, just not the number currently in your draft.
- Original files backed up as `data/classifier_model.ORIGINAL_GBM.pkl.bak`
  / `data/classifier_scaler.ORIGINAL.pkl.bak` if you want to compare or revert.
- `models/ktas_predictor.py` — fixed a label-off-by-one bug this swap
  introduced (xgboost's sklearn API requires 0-indexed classes; predictions
  needed +1 to map back to KTAS 1-5). Also removed the hard-coded
  `evaluate_xgboost_model()` numbers; it now reads real numbers from
  `evaluation/evaluation_report.json`.

**Open question for you:** if you actually have the Moon et al. dataset (or
any real patient data) sitting locally and it just wasn't in this zip,
upload it and I'll retrain/re-evaluate on that instead — real data any day
over synthetic. Otherwise Section 5 will need its numbers updated to match
what's real (see `evaluation/` below).

## 2. Symptom extraction was silently broken

`app_production.py` built the symptoms dict as
`{'symptoms': [patient['complaint']]}` — the *entire raw sentence* as one
list item. `ktas_predictor.py` checks for exact strings like `'chest pain'`
in that list, so this was always `False`. **Every symptom flag was always
zero, for every patient, regardless of what they said** — only vitals
ever affected predictions.

**Fixed:**
- `models/ai_orchestrator.py` now calls real symptom extraction itself
  before prediction, so this class of bug can't recur at the call site.
- `models/distilbert_processor.py`: the "DistilBERT" model was actually
  loading `facebook/bart-large-mnli` (different model, ~6x the params).
  Now genuinely loads `distilbert-base-uncased` and uses mean-pooled
  embedding + cosine similarity for semantic symptom matching, layered on
  top of the keyword matcher (which needs no download and is the reliable
  baseline).
- Found and fixed real false positives in the keyword matcher while testing
  (e.g. "lightheaded" matched the bare keyword "head" → false headache;
  "abdominal pain" matched bare "pain" → false chest pain). Switched to
  word-boundary/phrase matching. This is a genuine accuracy fix, not just
  cosmetic — a false chest-pain flag changes the KTAS prediction.

## 3. Explainability wasn't wired in at all

`ai_orchestrator.py` hard-coded `'explanation': None`. The Explainable AI
page's "ROBUST COMPATIBILITY PATCH" fallback logic existed specifically to
survive that. The sidebar caption claiming "MEDA System Accuracy: 92.0%"
was static text, unconnected to any computation.

**Fixed — added `models/shap_explainer.py`, all genuinely computed, not templated:**
- **Real SHAP**: `shap.TreeExplainer` against the actual trained model
  (exact, not approximated — takes ~2ms per prediction after a one-time
  warm-up).
- **Real uncertainty**: confidence margin (top-1 vs top-2 probability) and
  entropy, computed from the model's own output.
- **Real counterfactuals**: the input is actually perturbed (e.g. "HR
  normalized to 75") and re-run through the real model — the reported
  outcome is whatever the model actually predicts, not a canned sentence.
  Only clinically-relevant scenarios for that specific patient are shown.
- `explainability_engine.py` (the rule-based clinical-narrative layer) is
  kept and relabeled honestly — it's a genuine, useful complement (plain-
  language explanation) but was previously mislabeled as "SHAP-inspired"
  when it's hand-authored keyword weights, not SHAP.

## 4. Voice input didn't exist

"Voice Transcript Simulation" was a text box. The "Whisper" pipeline stage
was a 3-second `time.sleep()` progress bar. `models/whisper_handler.py`
(a genuinely solid, real Whisper integration) and `audio_recorder.py`
existed but were never imported anywhere.

**Fixed:**
- `app_production.py` now uses `st.audio_input` (browser mic capture) and
  actually calls `whisper_handler.transcribe_audio_improved()` on
  submission. Requires `streamlit>=1.40` (was pinned to 1.28.1 — bumped in
  `REBUILD_requirements.txt`).
- Degrades gracefully to text-only if `openai-whisper`/`torch` aren't
  installed (tested — shows a clear warning, doesn't crash).
- **I could not test actual transcription in this sandbox** — Whisper's
  model weights download from a domain outside what this sandbox can
  reach. The code is correct and standard; test it on your machine with
  internet access. Let me know if it errors and I'll help debug.

## 5. Other real bugs found while fixing the above

- Temperature wasn't collected in the UI at all — hard-coded to 36.5
  regardless of the actual patient. Now a real form field.
- Respiratory rate *was* collected in the form but silently discarded —
  `ai_orchestrator.py`'s old signature didn't even accept it, defaulting to
  16. Now passed through properly.
- The import-failure fallback silently returned a fake "KTAS 3, 88%
  confidence" result if the AI orchestrator failed to load. For a clinical
  triage tool, silently faking a result is worse than visibly failing —
  replaced with a clear error banner.
- `database/__init__.py` re-exported two functions (`get_patient_record`,
  `get_feedback`) that don't exist in `database.py` — `import database`
  the normal way would have raised `ImportError`. `app_production.py`
  happened to sidestep this via a direct file-path import, so it never
  surfaced, but it's fixed now.
- Command Center's "Pending Intake Queue" was a hard-coded 3-row mock
  table shown regardless of actual data; "Avg Triage Time: 2.1s" was
  static text. Both now pull real data from `medtriage.db` / real timing
  from this session.

## 6. Evaluation — real numbers and real graphs

`training/run_evaluation.py` runs the actual trained model against a
held-out test set (never seen during training) plus a second, deliberately
harder "edge-case" set (boundary-blended between adjacent KTAS levels —
this is what your report's "Dataset 2: Synthetic Diverse Cases" already
described; it just never had real data or a real run behind it).

Produces, all genuinely computed:
- `evaluation/evaluation_report.json` — accuracy, per-class accuracy,
  undertriage/overtriage rates, confidence calibration
- `evaluation/confusion_matrix.png`
- `evaluation/roc_curves.png` (per-class AUC 0.94–0.99)
- `evaluation/confidence_calibration.png`
- `evaluation/dataset_comparison.png` (standard 84.8% vs edge-case 79.3% —
  the harder set being harder is itself a sanity check that this is real)

These are the 4 graphs Section 5 has a placeholder for. **They will not
match the 92.1%-accuracy narrative currently drafted in Section 5** — 84.8%
is a real, defensible, genuinely-computed number; 92.1% wasn't backed by
anything. You'll want to reconcile the prose with whichever numbers you
end up using once you're back in the report (I know you said not to touch
it right now — just flagging this so it's not a surprise later).

## 7. Files that exist but aren't used (not deleted, just labelled)

Your project had three overlapping "main app" drafts and two overlapping
"explainability" modules from earlier iterations. I didn't delete
anything, just added a clear header comment to each so nobody (including
an examiner browsing the repo) mistakes them for live code:

- `ktas_wrapper.py`, `medtriage_stage1_pro.py`, `models/explainability.py`
  — superseded, not imported by `app_production.py`.
- `REBUILD_app.py` — this one is worth knowing about specifically: it's
  not just unused, it's **not valid Python as saved** (starts with
  unstripped chat text — "Here is the complete, production-ready..." and
  a markdown ` ```python ` fence). Confirmed with `ast.parse()`. Harmless
  since nothing imports it, but delete it whenever convenient.

## Verification

Every change above was tested, not just written:
- All 21 Python files parse (except the pre-existing broken `REBUILD_app.py`).
- Full pipeline tested end-to-end (text input → NLP → prediction → SHAP →
  counterfactuals → save to DB) with real, clinically-sensible output.
- All 6 pages smoke-tested headlessly via Streamlit's `AppTest` — zero
  exceptions, including the voice-input path degrading gracefully.
- Training → evaluation → app was re-run from a clean checkout (all
  generated files deleted first) to confirm nothing depends on leftover
  state — reproduced identically (84.8% accuracy both times, fixed seed).

## What you should do next

1. `pip install -r REBUILD_requirements.txt` in your local environment.
2. Run the app (`RUN_APP.bat` or `streamlit run app_production.py`) and
   test the voice input path for real — that's the one piece I couldn't
   verify end-to-end in this sandbox (no internet access to download the
   Whisper model weights here).
3. Take your screenshots for Figures 4.1–4.6 from the real running app.
4. The four graphs in `evaluation/` are ready to drop into Section 5.
5. Let me know if you do have real Moon et al. data somewhere, or if
   you'd like help reconciling Section 5's prose with the real numbers.
