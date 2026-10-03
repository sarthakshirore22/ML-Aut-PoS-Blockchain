"""
ROLE 2 HANDOFF: Independent Verification Test Suite
===================================================
Verifies all 10 core requirements for the authoritative Role 2 handoff package:
1. Model loads successfully from trust_classifier.pkl.
2. Model is an authentic sklearn RandomForestClassifier.
3. Model features exactly match the 5 behavioral dimensions in exact order.
4. .predict_proba(X) functions and returns valid 2-class probability distributions.
5. Output probabilities are bounded strictly in [0.0, 1.0].
6. get_trust_score() executes correctly for single transactions and historical sequences.
7. Cold-start policy fallback returns 0.80 when no prior history is available.
8. Multi-transaction historical scoring equals the exact arithmetic mean of individual model predictions.
9. current_transaction_id enforces temporal causality by strictly excluding transactions >= T.
10. Forbidden fields (true_label, validator_id, transaction_id, stake) are never passed as model features.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Configure imports relative to handoff package root
PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PACKAGE_DIR)

from get_trust_score import (
    get_trust_score,
    classify_trust_tier,
    load_trust_model,
    FEATURE_COLUMNS,
    FORBIDDEN_COLUMNS
)

def run_tests():
    print("=" * 75)
    print("ROLE 2 HANDOFF: INDEPENDENT VERIFICATION SUITE")
    print(f"Target Package Directory: {PACKAGE_DIR}")
    print("=" * 75)

    # 1. Model loads
    print("\n[Test 1] Loading model from trust_classifier.pkl...")
    model_path = os.path.join(PACKAGE_DIR, "trust_classifier.pkl")
    assert os.path.exists(model_path), f"Model file missing: {model_path}"
    model, honest_idx = load_trust_model(model_path)
    print(f"  -> Model loaded successfully. Source: {getattr(model, '_source_path', model_path)}")

    # 2. Model is RandomForestClassifier
    print("\n[Test 2] Verifying model object type...")
    assert isinstance(model, RandomForestClassifier), f"Expected RandomForestClassifier, got {type(model)}"
    print(f"  -> Model type confirmed: {type(model)}")
    print(f"  -> n_estimators: {model.n_estimators}, classes: {model.classes_}")
    assert list(model.classes_) == [0, 1], f"Expected classes [0, 1], got {model.classes_}"
    assert honest_idx == 1, f"Expected honest_idx 1, got {honest_idx}"

    # 3. Exactly 5 expected features in order
    print("\n[Test 3] Verifying 5 feature columns in exact training order...")
    expected_features = ["uptime", "response_time_ms", "verification_correct", "transaction_outcome", "suspicious_flag"]
    assert list(model.feature_names_in_) == expected_features, f"Feature mismatch: {model.feature_names_in_}"
    assert FEATURE_COLUMNS == expected_features, f"Module FEATURE_COLUMNS mismatch: {FEATURE_COLUMNS}"
    print(f"  -> 5 Features verified: {list(model.feature_names_in_)}")

    # 4. predict_proba() works
    print("\n[Test 4] Verifying predict_proba() functionality...")
    sample_df = pd.DataFrame([[0.98, 42.0, 1, 1, 0]], columns=expected_features)
    probs = model.predict_proba(sample_df)
    assert probs.shape == (1, 2), f"Expected shape (1, 2), got {probs.shape}"
    assert np.isclose(probs.sum(), 1.0), "Probabilities do not sum to 1.0"
    print(f"  -> predict_proba output: {probs}")

    # 5. Output probability is bounded in [0, 1]
    print("\n[Test 5] Verifying probability bounds in [0.0, 1.0]...")
    p_hon = probs[0, honest_idx]
    assert 0.0 <= p_hon <= 1.0, f"Probability out of bounds: {p_hon}"
    print(f"  -> P(HONEST) = {p_hon:.4f} (strictly in [0.0, 1.0])")

    # 6. get_trust_score() works
    print("\n[Test 6] Verifying get_trust_score() interface...")
    tx_honest = {"uptime": 0.99, "response_time_ms": 30.0, "verification_correct": 1, "transaction_outcome": 1, "suspicious_flag": 0}
    tx_malicious = {"uptime": 0.55, "response_time_ms": 400.0, "verification_correct": 0, "transaction_outcome": 0, "suspicious_flag": 1}
    s_hon = get_trust_score(tx_honest)
    s_mal = get_trust_score(tx_malicious)
    print(f"  -> Honest tx score: {s_hon:.4f} ({classify_trust_tier(s_hon)})")
    print(f"  -> Malicious tx score: {s_mal:.4f} ({classify_trust_tier(s_mal)})")
    assert s_hon >= 0.90, f"Honest node score too low: {s_hon}"
    assert s_mal < 0.70, f"Malicious node score too high: {s_mal}"

    # 7. Empty history returns 0.80 (cold start)
    print("\n[Test 7] Verifying cold-start policy fallback (0.80)...")
    assert get_trust_score(None) == 0.80, "None input did not return 0.80"
    assert get_trust_score({}) == 0.80, "Empty dict did not return 0.80"
    assert get_trust_score([]) == 0.80, "Empty list did not return 0.80"
    assert get_trust_score(pd.DataFrame()) == 0.80, "Empty DataFrame did not return 0.80"
    print("  -> Cold-start fallback cleanly verified as 0.80 across None, {}, [], and empty DataFrame.")

    # 8. Historical score matches mean of individual predictions
    print("\n[Test 8] Verifying multi-transaction history prediction averaging...")
    history = [
        {"uptime": 0.98, "response_time_ms": 40.0, "verification_correct": 1, "transaction_outcome": 1, "suspicious_flag": 0},
        {"uptime": 0.60, "response_time_ms": 350.0, "verification_correct": 0, "transaction_outcome": 0, "suspicious_flag": 1},
        {"uptime": 0.95, "response_time_ms": 55.0, "verification_correct": 1, "transaction_outcome": 1, "suspicious_flag": 0}
    ]
    p1 = get_trust_score(history[0])
    p2 = get_trust_score(history[1])
    p3 = get_trust_score(history[2])
    hist_score = get_trust_score(history)
    expected_mean = np.mean([p1, p2, p3])
    diff = abs(hist_score - expected_mean)
    print(f"  -> Individual scores: {p1:.4f}, {p2:.4f}, {p3:.4f}")
    print(f"  -> Historical score: {hist_score:.6f} vs Expected mean: {expected_mean:.6f} (diff = {diff:.2e})")
    assert diff < 1e-9, f"History aggregation mismatch: diff = {diff}"

    # 9. current_transaction_id excludes current/future transactions
    print("\n[Test 9] Verifying temporal filtering with current_transaction_id...")
    temporal_data = pd.DataFrame([
        {"transaction_id": 5, "uptime": 0.99, "response_time_ms": 32.0, "verification_correct": 1, "transaction_outcome": 1, "suspicious_flag": 0},
        {"transaction_id": 8, "uptime": 0.98, "response_time_ms": 35.0, "verification_correct": 1, "transaction_outcome": 1, "suspicious_flag": 0},
        {"transaction_id": 12, "uptime": 0.50, "response_time_ms": 400.0, "verification_correct": 0, "transaction_outcome": 0, "suspicious_flag": 1},
        {"transaction_id": 20, "uptime": 0.40, "response_time_ms": 500.0, "verification_correct": 0, "transaction_outcome": 0, "suspicious_flag": 1}
    ])
    # At T = 10, only tx 5 and 8 (< 10) are included
    score_t10 = get_trust_score(temporal_data, current_transaction_id=10)
    expected_t10 = get_trust_score(temporal_data.iloc[:2])
    assert np.isclose(score_t10, expected_t10), f"Temporal score mismatch: {score_t10} vs {expected_t10}"
    print(f"  -> Score at T=10 (evaluates tx < 10): {score_t10:.4f} (Matches past-only score: {expected_t10:.4f})")

    # Boundary test: At T = 5, no transactions exist strictly before T=5 -> returns 0.80
    score_t5 = get_trust_score(temporal_data, current_transaction_id=5)
    assert score_t5 == 0.80, f"Expected cold start 0.80, got {score_t5}"
    print(f"  -> Boundary score at T=5 (no prior txs): {score_t5:.4f} (Cold-start verified)")

    # 10. Forbidden columns are not passed as model features
    print("\n[Test 10] Verifying forbidden columns leakage wall...")
    forbidden = ["true_label", "validator_id", "transaction_id", "stake"]
    for col in forbidden:
        assert col not in FEATURE_COLUMNS, f"Forbidden column found in FEATURE_COLUMNS: {col}"
        assert col not in model.feature_names_in_, f"Forbidden column found in model.feature_names_in_: {col}"
    print(f"  -> Leakage wall confirmed: {forbidden} are strictly excluded from model feature matrix X.")

    print("\n" + "=" * 75)
    print("ALL 10 HANDOFF VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 75)

if __name__ == "__main__":
    run_tests()
