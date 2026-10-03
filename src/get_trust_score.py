"""
Role 2 Part 2: ML-Based Trust Score Generator
=============================================
Computes validator Trust Score directly from the trained Random Forest model.

    Trust Score = P(HONEST) = model.predict_proba(X)[:, honest_class_index]

Supports:
- Single observation dictionary / row: scores the transaction directly.
- Historical transaction log (pandas DataFrame or list of dicts):
  Scores each past transaction row via the Random Forest (matching the model's
  exact training representation), and computes the historical average honesty probability:
  Trust(v) = (1/k) * sum(P(HONEST | tx_i))
- Temporal Safety: If current_transaction_id is provided, automatically filters
  out any observation where transaction_id >= current_transaction_id.
- Cold-start fallback (0.80): Returned when no historical observations prior to
  the current transaction are available.
"""

import os
import joblib
import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "uptime",
    "response_time_ms",
    "verification_correct",
    "transaction_outcome",
    "suspicious_flag"
]

FORBIDDEN_COLUMNS = ["true_label", "validator_id", "stake"]

# Authoritative model path in role2_final_handoff/
_HANDOFF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "role2_final_handoff")
if os.path.exists(os.path.join(_HANDOFF_DIR, "trust_classifier_FINAL.pkl")):
    DEFAULT_MODEL_PATH = os.path.join(_HANDOFF_DIR, "trust_classifier_FINAL.pkl")
else:
    DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trust_classifier_FINAL.pkl")

_CACHED_MODEL = None
_HONEST_IDX = None

def load_trust_model(model_path=None):
    """Loads and caches the authoritative trained Random Forest trust classifier."""
    global _CACHED_MODEL, _HONEST_IDX
    if _CACHED_MODEL is None or (model_path is not None and model_path != getattr(_CACHED_MODEL, "_source_path", None)):
        path = model_path or DEFAULT_MODEL_PATH
        if not os.path.exists(path):
            candidates = [
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "role2_final_handoff", "trust_classifier_FINAL.pkl"),
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "trust_classifier_FINAL.pkl"),
                "role2_final_handoff/trust_classifier_FINAL.pkl",
                "trust_classifier_FINAL.pkl"
            ]
            for cand in candidates:
                if os.path.exists(cand):
                    path = cand
                    break
            else:
                raise FileNotFoundError(f"Authoritative model file not found at {path} or candidates {candidates}")
        
        _CACHED_MODEL = joblib.load(path)
        _CACHED_MODEL._source_path = path
        
        classes_list = list(_CACHED_MODEL.classes_)
        if 1 not in classes_list:
            raise ValueError(f"HONEST class (1) not found in model classes: {classes_list}")
        _HONEST_IDX = classes_list.index(1)
        
    return _CACHED_MODEL, _HONEST_IDX

def get_trust_score(validator_behavior_data, current_transaction_id=None, model_path=None):
    """
    Computes the ML-based Trust Score for a validator based on observed behavior.

    Parameters:
    -----------
    validator_behavior_data : dict, list of dicts, pd.DataFrame, pd.Series, or None
        Validator behavioral observations containing the 5 features:
        - uptime (float in [0, 1])
        - response_time_ms (float >= 0)
        - verification_correct (int: 0 or 1, or float)
        - transaction_outcome (int: 0 or 1, or float)
        - suspicious_flag (int: 0 or 1, or float)
        May optionally include 'transaction_id' for temporal filtering.

    current_transaction_id : int, optional
        Current consensus transaction/round index. If provided and transaction_id is present
        in validator_behavior_data, strictly filters for historical records where
        transaction_id < current_transaction_id.

    model_path : str, optional
        Path to trust_classifier_FINAL.pkl (defaults to local/cached model).

    Returns:
    --------
    trust_score : float in [0.0, 1.0]
        Estimated probability P(HONEST) from the trained Random Forest classifier.
    """
    # 1. Cold-start policy fallback for nodes with no historical observations
    if validator_behavior_data is None:
        return 0.80

    # 2. Parse and filter input based on data type
    if isinstance(validator_behavior_data, dict):
        if len(validator_behavior_data) == 0:
            return 0.80
        # Check temporal filtering for single dict
        if "transaction_id" in validator_behavior_data and current_transaction_id is not None:
            if validator_behavior_data["transaction_id"] >= current_transaction_id:
                # No prior history exists before current_transaction_id
                return 0.80
        missing = [c for c in FEATURE_COLUMNS if c not in validator_behavior_data]
        if missing:
            raise ValueError(f"Missing required behavioral features: {missing}")
        X = pd.DataFrame([[validator_behavior_data[col] for col in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)

    elif isinstance(validator_behavior_data, pd.DataFrame):
        if len(validator_behavior_data) == 0:
            return 0.80
        data = validator_behavior_data
        # Enforce temporal filtering if transaction_id and current_transaction_id are present
        if "transaction_id" in data.columns and current_transaction_id is not None:
            data = data[data["transaction_id"] < current_transaction_id]
            if len(data) == 0:
                return 0.80  # Cold start: no observations prior to current transaction
        missing = [c for c in FEATURE_COLUMNS if c not in data.columns]
        if missing:
            raise ValueError(f"Missing required behavioral features: {missing}")
        X = data[FEATURE_COLUMNS].copy()

    elif isinstance(validator_behavior_data, list):
        if len(validator_behavior_data) == 0:
            return 0.80
        data = pd.DataFrame(validator_behavior_data)
        if "transaction_id" in data.columns and current_transaction_id is not None:
            data = data[data["transaction_id"] < current_transaction_id]
            if len(data) == 0:
                return 0.80
        missing = [c for c in FEATURE_COLUMNS if c not in data.columns]
        if missing:
            raise ValueError(f"Missing required behavioral features: {missing}")
        X = data[FEATURE_COLUMNS].copy()

    elif isinstance(validator_behavior_data, pd.Series):
        if len(validator_behavior_data) == 0:
            return 0.80
        if "transaction_id" in validator_behavior_data.index and current_transaction_id is not None:
            if validator_behavior_data["transaction_id"] >= current_transaction_id:
                return 0.80
        missing = [c for c in FEATURE_COLUMNS if c not in validator_behavior_data.index]
        if missing:
            raise ValueError(f"Missing required behavioral features: {missing}")
        X = validator_behavior_data[FEATURE_COLUMNS].to_frame().T

    else:
        raise TypeError(f"Unsupported data type for validator_behavior_data: {type(validator_behavior_data)}")

    # 3. Model inference
    model, honest_idx = load_trust_model(model_path)
    
    # When multiple historical transactions are provided, score each transaction row
    # matching the model's transaction-level training representation, then compute the average P(HONEST)
    probs = model.predict_proba(X)[:, honest_idx]
    trust_score = float(np.mean(probs))
    
    return trust_score

def classify_trust_tier(trust_score):
    """
    Classifies a Trust Score into standard consensus operational tiers:
    - >= 0.90 -> 'HIGH TRUST'
    - 0.70 to < 0.90 -> 'NORMAL / MONITOR'
    - < 0.70 -> 'LOW TRUST / SUSPICIOUS'
    """
    if trust_score >= 0.90:
        return "HIGH TRUST"
    elif trust_score >= 0.70:
        return "NORMAL / MONITOR"
    else:
        return "LOW TRUST / SUSPICIOUS"
