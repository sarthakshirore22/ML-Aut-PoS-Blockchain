"""
ML-Based Trust-Aware Validator Selection for Blockchain Consensus
==================================================================
Role 1 — Blockchain Simulation & Data Collection (FINAL INTEGRATED VERSION)

This is the final, verified implementation combining:
  - The explicit np.random.Generator architecture (robust reproducibility)
  - Corrected statistical distributions matching Role 4's authoritative source code
  - Enforced unique defection points for on-off attackers
  - INTEGRATION: Uses Role 2's actual Random Forest model for live trust scoring
    (Task 1) and applies the 0.80 cold-start policy (Task 2).
  - FILE I/O FIX: Forces all reads/writes to the script's absolute directory.
"""

import os
import numpy as np
import pandas as pd
import joblib

# Automatically get the directory where this script file is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# PART 1: GLOBAL CONSTANTS (fixed by spec -- do not change without team sign-off)
# ---------------------------------------------------------------------------
GLOBAL_SEED = 42

N_VALIDATORS = 100
N_HONEST = 80
N_MALICIOUS = 20
N_TRANSACTIONS = 1000

STAKE_MIN = 10.0
STAKE_MAX = 1000.0

ONOFF_COUNT = 10                 # of the 20 malicious validators
DEFECTION_RANGE = (200, 800)     # inclusive range for random defection point

FEATURE_COLUMNS = [
    "uptime", "response_time_ms", "verification_correct",
    "transaction_outcome", "suspicious_flag",
]

SCHEMA_COLUMNS = [
    "transaction_id", "validator_id", "stake",
    "uptime", "response_time_ms", "verification_correct",
    "transaction_outcome", "suspicious_flag", "true_label",
]


# ---------------------------------------------------------------------------
# VALIDATOR POOL
# ---------------------------------------------------------------------------
def create_validator_pool(rng: np.random.Generator) -> pd.DataFrame:
    """
    Build the fixed pool of 100 validators shared by both datasets.
    80 honest (true_label=1), 20 malicious (true_label=0), stake from a
    Pareto distribution clipped to [10, 1000].
    """
    validator_ids = np.arange(1, N_VALIDATORS + 1)

    raw_stakes = rng.pareto(a=1.5, size=N_VALIDATORS) * 150.0 + 20.0
    stakes = np.clip(raw_stakes, STAKE_MIN, STAKE_MAX)

    true_labels = np.array([1] * N_HONEST + [0] * N_MALICIOUS)
    rng.shuffle(true_labels)

    return pd.DataFrame({
        "validator_id": validator_ids,
        "stake": np.round(stakes, 2),
        "true_label": true_labels,
    })


# ---------------------------------------------------------------------------
# BEHAVIORAL DISTRIBUTIONS
# ---------------------------------------------------------------------------
def generate_honest_behavior(rng: np.random.Generator, n: int) -> dict:
    """n independent rows of HONEST validator behavior (5 features)."""
    uptime = np.clip(rng.normal(0.98, 0.015, n), 0.85, 1.00)
    response_time_ms = np.clip(rng.lognormal(3.8, 0.35, n), 15.0, 300.0)
    verification_correct = rng.binomial(1, 0.985, n)
    transaction_outcome = rng.binomial(1, 0.940, n)
    suspicious_flag = rng.binomial(1, 0.020, n)
    return {
        "uptime": np.round(uptime, 4),
        "response_time_ms": np.round(response_time_ms, 2),
        "verification_correct": verification_correct,
        "transaction_outcome": transaction_outcome,
        "suspicious_flag": suspicious_flag,
    }


def generate_malicious_behavior(rng: np.random.Generator, n: int) -> dict:
    """n independent rows of MALICIOUS validator behavior (5 features)."""
    uptime = np.clip(rng.beta(4.0, 1.5, n), 0.40, 0.95)
    response_time_ms = np.clip(rng.lognormal(5.3, 0.60, n), 40.0, 1400.0)
    verification_correct = rng.binomial(1, 0.680, n)
    transaction_outcome = rng.binomial(1, 0.720, n)
    suspicious_flag = rng.binomial(1, 0.320, n)
    return {
        "uptime": np.round(uptime, 4),
        "response_time_ms": np.round(response_time_ms, 2),
        "verification_correct": verification_correct,
        "transaction_outcome": transaction_outcome,
        "suspicious_flag": suspicious_flag,
    }


def _quality_check(df: pd.DataFrame, name: str) -> None:
    """Full validation: zero nulls, in-range values, correct binary types."""
    assert df.isnull().sum().sum() == 0, f"{name}: found missing values!"
    assert df["uptime"].between(0.0, 1.0).all(), f"{name}: uptime out of range!"
    assert df["response_time_ms"].between(10.0, 1500.0).all(), f"{name}: response_time_ms out of range!"
    for col in ["verification_correct", "transaction_outcome", "suspicious_flag", "true_label"]:
        assert df[col].isin([0, 1]).all(), f"{name}: {col} has non-binary values!"
    print(f"  [OK] {name}: zero missing values, all ranges valid, all binary columns clean.")


# ---------------------------------------------------------------------------
# DELIVERABLE 1 -- TRAINING DATA
# ---------------------------------------------------------------------------
def generate_training_data(pool: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    honest_ids = pool.loc[pool["true_label"] == 1, "validator_id"].values
    malicious_ids = pool.loc[pool["true_label"] == 0, "validator_id"].values
    n_honest, n_malicious = len(honest_ids), len(malicious_ids)

    honest_behavior = generate_honest_behavior(rng, n_honest * N_TRANSACTIONS)
    malicious_behavior = generate_malicious_behavior(rng, n_malicious * N_TRANSACTIONS)

    honest_df = pd.DataFrame(honest_behavior)
    honest_df["validator_id"] = np.tile(honest_ids, N_TRANSACTIONS)
    honest_df["transaction_id"] = np.repeat(np.arange(1, N_TRANSACTIONS + 1), n_honest)
    honest_df["true_label"] = 1

    malicious_df = pd.DataFrame(malicious_behavior)
    malicious_df["validator_id"] = np.tile(malicious_ids, N_TRANSACTIONS)
    malicious_df["transaction_id"] = np.repeat(np.arange(1, N_TRANSACTIONS + 1), n_malicious)
    malicious_df["true_label"] = 0

    df = pd.concat([honest_df, malicious_df], ignore_index=True)
    df = df.merge(pool[["validator_id", "stake"]], on="validator_id", how="left")
    df = df[SCHEMA_COLUMNS].sort_values(["transaction_id", "validator_id"]).reset_index(drop=True)

    # Force save to SCRIPT_DIR
    out_path = os.path.join(SCRIPT_DIR, "validator_behavior_log_TRAINING.csv")
    df.to_csv(out_path, index=False)

    print("\n=== DELIVERABLE 1: Training Data ===")
    print(f"  Total rows:      {len(df):,} (expected {N_VALIDATORS * N_TRANSACTIONS:,})")
    print(f"  Honest rows:     {(df['true_label'] == 1).sum():,}")
    print(f"  Malicious rows:  {(df['true_label'] == 0).sum():,}")
    _quality_check(df, "TRAINING dataset")
    print(f"  Saved to: {out_path}")

    return df


# ---------------------------------------------------------------------------
# DELIVERABLE 2 -- COMPARISON DATA (with on-off camouflage attackers)
# ---------------------------------------------------------------------------
def generate_comparison_data(pool: pd.DataFrame, rng: np.random.Generator):
    honest_ids = pool.loc[pool["true_label"] == 1, "validator_id"].values
    malicious_ids = pool.loc[pool["true_label"] == 0, "validator_id"].values
    n_honest = len(honest_ids)

    shuffled_malicious = malicious_ids.copy()
    rng.shuffle(shuffled_malicious)
    onoff_ids = shuffled_malicious[:ONOFF_COUNT]
    always_malicious_ids = shuffled_malicious[ONOFF_COUNT:]
    n_onoff, n_always = len(onoff_ids), len(always_malicious_ids)

    low, high = DEFECTION_RANGE[0], DEFECTION_RANGE[1] + 1
    defection_points = rng.integers(low, high, size=n_onoff)
    while len(np.unique(defection_points)) < n_onoff:
        defection_points = rng.integers(low, high, size=n_onoff)

    switch_df = pd.DataFrame({
        "validator_id": onoff_ids,
        "defection_transaction_id": defection_points,
    }).sort_values("validator_id").reset_index(drop=True)

    honest_behavior = generate_honest_behavior(rng, n_honest * N_TRANSACTIONS)
    honest_df = pd.DataFrame(honest_behavior)
    honest_df["validator_id"] = np.tile(honest_ids, N_TRANSACTIONS)
    honest_df["transaction_id"] = np.repeat(np.arange(1, N_TRANSACTIONS + 1), n_honest)
    honest_df["true_label"] = 1

    always_behavior = generate_malicious_behavior(rng, n_always * N_TRANSACTIONS)
    always_df = pd.DataFrame(always_behavior)
    always_df["validator_id"] = np.tile(always_malicious_ids, N_TRANSACTIONS)
    always_df["transaction_id"] = np.repeat(np.arange(1, N_TRANSACTIONS + 1), n_always)
    always_df["true_label"] = 0

    onoff_blocks = []
    tx_ids_full = np.arange(1, N_TRANSACTIONS + 1)
    for vid, defect_at in zip(onoff_ids, defection_points):
        n_before = int(defect_at) - 1
        n_after = N_TRANSACTIONS - n_before
        honest_part = generate_honest_behavior(rng, n_before)
        malicious_part = generate_malicious_behavior(rng, n_after)
        block = {col: np.concatenate([honest_part[col], malicious_part[col]]) for col in FEATURE_COLUMNS}
        block_df = pd.DataFrame(block)
        block_df["validator_id"] = vid
        block_df["transaction_id"] = tx_ids_full
        block_df["true_label"] = 0
        onoff_blocks.append(block_df)
    onoff_df = pd.concat(onoff_blocks, ignore_index=True)

    df = pd.concat([honest_df, always_df, onoff_df], ignore_index=True)
    df = df.merge(pool[["validator_id", "stake"]], on="validator_id", how="left")
    df = df[SCHEMA_COLUMNS].sort_values(["transaction_id", "validator_id"]).reset_index(drop=True)

    out_path_df = os.path.join(SCRIPT_DIR, "validator_behavior_log_COMPARISON.csv")
    out_path_sw = os.path.join(SCRIPT_DIR, "onoff_validator_switch_points.csv")
    df.to_csv(out_path_df, index=False)
    switch_df.to_csv(out_path_sw, index=False)

    print("\n=== DELIVERABLE 2: Comparison Data ===")
    print(f"  Total rows:      {len(df):,} (expected {N_VALIDATORS * N_TRANSACTIONS:,})")
    print(f"  Honest rows:     {(df['true_label'] == 1).sum():,}")
    print(f"  Malicious rows:  {(df['true_label'] == 0).sum():,}")
    print(f"  Always-malicious IDs: {sorted(always_malicious_ids.tolist())}")
    print("  On-off (camouflage) IDs & defection points:")
    for _, row in switch_df.iterrows():
        print(f"    validator {int(row['validator_id'])}: defects at tx {int(row['defection_transaction_id'])}")
    _quality_check(df, "COMPARISON dataset")
    print(f"  Saved to: {out_path_df}")
    print(f"  Saved to: {out_path_sw}")

    return df, switch_df


# ---------------------------------------------------------------------------
# DELIVERABLE 3 -- LIVE TRUST SCORING + DUAL SELECTION ENGINE
# ---------------------------------------------------------------------------
def compute_live_trust_and_run_selection(model, comparison_df: pd.DataFrame,
                                          pool: pd.DataFrame, rng: np.random.Generator):
    validator_ids = np.sort(pool["validator_id"].values)
    n_validators = len(validator_ids)

    static_info = pool.set_index("validator_id").loc[validator_ids]
    stakes = static_info["stake"].values.astype(float)
    true_labels = static_info["true_label"].values

    grouped_features = {
        t: g.set_index("validator_id").loc[validator_ids, FEATURE_COLUMNS].values
        for t, g in comparison_df.groupby("transaction_id")
    }
    transactions = sorted(grouped_features.keys())

    running_prob_sum = np.zeros(n_validators)
    running_count = np.zeros(n_validators)
    honest_class_idx = int(np.where(model.classes_ == 1)[0][0])

    run_a_rows, run_b_rows = [], []

    for t in transactions:
        # Task 2 Fix: Cold start explicitly set to 0.80 instead of 0.50
        trust = np.where(
            running_count == 0,
            0.80,
            running_prob_sum / np.where(running_count == 0, 1, running_count),
        )

        p_a = stakes / stakes.sum()
        idx_a = rng.choice(n_validators, p=p_a)
        run_a_rows.append({
            "transaction_id": t,
            "selected_validator_id": int(validator_ids[idx_a]),
            "selected_validator_true_label": int(true_labels[idx_a]),
        })

        weights_b = stakes * trust
        p_b = weights_b / weights_b.sum()
        idx_b = rng.choice(n_validators, p=p_b)
        run_b_rows.append({
            "transaction_id": t,
            "selected_validator_id": int(validator_ids[idx_b]),
            "selected_validator_true_label": int(true_labels[idx_b]),
            "trust_score_of_selected": round(float(trust[idx_b]), 4),
        })

        features_t = grouped_features[t]
        features_t_df = pd.DataFrame(features_t, columns=FEATURE_COLUMNS)
        proba = model.predict_proba(features_t_df)
        p_honest_t = proba[:, honest_class_idx]
        running_prob_sum += p_honest_t
        running_count += 1

    run_a_df = pd.DataFrame(run_a_rows)
    run_b_df = pd.DataFrame(run_b_rows)

    assert run_a_df.isnull().sum().sum() == 0, "Run A log has missing values!"
    assert run_b_df.isnull().sum().sum() == 0, "Run B log has missing values!"
    assert len(run_a_df) == N_TRANSACTIONS and len(run_b_df) == N_TRANSACTIONS

    out_path_a = os.path.join(SCRIPT_DIR, "run_A_traditional_selection_log.csv")
    out_path_b = os.path.join(SCRIPT_DIR, "run_B_trust_aware_selection_log.csv")
    
    run_a_df.to_csv(out_path_a, index=False)
    run_b_df.to_csv(out_path_b, index=False)

    print("\n=== DELIVERABLE 3: Live Trust & Selection ===")
    print(f"  Transactions processed: {len(transactions):,}")
    mal_a = (run_a_df["selected_validator_true_label"] == 0).sum()
    mal_b = (run_b_df["selected_validator_true_label"] == 0).sum()
    print(f"  Run A (Traditional) -- malicious selected: {mal_a} / {len(run_a_df)} ({mal_a/10:.1f}%)")
    print(f"  Run B (Trust-Aware) -- malicious selected: {mal_b} / {len(run_b_df)} ({mal_b/10:.1f}%)")
    print(f"  Saved to: {out_path_a}")
    print(f"  Saved to: {out_path_b}")

    return run_a_df, run_b_df


# ---------------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Suppress scikit-learn InconsistentVersion warnings cleanly during inference
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    rng = np.random.default_rng(GLOBAL_SEED)

    print("#" * 70)
    print("# Blockchain Validator Simulation Pipeline (Final Integrated Version)")
    print("#" * 70)

    print("\n--- Building validator pool ---")
    pool = create_validator_pool(rng)
    print(f"  {N_VALIDATORS} validators created "
          f"({(pool['true_label'] == 1).sum()} honest, {(pool['true_label'] == 0).sum()} malicious)")
    print(f"  Stake range: [{pool['stake'].min():.2f}, {pool['stake'].max():.2f}]")
    print(f"  Honest-group stake mean:    {pool.loc[pool['true_label']==1, 'stake'].mean():.2f}")
    print(f"  Malicious-group stake mean: {pool.loc[pool['true_label']==0, 'stake'].mean():.2f}")

    training_df = generate_training_data(pool, rng)
    comparison_df, switch_df = generate_comparison_data(pool, rng)

    # Task 1 Fix: Load the real model using the absolute script path
    model_path = os.path.join(SCRIPT_DIR, "trust_classifier_FINAL.pkl")
    
    print("\n--- Loading Role 2's real trained Random Forest trust model ---")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"\nERROR: Could not find '{model_path}'.\n"
                                f"Please ensure 'trust_classifier_FINAL.pkl' is placed in the exact same folder as this script.")
    
    real_model = joblib.load(model_path)
    assert list(real_model.classes_) == [0, 1], "Unexpected model classes!"
    print(f"  Loaded: {type(real_model).__name__}, n_estimators={real_model.n_estimators}")
    
    run_a_df, run_b_df = compute_live_trust_and_run_selection(real_model, comparison_df, pool, rng)

    print("\n" + "#" * 70)
    print("# PIPELINE COMPLETE -- Sanity Stats")
    print("#" * 70)
    print(f"Training rows:     {len(training_df):,}")
    print(f"Comparison rows:   {len(comparison_df):,}")
    print(f"On-off validators: {len(switch_df)}")
    print(f"Run A selections:  {len(run_a_df):,}")
    print(f"Run B selections:  {len(run_b_df):,}")
    print("\nMean uptime by group (TRAINING data) -- expect honest~0.98, malicious~0.73:")
    print(training_df.groupby("true_label")["uptime"].mean().rename(index={0: "malicious", 1: "honest"}))
    print("\nMean uptime by group (COMPARISON data, includes on-off camouflage):")
    print(comparison_df.groupby("true_label")["uptime"].mean().rename(index={0: "malicious", 1: "honest"}))
    print("\nAll files successfully written to:")
    print(f"  {SCRIPT_DIR}")