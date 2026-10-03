# Role 2 Final Model Test & Behavioral Verification Report

**Project:** ML-Based Trust-Aware Validator Selection for Blockchain Consensus  
**Component:** Role 2 (Machine Learning Trust Scoring)  
**Package:** `ROLE 2 HANDOFF/`  
**Model Artifact:** `ROLE 2 HANDOFF/trust_classifier.pkl`  
**Dataset:** `ROLE 2 HANDOFF/validator_behavior_log_TRAINING.csv`  
**Date:** 2026-09-15  
**Evaluation Status:** **VERIFIED & AUDITED**  

---

## Executive Summary

This report documents the empirical evaluation and forensic behavioral testing of the final, authoritative Random Forest model (`trust_classifier.pkl`) delivered by Role 2.

All reported figures were directly measured by executing code against the actual 100,000-row CSV dataset and the serialized scikit-learn model object.

---

## A. Dataset Verification

The provided dataset was validated programmatically before and during model testing:

| Dataset Dimension | Measured Value | Verification Notes |
|---|---|---|
| **Total Rows** | **100,000** | Zero missing values across all 9 columns |
| **Total Unique Validators** | **100** | Validator IDs 1 through 100 |
| **Transactions per Validator** | **1,000** | Min = 1,000, Max = 1,000 (exactly 1,000 each) |
| **Validators with `true_label = 1`** | **80** | Fair / Honest validator nodes |
| **Validators with `true_label = 0`** | **20** | Malicious / Unfair validator nodes |
| **Class Distribution Ratio** | **80% / 20%** | Exactly matches the intended consensus ratio |
| **Ground Truth Label Consistency** | **100.0%** | Every validator maintains a single constant label across all 1,000 rounds |
| **Total Fair Transactions** | **80,000** | 80 nodes × 1,000 transactions |
| **Total Malicious Transactions** | **20,000** | 20 nodes × 1,000 transactions |

> **Dataset Context:** This dataset is the synthetic benchmark dataset generated for Role 2. It models consensus telemetry across validator uptime, network latency, cryptographic block verification, transaction completion, and peer anomaly flags.

---

## B. Model Verification

The final model artifact `trust_classifier.pkl` was loaded and inspected using scikit-learn:

- **Model Class:** `sklearn.ensemble._forest.RandomForestClassifier`
- **Class Implementation:** Authentic scikit-learn estimator (not a dictionary, JSON config, wrapper, or heuristic approximation)
- **Estimators (`n_estimators`):** `100` decision trees
- **Class Weighting:** `balanced` (inverse class frequency weighting)
- **Model Classes (`classes_`):** `[0, 1]` (`0 = MALICIOUS`, `1 = HONEST`)
- **Target Index:** Index `1` corresponds strictly to the `HONEST` class
- **Number of Input Features (`n_features_in_`):** `5`
- **Exact Input Feature Columns (`feature_names_in_`):**
  1. `uptime`
  2. `response_time_ms`
  3. `verification_correct`
  4. `transaction_outcome`
  5. `suspicious_flag`
- **Strict Data Leakage Wall (Excluded Fields):**
  - `true_label`: Excluded from feature matrix $X$ (used strictly as training target and evaluation ground truth)
  - `validator_id`: Excluded from feature matrix $X$ (prevents node ID memorization)
  - `transaction_id`: Excluded from feature matrix $X$ (used strictly as temporal filter)
  - `stake`: Excluded from feature matrix $X$ (reserved for downstream consensus selection formula $	ext{Stake} 	imes 	ext{Trust}$)

---

## C. Full 100K Behavioral Test

> [!WARNING]
> **METHODOLOGICAL NOTICE: FULL DATASET EVALUATION**  
> The metrics in this section were computed across the **entire 100,000 rows** of `validator_behavior_log_TRAINING.csv`. Because this full dataset includes the 80,000 transactions used during supervised model training (alongside the 20,000 held-out test split), these results demonstrate full-dataset behavioral compliance and fitting fidelity. They must **NOT** be interpreted as an unbiased out-of-sample generalization test.

### Measured Performance Metrics (100,000 Transactions)

| Evaluation Metric | Measured Value | Theoretical Meaning |
|---|---|---|
| **Accuracy** | **0.999750 (99.98%)** | Overall fraction of correct binary classifications |
| **Precision (Honest Class 1)** | **0.999875** | Fraction of predicted honest transactions that were genuinely honest |
| **Recall (Honest Class 1)** | **0.999812** | Fraction of genuinely honest transactions correctly identified |
| **F1-Score (Honest Class 1)** | **0.999844** | Harmonic mean of precision and recall |
| **ROC-AUC** | **0.999935** | Probability that a random honest transaction scores higher than a random malicious one |

### Full 100K Confusion Matrix

```
                         Predicted Malicious (0)    Predicted Honest (1)
Actual Malicious (0)             19,990                        10
Actual Honest (1)                    15                    79,985
```

- **True Negatives (TN):** `19,990` malicious transactions correctly identified as Malicious.
- **False Positives (FP):** `10` malicious transactions misclassified as Honest.
- **False Negatives (FN):** `15` honest transactions misclassified as Malicious.
- **True Positives (TP):** `79,985` honest transactions correctly identified as Honest.
- **Total Misclassifications:** Exactly **25 out of 100,000 transactions (0.025%)**.

---

## D. Validator-Level Trust Score Analysis

Consensus selection operates on **validators**, not individual disconnected transactions. For each of the 100 validators, all 1,000 transaction rounds were evaluated, computing individual honesty probabilities $P(	ext{HONEST})$ and calculating cumulative validator-level statistics:

### Summary Comparison: Fair vs. Malicious Nodes

| Metric | Fair / Honest Validators (80 nodes) | Malicious / Unfair Validators (20 nodes) | Separation Gap |
|---|:---:|:---:|:---:|
| **Mean Validator Trust** | **0.999389** | **0.002237** | **+0.997152** |
| **Median Validator Trust** | **0.999480** | **0.002155** | **+0.997325** |
| **Minimum Validator Trust** | **0.997350** (Validator 73) | **0.000520** (Validator 40) | — |
| **Maximum Validator Trust** | **0.999980** (Validator 57) | **0.003840** (Validator 69) | **+0.993510** |

### Operational Tier Distribution (at Validator Level)

| Operational Tier | Trust Score Definition | Fair Validators | Malicious Validators |
|---|---|:---:|:---:|
| **HIGH TRUST** | $	ext{Trust} \ge 0.90$ | **80 / 80 (100.0%)** | **0 / 20 (0.0%)** |
| **NORMAL / MONITOR** | $0.70 \le 	ext{Trust} < 0.90$ | **0 / 80 (0.0%)** | **0 / 20 (0.0%)** |
| **LOW TRUST / SUSPICIOUS** | $	ext{Trust} < 0.70$ | **0 / 80 (0.0%)** | **20 / 20 (100.0%)** |

### Key Observations
1. **Zero Misclassifications at Validator Level:** While individual transactions have 25 edge-case errors, averaging over multiple rounds eliminates transient noise completely. Every honest validator achieves $\ge 0.99735$ trust, and every malicious validator achieves $\le 0.00384$ trust.
2. **259× Discrimination Ratio:** The lowest fair validator score (0.99735) is 259 times higher than the highest malicious validator score (0.00384).

---

## E. Always-Denying Behavioral Test

### Fact-Check: Real CSV vs. Always-Denying Hypothesis
> [!IMPORTANT]
> The real CSV dataset does **NOT** contain a validator that denies 100% of transactions. In the actual dataset, all 20 malicious nodes exhibit probabilistic adversarial behavior (verification denial rates between 28.7% and 35.7%, block failure rates between 25.9% and 30.0%, and peer suspicious flags between 29.6% and 36.3%).

### 1. Real CSV: Highest Observed Denial Frequency Node (Validator 55)
The node exhibiting the highest verification denial frequency in `validator_behavior_log_TRAINING.csv` is **Validator 55**:

- **Validator ID:** `55`
- **True Ground Truth Label:** `0` (Malicious)
- **Total Transactions:** `1,000`
- **Verification Denials (`verification_correct == 0`):** `357` out of 1,000 (**35.70%**)
- **Failed Block Outcomes (`transaction_outcome == 0`):** `268` out of 1,000 (**26.80%**)
- **Peer Suspicious Flags (`suspicious_flag == 1`):** `300` out of 1,000 (**30.00%**)
- **Mean Uptime:** `0.7230` (72.30%)
- **Mean Latency:** `239.52 ms`
- **Average $P(	ext{HONEST})$ (Historical Trust):** **`0.002340`**
- **Median $P(	ext{HONEST})$:** `0.000000`
- **Minimum $P(	ext{HONEST})$:** `0.000000`
- **Maximum $P(	ext{HONEST})$:** `0.330000`
- **Transactions Classified as Malicious:** **1,000 / 1,000 (100.00%)**
- **Final Classification:** **`LOW TRUST / SUSPICIOUS`**

### 2. Synthetic Pure Always-Denial Profile Test
To test how the model behaves when faced with an uninterrupted, pure always-denying adversarial node, the model was tested against a pure denial input vector:

```python
pure_denial_tx = {
    'uptime': 0.50,
    'response_time_ms': 450.0,
    'verification_correct': 0,
    'transaction_outcome': 0,
    'suspicious_flag': 1
}
```

- **$P(	ext{MALICIOUS})$:** **`1.000000`**
- **$P(	ext{HONEST})$ (Trust Score):** **`0.000000`**
- **Trust Tier:** **`LOW TRUST / SUSPICIOUS`**
- **Conclusion:** The Random Forest drives the Trust Score of an always-denying behavioral profile to **exactly 0.000000**, ensuring complete penalization in downstream stake-weighting ($	ext{Stake} 	imes 0.0 = 0.0$).

---

## F. History Function Verification

The reusable scoring function [`get_trust_score(history)`](file:///C:/Users/thaku/OneDrive/Desktop/New%20folder/ROLE%202%20HANDOFF/get_trust_score.py) was tested against manual mathematical computation across 1,000-round validator logs:

$$	ext{Trust}(v) = rac{1}{k} \sum_{i=1}^k P(	ext{HONEST} \mid 	ext{tx}_i)$$

| Validator Evaluated | Ground Truth Class | `get_trust_score(history)` | Manual $rac{1}{k}\sum P(	ext{HONEST}\mid 	ext{tx}_i)$ | Numerical Difference | Verification |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **Validator 1** | Fair (1) | `0.997830` | `0.997830` | $0.00 	imes 10^0$ | **MATCH** |
| **Validator 2** | Fair (1) | `0.999580` | `0.999580` | $0.00 	imes 10^0$ | **MATCH** |
| **Validator 55** | Malicious (0) | `0.002340` | `0.002340` | $0.00 	imes 10^0$ | **MATCH** |
| **Validator 22** | Malicious (0) | `0.002080` | `0.002080` | $0.00 	imes 10^0$ | **MATCH** |
| **Validator 69** | Malicious (0) | `0.003840` | `0.003840` | $0.00 	imes 10^0$ | **MATCH** |

### Verification of No Pre-Inference Feature Averaging
Passing pre-averaged raw features into the Random Forest alters split conditions on binary variables. For Validator 69, pre-averaging features into floats yields `0.000000`, whereas evaluating each transaction on its true binary inputs and averaging probabilities yields `0.003840`. The function correctly evaluates each transaction row independently.

---

## G. Temporal Filtering Verification

Temporal safety was verified using Validator 1's chronological records from the dataset:

| Test Condition | Historical Observations Included | Generated Trust Score | Verification vs Manual Slice | Temporal Causality Status |
|---|:---:|:---:|:---:|:---:|
| `current_transaction_id = 10` | 9 past txs (`id < 10`) | **`1.000000`** | Exact match with `df[id < 10]` | **PASS** (txs $\ge 10$ excluded) |
| `current_transaction_id = 50` | 49 past txs (`id < 50`) | **`0.993469`** | Exact match with `df[id < 50]` | **PASS** (txs $\ge 50$ excluded) |
| `current_transaction_id = 100` | 99 past txs (`id < 100`) | **`0.996768`** | Exact match with `df[id < 100]` | **PASS** (txs $\ge 100$ excluded) |
| `current_transaction_id = 1` | 0 past txs (boundary) | **`0.800000`** | Cold-start fallback policy | **PASS** (No prior history) |

---

## H. Feature Importance (Learned Random Forest Weights)

Extracted directly from `model.feature_importances_` of the final model:

| Rank | Behavioral Feature Column | RF Gini Importance | Percentage |
|:---:|---|:---:|:---:|
| 1 | `uptime` | `0.555802` | **55.58%** |
| 2 | `response_time_ms` | `0.372165` | **37.22%** |
| 3 | `verification_correct` | `0.042612` | **4.26%** |
| 4 | `suspicious_flag` | `0.024466` | **2.45%** |
| 5 | `transaction_outcome` | `0.004956` | **0.50%** |
| **Total** | **All 5 Input Features** | **1.000000** | **100.00%** |

> [!NOTE]
> These values represent **Random Forest learned Gini impurity reductions**. They are **NOT** linear weights or multipliers. The model evaluates features non-linearly through multi-threshold decision trees.

---

## I. Data Leakage Checks

Independent code inspection verified that:
1. `true_label` is **never** passed into feature matrix $X$ during inference or testing.
2. `validator_id` is **never** passed into feature matrix $X$.
3. `transaction_id` is **never** passed into feature matrix $X$; it is used solely as a temporal row filter.
4. `stake` is **never** passed into feature matrix $X$; it is strictly reserved for downstream consensus selection ($	ext{Stake} 	imes 	ext{Trust}$).

---

## J. Generalization Evaluation (Validator-Disjoint Test)

### Why Validator-Disjoint Evaluation Matters
In a blockchain network, new validators join over time. A standard random train/test split mixes transactions from the *same* 100 validators across both training and test sets. To rigorously test whether the model generalizes to **completely unseen validator nodes**, a **Validator-Disjoint Group Split** (`GroupShuffleSplit`) was executed:

- **Total Dataset:** 100 validators (100,000 transactions)
- **Train Group:** 80 validators (80,000 transactions)
- **Test Group:** **20 completely unseen validators** (20,000 transactions)
- **Validator Overlap:** **0 validators (0.0% overlap)**

### Measured Out-of-Sample Performance on 20 Unseen Validators

| Metric | Out-of-Sample Result | Benchmark Requirement | Status |
|---|:---:|:---:|:---:|
| **ROC-AUC** | **0.999999** | $\ge 0.85$ | **PASS (+0.1499)** |
| **F1-Score** | **0.999750** | $\ge 0.80$ | **PASS (+0.1998)** |
| **Precision** | **0.999750** | $\ge 0.80$ | **PASS** |
| **Recall** | **0.999750** | — | **PASS** |
| **Accuracy** | **0.999600** (99.96%) | — | **PASS** |

### Confusion Matrix on 20 Unseen Validators (20,000 Transactions)
```
                         Predicted Malicious (0)    Predicted Honest (1)
Actual Malicious (0)              3,996                         4
Actual Honest (1)                     4                    15,996
```

### Forensic Generalization Conclusion
The model's near-perfect performance on unseen validators is not due to memorizing node IDs. Rather, it is because the synthetic dataset generator created clear, consistent behavioral distributions for honest vs. malicious nodes:
- Honest nodes consistently operate with high uptime ($\sim 98\%$) and low latency ($\sim 47	ext{ ms}$).
- Malicious nodes consistently operate with degraded uptime ($\sim 73\%$) and high latency ($\sim 240	ext{ ms}$).

The Random Forest learns these generalized behavioral boundaries, enabling it to classify new, unseen validators with high accuracy.
