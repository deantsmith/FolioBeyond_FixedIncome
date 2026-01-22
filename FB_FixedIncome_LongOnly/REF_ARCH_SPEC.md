# REF_ARCH_SPEC.md

## Overview
Technical audit and refactoring specification for `folio_beyond_fixed_income.py`. 
**Objective:** Eliminate look-ahead bias, ensure constraint integrity, enforce strictly point-in-time data usage, and vectorize optimization logic for performance.

---

## 1. Critical Defect Remediation (Priority: High)

### A. Eliminate Look-Ahead Bias (Stress Testing)
* **Issue:** `_load_stress_test_data` reads static values derived from full historical context (including future events relative to simulation date).
* **Requirement:** * Remove dependency on static `ETF Stress Test Final (values)...xlsx`.
    * Implement **Point-in-Time (PIT)** stress testing: Scenarios must be defined *ex-ante* or strictly derived from data available prior to `current_date`.

### B. Enforce Constraint Integrity
* **Issue:** `_optimize_single_date` silently drops constraints (Stress Test, Duration, Risk) upon SLSQP failure to force a solution.
* **Requirement:**
    * Remove logic that re-runs optimization with `simple_constraints` upon failure.
    * **Failure Protocol:** If constraints cannot be met, default to:
        1.  Hold previous weights, OR
        2.  Exit to Cash/Benchmark.
    * Log specific constraint violations rather than masking them.

### C. Standardize Expected Returns
* **Issue:** Logic mixes forward-looking `projected_returns` (YTM) with backward-looking `scaled_returns.mean()` based on data availability.
* **Requirement:**
    * Enforce a single methodology for $\mu$ (Expected Returns) vector.
    * Do not fallback to historically derived means if projected yields are missing; treat as data gap error.

### D. Correct Covariance Scaling
* **Issue:** Momentum scalars are applied effectively to the Covariance Matrix ($\Sigma$), implying uniform volatility shifts based on price trends.
* **Requirement:**
    * Apply Momentum signals to the Expected Returns vector ($\mu$), not the Covariance Matrix.
    * Use standard volatility forecasting (EWMA, GARCH) for $\Sigma$ adjustments if regime scaling is required.

---

## 2. Architectural Refactoring (Priority: Medium)

### A. Vectorization & Optimization Engine
* **Current:** `_optimize_single_date` reconstructs covariance matrices and constraints inside the temporal loop ($O(N)$ complexity).
* **Target:** * **Pre-computation:** Calculate rolling covariance matrices (`df.rolling().cov()`) and rolling mean returns for the entire history *before* entering the optimization loop.
    * **Loop Logic:** The loop should only slice pre-computed matrices at index $t$ and pass them to `scipy.optimize`.

### B. Separation of Concerns
* **Refactor `optimize_portfolio` into:**
    1.  `SignalGenerator`: Generates $\mu$ (Expected Returns) and $\Sigma$ (Covariance) for all timesteps.
    2.  `OptimizationEngine`: Pure solver logic accepting $\mu$, $\Sigma$, and Constraints.

### C. Modular Data Loading
* **Current:** Repetitive functions (`_load_returns_data`, `_load_projected_returns`, etc.).
* **Target:** Implement single generic loader:
    ```python
    def load_time_series(path: Path, index_col: str = 'Date') -> pd.DataFrame:
        ...
    ```

### D. Environment Configuration
* **Issue:** Hardcoded paths (`/Users/deansmith/...`).
* **Requirement:**
    * Implement `os.environ` or `python-dotenv` for base paths.
    * Use relative paths for repository structure.

---

## 3. Algorithmic Enhancements

### A. Transaction Cost Awareness
* **Current:** Costs applied post-hoc in performance analysis.
* **Requirement:** Incorporate transaction costs into the solver objective function to prevent signal churning:
    $$ \text{Maximize: } w^T \mu - \lambda (w^T \Sigma w) - \text{TC} \times \sum |w_t - w_{t-1}| $$

### B. Solver Robustness
* **Target:** Wrap `scipy.optimize.minimize` in a robust handler that checks strictly for `status=0` (Success).
* **Validation:** Verify Positive Definiteness of $\Sigma$ matrix before passing to solver; apply regularization (shrinkage) if singular.