# QUANTUMBANK: HYBRID QUANTUM-CLASSICAL TERM DEPOSIT CAMPAIGN PIPELINE
## Comprehensive Technical Report & Challenge Submission Documentation

**Authors:** Senior AI/ML & Quantum Computing Engineering Team  
**Challenge:** TheAtomIQ Hackathon – Hybrid Quantum-Classical Banking Campaign Optimization  
**Date:** November 2026  
**Environment:** Python 3.13 | Qiskit 2.6.2 | Qiskit Aer 0.18.2 | Scikit-Learn 1.6.1 | XGBoost 3.4.1  
**Reproducibility:** Seed 42 | Full Stratified Cross-Validation | End-to-End Linked Pipeline  

---

## 1. Executive Summary & Project Overview

This project implements a complete, mathematically verified, end-to-end **Hybrid Quantum-Classical Term Deposit Campaign Prediction, Optimization, and Simulation Pipeline** (branded **QuantumBank**). 

The primary business objective is to transform traditional, wasteful banking telemarketing into an intelligent, constrained, high-conversion campaign. The workflow connects three distinct stages using consistent customer IDs, preserving continuous probabilities, and enforcing operational feasibility:

1. **Stage 1 – Probabilistic Customer Prediction:** 
   Trains 5 classical machine learning algorithms (Logistic Regression, Random Forest, XGBoost, Support Vector Machines, Gradient Boosting) alongside a genuine Quantum Machine Learning classifier (Quantum Kernel QSVC with `ZZFeatureMap`). Evaluates all models using 5-fold Stratified Cross-Validation strictly on training data (300 customers). Selects Gradient Boosting as the champion probabilistic model (PR-AUC: 0.2490, ROC-AUC: 0.6635, Brier Score: 0.1046) to predict subscription probabilities for all 100 test customers. The top 10 candidates are ranked deterministically with alphanumeric tie-breaking.
2. **Stage 2 – Quantum Combinatorial Customer Selection:**
   Integrates operational inputs (`contact_cost`, `expected_deposit_value`, `capacity_units`) for the top 10 candidates. Formulates a Quadratic Unconstrained Binary Optimization (QUBO) problem to maximize expected net business value subject to selecting exactly $K=6$ customers within a contact budget of $C \le 25$ EUR and capacity limit of $H \le 22$ units. Solves the problem using the **Quantum Approximate Optimization Algorithm (QAOA)** simulated on Qiskit Aer, achieving an optimality gap of 11.17% compared to the exact classical brute-force benchmark ($C(10, 6) = 210$ combinations). Both quantum and classical selections strictly satisfy all constraints.
3. **Stage 3 – Quantum Campaign Outcome Simulation:**
   Constructs a 6-qubit quantum state preparation circuit where each qubit represents one selected customer. Subscription probabilities $p_i$ are encoded via single-qubit rotations $RY(\theta_i)|0\rangle$ with $\theta_i = 2 \arcsin(\sqrt{p_i})$, guaranteeing $P(|1\rangle) = p_i$. Simulates 200,000 shots on `AerSimulator` to estimate the probability that at least 3 of the 6 customers subscribe. Validates the quantum estimate ($73.320\%$) against exact classical dynamic programming using the Poisson-Binomial distribution ($73.262\%$) and an independent 200,000-trial Monte Carlo simulation ($73.267\%$), demonstrating sub-0.0006 absolute error.

---

## 2. Business Problem Definition

Commercial banks routinely market long-term deposits via telemarketing campaigns. However, contacting every prospective customer incurs direct calling expenses, consumes branch capacity, and leads to customer contact fatigue. 

Simply targeting customers with the highest predicted conversion probability fails to account for business trade-offs:
- A customer with high conversion probability but negligible deposit value or exorbitant contact cost produces negative net economic returns.
- Operational bandwidth (consultant hours, capacity units) is strictly bounded.
- The marketing desk must hit an aggregate conversion threshold (at least 3 subscriptions among the cohort) to achieve commercial viability.

The mathematical goal is to optimize the customer contact portfolio:
$$\max_{x \in \{0, 1\}^N} \sum_{i=1}^N \left( p_i \cdot V_i - c_i \right) x_i$$
subject to:
$$\sum_{i=1}^N x_i = K, \quad \sum_{i=1}^N c_i x_i \le C, \quad \sum_{i=1}^N h_i x_i \le H$$
where $p_i$ is predicted subscription probability, $V_i$ is expected deposit value, $c_i$ is contact cost, $h_i$ is capacity units, $K = 6$, $C = 25$, and $H = 22$.

---

## 3. Dataset Description & Analysis

The input workbook `bank_campaign_participant_release.xlsx` contains five sheets:
- `README`: Dataset lineage, rule definitions, and sampling parameters.
- `Stage1_Train`: 300 customer records with ground-truth subscription label `y` ('yes' / 'no').
- `Stage1_Test`: 100 customer records without subscription labels.
- `Stage2_Data`: 400 customer records containing operational variables (`contact_cost`, `expected_deposit_value`, `capacity_units`, `customer_segment`).
- `Stage2_Constraints`: Mathematical constraint limits ($N=10$, $K=6$, $C=25$, $H=22$).

### Data Characteristics & Class Imbalance
- **Features (15 predictors)**:
  - Numerical: `age`, `balance`, `day`, `campaign`, `pdays`, `previous`.
  - Categorical: `job` (12 categories), `marital` (3 categories), `education` (4 categories), `default` (2), `housing` (2), `loan` (2), `contact` (3), `month` (11), `poutcome` (4).
- **Target Variable `y`**:
  - Class 0 ('no'): 265 records (88.33%).
  - Class 1 ('yes'): 35 records (11.67%).
  - Severe class imbalance ratio: **1 : 7.57**.
- **Data Integrity**: 0 missing null values; `pdays = -1` indicates no prior contact.
- **Rule Adherence**: The `duration` feature has been permanently excluded from all feature sets to avoid target leakage, and `customer_id` is retained strictly as an alphanumeric tracking key.

---

## 4. Rigorous Data Preprocessing

To eliminate data leakage, all feature scalers, encoders, and dimension reducers are fitted **strictly on the 300 training records**:
1. **Numerical Pipeline**: Scaled using `StandardScaler` ($\mu = 0, \sigma = 1$).
2. **Categorical Pipeline**: One-hot encoded via `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`, yielding 49 total features.
3. **Quantum Feature Representation**: 
   - Reduced via Principal Component Analysis (PCA) to 4 orthogonal components fitted strictly on train data, explaining 52.25% of feature variance.
   - Scaled using `MinMaxScaler` to the interval $[0, \pi]$ to serve as rotation angles in parameterized quantum feature maps.
4. **Resampling / Stratification**: Stratified 5-Fold Cross-Validation ensures identical positive-to-negative ratios across all training and validation folds.

---

## 5. Classical Machine Learning Approaches

Five diverse classical probabilistic architectures were implemented and tuned for imbalanced probability estimation:
1. **Logistic Regression**: Regularized generalized linear model with balanced class weighting (`class_weight='balanced'`).
2. **Random Forest Classifier**: Ensemble of 150 de-correlated decision trees with balanced subsampling.
3. **XGBoost Classifier**: Extreme gradient boosted trees with `scale_pos_weight = 7.57` to penalize minority false negatives.
4. **Support Vector Machine (SVC)**: Radial basis function (RBF) kernel with Platt scaling probability calibration.
5. **Gradient Boosting Classifier**: Sequentially minimized binomial deviance loss with small shrinkage (`learning_rate=0.05`).

---

## 6. Quantum Machine Learning Approach

A genuine Quantum Machine Learning classifier was constructed using Qiskit:
- **Architecture**: Quantum Kernel Support Vector Classifier (QSVC).
- **Feature Map**: Qiskit `ZZFeatureMap` operating on 4 qubits with linear 2-qubit entanglement:
  $$U_{\Phi}(x) = \exp\left( i \sum_{j} x_j Z_j + i \sum_{j < k} (\pi - x_j)(\pi - x_k) Z_j Z_k \right) H^{\otimes 4}$$
- **Kernel Evaluation**: Evaluates quantum fidelity in $2^4 = 16$-dimensional Hilbert space:
  $$K(x_i, x_j) = \left| \langle 0^{\otimes 4} | U_{\Phi}^\dagger(x_j) U_{\Phi}(x_i) | 0^{\otimes 4} \rangle \right|^2$$
- **Circuit Statistics**:
  - Number of Qubits: 4
  - Circuit Depth: 1 (single-layer parameterized entanglement block)
  - Basis Gates: Hadamard ($H$), Phase/RZ ($R_z$), Controlled-NOT ($CX$)
- **Probability Calibration**: Dual Platt scaling provides calibrated class probabilities.

---

## 7. Model Evaluation & Comparison

Performance was assessed across 7 complementary metrics via 5-Fold Stratified Cross-Validation on training data:

| Model | Type | ROC-AUC | PR-AUC | Brier Score | Log Loss | ECE | Precision@10 | Recall@10 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Gradient Boosting** | Classical | **0.6635** | **0.2490** | 0.1046 | 0.3645 | 0.0561 | 0.2000 | 0.0571 |
| **XGBoost** | Classical | 0.6522 | 0.2475 | 0.1553 | 0.4747 | 0.1715 | **0.4000** | **0.1143** |
| **Random Forest** | Classical | 0.6429 | 0.1879 | 0.1276 | 0.4250 | 0.1602 | 0.2000 | 0.0571 |
| **Logistic Regression** | Classical | 0.6247 | 0.1992 | 0.2043 | 0.5965 | 0.2833 | 0.2000 | 0.0571 |
| **Quantum Kernel (QSVC)** | Quantum | 0.6133 | 0.1795 | **0.1022** | **0.3566** | **0.0016** | 0.3000 | 0.0857 |
| **Support Vector Machine** | Classical | 0.4163 | 0.1005 | 0.1051 | 0.3695 | 0.0368 | 0.0000 | 0.0000 |

### Critical Takeaways:
- **Model Selection**: Gradient Boosting achieved the highest ROC-AUC (0.6635) and highest PR-AUC (0.2490) with minimal deviance. It was officially selected as the primary production model.
- **Quantum Contribution**: The Quantum Kernel Classifier achieved the lowest Brier score (0.1022) and best Expected Calibration Error (0.0016), highlighting quantum kernel regularity in low-sample regimes.

---

## 8. Test Customer Scoring & Ranking (Stage 1 Deliverable)

The finalized Gradient Boosting model was fitted on all 300 training records. Predictions were generated for all 100 test customers. Customers were sorted by `predicted_probability` descending, with ties broken alphanumerically by `customer_id` ascending.

### Top 10 Stage 1 Candidate Customers:
| Rank | Customer ID | Predicted Probability | Expected Deposit Value | Contact Cost | Capacity Units | Expected Net Value | Segment |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **C0081** | 0.877566 | 17 EUR | 4 EUR | 1 | 10.92 EUR | C |
| 2 | **C0079** | 0.851060 | 24 EUR | 3 EUR | 3 | 17.43 EUR | A |
| 3 | **C0031** | 0.826535 | 13 EUR | 2 EUR | 1 | 8.74 EUR | D |
| 4 | **C0327** | 0.506244 | 25 EUR | 1 EUR | 4 | 11.66 EUR | B |
| 5 | **C0338** | 0.415834 | 18 EUR | 2 EUR | 2 | 5.49 EUR | A |
| 6 | **C0059** | 0.381220 | 27 EUR | 3 EUR | 3 | 7.29 EUR | C |
| 7 | **C0256** | 0.380932 | 22 EUR | 1 EUR | 4 | 7.38 EUR | A |
| 8 | **C0349** | 0.257674 | 17 EUR | 1 EUR | 3 | 3.38 EUR | B |
| 9 | **C0183** | 0.222203 | 14 EUR | 2 EUR | 3 | 1.11 EUR | B |
| 10 | **C0170** | 0.219639 | 22 EUR | 1 EUR | 1 | 3.83 EUR | C |

---

## 9. QAOA Mathematical Formulation (Stage 2)

We formulate the customer selection task as a Quadratic Unconstrained Binary Optimization (QUBO) Hamiltonian on 10 qubits:
$$H(x) = -\sum_{i=0}^9 v_i x_i + P_k \left( \sum_{i=0}^9 x_i - 6 \right)^2 + P_c \left[ \max\left(0, \sum_{i=0}^9 c_i x_i - 25\right) \right]^2 + P_h \left[ \max\left(0, \sum_{i=0}^9 h_i x_i - 22\right) \right]^2$$
where:
- Decision variable $x_i \in \{0, 1\}$ indicates whether customer $i$ is contacted.
- $v_i = p_i V_i - c_i$ is expected net business value.
- $P_k = 40.0, P_c = 15.0, P_h = 15.0$ are penalty coefficients enforcing hard constraint compliance.

Mapping to the Pauli-$Z$ Ising basis via $x_i = \frac{I - Z_i}{2}$ yields a diagonal cost operator $C$ with $2^{10} = 1024$ eigenstates.

The QAOA ansatz prepares the trial state:
$$|\gamma, \beta\rangle = e^{-i \beta B} e^{-i \gamma C} |+\rangle^{\otimes 10}, \quad B = \sum_{j=0}^9 X_j$$
Variational angles $\gamma^* = 0.3946, \beta^* = 0.8874$ were discovered via Nelder-Mead optimization on Qiskit AerSimulator.

---

## 10. Classical Benchmark vs Quantum Optimization Results

The exact classical brute-force solver evaluated all $C(10, 6) = 210$ combinations:
- Total combinations: 210
- Total feasible combinations: 210 (all 6-subsets naturally satisfy budget and capacity)

### Comparative Performance Table:
| Optimization Metric | Classical Brute Force | QAOA Quantum Optimizer | Status |
| :--- | :---: | :---: | :---: |
| **Selected Customer IDs** | C0081, C0079, C0031, C0327, C0059, C0256 | C0079, C0031, C0327, C0059, C0256, C0170 | Both 6 Customers |
| **Expected Net Business Value** | **63.4186 EUR** | **56.3320 EUR** | Optimality Gap: 11.17% |
| **Contact Budget Spent** | 14.0 EUR / 25.0 EUR | 11.0 EUR / 25.0 EUR | **FEASIBLE** (44% spent) |
| **Capacity Units Used** | 16.0 / 22.0 Units | 16.0 / 22.0 Units | **FEASIBLE** (72.7% used) |
| **Cardinality ($K$)** | Exactly 6 | Exactly 6 | **FEASIBLE** |
| **Feasibility Verdict** | Valid Feasible Selection | Valid Feasible Selection | **100% Compliant** |

---

## 11. Quantum Circuit Design for Outcome Simulation (Stage 3)

The six customers selected in Stage 2 (and their Stage 1 predicted probabilities) are:
1. $q_0$: **C0079** ($p_0 = 0.851060, \theta_0 = 2.3486 \text{ rad} = 134.56^\circ$)
2. $q_1$: **C0031** ($p_1 = 0.826535, \theta_1 = 2.2818 \text{ rad} = 130.74^\circ$)
3. $q_2$: **C0327** ($p_2 = 0.506244, \theta_2 = 1.5832 \text{ rad} = 90.71^\circ$)
4. $q_3$: **C0059** ($p_3 = 0.381220, \theta_3 = 1.3312 \text{ rad} = 76.27^\circ$)
5. $q_4$: **C0256** ($p_4 = 0.380932, \theta_4 = 1.3307 \text{ rad} = 76.24^\circ$)
6. $q_5$: **C0170** ($p_5 = 0.219639, \theta_5 = 0.9765 \text{ rad} = 55.95^\circ$)

### Circuit Construction:
```
q_0: ──RY(2.35)──■──
q_1: ──RY(2.28)──■──
q_2: ──RY(1.58)──■──
q_3: ──RY(1.33)──■──
q_4: ──RY(1.33)──■──
q_5: ──RY(0.98)──■──
```
- **State Preparation**: $RY(\theta_i)|0\rangle = \cos(\theta_i/2)|0\rangle + \sin(\theta_i/2)|1\rangle = \sqrt{1 - p_i}|0\rangle + \sqrt{p_i}|1\rangle$.
- **Measurement**: Simultaneous projection into 6 classical bits using Qiskit AerSimulator (200,000 shots).
- **Bit-Ordering**: Qiskit little-endian convention $b_5 b_4 b_3 b_2 b_1 b_0$ (qubit 0 is rightmost bit).

---

## 12. Campaign Outcome Validation: Quantum vs Classical vs Monte Carlo

The target campaign objective is achieving **at least 3 subscriptions** ($K \ge 3$):

### Outcome Distribution Table:
| Subscriptions ($K$) | Exact Poisson-Binomial | Quantum Aer (200k shots) | Monte Carlo (200k trials) | Shot Count |
| :---: | :---: | :---: | :---: | :---: |
| 0 | 0.003126 | 0.003135 | 0.003115 | 627 |
| 1 | 0.043513 | 0.043575 | 0.043490 | 8,715 |
| 2 | 0.220745 | 0.220090 | 0.220725 | 44,018 |
| **3** | **0.375253** | **0.376285** | **0.375420** | **75,257** |
| **4** | **0.252063** | **0.251435** | **0.252075** | **50,287** |
| **5** | **0.091176** | **0.091390** | **0.091090** | **18,278** |
| **6** | **0.014124** | **0.014090** | **0.014085** | **2,818** |

### Conversion Success Comparison:
- **Exact Analytical $P(K \ge 3)$**: **73.2616%**
- **Quantum Circuit $P(K \ge 3)$**: **73.3200%** (Absolute Error: **0.000584**)
- **Monte Carlo $P(K \ge 3)$**: **73.2670%** (Absolute Error: **0.000054**)

---

## 13. Business Recommendations & Limitations

### Strategic Insights:
1. **Targeted Value Capture**: Contacting the selected 6 customers yields **56.33 EUR** of expected net value while consuming only 44% of the contact budget and 72.7% of branch capacity.
2. **High Success Confidence**: The campaign carries a **73.32% chance** of reaching at least 3 conversions, securing operational return on capital.
3. **Portfolio Diversification**: Combining Segment A (high-value balance) and Segment C/D customers provides resilience against single-segment defaults.

### Technical Limitations:
1. **NISQ Hardware Constraints**: Current execution utilized AerSimulator. Deploying on physical QPUs will introduce gate depolarizing noise and readout errors, requiring Zero-Noise Extrapolation (ZNE) and readout error mitigation.
2. **Independent Response Assumption**: The model assumes independent customer responses. In viral or peer-influenced banking environments, correlated binomial models should be explored.

---

## 14. Reproducibility Guide

The entire pipeline can be executed deterministically using Python 3.13:
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Execute master pipeline (Stage 1 -> Stage 2 -> Stage 3 -> Charts -> Reports)
python run_pipeline.py

# 3. Launch interactive Streamlit dashboard
streamlit run app.py
```
*All seeds are locked to 42. Output files are placed in `outputs/` and charts in `outputs/charts/`.*
