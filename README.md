# QuantumBank: Hybrid Quantum-Classical Term Deposit Campaign Optimizer
## Comprehensive Technical Documentation, Reports & Local Execution Guide

[![Python 3.13](https://img.shields.io/badge/Python-3.13-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Qiskit 2.6.2](https://img.shields.io/badge/Qiskit-2.6.2-6929C4.svg?logo=qiskit&logoColor=white)](https://qiskit.org/)
[![Qiskit Aer](https://img.shields.io/badge/Qiskit--Aer-0.18.2-8A3FFC.svg)](https://github.com/Qiskit/qiskit-aer)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.6.1-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-EB5424.svg)](https://xgboost.ai/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![TheAtomIQ Hackathon](https://img.shields.io/badge/Challenge-TheAtomIQ%20Hackathon-blue.svg)](data/PROBLEM_STATEMENT_THEATOMIQHACKTHON.pdf)

> **QuantumBank** is an enterprise-grade, mathematically verified, end-to-end **Hybrid Quantum-Classical Banking Campaign System**. It couples classical machine learning with Quantum Machine Learning (QML) kernels, solves a constrained combinatorial customer contact portfolio using the **Quantum Approximate Optimization Algorithm (QAOA)**, and simulates campaign conversion outcomes via a **6-qubit parameterized state-preparation quantum circuit** sampled on `AerSimulator`.

---

## 📑 Table of Contents
1. [Local Setup & Execution Guide](#-local-setup--execution-guide)
   - [Prerequisites](#prerequisites)
   - [Quick Start: macOS & Linux](#step-by-step-macos--linux)
   - [Quick Start: Windows](#step-by-step-windows)
   - [Running the Streamlit Web Application](#running-the-interactive-dashboard)
   - [Jupyter Notebooks Execution](#running-the-step-by-step-notebooks)
   - [Troubleshooting & FAQs](#troubleshooting--faqs)
2. [Reports & Downloads Catalog (Page 7 Deliverables)](#-reports--downloads-catalog-page-7-deliverables)
   - [Stage Result CSV Files](#1-stage-result-csv-files)
   - [Model Benchmarks & Documentation](#2-models-benchmarks--documentation)
   - [Publication Charts Gallery (18 Charts)](#3-publication-quality-charts-gallery)
   - [All-in-One Project Results Bundle](#4-all-in-one-project-archive)
3. [System Architecture & Workflow](#-system-architecture--workflow)
4. [Executive Summary & Technical Report](#-executive-summary--technical-report)
   - [Business Problem Definition](#business-problem-definition)
   - [Dataset & Imbalance Handling](#dataset-characteristics--imbalance-handling)
   - [Data Preprocessing Pipeline](#rigorous-data-preprocessing)
   - [Stage 1: Machine Learning & Quantum Kernel Benchmark](#stage-1-probabilistic-customer-prediction)
   - [Stage 1: Test Customer Scoring & Ranking](#stage-1-top-10-candidate-ranking)
   - [Stage 2: QAOA Combinatorial Optimization Formulation](#stage-2-qaoa-mathematical-formulation)
   - [Stage 2: Quantum vs Classical Benchmark Results](#stage-2-quantum-vs-classical-benchmark-results)
   - [Stage 3: 6-Qubit Quantum Outcome Simulation Circuit](#stage-3-quantum-circuit-design--outcome-simulation)
   - [Stage 3: Validation: Quantum vs Exact Poisson-Binomial vs Monte Carlo](#stage-3-validation-quantum-vs-exact-poisson-binomial-vs-monte-carlo)
   - [Strategic Business Impact & Limitations](#strategic-business-recommendations--limitations)
5. [Streamlit Dashboard Multi-Page Tour](#-streamlit-dashboard-multi-page-tour)
6. [Repository Structure](#-repository-structure)
7. [Hackathon Verification Checklist](#-hackathon-verification-checklist)

---

## 💻 Local Setup & Execution Guide

### Prerequisites
- **Python**: Version `3.10`, `3.11`, `3.12`, or `3.13` (Tested and verified on Python 3.13).
- **Package Manager**: `pip` (or `conda` / `miniconda`).
- **RAM**: Minimum 4 GB (8 GB recommended for 200,000-shot quantum simulation).
- **OS**: macOS (Apple Silicon M-series or Intel), Windows 10/11, or Linux (Ubuntu 20.04+).

---

### Step-by-Step: macOS & Linux

#### 1. Open Terminal and Navigate to Project Root
```bash
cd "/Users/USER/Library/Mobile Documents/com~apple~CloudDocs/HYBRID_TERM_DEPOSIT"
```

#### 2. Create and Activate a Dedicated Virtual Environment
Using a clean virtual environment ensures dependency isolation:
```bash
# Create virtual environment named 'venv'
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate
```
*(Your terminal prompt will now show `(venv)`).*

#### 3. Install All Required Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### 4. Run the Full End-to-End Pipeline
Execute the automated master pipeline script:
```bash
# Option A: Direct Python runner
python run_pipeline.py

# Option B: Using the automated shell script
./run.sh
```
**What happens during this step (completed in ~12-15 seconds):**
- Loads and validates `bank_campaign_participant_release.xlsx` (train, test, and operational sheets).
- Applies train-fitted preprocessing (StandardScaler, OneHotEncoder, PCA to 4 components, MinMax to $[0, \pi]$).
- Executes 5-fold Stratified Cross-Validation on 5 classical models + 1 QML Quantum Kernel Classifier (`ZZFeatureMap`).
- Generates `outputs/model_comparison.csv`.
- Scores all 100 test customers and ranks the Top 10 -> `outputs/stage1_predictions.csv`.
- Formulates QUBO Hamiltonian and runs QAOA on `AerSimulator` alongside classical brute-force ($C(10, 6) = 210$) -> `outputs/stage2_optimization_results.csv`.
- Constructs 6-qubit $RY(\theta_i)$ quantum circuit and runs 200,000 shots on `AerSimulator` -> `outputs/stage3_quantum_simulation_results.csv`.
- Computes exact Poisson-Binomial dynamic programming distribution and 200,000 Monte Carlo trials.
- Generates all 18 publication-quality charts into `outputs/charts/`.
- Compiles the professional multi-page Executive Summary PDF -> `reports/executive_summary.pdf`.

#### 5. Launch the Streamlit Interactive Dashboard
```bash
# Option A: Direct Streamlit runner
streamlit run app.py

# Option B: Using the dashboard shell script
./start_dashboard.sh
```
Open your browser and navigate to:
```
http://localhost:8501
```

---

### Step-by-Step: Windows

#### 1. Open Command Prompt (`cmd.exe`) or PowerShell
Navigate to the project folder:
```cmd
cd "C:\path\to\HYBRID_TERM_DEPOSIT"
```

#### 2. Create and Activate Virtual Environment
```cmd
python -m venv venv
venv\Scripts\activate
```

#### 3. Install Dependencies
```cmd
pip install --upgrade pip
pip install -r requirements.txt
```

#### 4. Run Pipeline and Launch Dashboard
You can run the provided Windows batch files directly:
```cmd
:: Run the master pipeline
run.bat

:: Launch the Streamlit web dashboard
start_dashboard.bat
```
Alternatively:
```cmd
python run_pipeline.py
streamlit run app.py
```

---

### Running the Interactive Dashboard
The Streamlit application (`app.py`) provides a 7-page interactive analytics platform:
```bash
streamlit run app.py
```
Key features:
- **Real-time Filters**: Select customer segments, change QAOA simulation shots, inspect individual candidate qubits.
- **Interactive Visualizations**: Interactive Plotly charts for the Poisson-Binomial distribution, ROC curves, and candidate trade-offs.
- **One-Click File Downloads**: Instant download buttons for all CSV results, PDF summaries, and technical reports on Page 7.

---

### Running the Step-by-Step Notebooks
For research, academic review, or experimentation, run the 5 modular Jupyter notebooks in the `notebooks/` directory:
```bash
pip install jupyterlab
jupyter lab
```
Navigate to `QuantumBank_Term_Deposit_Optimizer/notebooks/`:
1. `01_data_exploration.ipynb` — Exploratory Data Analysis, class imbalance, and data schema checks.
2. `02_classical_machine_learning.ipynb` — 5 Classical probabilistic classifiers, ROC/PR curves, calibration.
3. `03_quantum_machine_learning.ipynb` — Quantum Kernel Construction (`ZZFeatureMap`), QSVC training & metrics.
4. `04_quantum_optimization.ipynb` — QUBO formulation, QAOA circuit implementation, and brute-force validation.
5. `05_quantum_simulation.ipynb` — 6-Qubit $RY$ state preparation, AerSimulator execution, Poisson-Binomial DP, and Monte Carlo.

---

### Troubleshooting & FAQs

| Issue | Cause | Solution |
| :--- | :--- | :--- |
| `ModuleNotFoundError: No module named 'qiskit'` | Running Python outside the activated virtual environment | Run `source venv/bin/activate` (Mac/Linux) or `venv\Scripts\activate` (Windows), then install with `pip install -r requirements.txt`. |
| `ModuleNotFoundError: No module named 'qiskit_aer'` on Apple Silicon (M1/M2/M3/M4) | Prebuilt binary architecture mismatch | Upgrade pip and reinstall: `pip install --upgrade pip` followed by `pip install qiskit-aer>=0.16.0`. Wheels for macOS `arm64` are officially supported. |
| `Port 8501 is already in use` | Another instance of Streamlit is running | Specify a custom port: `streamlit run app.py --server.port 8502`. |
| `FileNotFoundError: bank_campaign_participant_release.xlsx` | Pipeline executed from an unexpected nested path | Both `run_pipeline.py` and `src/utils.py` automatically detect either the root folder or `QuantumBank_Term_Deposit_Optimizer`. Always run from either the repository root or `QuantumBank_Term_Deposit_Optimizer/`. |
| Batch scripts reference `Anaconda3` on Windows | Default Anaconda path differed | Activate your own virtual environment and run `python run_pipeline.py` or edit `run.bat` to reference `python`. |

---

## 📑 Reports & Downloads Catalog (Page 7 Deliverables)

Page 7 of the dashboard (**7. Reports & Downloads**) compiles all final artifacts, downloadable CSV tables, technical documentation, and visual figures generated by the pipeline:

### 1. Stage Result CSV Files

All output CSV files are stored in `outputs/` (or `QuantumBank_Term_Deposit_Optimizer/outputs/`):

#### 📄 `stage1_predictions.csv` (100 Test Customers)
- **Purpose**: Scores all 100 unlabeled test customers using the champion Gradient Boosting model.
- **Key Columns**:
  - `customer_id`: Unique identifier (e.g., `C0081`, `C0079`).
  - `predicted_probability`: Continuous term-deposit conversion probability $[0, 1]$.
  - `rank`: Integer rank from 1 to 100 sorted descending by probability (with deterministic alphanumeric tie-breaking).
  - `top_10_flag`: Boolean indicator (`True` for ranks 1–10).
- **Top 3 Customers**:
  1. `C0081` — $P = 87.76\%$ (Rank 1)
  2. `C0079` — $P = 85.11\%$ (Rank 2)
  3. `C0031` — $P = 82.65\%$ (Rank 3)

#### 📄 `stage2_optimization_results.csv` (Top 10 Candidates)
- **Purpose**: Combinatorial customer contact portfolio optimization comparing QAOA with the exact Classical Brute-Force benchmark.
- **Key Columns**:
  - `customer_id`: Candidate identifier.
  - `predicted_probability`: Probability from Stage 1.
  - `contact_cost`: Calling expenditure in EUR ($c_i$).
  - `expected_deposit_value`: Expected deposit balance in EUR ($V_i$).
  - `capacity_units`: Operational branch consultant units ($h_i$).
  - `customer_segment`: Assigned segment (`A`, `B`, `C`, `D`).
  - `expected_net_value`: $v_i = p_i \cdot V_i - c_i$.
  - `qaoa_decision`: Binary flag (`1` = Selected by QAOA, `0` = Rejected).
  - `classical_decision`: Binary flag (`1` = Selected by Brute Force, `0` = Rejected).
  - `selected_flag`: Final cohort selection indicator.
- **Optimization Outcome**:
  - **QAOA Selected 6**: `C0079`, `C0031`, `C0327`, `C0059`, `C0256`, `C0170` (Net Value: **56.33 EUR** | Budget: **11.0 / 25.0 EUR** | Capacity: **16.0 / 22.0 Units**).
  - **Classical Brute Force 6**: `C0081`, `C0079`, `C0031`, `C0327`, `C0059`, `C0256` (Net Value: **63.42 EUR** | Budget: **14.0 / 25.0 EUR** | Capacity: **16.0 / 22.0 Units**).
  - **Optimality Gap**: **11.17%** (Feasibility: 100% compliant with all constraints).

#### 📄 `stage3_quantum_simulation_results.csv` (6 Selected Customers & Verification)
- **Purpose**: Qubit encoding parameters, 200,000-shot quantum Aer simulation results, exact Poisson-Binomial analytical distribution, and Monte Carlo validation.
- **Key Columns**:
  - `customer_id`: Selected customer.
  - `qubit_index`: Assigned qubit ($q_0$ through $q_5$).
  - `qubit_order_little_endian`: Bitstring mapping convention ($q_0$ rightmost).
  - `predicted_probability`: Continuous conversion probability $p_i$.
  - `rotation_angle_radians`: $\theta_i = 2 \arcsin(\sqrt{p_i})$ (radians).
  - `rotation_angle_degrees`: $\theta_i$ in degrees ($55.89^\circ$ to $134.60^\circ$).
  - `total_shots`: 200,000 shots.
  - `quantum_p_at_least_3`: Quantum Aer simulated probability of $\ge 3$ subscriptions (**0.7332**).
  - `exact_p_at_least_3`: Exact Poisson-Binomial dynamic programming probability (**0.732616**).
  - `monte_carlo_p_at_least_3`: 200,000-trial Monte Carlo benchmark (**0.732670**).
  - `quantum_absolute_error`: $|P_{\text{quantum}} - P_{\text{exact}}| = \mathbf{0.000584}$ (Sub-0.0006 error).
  - `monte_carlo_absolute_error`: $|P_{\text{MC}} - P_{\text{exact}}| = \mathbf{0.000054}$.

---

### 2. Models, Benchmarks & Documentation

#### 📊 `outputs/model_comparison.csv`
Comprehensive 5-fold Stratified Cross-Validation benchmark evaluating 5 classical algorithms and 1 Quantum Machine Learning classifier across 10 evaluation metrics:

| Model | Model Family | ROC-AUC | PR-AUC | Brier Score | Log Loss | ECE | Precision@10 | Recall@10 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Gradient Boosting** ⭐ | Classical | **0.6635** | **0.2490** | 0.1046 | 0.3645 | 0.0561 | 0.2000 | 0.0571 |
| **XGBoost** | Classical | 0.6522 | 0.2475 | 0.1553 | 0.4747 | 0.1715 | **0.4000** | **0.1143** |
| **Random Forest** | Classical | 0.6429 | 0.1879 | 0.1276 | 0.4250 | 0.1602 | 0.2000 | 0.0571 |
| **Logistic Regression** | Classical | 0.6247 | 0.1992 | 0.2043 | 0.5965 | 0.2833 | 0.2000 | 0.0571 |
| **Quantum Kernel (QSVC)** ⚛️ | Quantum | 0.6133 | 0.1795 | **0.1022** | **0.3566** | **0.0016** | 0.3000 | 0.0857 |
| **Support Vector Machine** | Classical | 0.4163 | 0.1005 | 0.1051 | 0.3695 | 0.0368 | 0.0000 | 0.0000 |

*Key Findings*:
1. **Gradient Boosting** achieved highest ROC-AUC (0.6635) and PR-AUC (0.2490) and was selected as champion.
2. **Quantum Kernel QSVC** achieved the lowest Brier Score (0.1022) and superior Expected Calibration Error (0.0016), demonstrating that quantum feature spaces provide smooth, well-calibrated margins in small sample regimes.

#### 📑 `reports/technical_report.md`
The complete, unabridged 14-section technical report containing all mathematical formulations, circuit diagrams, empirical tables, and reproducibility instructions.

#### 📑 `reports/executive_summary.pdf`
A professionally typeset multi-page PDF generated via ReportLab designed for banking executives and business stakeholders, presenting high-level ROI, capital allocation efficiency, and risk metrics.

---

### 3. Publication-Quality Charts Gallery

All 18 high-resolution visualizations are saved in `outputs/charts/`:

| File Name | Stage / Category | Description |
| :--- | :--- | :--- |
| `01_target_class_distribution.png` | EDA | Severe class imbalance visualization (88.33% 'no' vs 11.67% 'yes', 1:7.57 ratio). |
| `02_customer_age_distribution.png` | EDA | Age histogram and kernel density estimation by subscription status. |
| `03_customer_balance_distribution.png` | EDA | Log-transformed bank account balance distribution. |
| `04_job_wise_subscription_analysis.png` | EDA | Conversion rates across 12 occupation categories (students & retirees highest). |
| `05_education_wise_subscription_analysis.png`| EDA | Conversion rates across education tiers (tertiary education leading). |
| `06_correlation_heatmap.png` | EDA | Feature correlation matrix confirming zero multicollinearity. |
| `07_model_performance_comparison.png` | Stage 1 Benchmarking | Grouped bar chart comparing ROC-AUC, PR-AUC, and Brier Score across all 6 models. |
| `08_roc_curves.png` | Stage 1 Benchmarking | 5-Fold cross-validated ROC curves with random baseline comparison. |
| `09_precision_recall_curves.png` | Stage 1 Benchmarking | Precision-Recall curves highlighting performance under minority class imbalance. |
| `10_calibration_plots.png` | Stage 1 Benchmarking | Reliability diagrams illustrating calibration error (QSVC closest to diagonal). |
| `11_top_10_predicted_customers.png` | Stage 1 Output | Ranked conversion probabilities for the top 10 test customers ($P = 0.88$ to $0.22$). |
| `12_selected_six_customer_comparison.png` | Stage 2 Optimization | Expected deposit value vs contact cost trade-off for selected candidates. |
| `13_budget_utilization_chart.png` | Stage 2 Optimization | Contact budget expenditure: Classical (14.0€ / 25€) vs QAOA (11.0€ / 25€). |
| `14_capacity_utilization_chart.png` | Stage 2 Optimization | Operational capacity utilization: 16.0 / 22.0 units (72.7% consumed). |
| `15_qaoa_vs_classical_objective_comparison.png` | Stage 2 Optimization | Expected net business value comparison (Classical 63.42€ vs QAOA 56.33€). |
| `16_quantum_measurement_distribution.png` | Stage 3 Simulation | Histogram of 64 measured bitstring states sampled across 200,000 shots. |
| `17_exact_probability_distribution.png` | Stage 3 Simulation | Exact Poisson-Binomial distribution $P(K=k)$ with $K \ge 3$ shaded. |
| `18_quantum_vs_classical_probability_comparison.png` | Stage 3 Simulation | Side-by-side bar chart showing sub-0.0006 error across Quantum Aer, Exact DP, and MC. |

---

### 4. All-in-One Project Archive

- **Archive File**: `project_results.zip`
- **Location**: Root directory and `QuantumBank_Term_Deposit_Optimizer/project_results.zip`
- **Contents**: Complete bundle of all 4 output CSVs, 18 charts, technical report, and executive summary PDF for distribution.

---

## 🏛️ System Architecture & Workflow

```
   ┌────────────────────────────────────────────────────────┐
   │             Stage 1: Predict & Rank                    │
   │  • 300 Train (1:7.6 Imbalance) -> 5-Fold Stratified CV │
   │  • Models: LogisticReg, RF, XGBoost, SVM, GBDT, QML   │
   │  • Champion Model: Gradient Boosting (PR-AUC: 0.2490)  │
   │  • Predict 100 Test Customers -> Extract Top 10        │
   └───────────────────────────┬────────────────────────────┘
                               │ (Top 10 Candidates & Continuous Probabilities)
                               ▼
   ┌────────────────────────────────────────────────────────┐
   │          Stage 2: Constrained Optimization             │
   │  • Maximize Expected Net Value = p_i * V_i - c_i       │
   │  • Hard Constraints: K=6, Budget<=25€, Capacity<=22U   │
   │  • QAOA Solver (10 Qubits, AerSimulator, Nelder-Mead)  │
   │  • Classical Brute Force Benchmark (210 combinations)  │
   │  • Final Selection: 6 Feasible Customers (56.33€ Net)  │
   └───────────────────────────┬────────────────────────────┘
                               │ (Selected 6 Customers & Original Probabilities)
                               ▼
   ┌────────────────────────────────────────────────────────┐
   │          Stage 3: Quantum Outcome Simulation           │
   │  • 6-Qubit Circuit: RY(theta_i)|0> State Preparation   │
   │  • theta_i = 2 * arcsin(sqrt(p_i)) => P(|1>) = p_i     │
   │  • AerSimulator Sampling (200,000 shots)               │
   │  • Target: P(Subscribers >= 3)                         │
   │  • Quantum: 73.32% | Exact Analytical: 73.26%          │
   └────────────────────────────────────────────────────────┘
```

---

## 🔬 Executive Summary & Technical Report

### Business Problem Definition
Commercial telemarketing campaigns for bank deposits face strict operational constraints:
1. Contacting customers incurs direct calling expenses ($c_i$).
2. Customer relationship managers have limited branch capacity ($h_i$).
3. The marketing desk requires at least 3 successful subscriptions among the contacted cohort ($K \ge 3$) to justify campaign overhead.

Targeting customers based solely on predicted conversion probability ignores deposit value and operational friction. QuantumBank formulates a constrained portfolio optimization:
$$\max_{x \in \{0, 1\}^N} \sum_{i=1}^N \left( p_i \cdot V_i - c_i \right) x_i$$
subject to:
$$\sum_{i=1}^N x_i = 6, \quad \sum_{i=1}^N c_i x_i \le 25 \text{ EUR}, \quad \sum_{i=1}^N h_i x_i \le 22 \text{ Units}$$

---

### Dataset Characteristics & Imbalance Handling
The dataset `bank_campaign_participant_release.xlsx` contains:
- **`Stage1_Train`**: 300 customer records with ground-truth subscription label `y` (`yes` / `no`).
- **`Stage1_Test`**: 100 customer records without subscription labels.
- **`Stage2_Data`**: 400 customer records containing operational variables (`contact_cost`, `expected_deposit_value`, `capacity_units`, `customer_segment`).
- **`Stage2_Constraints`**: Constraint limits ($N=10$, $K=6$, $C=25$, $H=22$).

**Data Characteristics**:
- **Features (15 predictors)**: `age`, `balance`, `day`, `campaign`, `pdays`, `previous`, `job`, `marital`, `education`, `default`, `housing`, `loan`, `contact`, `month`, `poutcome`.
- **Target Imbalance**: 265 negatives (`no`, 88.33%) vs 35 positives (`yes`, 11.67%) $\to$ **1 : 7.57 severe imbalance ratio**.
- **Data Leakage Safeguards**: The `duration` attribute was permanently excluded from all feature sets, as call duration is unknown prior to placing the call. `customer_id` is tracked strictly as an alphanumeric key.

---

### Rigorous Data Preprocessing
All transformers are fitted **strictly on the 300 training records** to eliminate data leakage:
1. **Numerical Pipeline**: Scaled using `StandardScaler` ($\mu=0, \sigma=1$).
2. **Categorical Pipeline**: One-hot encoded via `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`, producing 49 total features.
3. **Quantum Feature Representation**:
   - Dimensionality reduced via Principal Component Analysis (PCA) to 4 orthogonal components fitted strictly on train data, explaining 52.25% of feature variance.
   - Scaled using `MinMaxScaler` to the interval $[0, \pi]$ for rotation angles in parameterized quantum feature maps.
4. **Stratified Cross-Validation**: 5-Fold Stratified CV preserves the 11.67% minority ratio across every split.

---

### Stage 1: Probabilistic Customer Prediction
Five classical machine learning algorithms were trained alongside a genuine Quantum Machine Learning classifier:
- **Logistic Regression**: Balanced class weighting (`class_weight='balanced'`).
- **Random Forest**: 150 de-correlated decision trees with balanced subsampling.
- **XGBoost**: Scale position weight `scale_pos_weight = 7.57` to penalize minority false negatives.
- **Support Vector Machine (SVC)**: RBF kernel with Platt scaling probability calibration.
- **Gradient Boosting**: Sequentially minimized binomial deviance loss (`learning_rate=0.05`).
- **Quantum Kernel QSVC**: Qiskit `ZZFeatureMap` operating on 4 qubits with linear 2-qubit entanglement:
  $$U_{\Phi}(x) = \exp\left( i \sum_{j} x_j Z_j + i \sum_{j < k} (\pi - x_j)(\pi - x_k) Z_j Z_k \right) H^{\otimes 4}$$
  Evaluates quantum state overlap in 16-dimensional Hilbert space:
  $$K(x_i, x_j) = \left| \langle 0^{\otimes 4} | U_{\Phi}^\dagger(x_j) U_{\Phi}(x_i) | 0^{\otimes 4} \rangle \right|^2$$

**Champion Model**: **Gradient Boosting Classifier** achieved the highest ROC-AUC (**0.6635**) and highest PR-AUC (**0.2490**).

---

### Stage 1: Top 10 Candidate Ranking
Scoring all 100 test customers with the champion Gradient Boosting model produced the Top 10 Candidate Pool:

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

### Stage 2: QAOA Mathematical Formulation
The customer selection task is formulated as a Quadratic Unconstrained Binary Optimization (QUBO) Hamiltonian on 10 qubits:
$$H(x) = -\sum_{i=0}^9 v_i x_i + P_k \left( \sum_{i=0}^9 x_i - 6 \right)^2 + P_c \left[ \max\left(0, \sum_{i=0}^9 c_i x_i - 25\right) \right]^2 + P_h \left[ \max\left(0, \sum_{i=0}^9 h_i x_i - 22\right) \right]^2$$
- Penalty parameters: $P_k = 40.0, P_c = 15.0, P_h = 15.0$.
- Qubit mapping: $x_i = \frac{I - Z_i}{2}$ creates an Ising diagonal cost operator $C$.
- QAOA ansatz trial state:
  $$|\gamma, \beta\rangle = e^{-i \beta B} e^{-i \gamma C} |+\rangle^{\otimes 10}, \quad B = \sum_{j=0}^9 X_j$$
- Variational angles: $\gamma^* = 0.3946, \beta^* = 0.8874$ (Nelder-Mead optimization on `AerSimulator`).

---

### Stage 2: Quantum vs Classical Benchmark Results
The exact classical brute-force solver evaluated all $C(10, 6) = 210$ combinations:

| Optimization Metric | Classical Brute Force | QAOA Quantum Optimizer | Status |
| :--- | :---: | :---: | :---: |
| **Selected Customers** | `C0081, C0079, C0031, C0327, C0059, C0256` | `C0079, C0031, C0327, C0059, C0256, C0170` | Exactly 6 Customers |
| **Expected Net Value** | **63.4186 EUR** | **56.3320 EUR** | Optimality Gap: **11.17%** |
| **Contact Cost Spent** | 14.0 EUR / 25.0 EUR | 11.0 EUR / 25.0 EUR | **FEASIBLE** (44% spent) |
| **Capacity Used** | 16.0 / 22.0 Units | 16.0 / 22.0 Units | **FEASIBLE** (72.7% used) |
| **Cardinality ($K$)** | Exactly 6 | Exactly 6 | **FEASIBLE** |
| **Feasibility Verdict**| 100% Compliant | 100% Compliant | **Valid Selection** |

---

### Stage 3: Quantum Circuit Design & Outcome Simulation
The six customers selected by QAOA were encoded into a 6-qubit quantum state preparation circuit:
1. $q_0$: **C0079** ($p_0 = 0.851060 \to \theta_0 = 2.3492 \text{ rad} = 134.60^\circ$)
2. $q_1$: **C0031** ($p_1 = 0.826535 \to \theta_1 = 2.2824 \text{ rad} = 130.77^\circ$)
3. $q_2$: **C0327** ($p_2 = 0.506244 \to \theta_2 = 1.5833 \text{ rad} = 90.72^\circ$)
4. $q_3$: **C0059** ($p_3 = 0.381220 \to \theta_3 = 1.3309 \text{ rad} = 76.26^\circ$)
5. $q_4$: **C0256** ($p_4 = 0.380932 \to \theta_4 = 1.3304 \text{ rad} = 76.22^\circ$)
6. $q_5$: **C0170** ($p_5 = 0.219639 \to \theta_5 = 0.9755 \text{ rad} = 55.89^\circ$)

**State Preparation**:
$$RY(\theta_i)|0\rangle = \cos(\theta_i/2)|0\rangle + \sin(\theta_i/2)|1\rangle = \sqrt{1 - p_i}|0\rangle + \sqrt{p_i}|1\rangle$$
$$P(\text{qubit}_i = 1) = \sin^2(\theta_i/2) = p_i$$

```
q_0: ──RY(2.35)──■──
q_1: ──RY(2.28)──■──
q_2: ──RY(1.58)──■──
q_3: ──RY(1.33)──■──
q_4: ──RY(1.33)──■──
q_5: ──RY(0.98)──■──
```
Simulated using 200,000 measurement shots on `AerSimulator` under Qiskit's little-endian bit-ordering convention ($q_0$ is the least significant bit).

---

### Stage 3: Validation: Quantum vs Exact Poisson-Binomial vs Monte Carlo
The campaign success condition requires **at least 3 subscriptions** ($K \ge 3$):

| Subscriptions ($K$) | Exact Poisson-Binomial | Quantum Aer (200k shots) | Monte Carlo (200k trials) | Shot Count |
| :---: | :---: | :---: | :---: | :---: |
| 0 | 0.003126 | 0.003135 | 0.003115 | 627 |
| 1 | 0.043513 | 0.043575 | 0.043490 | 8,715 |
| 2 | 0.220745 | 0.220090 | 0.220725 | 44,018 |
| **3** | **0.375253** | **0.376285** | **0.375420** | **75,257** |
| **4** | **0.252063** | **0.251435** | **0.252075** | **50,287** |
| **5** | **0.091176** | **0.091390** | **0.091090** | **18,278** |
| **6** | **0.014124** | **0.014090** | **0.014085** | **2,818** |

**Conversion Success Comparison**:
- **Exact Analytical Poisson-Binomial $P(K \ge 3)$**: **73.2616%**
- **Quantum Circuit $P(K \ge 3)$**: **73.3200%** (Absolute Error: **0.000584**)
- **Monte Carlo $P(K \ge 3)$**: **73.2670%** (Absolute Error: **0.000054**)

---

### Strategic Business Recommendations & Limitations

#### Strategic Business Recommendations
1. **Targeted Value Capture**: Contacting the selected 6 customers yields **56.33 EUR** of expected net value while consuming only 44% of the contact budget and 72.7% of branch capacity.
2. **High Success Confidence**: The campaign carries a **73.32% probability** of achieving at least 3 conversions, securing operational return on invested capital.
3. **Cross-Segment Resilience**: Combining Segment A (high-value balance) and Segment C/D customers provides resilience against single-segment defaults.

#### Technical Limitations & Future Work
1. **NISQ QPU Deployment**: Current simulation was performed on `AerSimulator`. Execution on physical superconducting QPUs will introduce gate errors and readout fidelity decay. Future work should apply Zero-Noise Extrapolation (ZNE) and Readout Error Mitigation (M3).
2. **Independent Response Assumption**: The simulation models independent customer conversion events. Incorporating multi-qubit entangling gates ($CX, CZ$) can model peer-influence and social contagion dynamics in future iterations.

---

## 🖥️ Streamlit Dashboard Multi-Page Tour

When launching `streamlit run app.py`, the user interface provides 7 dedicated navigation views:

| Page | Section Name | Key Visuals & Interactions |
| :---: | :--- | :--- |
| **1** | **Executive Overview** | High-level KPI metric cards, stage-by-stage architecture pipeline, workflow summary. |
| **2** | **Exploratory Data Analysis** | Interactive age/balance distribution filters, job-wise conversion rates, correlation heatmap. |
| **3** | **Model Benchmarking** | 6-model cross-validation leaderboard, interactive ROC/PR curves, reliability calibration plots. |
| **4** | **Stage 1 Predictions** | 100 test customer search table, top 10 candidate ranking, segment distribution breakdown. |
| **5** | **Stage 2 QAOA Optimization** | QAOA vs Classical convergence comparison, budget/capacity meters, candidate value trade-offs. |
| **6** | **Stage 3 Quantum Simulation** | Interactive 6-qubit Bloch sphere/angle viewer, 200,000-shot AerSimulator distribution, exact DP curves. |
| **7** | **Reports & Downloads** | Instant download buttons for all 4 CSV deliverables, executive summary PDF, technical report markdown, and full report preview. |

---

## 📁 Repository Structure

```
HYBRID_TERM_DEPOSIT/
├── README.md                                  # Comprehensive documentation & execution guide
├── requirements.txt                           # Frozen package dependencies
├── app.py                                     # Root Streamlit dashboard entry-point
├── run_pipeline.py                            # Root master pipeline execution script
├── run.sh                                     # macOS/Linux one-click pipeline runner
├── start_dashboard.sh                         # macOS/Linux one-click dashboard launcher
├── run.bat                                    # Windows one-click pipeline runner
├── start_dashboard.bat                        # Windows one-click dashboard launcher
├── project_results.zip                        # Complete bundle of all generated outputs
│
├── .streamlit/
│   └── config.toml                            # Dashboard theme and server settings
│
└── QuantumBank_Term_Deposit_Optimizer/        # Core project module
    ├── requirements.txt
    ├── run_pipeline.py
    ├── app.py
    │
    ├── data/
    │   ├── bank_campaign_participant_release.xlsx  # Official hackathon dataset
    │   └── PROBLEM_STATEMENT_THEATOMIQHACKTHON.pdf # Challenge problem statement
    │
    ├── notebooks/                                  # 5 Step-by-step Jupyter notebooks
    │   ├── 01_data_exploration.ipynb
    │   ├── 02_classical_machine_learning.ipynb
    │   ├── 03_quantum_machine_learning.ipynb
    │   ├── 04_quantum_optimization.ipynb
    │   └── 05_quantum_simulation.ipynb
    │
    ├── src/                                        # Modular source code
    │   ├── utils.py                                # Seeds, logging, paths
    │   ├── data_preprocessing.py                   # Train-only preprocessing pipeline
    │   ├── classical_models.py                     # Classical ML models & CV
    │   ├── quantum_classifier.py                   # QML QSVC & ZZFeatureMap
    │   ├── customer_prediction.py                  # Test scoring & top 10 ranking
    │   ├── classical_optimizer.py                  # Classical brute-force solver (210 combinations)
    │   ├── quantum_optimizer.py                    # QAOA QUBO solver & bitstring decoding
    │   ├── probability_calculation.py              # Poisson-Binomial DP & Monte Carlo
    │   ├── quantum_simulation.py                   # 6-qubit RY quantum circuit simulator
    │   └── visualization.py                        # 18 Publication-quality charts
    │
    ├── outputs/                                    # Official generated deliverables
    │   ├── stage1_predictions.csv                  # 100 test customer probabilities & ranks
    │   ├── stage2_optimization_results.csv         # Top 10 candidate optimization results
    │   ├── stage3_quantum_simulation_results.csv   # 6-qubit simulation & error comparisons
    │   ├── model_comparison.csv                    # 6-model cross-validation comparison
    │   └── charts/                                 # 18 High-resolution PNG charts
    │
    └── reports/                                    # Official reports & summaries
        ├── technical_report.md                     # Comprehensive technical documentation
        └── executive_summary.pdf                   # Multi-page executive summary PDF
```

---

## 🏆 Hackathon Verification Checklist

- [x] **Zero Data Leakage**: All scalers, encoders, and PCA transformers fitted strictly on training data (300 records).
- [x] **Duration Excluded**: Call duration permanently excluded from all predictive models.
- [x] **Severe Imbalance Handled**: 5-Fold Stratified Cross-Validation with minority class balancing.
- [x] **Stage 1 Complete**: All 100 test customers scored with continuous probabilities; Top 10 ranked deterministically with alphanumeric tie-breaker.
- [x] **Stage 2 QAOA Executed**: QUBO formulated on 10 qubits; QAOA simulated and benchmarked against all 210 classical brute-force combinations.
- [x] **Stage 2 Feasibility Verified**: Exactly $K=6$ customers selected; contact budget ($\le 25$ EUR) and capacity ($\le 22$ units) 100% satisfied.
- [x] **Stage 3 Quantum Circuit**: 6-Qubit $RY(\theta_i)|0\rangle$ state preparation circuit simulated with 200,000 shots on `AerSimulator`.
- [x] **Triangulated Validation**: Campaign success $P(K \ge 3)$ validated across Quantum Aer ($73.32\%$), Exact Poisson-Binomial DP ($73.26\%$), and Monte Carlo ($73.27\%$) with sub-0.0006 error.
- [x] **All Deliverables Exported**: `stage1_predictions.csv`, `stage2_optimization_results.csv`, `stage3_quantum_simulation_results.csv`, `model_comparison.csv`, 18 charts, and `executive_summary.pdf`.
- [x] **Interactive Web App**: Fully functional 7-page Streamlit dashboard with Page 7 reports preview and download buttons.
- [x] **Strict Reproducibility**: Fixed random seed 42 enforced across all classical and quantum seeds.
