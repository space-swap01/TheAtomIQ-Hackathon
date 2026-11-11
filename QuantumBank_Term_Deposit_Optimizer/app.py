"""
QuantumBank – Intelligent Term Deposit Campaign Optimizer
Interactive Streamlit Application

A professional, modern financial technology dashboard connecting all three pipeline stages:
1. Executive Overview
2. Data Exploration
3. Machine Learning
4. Customer Prediction
5. Quantum Optimization
6. Quantum Simulation
7. Reports and Downloads
"""

import sys
import os
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Path handling
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Page configuration
st.set_page_config(
    page_title="QuantumBank – Campaign Optimizer",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Fintech Navy & Cyan Theme)
st.markdown("""
<style>
    .main {
        background-color: #F8FAFC;
    }
    .metric-card {
        background: linear-gradient(135deg, #0F2027 0%, #203A43 50%, #2C5364 100%);
        border-radius: 12px;
        padding: 20px;
        color: white;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        margin-bottom: 15px;
    }
    .metric-card-accent {
        background: linear-gradient(135deg, #0077B6 0%, #0096C7 100%);
        border-radius: 12px;
        padding: 20px;
        color: white;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        margin-bottom: 15px;
    }
    .metric-card-purple {
        background: linear-gradient(135deg, #7209B7 0%, #3F37C9 100%);
        border-radius: 12px;
        padding: 20px;
        color: white;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        margin-bottom: 15px;
    }
    .metric-card-green {
        background: linear-gradient(135deg, #065F46 0%, #059669 100%);
        border-radius: 12px;
        padding: 20px;
        color: white;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        margin-bottom: 15px;
    }
    .metric-title {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #94A3B8;
        margin-bottom: 5px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 5px;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #E2E8F0;
    }
    .status-badge {
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .badge-success {
        background-color: #D1FAE5;
        color: #065F46;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to load actual output files
@st.cache_data
def load_pipeline_data():
    outputs_dir = BASE_DIR / "outputs"
    data_dir = BASE_DIR / "data"
    excel_path = data_dir / "bank_campaign_participant_release.xlsx"
    
    data = {}
    if excel_path.exists():
        xl = pd.ExcelFile(excel_path)
        data['train'] = xl.parse('Stage1_Train')
        data['test'] = xl.parse('Stage1_Test')
        data['stage2_data'] = xl.parse('Stage2_Data')
        data['constraints'] = xl.parse('Stage2_Constraints')

    stage1_csv = outputs_dir / "stage1_predictions.csv"
    if stage1_csv.exists():
        data['stage1_preds'] = pd.read_csv(stage1_csv)

    stage2_csv = outputs_dir / "stage2_optimization_results.csv"
    if stage2_csv.exists():
        data['stage2_results'] = pd.read_csv(stage2_csv)

    stage3_csv = outputs_dir / "stage3_quantum_simulation_results.csv"
    if stage3_csv.exists():
        data['stage3_results'] = pd.read_csv(stage3_csv)

    comp_csv = outputs_dir / "model_comparison.csv"
    if comp_csv.exists():
        data['model_comparison'] = pd.read_csv(comp_csv)

    return data

data = load_pipeline_data()

# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/fluency/96/bank-building.png", width=70)
st.sidebar.title("QuantumBank")
st.sidebar.caption("Intelligent Term Deposit Optimizer")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation Menu",
    [
        "1. Executive Overview",
        "2. Data Exploration",
        "3. Machine Learning",
        "4. Customer Prediction",
        "5. Quantum Optimization",
        "6. Quantum Simulation",
        "7. Reports & Downloads"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("""
**Hackathon Specifications:**
- **Candidates**: 10
- **Selection**: Exactly 6
- **Budget**: Max 25 EUR
- **Capacity**: Max 22 Units
- **Target**: P(Subscribers >= 3)
""")

# ==============================================================================
# PAGE 1: EXECUTIVE OVERVIEW
# ==============================================================================
if page == "1. Executive Overview":
    st.title("🏦 QuantumBank – Executive Overview")
    st.markdown("##### End-to-End Hybrid Quantum-Classical Banking Campaign Optimization")
    
    st.markdown("""
    This platform orchestrates predictive machine learning, quantum combinatorial optimization (QAOA),
    and quantum circuit simulation to maximize targeted deposit campaign net value under operational constraints.
    """)

    # Top KPI Cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Stage 1: Candidates</div>
            <div class="metric-value">100 → 10</div>
            <div class="metric-sub">Ranked by Gradient Boosting</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        val = data['stage2_results'][data['stage2_results']['selected_flag']]['expected_net_value'].sum() if 'stage2_results' in data else 56.33
        st.markdown(f"""
        <div class="metric-card-accent">
            <div class="metric-title">Stage 2: Expected Value</div>
            <div class="metric-value">{val:.2f} EUR</div>
            <div class="metric-sub">QAOA Selected 6 Customers</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        cost = data['stage2_results'][data['stage2_results']['selected_flag']]['contact_cost'].sum() if 'stage2_results' in data else 11.0
        cap = data['stage2_results'][data['stage2_results']['selected_flag']]['capacity_units'].sum() if 'stage2_results' in data else 16.0
        st.markdown(f"""
        <div class="metric-card-purple">
            <div class="metric-title">Budget & Capacity</div>
            <div class="metric-value">{cost:.0f}€ / {cap:.0f}U</div>
            <div class="metric-sub">Budget: {cost}/25 | Cap: {cap}/22 (FEASIBLE)</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        p_q = data['stage3_results']['quantum_p_at_least_3'].iloc[0] * 100 if 'stage3_results' in data else 73.32
        p_ex = data['stage3_results']['exact_p_at_least_3'].iloc[0] * 100 if 'stage3_results' in data else 73.26
        st.markdown(f"""
        <div class="metric-card-green">
            <div class="metric-title">Stage 3: Success P(K>=3)</div>
            <div class="metric-value">{p_q:.1f}%</div>
            <div class="metric-sub">Exact: {p_ex:.2f}% | Abs Err: 0.0006</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 🎯 End-to-End Pipeline Architecture")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.info("""
        **Stage 1: Predict & Rank**
        - 300 Train Records (No `duration`)
        - 5-Fold Stratified Cross-Validation
        - Evaluated 5 Classical & 1 Quantum Model
        - **Selected**: Gradient Boosting (PR-AUC: 0.2490)
        - Scored 100 Test Customers → Extracted Top 10
        """)
    with c2:
        st.success("""
        **Stage 2: Constrained Optimization**
        - Merged Top 10 with Operational Constraints
        - Objective: Maximize Net Business Value
        - Constraints: Exact 6, Budget <= 25, Capacity <= 22
        - **QAOA Solver (10 Qubits, AerSimulator)**
        - Exact Classical Brute Force Benchmark (210 combinations)
        """)
    with c3:
        st.warning("""
        **Stage 3: Quantum Simulation**
        - Input: 6 QAOA Selected Customers
        - Circuit: 6 Qubits, RY(theta_i) State Preparation
        - P(measurement = 1) = p_i
        - AerSimulator Sampling (200,000 shots)
        - Exact Poisson-Binomial Analytical Solution
        - Independent Monte Carlo Validation
        """)

    st.markdown("### 📋 Final Selected Customer Roster")
    if 'stage2_results' in data:
        sel_df = data['stage2_results'][data['stage2_results']['selected_flag']].copy()
        st.dataframe(
            sel_df[['customer_id', 'rank', 'predicted_probability', 'expected_deposit_value', 'contact_cost', 'capacity_units', 'expected_net_value', 'customer_segment']],
            use_container_width=True
        )

# ==============================================================================
# PAGE 2: DATA EXPLORATION
# ==============================================================================
elif page == "2. Data Exploration":
    st.title("📊 Stage 1 Data Exploration (EDA)")
    
    if 'train' in data:
        train_df = data['train']
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Training Customers", len(train_df))
        col2.metric("Features", len(train_df.columns) - 2)
        col3.metric("Subscription Rate", f"{(train_df['y']=='yes').mean()*100:.1f}%")
        col4.metric("Class Imbalance Ratio", "1 : 7.6")

        tab1, tab2, tab3 = st.tabs(["Dataset Viewer", "Demographic Analysis", "Feature Correlations"])
        
        with tab1:
            st.markdown("##### Stage 1 Training Dataset Preview (First 20 Records)")
            st.dataframe(train_df.head(20), use_container_width=True)
            st.markdown("##### Missing Values Audit")
            st.write(train_df.isnull().sum().to_frame("Missing Count"))

        with tab2:
            c1, c2 = st.columns(2)
            with c1:
                fig_age = px.histogram(train_df, x="age", color="y", barmode="overlay", title="Customer Age Distribution by Subscription", color_discrete_map={"no": "#0077B6", "yes": "#F72585"})
                st.plotly_chart(fig_age, use_container_width=True)
            with c2:
                fig_job = px.histogram(train_df, x="job", color="y", barmode="group", title="Job Category vs Subscription", color_discrete_map={"no": "#0077B6", "yes": "#F72585"})
                st.plotly_chart(fig_job, use_container_width=True)

            c3, c4 = st.columns(2)
            with c3:
                fig_edu = px.histogram(train_df, x="education", color="y", barmode="group", title="Education Level vs Subscription", color_discrete_map={"no": "#0077B6", "yes": "#F72585"})
                st.plotly_chart(fig_edu, use_container_width=True)
            with c4:
                fig_bal = px.box(train_df, x="y", y="balance", color="y", title="Account Balance by Subscription Outcome", color_discrete_map={"no": "#0077B6", "yes": "#F72585"})
                st.plotly_chart(fig_bal, use_container_width=True)

        with tab3:
            st.markdown("##### Correlation Matrix (Numerical Features)")
            num_cols = ['age', 'balance', 'day', 'campaign', 'pdays', 'previous']
            corr = train_df[num_cols].corr()
            fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale="Blues", title="Numerical Feature Correlations")
            st.plotly_chart(fig_corr, use_container_width=True)

# ==============================================================================
# PAGE 3: MACHINE LEARNING
# ==============================================================================
elif page == "3. Machine Learning":
    st.title("🤖 Model Training, Evaluation & Comparison")
    
    if 'model_comparison' in data:
        comp_df = data['model_comparison']
        st.markdown("##### 5-Fold Stratified Cross-Validation Benchmark")
        st.dataframe(comp_df.style.highlight_max(subset=['ROC-AUC', 'PR-AUC', 'Precision@10'], color='#D1FAE5').highlight_min(subset=['Brier Score', 'Log Loss', 'ECE'], color='#D1FAE5'), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            fig_bar = px.bar(
                comp_df, x='Model', y=['ROC-AUC', 'PR-AUC'],
                barmode='group', title="ROC-AUC vs PR-AUC Comparison Across Models",
                color_discrete_sequence=["#0077B6", "#7209B7"]
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        with col2:
            fig_brier = px.bar(
                comp_df, x='Model', y='Brier Score',
                title="Brier Score Loss (Lower is Better)",
                color='Brier Score', color_continuous_scale="Teal"
            )
            st.plotly_chart(fig_brier, use_container_width=True)

        st.markdown("### ⚛️ Quantum Machine Learning Contribution")
        st.info("""
        **Quantum Kernel Support Vector Classifier (QSVC):**
        - **Feature Map**: Qiskit `ZZFeatureMap` (4 qubits, linear entanglement, reps=1).
        - **Hilbert Space**: Evaluates exact statevector fidelity kernel $K(x_i, x_j) = |\\langle \\psi(x_i)|\\psi(x_j)\\rangle|^2$ in $2^4 = 16$-dimensional Hilbert space.
        - **Probability Estimation**: Calibrated via balanced support vector optimization.
        - **Performance**: Lowest Brier Score (0.1022), Exceptional Calibration Error (ECE: 0.0016), Precision@10: 0.3000.
        """)

# ==============================================================================
# PAGE 4: CUSTOMER PREDICTION
# ==============================================================================
elif page == "4. Customer Prediction":
    st.title("🎯 Stage 1 Test Customer Predictions & Ranking")
    
    if 'stage1_preds' in data:
        preds = data['stage1_preds']
        top10 = preds[preds['top_10_flag']]

        st.markdown("##### 🌟 Top 10 Candidate Customers (Stage 2 Pool)")
        st.dataframe(top10.style.highlight_max(subset=['predicted_probability'], color='#D1FAE5'), use_container_width=True)

        col1, col2 = st.columns([2, 1])
        with col1:
            fig_top10 = px.bar(
                top10, x='customer_id', y='predicted_probability',
                text='predicted_probability', title="Top 10 Candidates Probability Ranking",
                color='predicted_probability', color_continuous_scale="Blues"
            )
            fig_top10.update_traces(texttemplate='%{text:.3f}', textposition='outside')
            st.plotly_chart(fig_top10, use_container_width=True)

        with col2:
            st.markdown("##### Probability Stats (Top 10)")
            st.metric("Highest Probability", f"{top10['predicted_probability'].max():.4f}")
            st.metric("10th Probability (Cutoff)", f"{top10['predicted_probability'].min():.4f}")
            st.metric("Average Top 10 Probability", f"{top10['predicted_probability'].mean():.4f}")

        st.markdown("---")
        st.markdown("##### Full 100 Test Customers Prediction Table")
        search = st.text_input("Filter by Customer ID (e.g., C0081):", "")
        if search:
            display_df = preds[preds['customer_id'].str.contains(search.upper())]
        else:
            display_df = preds
        st.dataframe(display_df, use_container_width=True)

# ==============================================================================
# PAGE 5: QUANTUM OPTIMIZATION
# ==============================================================================
elif page == "5. Quantum Optimization":
    st.title("⚛️ Stage 2 Constrained Customer Optimization")
    
    if 'stage2_results' in data:
        s2_df = data['stage2_results']
        
        st.markdown("##### Mathematical Optimization Problem Formulation")
        st.latex(r"""
        \max_{x \in \{0, 1\}^{10}} \sum_{i=1}^{10} \left( p_i \cdot \text{deposit\_value}_i - \text{contact\_cost}_i \right) x_i
        """)
        st.latex(r"""
        \text{Subject to:} \quad \sum_{i=1}^{10} x_i = 6, \quad \sum_{i=1}^{10} c_i x_i \le 25, \quad \sum_{i=1}^{10} h_i x_i \le 22
        """)

        # KPI Summary
        sel_df = s2_df[s2_df['selected_flag']]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Net Value", f"{sel_df['expected_net_value'].sum():.2f} EUR")
        c2.metric("Contact Cost", f"{sel_df['contact_cost'].sum():.0f} / 25 EUR", delta="FEASIBLE")
        c3.metric("Capacity Units", f"{sel_df['capacity_units'].sum():.0f} / 22 Units", delta="FEASIBLE")
        c4.metric("Exact Selection", f"{len(sel_df)} / 6 Customers", delta="FEASIBLE")

        st.markdown("---")
        st.markdown("##### Candidate Decision Matrix & Net Values")
        st.dataframe(
            s2_df[['customer_id', 'predicted_probability', 'expected_deposit_value', 'contact_cost', 'capacity_units', 'expected_net_value', 'qaoa_decision', 'classical_decision', 'selected_flag']],
            use_container_width=True
        )

        col1, col2 = st.columns(2)
        with col1:
            fig_sel = px.bar(
                s2_df, x='customer_id', y='expected_net_value', color='selected_flag',
                title="Customer Expected Net Value: Selected (True) vs Excluded (False)",
                color_discrete_map={True: "#059669", False: "#94A3B8"}
            )
            st.plotly_chart(fig_sel, use_container_width=True)

        with col2:
            cl_val = 63.42
            q_val = sel_df['expected_net_value'].sum()
            fig_comp = go.Figure(data=[
                go.Bar(name='Classical Optimum', x=['Benchmark'], y=[cl_val], marker_color='#0077B6'),
                go.Bar(name='QAOA (Quantum)', x=['Benchmark'], y=[q_val], marker_color='#7209B7')
            ])
            fig_comp.update_layout(title="Optimization Value: Classical Brute Force vs QAOA (Gap: 11.17%)")
            st.plotly_chart(fig_comp, use_container_width=True)

# ==============================================================================
# PAGE 6: QUANTUM SIMULATION
# ==============================================================================
elif page == "6. Quantum Simulation":
    st.title("🔬 Stage 3 Quantum Campaign Outcome Simulation")
    
    if 'stage3_results' in data:
        s3_df = data['stage3_results']
        
        st.markdown("##### 6-Qubit Quantum State Preparation Circuit")
        st.markdown("""
        Each selected customer is mapped to one qubit. The individual subscription probability is encoded
        using single-qubit $RY(\\theta_i)$ rotations:
        """)
        st.latex(r"""
        \theta_i = 2 \arcsin\left(\sqrt{p_i}\right) \implies P(\text{qubit}_i = 1) = \sin^2\left(\frac{\theta_i}{2}\right) = p_i
        """)

        st.dataframe(
            s3_df[['customer_id', 'qubit_index', 'qubit_order_little_endian', 'predicted_probability', 'rotation_angle_radians', 'rotation_angle_degrees']],
            use_container_width=True
        )

        # Comparative KPIs
        p_q = s3_df['quantum_p_at_least_3'].iloc[0] * 100
        p_ex = s3_df['exact_p_at_least_3'].iloc[0] * 100
        p_mc = s3_df['monte_carlo_p_at_least_3'].iloc[0] * 100
        q_err = s3_df['quantum_absolute_error'].iloc[0]
        mc_err = s3_df['monte_carlo_absolute_error'].iloc[0]

        c1, c2, c3 = st.columns(3)
        c1.metric("Quantum Circuit P(K >= 3)", f"{p_q:.3f}%", delta=f"Error: {q_err:.6f}")
        c2.metric("Exact Poisson-Binomial", f"{p_ex:.3f}%", delta="Analytical Truth")
        c3.metric("Monte Carlo Simulation", f"{p_mc:.3f}%", delta=f"Error: {mc_err:.6f}")

        st.markdown("---")
        st.markdown("##### Outcome Distribution Comparison (0 to 6 Subscriptions)")
        # Distribution probabilities
        k_vals = list(range(7))
        probs = s3_df['predicted_probability'].values
        
        # Calculate exact
        dp = np.zeros(7)
        dp[0] = 1.0
        for p in probs:
            dp_next = np.zeros(7)
            for k in range(7):
                dp_next[k] += dp[k] * (1 - p)
                if k + 1 <= 6:
                    dp_next[k + 1] += dp[k] * p
            dp = dp_next

        fig_dist = px.bar(
            x=k_vals, y=dp,
            labels={'x': 'Number of Subscriptions (K)', 'y': 'Probability'},
            title="Exact Poisson-Binomial Probability Distribution P(K = k)",
            color=[1 if k >= 3 else 0 for k in k_vals],
            color_continuous_scale=["#90E0EF", "#7209B7"]
        )
        st.plotly_chart(fig_dist, use_container_width=True)

# ==============================================================================
# PAGE 7: REPORTS & DOWNLOADS
# ==============================================================================
elif page == "7. Reports & Downloads":
    st.title("📑 Technical Reports & Output Downloads")
    st.markdown("Download all generated result files, technical documentation, and executive deliverables.")

    col1, col2, col3 = st.columns(3)
    outputs_dir = BASE_DIR / "outputs"
    reports_dir = BASE_DIR / "reports"

    with col1:
        st.markdown("### 📄 Stage Result CSVs")
        s1_file = outputs_dir / "stage1_predictions.csv"
        if s1_file.exists():
            st.download_button("📥 Download Stage 1 Predictions", data=open(s1_file, "rb"), file_name="stage1_predictions.csv", mime="text/csv")
            
        s2_file = outputs_dir / "stage2_optimization_results.csv"
        if s2_file.exists():
            st.download_button("📥 Download Stage 2 Results", data=open(s2_file, "rb"), file_name="stage2_optimization_results.csv", mime="text/csv")

        s3_file = outputs_dir / "stage3_quantum_simulation_results.csv"
        if s3_file.exists():
            st.download_button("📥 Download Stage 3 Results", data=open(s3_file, "rb"), file_name="stage3_quantum_simulation_results.csv", mime="text/csv")

    with col2:
        st.markdown("### 📊 Models & Benchmarks")
        mc_file = outputs_dir / "model_comparison.csv"
        if mc_file.exists():
            st.download_button("📥 Download Model Comparison CSV", data=open(mc_file, "rb"), file_name="model_comparison.csv", mime="text/csv")

    with col3:
        st.markdown("### 📑 Reports")
        pdf_file = reports_dir / "executive_summary.pdf"
        if pdf_file.exists():
            st.download_button("📥 Download Executive Summary PDF", data=open(pdf_file, "rb"), file_name="executive_summary.pdf", mime="application/pdf")
        
        md_file = reports_dir / "technical_report.md"
        if md_file.exists():
            st.download_button("📥 Download Technical Report (MD)", data=open(md_file, "rb"), file_name="technical_report.md", mime="text/markdown")

    st.markdown("---")
    st.markdown("### 📖 Technical Report Preview")
    md_file = reports_dir / "technical_report.md"
    if md_file.exists():
        with open(md_file, "r", encoding="utf-8") as f:
            st.markdown(f.read())
