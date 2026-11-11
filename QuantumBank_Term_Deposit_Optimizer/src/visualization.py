"""
Visualization Module for QuantumBank Campaign Pipeline.

Generates 18 publication-quality charts saved to outputs/charts/ with a modern
fintech aesthetic (Navy blue, teal, purple accents, clean white cards, high DPI).
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple
import matplotlib
matplotlib.use('Agg') # Headless rendering
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from .utils import setup_logger, get_project_root

logger = setup_logger("Visualization")

# Professional FinTech Palette
PALETTE = {
    "primary": "#0F2027",
    "secondary": "#203A43",
    "accent": "#0077B6",
    "light_blue": "#00B4D8",
    "cyan": "#90E0EF",
    "purple": "#7209B7",
    "magenta": "#F72585",
    "green": "#2EC4B6",
    "gray_light": "#F8F9FA",
    "text": "#1E293B"
}

def set_style():
    """Apply consistent styling across all matplotlib/seaborn charts."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.sans-serif': 'Arial, DejaVu Sans, Helvetica',
        'font.family': 'sans-serif',
        'axes.edgecolor': '#CBD5E1',
        'axes.linewidth': 1.0,
        'axes.titlesize': 14,
        'axes.titleweight': 'bold',
        'axes.labelsize': 12,
        'axes.labelweight': 'bold',
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.titlesize': 16,
        'figure.titleweight': 'bold',
        'figure.facecolor': '#FFFFFF',
        'axes.facecolor': '#FAFCFF',
        'grid.color': '#E2E8F0',
        'grid.linestyle': '--',
        'grid.alpha': 0.7
    })

class PipelineVisualizer:
    """Creates and saves all 18 standard project visualization charts."""

    def __init__(self, output_dir: Path = None):
        if output_dir is None:
            self.output_dir = get_project_root() / "outputs" / "charts"
        else:
            self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        set_style()

    def generate_all_charts(
        self,
        raw_train_df: pd.DataFrame,
        model_comparison_df: pd.DataFrame,
        cv_stores: Dict[str, Any],
        stage1_top10_df: pd.DataFrame,
        stage2_results_df: pd.DataFrame,
        qaoa_sol: Dict[str, Any],
        classical_sol: Dict[str, Any],
        sim_results: Dict[str, Any],
        distribution_df: pd.DataFrame
    ) -> List[Path]:
        """Generate all 18 charts in sequence."""
        logger.info(f"Generating all 18 publication-quality charts into: {self.output_dir}")
        paths = []
        
        # 1-6: EDA Charts
        paths.append(self.plot_target_distribution(raw_train_df))
        paths.append(self.plot_age_distribution(raw_train_df))
        paths.append(self.plot_balance_distribution(raw_train_df))
        paths.append(self.plot_job_subscription(raw_train_df))
        paths.append(self.plot_education_subscription(raw_train_df))
        paths.append(self.plot_correlation_heatmap(raw_train_df))

        # 7-10: ML Evaluation Charts
        paths.append(self.plot_model_comparison(model_comparison_df))
        paths.append(self.plot_roc_curves(cv_stores))
        paths.append(self.plot_pr_curves(cv_stores))
        paths.append(self.plot_calibration_plots(cv_stores))

        # 11-15: Stage 2 Optimization Charts
        paths.append(self.plot_top10_customers(stage1_top10_df))
        paths.append(self.plot_selected_six(stage2_results_df))
        paths.append(self.plot_budget_utilization(qaoa_sol, classical_sol))
        paths.append(self.plot_capacity_utilization(qaoa_sol, classical_sol))
        paths.append(self.plot_qaoa_vs_classical(qaoa_sol, classical_sol))

        # 16-18: Stage 3 Quantum Simulation Charts
        paths.append(self.plot_quantum_measurement_distribution(sim_results))
        paths.append(self.plot_exact_probability_distribution(distribution_df))
        paths.append(self.plot_comparative_probabilities(distribution_df, sim_results))

        logger.info(f"Successfully generated {len(paths)} charts!")
        return paths

    # 1. Target Class Distribution
    def plot_target_distribution(self, df: pd.DataFrame) -> Path:
        fig, ax = plt.subplots(figsize=(7, 5))
        counts = df['y'].value_counts()
        colors = [PALETTE["accent"], PALETTE["magenta"]]
        bars = ax.bar(counts.index.str.upper(), counts.values, color=colors, width=0.45, edgecolor='black', linewidth=0.8)
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width()/2., h + 3, f"{h} ({h/len(df)*100:.1f}%)", ha='center', va='bottom', fontweight='bold')
        ax.set_title("Stage 1 Training Target Class Distribution (Imbalance: 1 : 7.6)", pad=15)
        ax.set_xlabel("Subscription Outcome (y)")
        ax.set_ylabel("Customer Count")
        ax.set_ylim(0, max(counts.values) * 1.15)
        plt.tight_layout()
        path = self.output_dir / "01_target_class_distribution.png"
        fig.savefig(path, dpi=300)
        plt.close(fig)
        return path

    # 2. Customer Age Distribution
    def plot_age_distribution(self, df: pd.DataFrame) -> Path:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(data=df, x='age', hue='y', kde=True, bins=15, palette=[PALETTE["accent"], PALETTE["magenta"]], ax=ax, alpha=0.6)
        ax.set_title("Customer Age Distribution by Subscription Outcome", pad=15)
        ax.set_xlabel("Age (Years)")
        ax.set_ylabel("Frequency")
        plt.tight_layout()
        path = self.output_dir / "02_customer_age_distribution.png"
        fig.savefig(path, dpi=300)
        plt.close(fig)
        return path

    # 3. Customer Balance Distribution
    def plot_balance_distribution(self, df: pd.DataFrame) -> Path:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(data=df, x='y', y='balance', hue='y', palette=[PALETTE["light_blue"], PALETTE["purple"]], ax=ax, width=0.4, legend=False)
        ax.set_title("Account Balance Distribution Across Target Classes", pad=15)
        ax.set_xlabel("Subscribed (y)")
        ax.set_ylabel("Account Balance (EUR)")
        ax.set_yscale('symlog')
        plt.tight_layout()
        path = self.output_dir / "03_customer_balance_distribution.png"
        fig.savefig(path, dpi=300)
        plt.close(fig)
        return path

    # 4. Job-wise Subscription Analysis
    def plot_job_subscription(self, df: pd.DataFrame) -> Path:
        fig, ax = plt.subplots(figsize=(10, 6))
        job_conv = df.groupby('job')['y'].apply(lambda s: (s == 'yes').mean() * 100).sort_values(ascending=False)
        bars = ax.barh(job_conv.index, job_conv.values, color=PALETTE["secondary"], edgecolor='black', linewidth=0.8)
        for b in bars:
            w = b.get_width()
            ax.text(w + 0.5, b.get_y() + b.get_height()/2., f"{w:.1f}%", ha='left', va='center', fontsize=9, fontweight='bold')
        ax.set_title("Subscription Rate by Job Category (%)", pad=15)
        ax.set_xlabel("Conversion Rate (%)")
        ax.set_ylabel("Job Category")
        ax.set_xlim(0, max(job_conv.values) * 1.2)
        plt.tight_layout()
        path = self.output_dir / "04_job_wise_subscription_analysis.png"
        fig.savefig(path, dpi=300)
        plt.close(fig)
        return path

    # 5. Education-wise Subscription Analysis
    def plot_education_subscription(self, df: pd.DataFrame) -> Path:
        fig, ax = plt.subplots(figsize=(8, 5))
        edu_conv = df.groupby('education')['y'].apply(lambda s: (s == 'yes').mean() * 100).sort_values(ascending=False)
        bars = ax.bar(edu_conv.index.str.capitalize(), edu_conv.values, color=PALETTE["purple"], width=0.45, edgecolor='black', linewidth=0.8)
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width()/2., h + 0.5, f"{h:.1f}%", ha='center', va='bottom', fontweight='bold')
        ax.set_title("Subscription Rate by Education Level (%)", pad=15)
        ax.set_xlabel("Education Level")
        ax.set_ylabel("Conversion Rate (%)")
        ax.set_ylim(0, max(edu_conv.values) * 1.25)
        plt.tight_layout()
        path = self.output_dir / "05_education_wise_subscription_analysis.png"
        fig.savefig(path, dpi=300)
        plt.close(fig)
        return path

    # 6. Correlation Heatmap
    def plot_correlation_heatmap(self, df: pd.DataFrame) -> Path:
        fig, ax = plt.subplots(figsize=(8, 6))
        num_df = df[['age', 'balance', 'day', 'campaign', 'pdays', 'previous']].copy()
        num_df['subscribed'] = (df['y'] == 'yes').astype(int)
        corr = num_df.corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="Blues", ax=ax, cbar=True, linewidths=0.5)
        ax.set_title("Feature Correlation Heatmap (Numerical & Target)", pad=15)
        plt.tight_layout()
        path = self.output_dir / "06_correlation_heatmap.png"
        fig.savefig(path, dpi=300)
        plt.close(fig)
        return path

    # 7. Model Performance Comparison
    def plot_model_comparison(self, comp_df: pd.DataFrame) -> Path:
        fig, ax = plt.subplots(figsize=(11, 6))
        x = np.arange(len(comp_df))
        width = 0.25
        
        b1 = ax.bar(x - width, comp_df['ROC-AUC'], width, label='ROC-AUC', color=PALETTE["accent"])
        b2 = ax.bar(x, comp_df['PR-AUC'], width, label='PR-AUC', color=PALETTE["purple"])
        b3 = ax.bar(x + width, 1.0 - comp_df['Brier Score'], width, label='1 - Brier Score', color=PALETTE["green"])

        ax.set_xticks(x)
        ax.set_xticklabels(comp_df['Model'], rotation=15, ha='right', fontweight='bold')
        ax.set_title("Cross-Validated Probabilistic Performance Across Models", pad=15)
        ax.set_ylabel("Metric Score")
        ax.set_ylim(0, 1.05)
        ax.legend(loc='lower right', frameon=True)
        plt.tight_layout()
        path = self.output_dir / "07_model_performance_comparison.png"
        fig.savefig(path, dpi=300)
        plt.close(fig)
        return path

    # 8. ROC Curves
    def plot_roc_curves(self, cv_stores: Dict[str, Any]) -> Path:
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = [PALETTE["accent"], PALETTE["secondary"], PALETTE["purple"], "#E76F51", PALETTE["green"], PALETTE["magenta"]]
        
        for idx, (name, store) in enumerate(cv_stores.items()):
            fpr, tpr = store["roc_curve"]
            auc_val = store["metrics"]["ROC-AUC"]
            ax.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.3f})", color=colors[idx % len(colors)], linewidth=2)
            
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Chance (AUC = 0.500)')
        ax.set_title("Stratified 5-Fold Cross-Validation ROC Curves", pad=15)
        ax.set_xlabel("False Positive Rate (1 - Specificity)")
        ax.set_ylabel("True Positive Rate (Recall)")
        ax.legend(loc='lower right', frameon=True)
        plt.tight_layout()
        path = self.output_dir / "08_roc_curves.png"
        fig.savefig(path, dpi=300)
        plt.close(fig)
        return path

    # 9. Precision-Recall Curves
    def plot_pr_curves(self, cv_stores: Dict[str, Any]) -> Path:
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = [PALETTE["accent"], PALETTE["secondary"], PALETTE["purple"], "#E76F51", PALETTE["green"], PALETTE["magenta"]]
        
        for idx, (name, store) in enumerate(cv_stores.items()):
            prec, rec = store["pr_curve"]
            pr_auc = store["metrics"]["PR-AUC"]
            ax.plot(rec, prec, label=f"{name} (PR-AUC = {pr_auc:.3f})", color=colors[idx % len(colors)], linewidth=2)
            
        ax.axhline(y=0.1167, color='k', linestyle='--', alpha=0.5, label='No-Skill Baseline (0.117)')
        ax.set_title("Precision-Recall Curves (Imbalanced Target Focus)", pad=15)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.legend(loc='upper right', frameon=True)
        plt.tight_layout()
        path = self.output_dir / "09_precision_recall_curves.png"
        fig.savefig(path, dpi=300)
        plt.close(fig)
        return path

    # 10. Calibration Plots
    def plot_calibration_plots(self, cv_stores: Dict[str, Any]) -> Path:
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = [PALETTE["accent"], PALETTE["secondary"], PALETTE["purple"], "#E76F51", PALETTE["green"], PALETTE["magenta"]]
        
        for idx, (name, store) in enumerate(cv_stores.items()):
            prob_true, prob_pred = store["calibration_curve"]
            ece_val = store["metrics"]["ECE"]
            ax.plot(prob_pred, prob_true, marker='o', label=f"{name} (ECE = {ece_val:.3f})", color=colors[idx % len(colors)], linewidth=1.8)
            
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.6, label='Perfect Calibration')
        ax.set_title("Probability Calibration Reliability Diagram", pad=15)
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives (Empirical)")
        ax.legend(loc='upper left', frameon=True)
        plt.tight_layout()
        path = self.output_dir / "10_calibration_plots.png"
        fig.savefig(path, dpi=300)
        plt.close(fig)
        return path

    # 11. Top 10 Predicted Customers
    def plot_top10_customers(self, top10_df: pd.DataFrame) -> Path:
        fig, ax = plt.subplots(figsize=(9, 5))
        bars = ax.bar(top10_df['customer_id'], top10_df['predicted_probability'], color=PALETTE["accent"], edgecolor='black', linewidth=0.8)
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width()/2., h + 0.02, f"{h:.3f}", ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.set_title("Top 10 Stage 1 Candidate Customers Ranked by Subscription Probability", pad=15)
        ax.set_xlabel("Candidate Customer ID")
        ax.set_ylabel("Predicted Subscription Probability")
        ax.set_ylim(0, 1.05)
        plt.tight_layout()
        path = self.output_dir / "11_top_10_predicted_customers.png"
        fig.savefig(path, dpi=300)
        plt.close(fig)
        return path

    # 12. Selected Six Customers Comparison
    def plot_selected_six(self, s2_df: pd.DataFrame) -> Path:
        fig, ax = plt.subplots(figsize=(10, 6))
        # Highlight selected vs non-selected
        colors = [PALETTE["green"] if s else "#94A3B8" for s in s2_df['selected_flag']]
        bars = ax.bar(s2_df['customer_id'], s2_df['expected_net_value'], color=colors, edgecolor='black', linewidth=0.8)
        for b, (_, row) in zip(bars, s2_df.iterrows()):
            h = b.get_height()
            tag = "SELECTED" if row['selected_flag'] else "Excluded"
            ax.text(b.get_x() + b.get_width()/2., h + 0.3, f"{h:.2f}\n({tag})", ha='center', va='bottom', fontsize=8, fontweight='bold')
        ax.set_title("Stage 2 Customer Expected Net Value: Selected 6 vs Excluded 4", pad=15)
        ax.set_xlabel("Candidate Customer ID")
        ax.set_ylabel("Expected Net Value (EUR)")
        ax.set_ylim(0, max(s2_df['expected_net_value']) * 1.25)
        plt.tight_layout()
        path = self.output_dir / "12_selected_six_customer_comparison.png"
        fig.savefig(path, dpi=300)
        plt.close(fig)
        return path

    # 13. Budget Utilization Chart
    def plot_budget_utilization(self, qaoa_sol: Dict[str, Any], cl_sol: Dict[str, Any]) -> Path:
        fig, ax = plt.subplots(figsize=(7, 5))
        methods = ['Classical Optimum', 'QAOA (Quantum)']
        costs = [cl_sol['contact_cost'], qaoa_sol['contact_cost']]
        limit = qaoa_sol['budget_limit']
        
        bars = ax.bar(methods, costs, color=[PALETTE["accent"], PALETTE["purple"]], width=0.45, edgecolor='black', linewidth=0.8)
        ax.axhline(y=limit, color='red', linestyle='--', linewidth=2, label=f'Max Budget = {limit} EUR')
        
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width()/2., h + 0.6, f"{h:.1f} / {limit} ({h/limit*100:.1f}%)", ha='center', va='bottom', fontweight='bold')
            
        ax.set_title("Stage 2 Contact Budget Utilization", pad=15)
        ax.set_ylabel("Contact Cost (EUR)")
        ax.set_ylim(0, limit * 1.25)
        ax.legend(loc='lower right', frameon=True)
        plt.tight_layout()
        path = self.output_dir / "13_budget_utilization_chart.png"
        fig.savefig(path, dpi=300)
        plt.close(fig)
        return path

    # 14. Capacity Utilization Chart
    def plot_capacity_utilization(self, qaoa_sol: Dict[str, Any], cl_sol: Dict[str, Any]) -> Path:
        fig, ax = plt.subplots(figsize=(7, 5))
        methods = ['Classical Optimum', 'QAOA (Quantum)']
        caps = [cl_sol['capacity_units'], qaoa_sol['capacity_units']]
        limit = qaoa_sol['capacity_limit']
        
        bars = ax.bar(methods, caps, color=[PALETTE["accent"], PALETTE["purple"]], width=0.45, edgecolor='black', linewidth=0.8)
        ax.axhline(y=limit, color='red', linestyle='--', linewidth=2, label=f'Max Capacity = {limit} Units')
        
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width()/2., h + 0.6, f"{h:.1f} / {limit} ({h/limit*100:.1f}%)", ha='center', va='bottom', fontweight='bold')
            
        ax.set_title("Stage 2 Operational Capacity Utilization", pad=15)
        ax.set_ylabel("Capacity Units")
        ax.set_ylim(0, limit * 1.25)
        ax.legend(loc='lower right', frameon=True)
        plt.tight_layout()
        path = self.output_dir / "14_capacity_utilization_chart.png"
        fig.savefig(path, dpi=300)
        plt.close(fig)
        return path

    # 15. QAOA vs Classical Objective Comparison
    def plot_qaoa_vs_classical(self, qaoa_sol: Dict[str, Any], cl_sol: Dict[str, Any]) -> Path:
        fig, ax = plt.subplots(figsize=(7, 5))
        methods = ['Classical Brute Force', 'QAOA (Quantum)']
        vals = [cl_sol['expected_net_value'], qaoa_sol['expected_net_value']]
        gap = (cl_sol['expected_net_value'] - qaoa_sol['expected_net_value']) / cl_sol['expected_net_value'] * 100
        
        bars = ax.bar(methods, vals, color=[PALETTE["accent"], PALETTE["green"]], width=0.45, edgecolor='black', linewidth=0.8)
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width()/2., h + 1.2, f"{h:.4f} EUR", ha='center', va='bottom', fontweight='bold')
            
        ax.set_title(f"Optimization Objective: QAOA vs Classical (Gap: {gap:.2f}%)", pad=15)
        ax.set_ylabel("Expected Net Business Value (EUR)")
        ax.set_ylim(0, max(vals) * 1.25)
        plt.tight_layout()
        path = self.output_dir / "15_qaoa_vs_classical_objective_comparison.png"
        fig.savefig(path, dpi=300)
        plt.close(fig)
        return path

    # 16. Quantum Measurement Distribution
    def plot_quantum_measurement_distribution(self, sim_results: Dict[str, Any]) -> Path:
        fig, ax = plt.subplots(figsize=(9, 5))
        q_pmf = sim_results["quantum_pmf"]
        k_vals = np.arange(len(q_pmf))
        
        colors = [PALETTE["light_blue"] if k < 3 else PALETTE["green"] for k in k_vals]
        bars = ax.bar(k_vals, q_pmf, color=colors, width=0.55, edgecolor='black', linewidth=0.8)
        
        for b, k in zip(bars, k_vals):
            h = b.get_height()
            lbl = f"{h*100:.1f}%\n" + ("Success" if k >= 3 else "Fail")
            ax.text(b.get_x() + b.get_width()/2., h + 0.01, lbl, ha='center', va='bottom', fontsize=8, fontweight='bold')
            
        p_succ = sim_results['p_quantum_at_least_3']
        ax.set_title(f"Stage 3 Quantum Circuit Measurement Distribution | P(K >= 3) = {p_succ*100:.2f}%", pad=15)
        ax.set_xlabel("Number of Customer Subscriptions (K)")
        ax.set_ylabel("Measurement Probability")
        ax.set_ylim(0, max(q_pmf) * 1.25)
        plt.tight_layout()
        path = self.output_dir / "16_quantum_measurement_distribution.png"
        fig.savefig(path, dpi=300)
        plt.close(fig)
        return path

    # 17. Exact Probability Distribution
    def plot_exact_probability_distribution(self, dist_df: pd.DataFrame) -> Path:
        fig, ax = plt.subplots(figsize=(9, 5))
        k_vals = dist_df['k_subscriptions']
        exact_p = dist_df['exact_prob']
        
        colors = [PALETTE["purple"] if k >= 3 else PALETTE["light_blue"] for k in k_vals]
        bars = ax.bar(k_vals, exact_p, color=colors, width=0.55, edgecolor='black', linewidth=0.8)
        
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width()/2., h + 0.01, f"{h*100:.2f}%", ha='center', va='bottom', fontsize=8, fontweight='bold')
            
        p_at_least_3 = exact_p[k_vals >= 3].sum()
        ax.set_title(f"Exact Poisson-Binomial Distribution | P(K >= 3) = {p_at_least_3*100:.2f}%", pad=15)
        ax.set_xlabel("Number of Subscriptions (K)")
        ax.set_ylabel("Analytical Probability")
        ax.set_ylim(0, max(exact_p) * 1.25)
        plt.tight_layout()
        path = self.output_dir / "17_exact_probability_distribution.png"
        fig.savefig(path, dpi=300)
        plt.close(fig)
        return path

    # 18. Quantum vs Classical Probability Comparison
    def plot_comparative_probabilities(self, dist_df: pd.DataFrame, sim_results: Dict[str, Any]) -> Path:
        fig, ax = plt.subplots(figsize=(10, 6))
        x = dist_df['k_subscriptions']
        width = 0.25
        
        ax.bar(x - width, dist_df['exact_prob'], width, label='Exact Poisson-Binomial', color=PALETTE["primary"])
        ax.bar(x, dist_df['quantum_prob'], width, label=f'Quantum Aer ({sim_results["shots"]:,} shots)', color=PALETTE["accent"])
        ax.bar(x + width, dist_df['monte_carlo_prob'], width, label=f'Monte Carlo ({sim_results["shots"]:,} trials)', color=PALETTE["green"])
        
        err_q = sim_results['abs_error_quantum']
        err_mc = sim_results['abs_error_monte_carlo']
        
        ax.set_title(f"Outcome Distribution Comparison (|Quantum-Exact|: {err_q:.5f}, |MC-Exact|: {err_mc:.5f})", pad=15)
        ax.set_xlabel("Number of Subscriptions (K)")
        ax.set_ylabel("Probability")
        ax.set_xticks(x)
        ax.legend(loc='upper right', frameon=True)
        plt.tight_layout()
        path = self.output_dir / "18_quantum_vs_classical_probability_comparison.png"
        fig.savefig(path, dpi=300)
        plt.close(fig)
        return path
