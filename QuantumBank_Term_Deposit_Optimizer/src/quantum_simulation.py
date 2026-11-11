"""
Quantum Campaign Outcome Simulation Module (Stage 3).

Responsibilities:
- Build a 6-qubit quantum circuit encoding subscription probabilities via RY(theta_i) gates:
    theta_i = 2 * arcsin(sqrt(p_i))
- Guarantee that P(qubit_i = 1) = p_i
- Document customer-to-qubit mapping and Qiskit little-endian bit-ordering convention
- Execute shot-based simulation on Qiskit AerSimulator
- Count bitstrings with >= 3 successful subscriptions
- Compare Quantum estimate against Exact Poisson-Binomial and Monte Carlo results
- Export stage3_quantum_simulation_results.csv
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

from .utils import RANDOM_SEED, setup_logger, get_project_root
from .probability_calculation import PoissonBinomialCalculator

logger = setup_logger("QuantumSimulation")

class QuantumCampaignSimulator:
    """Simulates campaign conversion outcome on a 6-qubit quantum circuit."""

    def __init__(self, selected_customers_df: pd.DataFrame, shots: int = 200000, random_state: int = RANDOM_SEED):
        assert len(selected_customers_df) == 6, f"Expected exactly 6 selected customers, got {len(selected_customers_df)}"
        self.customers_df = selected_customers_df.reset_index(drop=True)
        self.shots = shots
        self.random_state = random_state
        self.probabilities = self.customers_df['predicted_probability'].values.astype(float)
        self.n_qubits = len(self.probabilities)

        # Rotation angles: theta_i = 2 * arcsin(sqrt(p_i))
        self.thetas = 2.0 * np.arcsin(np.sqrt(np.clip(self.probabilities, 0.0, 1.0)))
        self.circuit = self._build_circuit()

    def _build_circuit(self) -> QuantumCircuit:
        """Construct the 6-qubit RY state preparation and measurement circuit."""
        qc = QuantumCircuit(self.n_qubits, self.n_qubits, name="QuantumBank_Campaign_RY6")
        
        # State preparation
        for q, theta in enumerate(self.thetas):
            qc.ry(theta, q)
            
        qc.barrier()
        # Measure all qubits into corresponding classical register bits
        qc.measure(range(self.n_qubits), range(self.n_qubits))
        return qc

    def get_circuit_metadata(self) -> Dict[str, Any]:
        """Return circuit details, depth, gate counts, and mapping."""
        mapping = {}
        for q in range(self.n_qubits):
            cid = self.customers_df.iloc[q]['customer_id']
            prob = float(self.probabilities[q])
            theta = float(self.thetas[q])
            mapping[f"qubit_{q}"] = {
                "customer_id": cid,
                "predicted_probability": prob,
                "theta_radians": theta,
                "theta_degrees": float(np.degrees(theta))
            }

        return {
            "circuit_name": self.circuit.name,
            "num_qubits": self.circuit.num_qubits,
            "num_clbits": self.circuit.num_clbits,
            "depth": self.circuit.depth(),
            "gate_counts": dict(self.circuit.count_ops()),
            "customer_qubit_mapping": mapping,
            "bit_ordering_note": (
                "Qiskit represents measurement bitstrings in little-endian order: "
                "b = b_5 b_4 b_3 b_2 b_1 b_0, where bit b_0 corresponds to qubit 0 (rightmost)."
            )
        }

    def run_simulation(self) -> Tuple[Dict[str, Any], pd.DataFrame]:
        """
        Execute simulation on AerSimulator, compute quantum PMF,
        and compare with Exact Poisson-Binomial and Monte Carlo estimates.
        """
        logger.info(f"Running 6-qubit quantum simulation with {self.shots:,} shots on AerSimulator (seed={self.random_state})...")
        backend = AerSimulator(seed_simulator=self.random_state)
        job = backend.run(self.circuit, shots=self.shots)
        result = job.result()
        raw_counts = result.get_counts()

        # Outcome aggregation by number of successful subscriptions (Hamming weight)
        q_k_counts = np.zeros(self.n_qubits + 1, dtype=int)
        success_shots = 0

        for bitstring, count in raw_counts.items():
            ones = bitstring.count('1')
            q_k_counts[ones] += count
            if ones >= 3:
                success_shots += count

        q_pmf = q_k_counts / self.shots
        p_quantum_at_least_3 = float(success_shots / self.shots)

        # 1. Exact Classical Calculation
        pb_calc = PoissonBinomialCalculator(self.probabilities.tolist())
        exact_pmf, p_exact_at_least_3 = pb_calc.compute_exact_distribution()

        # 2. Monte Carlo Simulation
        mc_pmf, p_mc_at_least_3, mc_stderr = pb_calc.run_monte_carlo(
            num_trials=self.shots,
            random_state=self.random_state
        )

        # Error metrics
        abs_err_quantum = abs(p_quantum_at_least_3 - p_exact_at_least_3)
        abs_err_mc = abs(p_mc_at_least_3 - p_exact_at_least_3)

        logger.info(f"=== STAGE 3 SUMMARY ===")
        logger.info(f"Quantum Circuit P(K >= 3):   {p_quantum_at_least_3:.6f}")
        logger.info(f"Exact Analytical P(K >= 3):  {p_exact_at_least_3:.6f}")
        logger.info(f"Monte Carlo P(K >= 3):       {p_mc_at_least_3:.6f}")
        logger.info(f"|Quantum - Exact| Error:     {abs_err_quantum:.6f}")
        logger.info(f"|Monte Carlo - Exact| Error: {abs_err_mc:.6f}")

        # Distribution comparison DataFrame
        dist_records = []
        for k in range(self.n_qubits + 1):
            dist_records.append({
                "k_subscriptions": k,
                "quantum_prob": float(q_pmf[k]),
                "exact_prob": float(exact_pmf[k]),
                "monte_carlo_prob": float(mc_pmf[k]),
                "quantum_shot_count": int(q_k_counts[k])
            })
        distribution_df = pd.DataFrame(dist_records)

        simulation_results = {
            "p_quantum_at_least_3": p_quantum_at_least_3,
            "p_exact_at_least_3": p_exact_at_least_3,
            "p_monte_carlo_at_least_3": p_mc_at_least_3,
            "monte_carlo_stderr": mc_stderr,
            "abs_error_quantum": abs_err_quantum,
            "abs_error_monte_carlo": abs_err_mc,
            "shots": self.shots,
            "raw_measurement_counts": raw_counts,
            "quantum_pmf": q_pmf,
            "exact_pmf": exact_pmf,
            "monte_carlo_pmf": mc_pmf,
            "circuit_metadata": self.get_circuit_metadata()
        }

        return simulation_results, distribution_df

def export_stage3_results(
    selected_customers_df: pd.DataFrame,
    sim_results: Dict[str, Any],
    output_dir: Path = None
) -> pd.DataFrame:
    """
    Format and export stage3_quantum_simulation_results.csv containing
    customer IDs, qubit order, individual probabilities, angles, counts,
    and comparative probability estimates.
    """
    if output_dir is None:
        output_dir = get_project_root() / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    cust_df = selected_customers_df.reset_index(drop=True)
    probs = cust_df['predicted_probability'].values.astype(float)
    thetas = 2.0 * np.arcsin(np.sqrt(np.clip(probs, 0.0, 1.0)))

    for q in range(len(cust_df)):
        row = cust_df.iloc[q]
        records.append({
            "customer_id": row["customer_id"],
            "qubit_index": q,
            "qubit_order_little_endian": f"q_{q}",
            "predicted_probability": float(probs[q]),
            "rotation_angle_radians": float(thetas[q]),
            "rotation_angle_degrees": float(np.degrees(thetas[q])),
            "total_shots": sim_results["shots"],
            "quantum_p_at_least_3": sim_results["p_quantum_at_least_3"],
            "exact_p_at_least_3": sim_results["p_exact_at_least_3"],
            "monte_carlo_p_at_least_3": sim_results["p_monte_carlo_at_least_3"],
            "quantum_absolute_error": sim_results["abs_error_quantum"],
            "monte_carlo_absolute_error": sim_results["abs_error_monte_carlo"]
        })

    out_df = pd.DataFrame(records)
    csv_path = output_dir / "stage3_quantum_simulation_results.csv"
    out_df.to_csv(csv_path, index=False)
    logger.info(f"Saved Stage 3 simulation results to: {csv_path}")
    return out_df
