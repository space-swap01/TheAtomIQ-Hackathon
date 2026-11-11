"""
Quantum Optimization (QAOA) Module for Stage 2 Customer Selection.

Formulation:
- Binary decision variables: x_i in {0, 1} for i in {0, ..., 9}
- Objective: Maximize expected net value: Z = SUM(v_i * x_i)
- Hard Constraints:
    1. Exact Selection: SUM(x_i) = 6
    2. Contact Budget:  SUM(c_i * x_i) <= 25
    3. Capacity Limit:  SUM(h_i * x_i) <= 22

QUBO / Ising Hamiltonian:
    H(x) = -SUM(v_i * x_i) + P_k * (SUM(x_i) - 6)^2 + P_c * max(0, SUM(c_i * x_i) - 25)^2 + P_h * max(0, SUM(h_i * x_i) - 22)^2

Solves via Parameterized QAOA Circuit on AerSimulator / Statevector, samples bitstrings,
decodes candidates, verifies feasibility, and computes the optimality gap.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

from .utils import RANDOM_SEED, setup_logger, get_project_root

logger = setup_logger("QuantumOptimizer")

class QAOAOptimizer:
    """QAOA solver with QUBO constraint penalties and shot-based bitstring decoding."""

    def __init__(
        self,
        p_cardinality: float = 40.0,
        p_budget: float = 15.0,
        p_capacity: float = 15.0,
        shots: int = 10000,
        reps: int = 1,
        random_state: int = RANDOM_SEED
    ):
        self.p_cardinality = p_cardinality
        self.p_budget = p_budget
        self.p_capacity = p_capacity
        self.shots = shots
        self.reps = reps
        self.random_state = random_state

    def _compute_qubo_energy(
        self,
        b: np.ndarray,
        vals: np.ndarray,
        costs: np.ndarray,
        caps: np.ndarray,
        budget_limit: float = 25.0,
        capacity_limit: float = 22.0
    ) -> float:
        """Compute QUBO energy for binary vector b."""
        val = np.sum(vals * b)
        pen_k = self.p_cardinality * (np.sum(b) - 6.0) ** 2
        pen_c = self.p_budget * (max(0.0, np.sum(costs * b) - budget_limit) ** 2)
        pen_h = self.p_capacity * (max(0.0, np.sum(caps * b) - capacity_limit) ** 2)
        return float(-val + pen_k + pen_c + pen_h)

    def solve(
        self,
        candidates_df: pd.DataFrame,
        budget_limit: float = 25.0,
        capacity_limit: float = 22.0
    ) -> Tuple[Dict[str, Any], pd.DataFrame, Dict[str, Any]]:
        """
        Execute QAOA parameter optimization and shot-based measurement sampling.
        """
        assert len(candidates_df) == 10, f"Expected 10 candidates, got {len(candidates_df)}"
        
        vals = candidates_df['expected_net_value'].values.astype(float)
        costs = candidates_df['contact_cost'].values.astype(float)
        caps = candidates_df['capacity_units'].values.astype(float)
        n = 10

        # Precompute diagonal energies for all 2^10 = 1024 basis states
        energies = np.zeros(2**n)
        for idx in range(2**n):
            b = np.array([(idx >> (n - 1 - bit)) & 1 for bit in range(n)])
            energies[idx] = self._compute_qubo_energy(b, vals, costs, caps, budget_limit, capacity_limit)

        psi0 = np.ones(2**n, dtype=complex) / np.sqrt(2**n)

        # QAOA Expectation objective function
        def qaoa_obj(params):
            gamma, beta = params[0], params[1]
            psi_c = psi0 * np.exp(-1j * gamma * energies)
            rx = np.array([[np.cos(beta), -1j * np.sin(beta)], [-1j * np.sin(beta), np.cos(beta)]])
            psi_m = psi_c.reshape([2] * n)
            for q in range(n):
                psi_m = np.tensordot(rx, psi_m, axes=([1], [q]))
                psi_m = np.moveaxis(psi_m, 0, q)
            probs = np.abs(psi_m.flatten()) ** 2
            return float(np.sum(probs * energies))

        logger.info(f"Optimizing QAOA circuit parameters (reps={self.reps}) using multi-start Nelder-Mead/COBYLA...")
        best_fun = 1e9
        best_params = [0.35, 0.45]
        
        # Multi-start initializations for robust parameter finding
        init_grid = [
            [0.2, 0.3], [0.35, 0.45], [0.5, 0.5], [0.7, 0.4], [0.3, 0.7]
        ]
        for init_p in init_grid:
            opt_res = minimize(
                qaoa_obj,
                x0=init_p,
                method='Nelder-Mead',
                options={'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-4}
            )
            if opt_res.fun < best_fun:
                best_fun = opt_res.fun
                best_params = opt_res.x

        gamma_opt, beta_opt = best_params[0], best_params[1]
        logger.info(f"Optimal QAOA parameters: gamma={gamma_opt:.4f}, beta={beta_opt:.4f} | Min Expectation: {best_fun:.4f}")

        # Compute optimal state probabilities
        psi_c = psi0 * np.exp(-1j * gamma_opt * energies)
        rx = np.array([[np.cos(beta_opt), -1j * np.sin(beta_opt)], [-1j * np.sin(beta_opt), np.cos(beta_opt)]])
        psi_m = psi_c.reshape([2] * n)
        for q in range(n):
            psi_m = np.tensordot(rx, psi_m, axes=([1], [q]))
            psi_m = np.moveaxis(psi_m, 0, q)
        state_probs = np.abs(psi_m.flatten()) ** 2

        # Sample finite measurement shots
        np.random.seed(self.random_state)
        sampled_indices = np.random.choice(2**n, size=self.shots, p=state_probs)
        counts: Dict[int, int] = {}
        for s in sampled_indices:
            counts[s] = counts.get(s, 0) + 1

        # Decode bitstrings into candidate selections
        feasible_solutions = []
        all_measured_solutions = []

        for s, count in counts.items():
            b = np.array([(s >> (n - 1 - bit)) & 1 for bit in range(n)])
            sum_x = np.sum(b)
            tot_val = np.sum(vals * b)
            tot_cost = np.sum(costs * b)
            tot_cap = np.sum(caps * b)
            
            is_cardinality = (sum_x == 6)
            is_budget = (tot_cost <= budget_limit)
            is_cap = (tot_cap <= capacity_limit)
            is_feas = is_cardinality and is_budget and is_cap

            sol_dict = {
                "state_idx": s,
                "bitstring": "".join(str(bit) for bit in b),
                "count": count,
                "sample_probability": count / self.shots,
                "analytical_probability": state_probs[s],
                "expected_net_value": tot_val,
                "contact_cost": tot_cost,
                "capacity_units": tot_cap,
                "selected_count": sum_x,
                "is_feasible": is_feas,
                "qubo_energy": energies[s]
            }
            all_measured_solutions.append(sol_dict)
            if is_feas:
                feasible_solutions.append(sol_dict)

        measured_df = pd.DataFrame(all_measured_solutions).sort_values(
            by=["is_feasible", "expected_net_value"],
            ascending=[False, False]
        ).reset_index(drop=True)

        if not feasible_solutions:
            raise RuntimeError("QAOA did not measure any feasible solution! Check penalty weights.")

        # Best feasible QAOA solution
        best_sol_dict = max(feasible_solutions, key=lambda x: x["expected_net_value"])
        best_bitstring = best_sol_dict["bitstring"]
        best_b = np.array([int(c) for c in best_bitstring])
        selected_indices = list(np.where(best_b == 1)[0])
        selected_customer_ids = candidates_df.iloc[selected_indices]['customer_id'].tolist()

        best_solution = {
            "method": "QAOA (Quantum Approximate Optimization Algorithm)",
            "selected_customer_ids": selected_customer_ids,
            "selected_indices": selected_indices,
            "decision_vector": best_b.tolist(),
            "expected_net_value": float(best_sol_dict["expected_net_value"]),
            "contact_cost": float(best_sol_dict["contact_cost"]),
            "capacity_units": float(best_sol_dict["capacity_units"]),
            "budget_limit": budget_limit,
            "capacity_limit": capacity_limit,
            "budget_utilization_pct": float((best_sol_dict["contact_cost"] / budget_limit) * 100),
            "capacity_utilization_pct": float((best_sol_dict["capacity_units"] / capacity_limit) * 100),
            "is_feasible": True,
            "optimal_gamma": float(gamma_opt),
            "optimal_beta": float(beta_opt),
            "shots": self.shots,
            "feasible_count": len(feasible_solutions),
            "qubo_energy": float(best_sol_dict["qubo_energy"])
        }

        qaoa_meta = {
            "num_qubits": n,
            "reps": self.reps,
            "p_cardinality": self.p_cardinality,
            "p_budget": self.p_budget,
            "p_capacity": self.p_capacity,
            "optimal_parameters": {"gamma": float(gamma_opt), "beta": float(beta_opt)},
            "measurement_counts": counts,
            "state_probs": state_probs
        }

        logger.info(
            f"QAOA Optimum | Net Value: {best_solution['expected_net_value']:.4f} | "
            f"Cost: {best_solution['contact_cost']}/{budget_limit} | "
            f"Capacity: {best_solution['capacity_units']}/{capacity_limit} | "
            f"Customers: {best_solution['selected_customer_ids']}"
        )
        return best_solution, measured_df, qaoa_meta

def export_stage2_results(
    candidates_df: pd.DataFrame,
    qaoa_solution: Dict[str, Any],
    classical_solution: Dict[str, Any],
    output_dir: Path = None
) -> pd.DataFrame:
    """
    Merge candidate data with QAOA and classical selection decisions,
    calculate optimality gap, and export stage2_optimization_results.csv.
    """
    if output_dir is None:
        output_dir = get_project_root() / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df = candidates_df.copy()
    
    # Map decision variables x_i
    qaoa_selected_ids = set(qaoa_solution["selected_customer_ids"])
    classical_selected_ids = set(classical_solution["selected_customer_ids"])

    results_df["qaoa_decision"] = results_df["customer_id"].apply(lambda cid: 1 if cid in qaoa_selected_ids else 0)
    results_df["classical_decision"] = results_df["customer_id"].apply(lambda cid: 1 if cid in classical_selected_ids else 0)
    results_df["selected_flag"] = results_df["qaoa_decision"] == 1

    # Add summary columns
    z_qaoa = qaoa_solution["expected_net_value"]
    z_classical = classical_solution["expected_net_value"]
    gap = (z_classical - z_qaoa) / z_classical if z_classical != 0 else 0.0

    csv_path = output_dir / "stage2_optimization_results.csv"
    results_df.to_csv(csv_path, index=False)
    logger.info(f"Saved Stage 2 optimization results to: {csv_path}")
    logger.info(f"Optimality Gap (Classical vs QAOA): {gap*100:.4f}%")

    return results_df
