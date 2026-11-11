"""
Classical Benchmark Optimization Module (Stage 2).

Implements exact brute-force combinatorial optimization:
- Evaluates all C(10, 6) = 210 combinations of candidate customers
- Computes expected net value, total contact cost, and total capacity units
- Evaluates hard constraints: SUM(cost) <= 25 and SUM(capacity) <= 22
- Identifies exact global optimal solution and full ranked combinations
"""

import itertools
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np

from .utils import setup_logger

logger = setup_logger("ClassicalOptimizer")

class ClassicalBruteForceOptimizer:
    """Exact combinatorial solver enumerating all candidate subsets."""

    def __init__(self, budget_limit: float = 25.0, capacity_limit: float = 22.0, k_select: int = 6):
        self.budget_limit = budget_limit
        self.capacity_limit = capacity_limit
        self.k_select = k_select

    def solve(self, candidates_df: pd.DataFrame) -> Tuple[Dict[str, Any], pd.DataFrame]:
        """
        Enumerate all combinations of k_select from candidates_df.
        Returns:
            best_solution: Dict of optimal customer IDs, value, cost, capacity
            all_combos_df: DataFrame ranking all 210 combinations
        """
        assert len(candidates_df) == 10, f"Expected 10 candidates, got {len(candidates_df)}"
        
        records = []
        n = len(candidates_df)
        indices = list(range(n))

        for comb in itertools.combinations(indices, self.k_select):
            subset = candidates_df.iloc[list(comb)]
            tot_val = subset['expected_net_value'].sum()
            tot_cost = subset['contact_cost'].sum()
            tot_cap = subset['capacity_units'].sum()
            
            is_budget_feasible = tot_cost <= self.budget_limit
            is_cap_feasible = tot_cap <= self.capacity_limit
            is_feasible = is_budget_feasible and is_cap_feasible

            records.append({
                "combination_indices": comb,
                "customer_ids": subset['customer_id'].tolist(),
                "expected_net_value": tot_val,
                "contact_cost": tot_cost,
                "capacity_units": tot_cap,
                "budget_feasible": is_budget_feasible,
                "capacity_feasible": is_cap_feasible,
                "is_feasible": is_feasible
            })

        all_combos_df = pd.DataFrame(records)
        # Sort by feasibility (True first), then expected_net_value descending
        all_combos_df = all_combos_df.sort_values(
            by=["is_feasible", "expected_net_value"],
            ascending=[False, False]
        ).reset_index(drop=True)

        # Optimal feasible solution
        feasible_df = all_combos_df[all_combos_df["is_feasible"]]
        if feasible_df.empty:
            raise RuntimeError("No feasible combination found within budget and capacity limits!")

        best_row = feasible_df.iloc[0]
        best_solution = {
            "method": "Classical Brute Force",
            "selected_customer_ids": best_row["customer_ids"],
            "selected_indices": list(best_row["combination_indices"]),
            "expected_net_value": float(best_row["expected_net_value"]),
            "contact_cost": float(best_row["contact_cost"]),
            "capacity_units": float(best_row["capacity_units"]),
            "budget_limit": self.budget_limit,
            "capacity_limit": self.capacity_limit,
            "budget_utilization_pct": float((best_row["contact_cost"] / self.budget_limit) * 100),
            "capacity_utilization_pct": float((best_row["capacity_units"] / self.capacity_limit) * 100),
            "is_feasible": True,
            "total_evaluated": len(all_combos_df),
            "total_feasible": len(feasible_df)
        }

        logger.info(
            f"Classical Optimum | Net Value: {best_solution['expected_net_value']:.4f} | "
            f"Cost: {best_solution['contact_cost']}/{self.budget_limit} | "
            f"Capacity: {best_solution['capacity_units']}/{self.capacity_limit} | "
            f"Customers: {best_solution['selected_customer_ids']}"
        )
        return best_solution, all_combos_df
