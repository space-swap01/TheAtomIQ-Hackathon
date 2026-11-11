"""
Probability Calculation and Monte Carlo Validation Module (Stage 3).

Responsibilities:
- Calculate exact Poisson-Binomial distribution for independent non-identical Bernoulli trials
- Dynamic programming implementation for exact P(K = k) for k in {0, ..., 6}
- Exact campaign success probability: P(K >= 3)
- High-sample Monte Carlo simulation with reproducible seed
- Error analysis and statistical bounds
"""

from typing import Dict, Any, List, Tuple
import numpy as np

from .utils import RANDOM_SEED, setup_logger

logger = setup_logger("ProbabilityCalculation")

class PoissonBinomialCalculator:
    """Exact and numerical distribution calculators for independent trials with distinct probabilities."""

    def __init__(self, probabilities: List[float]):
        self.probabilities = np.array(probabilities, dtype=float)
        self.n = len(self.probabilities)
        assert np.all((self.probabilities >= 0.0) & (self.probabilities <= 1.0)), "Probabilities must be in [0, 1]"

    def compute_exact_distribution(self) -> Tuple[np.ndarray, float]:
        """
        Exact Poisson-Binomial distribution via dynamic programming.
        Returns:
            pmf: array of size n+1 where pmf[k] = P(K = k)
            p_at_least_3: sum(pmf[3:])
        """
        # dp[k] stores P(K = k)
        dp = np.zeros(self.n + 1, dtype=float)
        dp[0] = 1.0

        for p in self.probabilities:
            dp_next = np.zeros(self.n + 1, dtype=float)
            for k in range(self.n + 1):
                # Did not subscribe: prob (1 - p)
                dp_next[k] += dp[k] * (1.0 - p)
                # Subscribed: prob p (adds 1 to success count)
                if k + 1 <= self.n:
                    dp_next[k + 1] += dp[k] * p
            dp = dp_next

        p_at_least_3 = float(np.sum(dp[3:]))
        logger.info(f"Exact Poisson-Binomial: P(K >= 3) = {p_at_least_3:.6f}")
        return dp, p_at_least_3

    def run_monte_carlo(self, num_trials: int = 200000, random_state: int = RANDOM_SEED) -> Tuple[np.ndarray, float, float]:
        """
        Independent Monte Carlo simulation.
        Returns:
            mc_pmf: estimated distribution of k successes
            mc_p_at_least_3: fraction of trials with >= 3 successes
            mc_std_error: standard error of the binomial proportion estimate
        """
        rng = np.random.RandomState(random_state)
        # Shape: (num_trials, n)
        random_draws = rng.rand(num_trials, self.n)
        success_matrix = (random_draws < self.probabilities).astype(int)
        
        # Total successes per trial
        trial_successes = np.sum(success_matrix, axis=1)

        # Compute empirical PMF
        counts = np.bincount(trial_successes, minlength=self.n + 1)
        mc_pmf = counts / num_trials

        mc_p_at_least_3 = float(np.mean(trial_successes >= 3))
        # Standard error: sqrt(p * (1 - p) / N)
        std_err = float(np.sqrt(mc_p_at_least_3 * (1.0 - mc_p_at_least_3) / num_trials))

        logger.info(f"Monte Carlo ({num_trials} trials): P(K >= 3) = {mc_p_at_least_3:.6f} +/- {std_err:.6f}")
        return mc_pmf, mc_p_at_least_3, std_err
