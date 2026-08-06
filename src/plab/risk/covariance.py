"""Covariance estimation.

Implements the Ledoit-Wolf (2004) shrinkage towards a scaled identity target. Shrinkage is
a stated methodological choice, not a silent rescue: a degenerate sample still raises.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from plab.risk.metrics import NUMERICAL_ZERO, PERIODS_PER_YEAR


class SingularCovarianceError(Exception):
    """Raised when the covariance of the sample carries no usable information."""


def _centered(returns: pd.DataFrame) -> np.ndarray:
    values = returns.to_numpy(dtype=float)
    return values - values.mean(axis=0)


def sample_covariance(returns: pd.DataFrame) -> pd.DataFrame:
    """Annualized sample covariance matrix."""
    cov = returns.cov(ddof=1) * PERIODS_PER_YEAR
    return cov


def shrinkage_intensity(returns: pd.DataFrame) -> float:
    """Ledoit-Wolf optimal shrinkage weight towards the scaled identity target."""
    x = _centered(returns)
    n_obs, n_assets = x.shape
    if n_obs < 2:
        raise SingularCovarianceError("at least two observations are required")

    sample = x.T @ x / n_obs  # maximum-likelihood covariance
    mu = float(np.trace(sample) / n_assets)
    if mu <= NUMERICAL_ZERO:
        raise SingularCovarianceError("sample covariance has zero trace")

    dispersion = float(np.sum((sample - mu * np.eye(n_assets)) ** 2) / n_assets)
    if dispersion <= NUMERICAL_ZERO:
        raise SingularCovarianceError("sample covariance is exactly the identity target")

    noise = float(
        np.sum([np.sum((np.outer(row, row) - sample) ** 2) for row in x])
        / (n_assets * n_obs**2)
    )
    return float(min(noise, dispersion) / dispersion)


def ledoit_wolf_covariance(returns: pd.DataFrame) -> pd.DataFrame:
    """Annualized covariance shrunk towards a scaled identity matrix."""
    intensity = shrinkage_intensity(returns)
    x = _centered(returns)
    n_obs, n_assets = x.shape
    sample = x.T @ x / n_obs
    mu = float(np.trace(sample) / n_assets)

    shrunk = intensity * mu * np.eye(n_assets) + (1.0 - intensity) * sample
    shrunk *= PERIODS_PER_YEAR
    # Rescale to the unbiased convention used by sample_covariance.
    shrunk *= n_obs / (n_obs - 1)

    # Unreachable by construction: shrinking a positive-semidefinite sample towards a
    # positive-definite target with an intensity in (0, 1] and mu > NUMERICAL_ZERO cannot
    # produce a non-positive-definite result, and the intensity is never exactly zero
    # because the noise term is never exactly zero. Kept as a guard so that a future
    # change to the shrinkage formula fails loudly instead of handing a singular matrix
    # to the optimizers in Tasks 8 and 9.
    if np.any(np.linalg.eigvalsh(shrunk) <= NUMERICAL_ZERO):  # pragma: no cover
        raise SingularCovarianceError("shrunk covariance is not positive definite")
    return pd.DataFrame(shrunk, index=returns.columns, columns=returns.columns)
